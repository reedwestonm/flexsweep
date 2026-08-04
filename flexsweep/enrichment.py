import os
import sys
import time

import polars_bio as pb
from numba import njit

from . import Parallel, delayed, np, pl


def compute_distances(vip_gene_ids: list, annotation_file: str) -> pl.DataFrame:
    """
    Compute distance from each gene to its nearest VIP (case) gene.

    Parameters
    ----------
    vip_gene_ids    : list of VIP gene ID strings
    annotation_file : path to BED file (chrom  start  end  gene_id  strand), tab-separated, no header

    Returns
    -------
    pl.DataFrame with columns: gene_id, distance
    """
    case_genes = set(vip_gene_ids)

    df_annotation = pl.read_csv(
        annotation_file,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "gene_id", "strand"],
        schema_overrides={"chrom": pl.Utf8},
    )

    # Use gene centre as a 1-bp point feature
    df_annotation = df_annotation.with_columns(
        [
            ((pl.col("start") + pl.col("end")) // 2).cast(pl.Int64).alias("start"),
            ((pl.col("start") + pl.col("end")) // 2 + 1).cast(pl.Int64).alias("end"),
        ]
    )

    df_cases_annot = df_annotation.filter(pl.col("gene_id").is_in(case_genes))

    df_all_pts = df_annotation.select(["chrom", "start", "end", "gene_id"])
    df_case_pts = df_cases_annot.select(["chrom", "start", "end", "gene_id"])
    df_all_pts.config_meta.set(coordinate_system_zero_based=True)
    df_case_pts.config_meta.set(coordinate_system_zero_based=True)

    result = (
        pb.nearest(
            df_all_pts,
            df_case_pts,
            cols1=["chrom", "start", "end"],
            cols2=["chrom", "start", "end"],
            suffixes=("_1", "_2"),
            output_type="polars.LazyFrame",
        )
        .select(
            [
                pl.col("gene_id_1").alias("gene_id"),
                pl.col("distance"),
            ]
        )
        .collect()
    )

    # polars_bio.nearest() returns gap distance (b−a−1); closestBed -d returns
    # start-to-start distance (b−a). Add 1 for non-overlapping pairs to match Perl.
    result = result.with_columns(
        pl.when(pl.col("distance") > 0)
        .then(pl.col("distance") + 1)
        .otherwise(pl.col("distance"))
        .alias("distance")
    )

    return result.sort("gene_id")


@njit(cache=True)
def iterative_control_set_fast(
    vip_X: np.ndarray,
    nonvip_X: np.ndarray,
    vip_number: int,
    tolerance: float,
    max_rep: int,
    seed: int = None,
    skip_factor_idx: int = 8,
    n_batches: int = 10,
) -> np.ndarray:
    """
    JIT-compiled bootstrap builder for the nomatchomega_fast variant.

    Runs n_batches sequential internal batches with **continuous state**,
    matching Perl's within-process behaviour exactly:
      - used_counts, current_means, fake_seed, sc_gn persist across batches.
      - Target for batch r: 100 + 11 * vip_number * r  (grows each batch).
      - 10 sets sliced from the end of the list after each batch.

    Additional semantic differences vs iterative_control_set (sweep_count.py):
      1. Factor skip_factor_idx (default 8, dN/dS) is unconstrained (±1e18).
      2. The +2 offset on factor 27 has already been applied by the caller.

    Parameters
    ----------
    vip_X           : (vip_number, n_factors) float64 — offset pre-applied
    nonvip_X        : (n_nonvips,  n_factors) float64 — offset pre-applied
    vip_number      : number of VIP genes
    tolerance       : ± fraction for confounder matching (e.g. 0.05)
    max_rep         : dynamic rep-limit numerator
    seed            : RNG seed for this parallel call
    skip_factor_idx : 0-based factor index to leave unconstrained
    n_batches       : number of sequential internal batches (≡ Perl Iterations_number/10)

    Returns
    -------
    np.ndarray of shape (n_batches * 10, vip_number), dtype int64.
    Each row is one control set of nonvip_X row indices.
    """
    n_nonvips = nonvip_X.shape[0]
    n_factors = nonvip_X.shape[1]

    # Compute VIP mean vector.
    vip_vec = np.zeros(n_factors)
    for i in range(vip_X.shape[0]):
        for f in range(n_factors):
            vip_vec[f] += vip_X[i, f]
    for f in range(n_factors):
        vip_vec[f] /= vip_X.shape[0]

    # Tolerance bounds; skip_factor_idx is unconstrained.
    inf_vec = (1.0 - tolerance) * vip_vec
    sup_vec = (1.0 + tolerance) * vip_vec
    inf_vec[skip_factor_idx] = -1e18
    sup_vec[skip_factor_idx] = 1e18

    init_fake = 100

    # Preallocate for the full continuous list across all batches.
    capacity = (init_fake + 11 * vip_number * n_batches) * 2
    good_idx = np.empty(capacity, dtype=np.int64)

    # Continuous state — persists across all n_batches (matches Perl).
    used_counts = np.zeros(n_nonvips, dtype=np.int32)
    current_means = vip_vec.copy()
    fake_seed = 100
    sc_gn = 0

    np.random.seed(seed)

    out = np.empty((n_batches * 10, vip_number), dtype=np.int64)
    out_row = 0

    for r in range(1, n_batches + 1):
        target_len = init_fake + 11 * vip_number * r  # grows each batch

        while sc_gn < target_len:
            # if sc_gn % 1000 == 0:
            #     print("batch", r, "progress", sc_gn, "/", target_len)
            i1 = np.random.randint(0, n_nonvips)
            i2 = np.random.randint(0, n_nonvips)
            while i2 == i1:
                i2 = np.random.randint(0, n_nonvips)

            rep_limit = max_rep * (sc_gn + 1) / vip_number
            if used_counts[i1] >= rep_limit or used_counts[i2] >= rep_limit:
                continue

            denom = sc_gn + 2 + fake_seed
            ok = True
            for f in range(n_factors):
                sub = (nonvip_X[i1, f] + nonvip_X[i2, f]) * 0.5
                new_m = (
                    fake_seed * vip_vec[f] + current_means[f] * sc_gn + sub * 2.0
                ) / denom
                if new_m < inf_vec[f] or new_m > sup_vec[f]:
                    ok = False
                    break

            if ok:
                if sc_gn + 2 > capacity:
                    new_cap = capacity * 2
                    new_buf = np.empty(new_cap, dtype=np.int64)
                    new_buf[:sc_gn] = good_idx[:sc_gn]
                    good_idx = new_buf
                    capacity = new_cap

                good_idx[sc_gn] = i1
                good_idx[sc_gn + 1] = i2
                used_counts[i1] += 1
                used_counts[i2] += 1

                for f in range(n_factors):
                    sub = (nonvip_X[i1, f] + nonvip_X[i2, f]) * 0.5
                    current_means[f] = (
                        fake_seed * vip_vec[f] + current_means[f] * sc_gn + sub * 2.0
                    ) / denom

                sc_gn += 2
                if fake_seed > 0:
                    fake_seed -= 1

        # Slice 10 sets from the end of good_idx for batch r
        total = sc_gn
        n_sets = 0
        for p in range(10):
            if total - (p + 1) * vip_number < 0:
                break
            n_sets += 1

        for p in range(n_sets):
            sup = total - p * vip_number
            inf_i = sup - vip_number
            for j in range(vip_number):
                out[out_row + (n_sets - 1 - p), j] = good_idx[inf_i + j]
        out_row += n_sets

    return out[:out_row]


def run_bootstrap_nomatchomega(
    case_genes: list,
    factors_file: str,
    annotation_file: str,
    runs_number: int,
    tolerance: float,
    min_dist: float,
    flip: bool,
    max_rep: int,
    seed: int = None,
    nthreads: int = 1,
    control_genes: list = None,
    distance_file: str = None,
    skip_factor_idx: int = 8,
    offset_factor_idx: int = 27,
    offset_value: float = 2.0,
    n_batches: int = 10,
):
    """
    Step 6: nomatchomega_fast variant of the matched bootstrap control set
    generator.

    Implements the behaviour of bootstrap_test_script_nomatchomega_fast.pl:
      - Factor skip_factor_idx (default 8, omega/dN/dS) is unconstrained.
      - Factor offset_factor_idx (default 27, Perl column 28) gets +offset_value.
      - n_batches sequential internal batches per parallel call, with continuous
        state (used_counts, current_means, fake_seed persist across batches),
        matching Perl's within-process continuous state.
      - Target for batch r: 100 + 11 * vip_number * r (grows each batch).
      - Total control sets = runs_number * n_batches * 10.

    Parameters
    ----------
    case_genes        : list of VIP gene IDs (HLA/histone excluded by caller)
    factors_file      : path to confounding factors table (TSV, first col gene_id)
    annotation_file   : path to BED gene coordinates file (chrom start end gene_id strand). Distances to nearest VIP are computed automatically via compute_distances.
    runs_number       : number of independent parallel calls (≡ Perl Runs_number)
    tolerance         : allowed ± fraction for confounder matching (e.g. 0.05)
    min_dist          : minimum distance (bp) from VIPs for control eligibility
    flip              : if True, swap VIP and control pools
    max_rep           : max times a control gene may be sampled on average
    nthreads          : number of parallel workers
    control_genes     : optional explicit candidate control gene IDs. If None, all genes in factors_file that are not case genes are used. Pass the "no"-labelled genes from genes_set_file to match Perl behaviour exactly.
    distance_file     : optional path to pre-computed distance TSV (gene_id \\t distance, no header). When provided, distances are read directly instead of calling compute_distances(). Use Perl's distance_genes_set_file.txt for exact control pool matching.
    skip_factor_idx   : 0-based factor index to leave unconstrained (default 8 = Perl factor 9, dN/dS / omega).
    offset_factor_idx : 0-based factor index to apply +offset_value to (default 27 = Perl column 28).
    offset_value      : value added to offset_factor_idx column (default 2.0).
    n_batches         : sequential internal batches per parallel call (≡ Perl Iterations_number/10; default 10).

    Returns
    -------
    df_set               : pl.DataFrame — VIP genes (gene_id column)
    df_bootstrap_control : pl.DataFrame — rows = control sets, cols = gene positions (sample_id + column_0..column_N)
    """
    case_ids = set(case_genes)

    df_factors = pl.read_csv(factors_file, separator=" ", has_header=False).rename(
        {"column_1": "gene_id"}
    )

    # Perl checks index 19 of the raw line. In Polars, this is column_20.
    df_factors = (
        pl.read_csv(factors_file, separator=" ", has_header=False)
        .rename({"column_1": "gene_id"})
        .filter(pl.col("column_20") >= -1e21)
    )

    # Background pool
    if control_genes is not None:
        background_ids = set(control_genes) - case_ids
    else:
        background_ids = set(df_factors["gene_id"].to_list()) - case_ids

    if distance_file is not None:
        distance_df = pl.read_csv(
            distance_file,
            separator="\t",
            has_header=False,
            new_columns=["gene_id", "distance"],
        )
    else:
        distance_df = compute_distances(case_genes, annotation_file)
    set_by_dist = set(
        distance_df.filter(pl.col("distance") >= min_dist)["gene_id"].to_list()
    )

    df_set = df_factors.filter(pl.col("gene_id").is_in(case_ids))
    df_control_pool = df_factors.filter(
        pl.col("gene_id").is_in(background_ids) & pl.col("gene_id").is_in(set_by_dist)
    )

    if flip:
        df_set, df_control_pool = df_control_pool, df_set

    factor_cols = [c for c in df_factors.columns if c != "gene_id"]
    control_factors = df_control_pool.select(["gene_id"] + factor_cols)

    vip_X = df_set.select(factor_cols).to_numpy().astype(np.float64)
    nonvip_X = control_factors.select(factor_cols).to_numpy().astype(np.float64)
    nonvip_genes = control_factors["gene_id"].to_numpy().astype(str)
    vip_number = df_set.height

    # Apply offset once in Python — shared read-only across all parallel workers.
    if 0 <= offset_factor_idx < vip_X.shape[1]:
        vip_X_off = vip_X.copy()
        nonvip_X_off = nonvip_X.copy()
        vip_X_off[:, offset_factor_idx] += offset_value
        nonvip_X_off[:, offset_factor_idx] += offset_value
    else:
        vip_X_off = vip_X
        nonvip_X_off = nonvip_X

    print(
        f"There are {df_set.height} genes of interest and {df_control_pool.height} potential control genes at distance of at least {min_dist} bases."
    )

    if df_control_pool.height <= 1.5 * df_set.height:
        print(
            "The number of control genes is less than 1.5× the number of VIPs. FDR may be high."
        )

    def _run_one_batch(run_idx: int) -> np.ndarray:
        _batch_seed = int(
            np.random.default_rng(None if seed is None else seed + run_idx).integers(
                0, 2**31
            )
        )
        idx = iterative_control_set_fast(
            vip_X_off,
            nonvip_X_off,
            vip_number,
            tolerance,
            max_rep,
            _batch_seed,
            skip_factor_idx,
            n_batches,
        )
        return nonvip_genes[idx]

    batch_results = Parallel(n_jobs=nthreads, backend="loky", verbose=10)(
        delayed(_run_one_batch)(run_idx) for run_idx in range(runs_number)
    )

    np_control = np.concatenate(batch_results, axis=0)
    df_bootstrap_control = pl.DataFrame(np_control)
    df_bootstrap_control = df_bootstrap_control.with_columns(
        (
            pl.lit("sample_")
            + pl.arange(1, df_bootstrap_control.height + 1).cast(pl.Utf8)
        ).alias("sample_id")
    ).select("sample_id", *df_bootstrap_control.columns)

    return df_set.select("gene_id"), df_bootstrap_control


def build_gene_neighbors(coord_file: str, valid_genes: list, dist: int) -> pl.DataFrame:
    """
    Find all gene neighbours within ±dist bp using polars-bio overlap.

    Uses the full gene body intervals (not centre approximation). Each gene's
    interval is expanded by dist on both sides, then overlapped against all
    gene centre points — equivalent to the original pybedtools bed.window(w=dist).

    Parameters
    ----------
    coord_file   : path to BED gene coordinates (chrom  start  end  gene_id  strand)
    valid_genes  : list of gene IDs to include (QC-passing universe)
    dist         : window radius in bp (e.g. 500_000)

    Returns
    -------
    pl.DataFrame with columns: gene_id, neighbors (space-joined string)
    """
    valid = set(valid_genes)

    df_coords = (
        pl.read_csv(
            coord_file,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end", "gene_id", "strand"],
            schema_overrides={"chrom": pl.Utf8},
        )
        .with_columns(
            [
                pl.col("start").cast(pl.Int64),
                pl.col("end").cast(pl.Int64),
            ]
        )
        .filter(pl.col("gene_id").is_in(valid))
        .with_columns(((pl.col("start") + pl.col("end")) // 2).alias("center"))
    )

    # Expand each gene body by ±dist (pybedtools window equivalent)
    df_windows = df_coords.select(
        [
            pl.col("chrom"),
            (pl.col("start") - dist).alias("win_start"),
            (pl.col("end") + dist).alias("win_end"),
            pl.col("gene_id"),
        ]
    )

    # Target: 1-bp gene centres for overlap detection
    df_centers = df_coords.select(
        [
            pl.col("chrom"),
            pl.col("center").alias("start"),
            (pl.col("center") + 1).alias("end"),
            pl.col("gene_id"),
        ]
    )

    df_windows.config_meta.set(coordinate_system_zero_based=True)
    df_centers.config_meta.set(coordinate_system_zero_based=True)

    pairs = (
        pb.overlap(
            df_windows,
            df_centers,
            cols1=["chrom", "win_start", "win_end"],
            cols2=["chrom", "start", "end"],
            suffixes=("_1", "_2"),
            output_type="polars.LazyFrame",
        )
        .select(
            [
                pl.col("gene_id_1").alias("gene_id"),
                pl.col("gene_id_2").alias("neighbor"),
            ]
        )
        .filter(pl.col("gene_id") != pl.col("neighbor"))
        .collect()
    )

    df_neighbors = pairs.group_by("gene_id").agg(
        pl.col("neighbor").sort().str.join(" ").alias("neighbors")
    )

    df_all_genes = df_coords.select("gene_id")
    return (
        df_all_genes.join(df_neighbors, on="gene_id", how="left")
        .with_columns(pl.col("neighbors").fill_null(""))
        .sort("gene_id")
    )


def build_gene_neighbors_numpy(
    coord_file: str, valid_genes: list, dist: int
) -> pl.DataFrame:
    """
    Find all gene neighbours within ±dist bp using pure NumPy (10 kb binning).

    Gene centres are binned to a 10 kb grid. For each gene, numpy.searchsorted
    finds all genes whose binned centre falls within [center - dist, center + dist].

    Parameters
    ----------
    coord_file  : path to BED gene coordinates (chrom  start  end  gene_id  strand)
    valid_genes : list of gene IDs to include (QC-passing universe)
    dist        : neighbourhood radius in bp (e.g. 500_000)

    Returns
    -------
    pl.DataFrame with columns: gene_id, neighbors (space-joined string)
    """
    valid = set(valid_genes)

    df = (
        pl.read_csv(
            coord_file,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end", "gene_id", "strand"],
            schema_overrides={"chrom": pl.Utf8},
        )
        .with_columns(
            [
                pl.col("start").cast(pl.Int64),
                pl.col("end").cast(pl.Int64),
            ]
        )
        .filter(pl.col("gene_id").is_in(valid))
        .with_columns(((pl.col("start") + pl.col("end")) // 2).alias("center"))
        .with_columns(((pl.col("center") / 10000).floor() * 10000).alias("bin"))
        .select(["chrom", "bin", "gene_id"])
        .sort(["chrom", "bin"])
    )

    results = []
    for chrom, sub in df.group_by("chrom", maintain_order=True):
        bins = sub["bin"].to_numpy()
        genes = sub["gene_id"].to_list()
        n = len(genes)
        left_idx = np.searchsorted(bins, bins - dist, side="left")
        right_idx = np.searchsorted(bins, bins + dist, side="right")
        for i in range(n):
            neighbors = [g for g in genes[left_idx[i] : right_idx[i]] if g != genes[i]]
            results.append((genes[i], " ".join(neighbors)))

    return pl.DataFrame(results, schema=["gene_id", "neighbors"], orient="row").sort(
        "gene_id"
    )


def simplify_sweeps_gz(
    rank_file: str,
    thresholds: list,
    col_index: int = 1,
    df: "pl.DataFrame | None" = None,
) -> pl.DataFrame:
    """
    Assign each gene the most stringent threshold it passes (smallest cutoff c
    such that rank <= c).

    Parameters
    ----------
    rank_file  : TSV with gene_id in column 0 and rank values in subsequent
                 columns (no header).  For single-population files the rank is
                 in column 1.  For multi-population files (e.g. all_ihsfreqafr
                 with ESN/GWD/LWK/MSL/YRI in columns 1-5) pass col_index to
                 select the population column (1-based, default 1).
    thresholds : list of int rank thresholds
    col_index  : 1-based index of the rank column to use (default 1 = second
                 column, i.e. first rank column after gene_id).
    df         : optional pre-loaded pl.DataFrame (same column layout as the file).
                 When provided, rank_file is ignored and df is used directly —
                 enables in-memory shuffle workflows without disk I/O.

    Returns
    -------
    pl.DataFrame with columns: gene_id, selected_cutoff (genes with no hit excluded)
    """
    if df is not None:
        df_raw = df
    else:
        # Auto-detect delimiter: try tab first, fall back to space.
        # (AFR/pop rank files use space; some single-pop files may use tab.)
        import gzip as _gzip

        _open = _gzip.open if rank_file.endswith(".gz") else open
        with _open(rank_file, "rt") as _fh:
            _raw = _fh.readline()
        _sep = "\t" if "\t" in _raw else " "
        df_raw = pl.read_csv(
            rank_file,
            separator=_sep,
            has_header=False,
            infer_schema_length=0,  # read all as Utf8 first to handle variable cols
        )
    col_name = df_raw.columns[col_index]
    df = df_raw.select(
        [
            pl.col(df_raw.columns[0]).alias("gene_id").cast(pl.Utf8),
            pl.col(col_name).alias("rank").cast(pl.Int64),
        ]
    )
    cuts = np.sort(np.array(thresholds, dtype=int))

    expr = None
    for c in cuts:
        if expr is None:
            expr = pl.when(pl.col("rank") <= c).then(pl.lit(c))
        else:
            expr = expr.when(pl.col("rank") <= c).then(pl.lit(c))

    return (
        df.with_columns(expr.otherwise(None).alias("selected_cutoff"))
        .drop_nulls("selected_cutoff")
        .select(["gene_id", "selected_cutoff"])
    )


def simplify_sweeps_gz_list(rank_file: str, thresholds: list) -> pl.DataFrame:
    """
    For a single-population rank file, return all thresholds each gene passes
    as a comma-separated string.

    Parameters
    ----------
    rank_file  : TSV with columns gene_id, rank (no header)
    thresholds : list of int rank thresholds

    Returns
    -------
    pl.DataFrame with columns: gene_id, cutoffs (comma-separated string)
    """
    df = pl.read_csv(
        rank_file, separator="\t", has_header=False, new_columns=["gene_id", "rank"]
    )
    cuts = np.sort(thresholds)[::-1]

    df_cut = df.with_columns(
        pl.when(pl.col("rank").is_not_null())
        .then(
            pl.struct(pl.col("rank")).map_elements(
                lambda s: ",".join(str(c) for c in cuts if s["rank"] <= c),
                return_dtype=pl.Utf8,
            )
        )
        .alias("cutoffs")
    )
    return df_cut.filter(pl.col("cutoffs") != "")


def _count_sweep_events(query_genes: set, sweep_genes: set, neighbor_map: dict) -> int:
    """
    Count distinct sweep events = connected components of the neighbour graph
    restricted to `sweep_genes`, that are touched by at least one gene from
    `query_genes`.

    Each connected component in the sub-graph of sweep_genes (using neighbor_map
    edges) is a single sweep event.  We count components that share at least one
    gene with query_genes.
    """
    visited: set = set()
    count = 0
    for g in query_genes:
        if g not in sweep_genes or g in visited:
            continue
        # DFS to find the connected component of g within sweep_genes
        stack = [g]
        while stack:
            node = stack.pop()
            if node in visited or node not in sweep_genes:
                continue
            visited.add(node)
            stack.extend(neighbor_map.get(node, set()) - visited)
        count += 1
    return count


def count_sweeps_singlepop(
    vip_genes: list,
    control_sets: list,
    simplified_sweeps: pl.DataFrame,
    neighbors: pl.DataFrame,
    thresholds: list,
    use_clust: bool = True,
    count_sweeps: bool = False,
    neighbor_col: str = None,
) -> pl.DataFrame:
    """
    Count sweeps overlapping VIPs vs control sets for a single population.

    Parameters
    ----------
    vip_genes        : list of VIP gene ID strings
    control_sets     : list of lists of control gene IDs (one list per control set)
    simplified_sweeps: pl.DataFrame from simplify_sweeps_gz (gene_id, selected_cutoff)
    neighbors        : pl.DataFrame from build_gene_neighbors_numpy (gene_id, neighbors)
    thresholds       : sorted list of int rank thresholds
    use_clust        : expand gene sets by neighbours before counting (de-duplication)
    count_sweeps     : if True, count distinct sweep events (connected components of the neighbour graph within sweep genes) rather than individual genes.
    neighbor_col     : column name for neighbours (auto-detected if None)

    Returns
    -------
    pl.DataFrame with columns: threshold, vip_count, ctrl_mean, ctrl_ci_lo, ctrl_ci_hi, ratio, ci_lo_ratio, ci_hi_ratio, p_value
    """
    cuts = sorted(int(x) for x in thresholds)
    _sw = simplified_sweeps.select(["gene_id", "selected_cutoff"])
    sel_map = dict(zip(_sw["gene_id"].to_list(), _sw["selected_cutoff"].to_list()))
    neighbor_map: dict = {}
    if use_clust or count_sweeps:
        if neighbor_col is None:
            if "neighbors" in neighbors.columns:
                neighbor_col = "neighbors"
            elif "gene_id_b" in neighbors.columns:
                neighbor_col = "gene_id_b"
        if neighbor_col and neighbor_col in neighbors.columns:
            for row in neighbors.iter_rows(named=True):
                raw = row.get(neighbor_col, None)
                neighbor_map[row["gene_id"]] = set(raw.split(" ")) if raw else set()

    def expand(ids: set) -> set:
        if not use_clust or not neighbor_map:
            return ids
        visited: set = set()
        expanded: set = set()
        for g in ids:
            if g in visited:
                continue
            cluster = neighbor_map.get(g, set()) | {g}
            expanded |= cluster
            visited |= cluster
        return expanded

    vip_set = expand(set(vip_genes))

    # Keep raw lists for counting (duplicates matter — Perl bootstrap samples with
    # replacement and scores = sum of repetition counts for sweep genes in ctrl list).
    # Only the set-expanded version is needed for use_clust / count_sweeps paths.
    ctrl_lists = [list(ctrl) for ctrl in control_sets]  # raw lists with dups
    ctrl_sets_expanded = [expand(set(ctrl)) for ctrl in control_sets]  # for clust paths

    # Vectorized path for the common case (use_clust=False, count_sweeps=False):
    # build a numpy int32 matrix once, then use boolean indexing per threshold.
    # This replaces O(n_sets × n_genes × n_thresholds) Python loops with numpy ops.
    if not use_clust and not count_sweeps and ctrl_lists:
        _all_genes = sorted({g for ctrl in ctrl_lists for g in ctrl})
        _g2i = {g: i for i, g in enumerate(_all_genes)}
        _n_vocab = len(_all_genes)
        _ctrl_matrix = np.array(
            [[_g2i[g] for g in ctrl] for ctrl in ctrl_lists], dtype=np.int32
        )  # shape: (n_ctrl_sets, n_genes_per_set)
    else:
        _ctrl_matrix = None
        _g2i = None
        _n_vocab = 0

    rows = []
    if not use_clust and not count_sweeps and _ctrl_matrix is not None:
        # === FAST PATH: batch all thresholds in one tensor operation ===
        # Build cutoff vector aligned to control-gene vocab (_g2i)
        _cutoff_arr = np.full(_n_vocab, np.iinfo(np.int32).max, dtype=np.int32)
        for g, c in sel_map.items():
            if g in _g2i:
                _cutoff_arr[_g2i[g]] = int(c)

        # VIP cutoffs from sel_map directly (independent of _g2i — covers all VIPs)
        _vip_sorted = sorted(vip_set)
        _vip_cutoffs = np.array(
            [int(sel_map.get(g, np.iinfo(np.int32).max)) for g in _vip_sorted],
            dtype=np.int32,
        )

        _thresh_arr = np.array(cuts, dtype=np.int32)  # (T,)
        _sweep_mask = _cutoff_arr[None, :] <= _thresh_arr[:, None]  # (T, V) bool
        _all_ctrl = _sweep_mask[:, _ctrl_matrix].sum(axis=2)  # (T, S)
        _all_vip = (_vip_cutoffs[None, :] <= _thresh_arr[:, None]).sum(axis=1)  # (T,)
        _all_means = _all_ctrl.mean(axis=1)  # (T,)
        _all_ci = np.percentile(_all_ctrl, [2.5, 97.5], axis=1)  # (2, T)
        _all_pvals = (_all_ctrl > _all_vip[:, None]).mean(axis=1)  # (T,)

        for t_idx, t in enumerate(cuts):
            vip_count = int(_all_vip[t_idx])
            ctrl_mean = float(_all_means[t_idx])
            ci_low = float(_all_ci[0, t_idx])
            ci_high = float(_all_ci[1, t_idx])
            p_val = float(_all_pvals[t_idx])
            denom = ctrl_mean + 0.1
            rows.append(
                {
                    "threshold": t,
                    "vip_count": vip_count,
                    "ctrl_mean": ctrl_mean,
                    "ctrl_ci_lo": ci_low,
                    "ctrl_ci_hi": ci_high,
                    "ratio": (vip_count + 0.1) / denom,
                    "ci_lo_ratio": (ci_low + 0.1) / denom,
                    "ci_hi_ratio": (ci_high + 0.1) / denom,
                    "p_value": p_val,
                }
            )
    else:
        # === SLOW PATHS: per-threshold loop (use_clust=True or count_sweeps=True) ===
        sweep_dict = {t: {g for g, c in sel_map.items() if c <= t} for t in cuts}
        for t in cuts:
            S = sweep_dict[t]
            if count_sweeps:
                ctrl_counts = [
                    _count_sweep_events(C, S, neighbor_map) for C in ctrl_sets_expanded
                ]
                vip_count = _count_sweep_events(vip_set, S, neighbor_map)
            else:
                # use_clust=True: expand by neighbours, count unique genes
                vip_count = len(vip_set & S)
                ctrl_counts = [len(C & S) for C in ctrl_sets_expanded]

            ctrl_counts_arr = np.asarray(ctrl_counts)
            ctrl_mean = float(np.mean(ctrl_counts_arr)) if ctrl_counts_arr.size else 0.0
            if ctrl_counts_arr.size > 1:
                ci_low, ci_high = np.percentile(ctrl_counts_arr, [2.5, 97.5])
            else:
                ci_low = ci_high = ctrl_mean
            denom = ctrl_mean + 0.1
            p_val = (
                float(np.mean(ctrl_counts_arr > vip_count))
                if ctrl_counts_arr.size > 0
                else float("nan")
            )
            rows.append(
                {
                    "threshold": t,
                    "vip_count": vip_count,
                    "ctrl_mean": ctrl_mean,
                    "ctrl_ci_lo": float(ci_low),
                    "ctrl_ci_hi": float(ci_high),
                    "ratio": float((vip_count + 0.1) / denom),
                    "ci_lo_ratio": float((ci_low + 0.1) / denom),
                    "ci_hi_ratio": float((ci_high + 0.1) / denom),
                    "p_value": p_val,
                }
            )

    return pl.DataFrame(rows)


def _resolve_target_pops(pop_interest: str, populations: list, groups: list) -> list:
    """Return the list of population names matching pop_interest."""
    if pop_interest == "All":
        return populations
    if pop_interest in populations:
        return [pop_interest]
    # treat as group name
    return [p for p, g in zip(populations, groups) if g == pop_interest]


def _group_sweep_sets(
    pops_in_group: list,
    simplified_sweeps_by_pop: dict,
    thresholds: list,
) -> pl.DataFrame:
    """
    Build a unified simplified-sweeps DataFrame for a group of populations:
    a gene is 'in sweep at threshold t' if it is in ANY member population's sweep set.
    The selected_cutoff is the minimum (most stringent) cutoff across member pops.
    """
    cuts = sorted(int(x) for x in thresholds)
    # For each gene collect the minimum selected_cutoff across member pops
    gene_cutoff: dict = {}
    for pop in pops_in_group:
        df_pop = simplified_sweeps_by_pop.get(pop)
        if df_pop is None:
            continue
        for gene_id, cutoff in df_pop.select(
            ["gene_id", "selected_cutoff"]
        ).iter_rows():
            if gene_id not in gene_cutoff or cutoff < gene_cutoff[gene_id]:
                gene_cutoff[gene_id] = cutoff
    if not gene_cutoff:
        return pl.DataFrame(
            {"gene_id": [], "selected_cutoff": []},
            schema={"gene_id": pl.Utf8, "selected_cutoff": pl.Int64},
        )
    return pl.DataFrame(
        list(gene_cutoff.items()),
        schema=["gene_id", "selected_cutoff"],
        orient="row",
    )


def count_sweeps_multipop(
    vip_genes: list,
    control_sets: list,
    simplified_sweeps_by_pop: dict,
    neighbors: pl.DataFrame,
    thresholds: list,
    populations: list,
    groups: list,
    pop_interest: str,
    count_sweeps: bool = False,
    use_clust: bool = True,
    nthreads: int = 1,
) -> pl.DataFrame:
    """
    Step 7: Multi-population sweep counting with group aggregation and adaptive depth.

    Populations are processed in parallel. For a group or 'All', the sweep sets
    of member populations are unioned before counting (each sweep counted once per
    group, not once per population). Adaptive depth escalates from 100 → 1000 →
    all control sets depending on observed p-values.

    Parameters
    ----------
    vip_genes               : list of VIP gene IDs
    control_sets            : list of lists of control gene IDs (up to 10 000)
    simplified_sweeps_by_pop: dict mapping population name → pl.DataFrame (output of simplify_sweeps_gz per pop)
    neighbors               : pl.DataFrame from build_gene_neighbors_numpy
    thresholds              : list of int rank thresholds
    populations             : ordered list of population codes
    groups                  : group label for each population (same length)
    pop_interest            : single pop name, group name, or "All"
    count_sweeps            : if True count distinct sweep events (connected components of the neighbour graph within sweep genes); if False count genes in sweeps (default; matches Perl behaviour)
    use_clust               : expand gene sets by neighbours before counting (de-duplication; maps to Perl Count_sweeps parameter)
    nthreads                : parallel workers

    Returns
    -------
    pl.DataFrame with columns: scope, threshold, vip_count, ctrl_mean, ctrl_ci_lo, ctrl_ci_hi, ratio, ci_lo_ratio, ci_hi_ratio, p_value. scope is the population name, group name, or "All".
    """
    target_pops = _resolve_target_pops(pop_interest, populations, groups)
    if not target_pops:
        raise ValueError(
            f"pop_interest '{pop_interest}' not found in populations or groups"
        )

    # Determine scopes to evaluate:
    # - each individual target population
    # - each group that appears among target pops (if > 1 pop in group)
    # - "All" if pop_interest == "All" or covers multiple groups
    # Group/All scopes use gene union across member pops (min cutoff), which matches
    # the Perl recurrence-weighted formula: each gene counted once regardless of
    # how many populations it appears in.
    scopes: dict = {}  # scope_name → simplified_sweeps df

    for pop in target_pops:
        if pop in simplified_sweeps_by_pop:
            scopes[pop] = simplified_sweeps_by_pop[pop]

    # Group-level scopes
    group_to_pops: dict = {}
    for pop in target_pops:
        g = groups[populations.index(pop)]
        group_to_pops.setdefault(g, []).append(pop)

    for grp, members in group_to_pops.items():
        if len(members) > 1:
            scopes[grp] = _group_sweep_sets(
                members, simplified_sweeps_by_pop, thresholds
            )

    # "All" scope
    if pop_interest == "All" and len(target_pops) > 1:
        scopes["All"] = _group_sweep_sets(
            target_pops, simplified_sweeps_by_pop, thresholds
        )

    # Adaptive depth rounds: 100 → 1000 → all
    adaptive_limits = [100, 1000, len(control_sets)]

    all_results = []

    for round_idx, n_ctrl in enumerate(adaptive_limits):
        ctrl_subset = control_sets[:n_ctrl]

        # Run each scope in parallel
        def _count_scope(scope_name, sw_df):
            df = count_sweeps_singlepop(
                vip_genes=vip_genes,
                control_sets=ctrl_subset,
                simplified_sweeps=sw_df,
                neighbors=neighbors,
                thresholds=thresholds,
                use_clust=use_clust,
                count_sweeps=count_sweeps,
            )
            return scope_name, df

        scope_items = list(scopes.items())
        results_round = Parallel(n_jobs=nthreads, backend="loky")(
            delayed(_count_scope)(name, df) for name, df in scope_items
        )

        # Collect into a single DataFrame for this round
        frames = []
        for scope_name, df in results_round:
            frames.append(df.with_columns(pl.lit(scope_name).alias("scope")))

        combined = pl.concat(frames).select(
            [
                "scope",
                "threshold",
                "vip_count",
                "ctrl_mean",
                "ctrl_ci_lo",
                "ctrl_ci_hi",
                "ratio",
                "ci_lo_ratio",
                "ci_hi_ratio",
                "p_value",
            ]
        )
        all_results = combined  # overwrite — later rounds have more control sets

        # Adaptive depth decision
        p_values = combined["p_value"].drop_nulls().to_list()
        if round_idx == 0:
            # Escalate if any p ≤ 0.05 or p ≥ 0.95
            if not any(p <= 0.05 or p >= 0.95 for p in p_values):
                break  # not significant at any threshold — stop here
        elif round_idx == 1:
            # Escalate if any p ≤ 0.002 or p ≥ 0.998
            if not any(p <= 0.002 or p >= 0.998 for p in p_values):
                break  # weakly significant only — no need for full 10 000 sets
        # round_idx == 2: always final

        # Don't escalate beyond what we have
        if n_ctrl >= len(control_sets):
            break

    return all_results


def shuffle_genome(
    coord_file: str,
    valid_genes: list,
    sweep_files: list,
    n_shuffles: int,
    shuffling_segments_number: int,
    output_dir: str = None,
    max_rank_boundary: int = 2000,
    seed: int = None,
) -> list:
    """
    Step 8: Shuffle genomic positions of sweep signals to build an FDR null.

    Genes are ordered by chromosomal position and divided into segments.
    Segments are randomly reordered (and optionally reversed), building
    n_shuffles independent gene→gene permutation maps. For each original
    sweep file and each permutation, a shuffled dataset is produced.

    n_shuffles must be a multiple of 8. Each call to the outer loop produces
    8 independent shufflings per outer iteration.

    Parameters
    ----------
    coord_file               : path to BED gene coordinates (no header)
    valid_genes              : list of gene IDs to include (QC-passing universe)
    sweep_files              : list of original sweep file paths (all window sizes)
    n_shuffles               : total number of shuffled replicates (multiple of 8)
    shuffling_segments_number: number of genomic segments to cut the genome into
    output_dir               : directory to write shuffled sweep files.
                               If None (default), data is kept in memory and a
                               list of {sweep_file: pl.DataFrame} dicts is returned
                               (one dict per replicate) — no files written to disk.
    max_rank_boundary        : genes ranked <= this are considered "strong signal"
                               and candidate cut points flanked by two such genes
                               are marked invalid

    Returns
    -------
    output_dir is not None  → list of str (paths to written shuffled files)
    output_dir is None      → list of dict[str, pl.DataFrame], one per replicate.
                              Each dict maps sweep_file path → shuffled DataFrame
                              with the same column layout as the original file.
                              Pass directly to estimate_fdr as
                              shuffled_sweep_files_by_shuffle.
    """
    if n_shuffles % 8 != 0:
        raise ValueError("n_shuffles must be a multiple of 8")
    in_memory = output_dir is None
    if not in_memory:
        os.makedirs(output_dir, exist_ok=True)

    # --- 1. Valid gene universe ---
    valid_genes = set(valid_genes)

    # --- 2. Load coordinates and build chromosomal order ---
    df_coords = (
        pl.read_csv(
            coord_file,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end", "gene_id", "strand"],
            schema_overrides={"chrom": pl.Utf8},
        )
        .filter(pl.col("gene_id").is_in(valid_genes))
        .with_columns(((pl.col("start") + pl.col("end")) // 2).alias("center"))
        .with_columns(
            ((pl.col("center") / 10000).floor() * 10000).cast(pl.Int64).alias("bin")
        )
    )

    # Build ordered gene list (autosomes 1–22, numeric chromosome order)
    ordered_genes = (
        df_coords.filter(pl.col("chrom").is_in([str(i) for i in range(1, 23)]))
        .with_columns(pl.col("chrom").cast(pl.Int32).alias("_chrom_int"))
        .sort(["_chrom_int", "bin"])["gene_id"]
        .to_list()
    )

    n_genes = len(ordered_genes)
    if n_genes == 0:
        raise ValueError("No valid genes found in coord_file")

    gene_to_idx = {g: i for i, g in enumerate(ordered_genes)}
    # Perl: $seg_size = $segcut (direct segment size, not count).
    # shuffling_segments_number IS the number of genes per segment, not the
    # number of segments.  Segments grow until a valid cut is found.
    seg_size = max(1, shuffling_segments_number)

    # --- 3. Load sweep data for cut-point validation ---
    # A gene is "strong signal" if its rank <= max_rank_boundary in ANY sweep file
    strong_signal_genes: set = set()
    sweep_data_by_file: dict = (
        {}
    )  # file → dict gene_id → tab-separated line (disk mode)
    rank_matrix_by_file: dict = {}  # file → (gene_col, rank_cols, numpy int32 matrix)

    for sf in sweep_files:
        gene_ranks: dict = {}
        _first_line = open(sf).readline()
        _sep = "\t" if "\t" in _first_line else " "
        df_sw = pl.read_csv(
            sf,
            has_header=False,
            separator=_sep,
            infer_schema_length=0,
        )
        # column_1 = gene_id, remaining columns = per-population ranks
        # A gene is strong-signal if its rank in ANY population <= max_rank_boundary
        if df_sw.width >= 2:
            # Vectorized: cast rank columns to Int64, take row-wise min, filter
            _rank_exprs = [
                pl.col(c).cast(pl.Int64, strict=False) for c in df_sw.columns[1:]
            ]
            strong_signal_genes |= set(
                df_sw.filter(pl.min_horizontal(_rank_exprs) <= max_rank_boundary)[
                    :, 0
                ].to_list()
            )
            # Disk-mode: store tab-separated lines for gene remapping
            if not in_memory:
                for row in df_sw.iter_rows():
                    gene_ranks[row[0]] = "\t".join(str(v) for v in row)

            # In-memory fast path (S1): pre-align sweep data to ordered_genes order
            # as a numpy int32 matrix so each shuffle is a single fancy-index op.
            _gcol = df_sw.columns[0]
            _rcols = df_sw.columns[1:]
            _odf = pl.DataFrame({"_g": ordered_genes, "_pos": list(range(n_genes))})
            _aligned = _odf.join(df_sw.rename({_gcol: "_g"}), on="_g", how="left").sort(
                "_pos"
            )
            rank_matrix_by_file[sf] = (
                _gcol,
                list(_rcols),
                _aligned.select(list(_rcols))
                .with_columns(
                    [
                        pl.col(c).cast(pl.Int32, strict=False).fill_null(10_000_000)
                        for c in _rcols
                    ]
                )
                .to_numpy(),  # shape: (n_genes, n_pops)
            )
        sweep_data_by_file[sf] = gene_ranks

    # --- 4. Detect valid cut points ---
    # Candidate cuts: every seg_size genes
    # Invalid: both the gene just before AND just after the cut are strong-signal genes
    cut_valid = []  # list of (cut_index, is_valid)
    for i in range(seg_size, n_genes, seg_size):
        gene_before = ordered_genes[i - 1]
        gene_after = ordered_genes[min(i, n_genes - 1)]
        both_strong = (
            gene_before in strong_signal_genes and gene_after in strong_signal_genes
        )
        cut_valid.append((i, not both_strong))

    # --- 5. Build segments respecting valid cuts (as numpy arrays for fast mirror) ---
    segments = []
    seg_start = 0
    for cut_idx, is_valid in cut_valid:
        if is_valid:
            segments.append(np.arange(seg_start, cut_idx, dtype=np.int32))
            seg_start = cut_idx
    # tail segment
    if seg_start < n_genes:
        segments.append(np.arange(seg_start, n_genes, dtype=np.int32))

    rng = np.random.default_rng(seed)
    written_paths = []
    inmem_replicates = []
    n_outer = n_shuffles // 8

    for outer in range(n_outer):
        # Produce 8 independent shufflings for this outer iteration
        for inner in range(8):
            shuffle_id = outer * 8 + inner + 1

            # Shuffle segment order
            seg_order = list(range(len(segments)))
            rng.shuffle(seg_order)

            # Per-segment: either keep forward indices or mirror to opposite end of genome.
            # Matches Perl dice mechanism: dice>5 → use inds[n-1-i] instead of inds[i].
            # Segments are numpy arrays — mirror is a single broadcast subtract (no loop).
            collected = []
            for si in seg_order:
                seg = segments[si].copy()
                if rng.integers(0, 2) == 1:
                    seg = (
                        np.int32(n_genes - 1) - seg
                    )  # mirror position (numpy broadcast)
                collected.append(seg)
            shuffled_indices = np.concatenate(collected)

            # Produce one shuffled dataset per original sweep file
            replicate_dict = {}
            for sf in sweep_files:
                if in_memory:
                    # S1: numpy fancy-index — single op replaces perm_map + gene loop
                    _gcol, _rcols, _rmat = rank_matrix_by_file[sf]
                    _reindexed = _rmat[shuffled_indices]  # (n_genes, n_pops)
                    replicate_dict[sf] = pl.DataFrame(
                        {_gcol: ordered_genes}
                        | {_rcols[j]: _reindexed[:, j] for j in range(len(_rcols))}
                    )
                else:
                    # Disk mode: build perm_map and remap gene by gene (string-based)
                    perm_map = {
                        ordered_genes[i]: ordered_genes[int(shuffled_indices[i])]
                        for i in range(min(n_genes, len(shuffled_indices)))
                    }
                    gene_ranks = sweep_data_by_file[sf]
                    rows = []
                    for orig_gene in ordered_genes:
                        shuffled_gene = perm_map.get(orig_gene, orig_gene)
                        sweep_line = gene_ranks.get(shuffled_gene)
                        if sweep_line is None:
                            continue
                        parts = sweep_line.split("\t")
                        parts[0] = orig_gene
                        rows.append(parts)
                    base = os.path.basename(sf)
                    out_path = os.path.join(
                        output_dir, f"fake_{base}_shuffle{shuffle_id}"
                    )
                    with open(out_path, "w") as fout:
                        for parts in rows:
                            fout.write("\t".join(parts) + "\n")
                    written_paths.append(out_path)

            if in_memory:
                inmem_replicates.append(replicate_dict)

    return inmem_replicates if in_memory else written_paths


def estimate_fdr(
    real_results: pl.DataFrame,
    shuffled_sweep_files_by_shuffle: list,
    vip_genes: list,
    control_sets: list,
    neighbors: pl.DataFrame,
    thresholds: list,
    populations: list,
    groups: list,
    pop_interest: str,
    simplified_sweeps_by_pop_fn=None,
    count_sweeps: bool = False,
    use_clust: bool = True,
    nthreads: int = 1,
    p_cutoff: float = 0.05,
    max_threshold: int = None,
    min_threshold: int = 0,
    min_vip_count: int = 0,
) -> pl.DataFrame:
    """
    Step 9: Estimate FDR by comparing real results to a null distribution built
    from genome-shuffled sweep files (output of shuffle_genome).

    Matches the behaviour of estimate_FPR.pl: for each scope (population / group /
    "All"), the test statistic is sum(vip_count − ctrl_mean) across rows that pass
    the significance and threshold filters.  FDR p-value = fraction of null
    replicates where that statistic ≥ the real statistic.

    Parameters
    ----------
    real_results                    : pl.DataFrame — output of count_sweeps_multipop on the real data (columns: scope, threshold, vip_count, ctrl_mean, p_value)
    shuffled_sweep_files_by_shuffle : list of lists — each inner list holds paths for one shuffle replicate. Layout A: one file per population [[esn_s1, gwd_s1, ...], ...]. Layout B: one multi-population file per replicate [[multi_s1], [multi_s2], ...].
    vip_genes                       : list of VIP gene IDs
    control_sets                    : control sets from run_bootstrap
    neighbors                       : pl.DataFrame from build_gene_neighbors_numpy
    thresholds                      : list of int thresholds
    populations                     : ordered population list
    groups                          : group label per population
    pop_interest                    : single pop, group name, or "All"
    simplified_sweeps_by_pop_fn     : callable(sweep_file, col_index) → pl.DataFrame, or callable(sweep_file) → dict[pop, pl.DataFrame] for multi-pop files. Defaults to simplify_sweeps_gz.
    use_clust                       : expand by neighbours before counting
    nthreads                        : parallel workers
    p_cutoff                        : p-value threshold for "significant" rows in the test statistic (Perl: cutoff, default 0.05)
    max_threshold                   : upper bound on rank threshold included in the statistic (Perl: limit; None = no limit)
    min_threshold                   : lower bound on rank threshold (Perl: cutoff2, default 0)
    min_vip_count                   : minimum vip_count for a row to contribute to the statistic (Perl: enough, default 0)

    Returns
    -------
    pl.DataFrame with columns: scope, p_value, n_replicates, real_stat, max_null_stat. One row per scope. real_stat = sum(vip_count − ctrl_mean) for significant rows. p_value = fraction of null replicates where null_stat ≥ real_stat.
    """
    target_pops = _resolve_target_pops(pop_interest, populations, groups)

    if simplified_sweeps_by_pop_fn is None:

        def simplified_sweeps_by_pop_fn(sweep_file, col_index=1):
            return simplify_sweeps_gz(sweep_file, thresholds, col_index=col_index)

    def _run_one_shuffle(shuffle_data) -> pl.DataFrame:
        """Run count_sweeps_multipop on one shuffled replicate.

        shuffle_data is either:
          - list[str]           — file paths (disk mode)
          - dict[str, pl.DataFrame] — {sweep_file: df} (in-memory mode from
                                      shuffle_genome(output_dir=None))
        """
        simplified_by_pop = {}

        if isinstance(shuffle_data, dict):
            # In-memory mode: shuffle_data maps sweep_file → shuffled DataFrame.
            # Use simplify_sweeps_gz with df= kwarg to avoid any disk I/O.
            # If more than one sweep file, union the simplified sets per pop.
            sf_keys = list(shuffle_data.keys())
            for i, pop in enumerate(target_pops):
                col_idx = i + 1
                frames = []
                for sf in sf_keys:
                    df_shuf = shuffle_data[sf]
                    if col_idx < len(df_shuf.columns):
                        frames.append(
                            simplify_sweeps_gz(
                                sf, thresholds, col_index=col_idx, df=df_shuf
                            )
                        )
                if frames:
                    simplified_by_pop[pop] = (
                        pl.concat(frames)
                        .group_by("gene_id")
                        .agg(pl.col("selected_cutoff").min())
                        if len(frames) > 1
                        else frames[0]
                    )
        elif len(shuffle_data) == 1:
            # Single multi-pop file (disk mode)
            for i, pop in enumerate(target_pops):
                try:
                    result = simplified_sweeps_by_pop_fn(shuffle_data[0], i + 1)
                except TypeError:
                    full = simplified_sweeps_by_pop_fn(shuffle_data[0])
                    if isinstance(full, dict):
                        simplified_by_pop = {
                            p: full[p] for p in target_pops if p in full
                        }
                    break
                simplified_by_pop[pop] = result
        else:
            # One file per population (disk mode)
            for i, pop in enumerate(target_pops):
                if i < len(shuffle_data):
                    try:
                        simplified_by_pop[pop] = simplified_sweeps_by_pop_fn(
                            shuffle_data[i], i + 1
                        )
                    except TypeError:
                        simplified_by_pop[pop] = simplified_sweeps_by_pop_fn(
                            shuffle_data[i]
                        )
        # For FDR, _compute_stat sums individual population rows (not group/All rows).
        # Only compute per-population scopes here — skip group/All overhead.
        frames = []
        for pop in target_pops:
            if pop not in simplified_by_pop:
                continue
            df_pop = count_sweeps_singlepop(
                vip_genes=vip_genes,
                control_sets=control_sets,
                simplified_sweeps=simplified_by_pop[pop],
                neighbors=neighbors,
                thresholds=thresholds,
                use_clust=use_clust,
                count_sweeps=count_sweeps,
            )
            frames.append(df_pop.with_columns(pl.lit(pop).alias("scope")))
        return pl.concat(frames) if frames else pl.DataFrame()

    null_results = Parallel(n_jobs=nthreads, backend="loky", verbose=1)(
        delayed(_run_one_shuffle)(shuf_files)
        for shuf_files in shuffled_sweep_files_by_shuffle
    )

    # --- Compute FDR per scope, matching estimate_FPR.pl ---
    # Test statistic = sum(vip_count − ctrl_mean) for rows passing all filters.
    # FDR p-value = fraction of null replicates where null_stat ≥ real_stat.

    def _compute_stat(df: pl.DataFrame, scope: str) -> float:
        """Perl estimate_FPR.pl: total_diff = sum(vip_count) - sum(ctrl_mean) for sig rows.

        For scope == "All": Perl accumulates counts PER population (sum-based), not a
        gene union. Summing individual population rows reproduces this exactly.
        """
        if "scope" in df.columns:
            if scope == "All":
                # Perl "All:" = sum of individual population contributions.
                # Each gene in k populations contributes k to the total, not 1.
                s = df.filter(pl.col("scope").is_in(populations))
            else:
                s = df.filter(pl.col("scope") == scope)
        else:
            s = df
        _pred = (
            (pl.col("p_value") <= p_cutoff)
            & (pl.col("vip_count") >= min_vip_count)
            & (pl.col("threshold") >= min_threshold)
        )
        if max_threshold is not None:
            _pred = _pred & (pl.col("threshold") <= max_threshold)
        s = s.filter(_pred)
        if s.is_empty():
            return 0.0
        return float((s["vip_count"].cast(pl.Float64) - s["ctrl_mean"]).sum())

    # Determine which scope(s) to report FDR for.
    # Matches estimate_FPR.pl: only the scope matching pop_interest is reported
    # (Perl uses regex match: if($pop =~ $splitter_line[1])).
    if "scope" in real_results.columns:
        all_scopes = real_results["scope"].unique().to_list()
        scopes = [s for s in all_scopes if s == pop_interest]
        if not scopes:
            scopes = [pop_interest]  # fallback if scope not found
    else:
        scopes = [pop_interest]

    rows = []
    for scope in sorted(scopes):
        real_stat = _compute_stat(real_results, scope)
        null_stats = [_compute_stat(df, scope) for df in null_results]

        n_reps = len(null_stats)
        pval = (
            float(np.sum(np.array(null_stats) >= real_stat) / n_reps)
            if n_reps > 0
            else float("nan")
        )
        max_null = float(np.max(null_stats)) if null_stats else float("nan")

        rows.append(
            {
                "scope": scope,
                "p_value": pval,
                "n_replicates": n_reps,
                "real_stat": real_stat,
                "max_null_stat": max_null,
            }
        )

    return pl.DataFrame(rows)


def run_enrichment(
    sweep_files: list,
    gene_set: str,
    factors_file: str,
    annotation_file: str,
    populations: list,
    groups: list,
    thresholds: list,
    pop_interest: str = "All",
    cluster_distance: int = 500_000,
    n_runs: int = 10,
    tolerance: float = 0.05,
    min_distance: int = 1_250_000,
    flip: bool = False,
    max_rep: int = 25,
    nthreads: int = 1,
    n_shuffles: int = 8,
    shuffling_segs: int = 2,
    bootstrap_dir: str = None,
    distance_file: str = None,
) -> list:
    """
    Run the full gene-set sweep enrichment pipeline.

    Steps:

    1. Load gene set and derive valid gene universe (factors ∩ sweep_files[0]).
    2. Bootstrap control sets matched on confounding factors
       (or load pre-computed sets from ``bootstrap_dir``).
    3. For each sweep file: count sweep overlaps, shuffle genome for null
       distribution, and estimate FDR.

    :param sweep_files: Paths to sweep rank files (gene_id + per-population
        rank columns, tab- or space-separated, optionally gzipped).
    :param gene_set: TSV with ``gene_id`` and ``yes``/``no`` label columns
        (no header). Genes labelled ``yes`` are VIPs.
    :param factors_file: TSV confounding factors file (gene_id + factor
        columns, no header).
    :param annotation_file: BED gene coordinates file (0-based, no header).
    :param populations: Population codes matching sweep file column order.
    :param groups: Group label per population (same length as populations).
    :param thresholds: Rank cutoffs for enrichment curve (e.g. [6000, ..., 20]).
    :param pop_interest: Population, group name, or ``'All'`` for FDR scope.
    :param cluster_distance: Max bp between genes to count as neighbours.
    :param n_runs: Bootstrap batches.
    :param tolerance: Allowed ± fraction deviation in factor averages for
        control gene matching.
    :param min_distance: Minimum bp distance from VIPs for control eligibility.
    :param flip: Flip test direction when ``len(VIPs) > len(controls)`` (increases power).
    :param max_rep: Max average resamples per control gene across bootstrap sets.
    :param nthreads: Parallel workers (joblib).
    :param n_shuffles: FDR shuffle replicates (must be a multiple of 8).
    :param shuffling_segs: Genes per genomic shuffle segment.
    :param bootstrap_dir: Folder with pre-computed ``VIPs/`` and ``nonVIPs/``
        sub-dirs. When non-empty, the bootstrap step is skipped.
    :param distance_file: Optional pre-computed distance TSV (gene_id \\t distance,
        no header). Passed to ``run_bootstrap_nomatchomega`` to bypass internal
        distance computation and match Perl's control pool exactly.
    :returns: List of FDR DataFrames, one per entry in ``sweep_files``.
    """
    import gzip as _gz

    _t_start = time.time()

    # --- Load gene set and derive valid universe from factors ∩ sweep_files[0] ---
    _df_geneset = pl.read_csv(
        gene_set,
        has_header=False,
        separator="\t",
        new_columns=["gene_id", "label"],
    )
    _exclude = set(hla_genes + hist_genes)

    with open(factors_file, "rt") as _fh:
        _fsep = "\t" if "\t" in _fh.readline() else " "
    _factors_ids = set(
        pl.read_csv(
            factors_file, has_header=False, separator=_fsep, infer_schema_length=0
        )[:, 0].to_list()
    )
    _sf0 = sweep_files[0]
    _open_fn = _gz.open if _sf0.endswith(".gz") else open
    with _open_fn(_sf0, "rt") as _fh:
        _sep0 = "\t" if "\t" in _fh.readline() else " "
    _sweep_ids = set(
        pl.read_csv(_sf0, has_header=False, separator=_sep0, infer_schema_length=0)[
            :, 0
        ].to_list()
    )
    _valid_genes_list = sorted(_factors_ids & _sweep_ids)

    _case_genes = (
        _df_geneset.filter(
            (pl.col("label") == "yes")
            & ~pl.col("gene_id").is_in(_exclude)
            & pl.col("gene_id").is_in(_valid_genes_list)
        )
        .drop_nulls()["gene_id"]
        .to_list()
    )
    _control_genes = _df_geneset.filter(
        (pl.col("label") == "no")
        & ~pl.col("gene_id").is_in(_exclude)
        & pl.col("gene_id").is_in(_valid_genes_list)
    )["gene_id"].to_list()
    # --- Step 6: Bootstrap ---
    if bootstrap_dir is not None:
        _vip_file = os.path.join(bootstrap_dir, "VIPs", "file_1")
        _ctrl_file = os.path.join(bootstrap_dir, "nonVIPs", "file_1")
        with open(_vip_file) as _fh:
            _vip_list_raw = [l.strip() for l in _fh if l.strip()]
        df_case = pl.DataFrame({"gene_id": _vip_list_raw})
        _ctrl_sets_raw = []
        with open(_ctrl_file) as _fh:
            for _line in _fh:
                _parts = _line.strip().split()
                if _parts:
                    _ctrl_sets_raw.append(_parts[1:])  # drop sample_N prefix
        _n_ctrl_genes = max(len(r) for r in _ctrl_sets_raw) if _ctrl_sets_raw else 0
        df_control = pl.DataFrame(
            {"sample_id": [f"sample_{i + 1}" for i in range(len(_ctrl_sets_raw))]}
            | {
                f"gene_{j}": [(r[j] if j < len(r) else None) for r in _ctrl_sets_raw]
                for j in range(_n_ctrl_genes)
            }
        )
        print("TIMING_BOOTSTRAP: 0.000", flush=True)
    else:
        df_case, df_control = run_bootstrap_nomatchomega(
            case_genes=_case_genes,
            factors_file=factors_file,
            annotation_file=annotation_file,
            runs_number=n_runs,
            tolerance=tolerance,
            min_dist=min_distance,
            flip=flip,
            max_rep=max_rep,
            nthreads=nthreads,
            control_genes=_control_genes,
            distance_file=distance_file,
        )
        print(f"TIMING_BOOTSTRAP: {time.time() - _t_start:.3f}", flush=True)

    _t_post_bootstrap = time.time()

    # --- Step 7: Sweep counting ---
    df_n = build_gene_neighbors_numpy(
        annotation_file, _valid_genes_list, cluster_distance
    )

    _vip_list = df_case["gene_id"].to_list()
    _ctrl_sets = df_control.drop("sample_id").to_numpy().tolist()

    fdr_results = []
    curve_results = []
    for _rank_file in sweep_files:
        _size = os.path.basename(_rank_file).rsplit("_", 1)[-1]

        _simplified_by_pop = {
            pop: simplify_sweeps_gz(_rank_file, thresholds, col_index=i + 1)
            for i, pop in enumerate(populations)
        }

        results = count_sweeps_multipop(
            vip_genes=_vip_list,
            control_sets=_ctrl_sets,
            simplified_sweeps_by_pop=_simplified_by_pop,
            neighbors=df_n,
            thresholds=thresholds,
            populations=populations,
            groups=groups,
            # FDR requires all-population scopes; matches Perl "All:"
            pop_interest="All",
            use_clust=False,
            count_sweeps=False,
            nthreads=nthreads,
        )

        # --- Step 8: Genome shuffling (in-memory) ---
        shuffled_data = shuffle_genome(
            coord_file=annotation_file,
            valid_genes=_valid_genes_list,
            sweep_files=[_rank_file],
            n_shuffles=n_shuffles,
            shuffling_segments_number=shuffling_segs,
            output_dir=None,
        )

        # --- Step 9: FDR estimation ---
        fdr = estimate_fdr(
            real_results=results,
            shuffled_sweep_files_by_shuffle=shuffled_data,
            vip_genes=_vip_list,
            control_sets=_ctrl_sets,
            neighbors=df_n,
            thresholds=thresholds,
            populations=populations,
            groups=groups,
            pop_interest="All",  # matches Perl estimate_FPR.pl hardcoded "All:"
            use_clust=False,
            count_sweeps=False,
            nthreads=nthreads,
            p_cutoff=1.0,
            max_threshold=max(thresholds),
            min_threshold=0,
            min_vip_count=1,
        )

        row = fdr.row(0, named=True)
        print(
            f"FDR {_size}: {row['p_value']:.4f} {row['n_replicates']} "
            f"{row['real_stat']:.2f} {row['max_null_stat']:.2f}",
            flush=True,
        )

        results = results.with_columns(pl.lit(_rank_file).alias("dataset"))
        fdr = fdr.with_columns(pl.lit(_rank_file).alias("dataset"))
        curve_results.append(results)
        fdr_results.append(fdr)

    print(f"TIMING_POSTBOOTSTRAP: {time.time() - _t_post_bootstrap:.3f}", flush=True)
    print(f"TIMING_TOTAL: {time.time() - _t_start:.3f}", flush=True)
    return df_case, df_control, pl.concat(curve_results), pl.concat(fdr_results)


thresholds = [
    6000,
    5000,
    4000,
    3000,
    2500,
    2000,
    1500,
    1000,
    900,
    800,
    700,
    600,
    500,
    450,
    400,
    350,
    300,
    250,
    200,
    150,
    100,
    50,
    20,
]


# hla_genes = [  # original expanded list — DO NOT restore
#     "ENSG00000233095", "ENSG00000223980", "ENSG00000230254", "ENSG00000204642",
#     "ENSG00000204632", "ENSG00000206503", "ENSG00000204592", "ENSG00000114455",
#     "ENSG00000235220", "ENSG00000235346", "ENSG00000235657", "ENSG00000229252",
#     "ENSG00000228299", "ENSG00000228987", "ENSG00000236884", "ENSG00000236418",
#     "ENSG00000233209", "ENSG00000223793", "ENSG00000228813", "ENSG00000243496",
#     "ENSG00000226264", "ENSG00000241394", "ENSG00000235744", "ENSG00000229685",
#     "ENSG00000237710", "ENSG00000206435", "ENSG00000228964", "ENSG00000234794",
#     "ENSG00000227357", "ENSG00000228080", "ENSG00000232062", "ENSG00000231939",
#     "ENSG00000231526", "ENSG00000226165", "ENSG00000239457", "ENSG00000241296",
#     "ENSG00000243215", "ENSG00000232957", "ENSG00000236177", "ENSG00000226826",
#     "ENSG00000237508", "ENSG00000226260", "ENSG00000227826", "ENSG00000229074",
#     "ENSG00000225890", "ENSG00000235680", "ENSG00000225824", "ENSG00000231834",
#     "ENSG00000231823", "ENSG00000229493", "ENSG00000239329", "ENSG00000243189",
#     "ENSG00000231558", "ENSG00000228163", "ENSG00000236632", "ENSG00000229295",
#     "ENSG00000237022", "ENSG00000224608", "ENSG00000229698", "ENSG00000132297",
#     "ENSG00000206509", "ENSG00000206506", "ENSG00000206505", "ENSG00000206493",
#     "ENSG00000206452", "ENSG00000206450", "ENSG00000204525", "ENSG00000234745",
#     "ENSG00000232962", "ENSG00000224103", "ENSG00000230708", "ENSG00000206308",
#     "ENSG00000196101", "ENSG00000206306", "ENSG00000206305", "ENSG00000206302",
#     "ENSG00000225103", "ENSG00000196610", "ENSG00000241674", "ENSG00000242685",
#     "ENSG00000206292", "ENSG00000206291", "ENSG00000215048", "ENSG00000233841",
#     "ENSG00000223532", "ENSG00000204287", "ENSG00000198502", "ENSG00000196126",
#     "ENSG00000196735", "ENSG00000179344", "ENSG00000237541", "ENSG00000232629",
#     "ENSG00000241106", "ENSG00000137403", "ENSG00000230413", "ENSG00000224320",
#     "ENSG00000197568", "ENSG00000242574", "ENSG00000204257", "ENSG00000204252",
#     "ENSG00000231389", "ENSG00000223865", "ENSG00000233904", "ENSG00000234487",
#     "ENSG00000237216", "ENSG00000229215", "ENSG00000227715", "ENSG00000230726",
#     "ENSG00000231021", "ENSG00000228284", "ENSG00000231286", "ENSG00000225201",
#     "ENSG00000206301", "ENSG00000228254", "ENSG00000241910", "ENSG00000242386",
#     "ENSG00000239463", "ENSG00000230141", "ENSG00000168384", "ENSG00000230763",
#     "ENSG00000230463", "ENSG00000233192", "ENSG00000230675", "ENSG00000227993",
#     "ENSG00000243612", "ENSG00000231679", "ENSG00000206240", "ENSG00000206237",
#     "ENSG00000204276", "ENSG00000224305", "ENSG00000241386", "ENSG00000225691",
#     "ENSG00000242092", "ENSG00000242361", "ENSG00000235844", "ENSG00000236693",
#     "ENSG00000232126", "ENSG00000234154", "ENSG00000243719",
# ]
#
# hist_genes = [  # original histone list — DO NOT restore
#     "ENSG00000164508", "ENSG00000146047", "ENSG00000124610", "ENSG00000198366",
#     "ENSG00000196176", "ENSG00000124529", "ENSG00000124693", "ENSG00000137259",
#     "ENSG00000196226", "ENSG00000196532", "ENSG00000187837", "ENSG00000197061",
#     "ENSG00000187475", "ENSG00000180596", "ENSG00000180573", "ENSG00000168298",
#     "ENSG00000158373", "ENSG00000197697", "ENSG00000188987", "ENSG00000197409",
#     "ENSG00000196866", "ENSG00000197846", "ENSG00000198518", "ENSG00000187990",
#     "ENSG00000168274", "ENSG00000196966", "ENSG00000124575", "ENSG00000198327",
#     "ENSG00000124578", "ENSG00000256316", "ENSG00000197459", "ENSG00000256018",
#     "ENSG00000168242", "ENSG00000158406", "ENSG00000124635", "ENSG00000196787",
#     "ENSG00000197903", "ENSG00000198339", "ENSG00000184825", "ENSG00000185130",
#     "ENSG00000196747", "ENSG00000203813", "ENSG00000182611", "ENSG00000196374",
#     "ENSG00000197238", "ENSG00000197914", "ENSG00000184348", "ENSG00000233822",
#     "ENSG00000198374", "ENSG00000184357", "ENSG00000182572", "ENSG00000198558",
#     "ENSG00000197153", "ENSG00000233224", "ENSG00000196331", "ENSG00000168148",
#     "ENSG00000181218", "ENSG00000196890", "ENSG00000203818", "ENSG00000203814",
#     "ENSG00000183598", "ENSG00000183941", "ENSG00000203811", "ENSG00000183558",
#     "ENSG00000203812", "ENSG00000203852", "ENSG00000182217", "ENSG00000184678",
#     "ENSG00000184260", "ENSG00000184270", "ENSG00000197837", "ENSG00000265232",
#     "ENSG00000263376", "ENSG00000265133", "ENSG00000265198", "ENSG00000266225",
#     "ENSG00000266725", "ENSG00000264719", "ENSG00000263521", "ENSG00000263965",
# ]

hla_genes = [
    "ENSG00000179344",
    "ENSG00000196126",
    "ENSG00000196735",
    "ENSG00000198502",
    "ENSG00000204252",
    "ENSG00000204257",
    "ENSG00000204287",
    "ENSG00000204525",
    "ENSG00000204592",
    "ENSG00000204632",
    "ENSG00000204642",
    "ENSG00000206503",
    "ENSG00000223865",
    "ENSG00000231389",
    "ENSG00000232629",
    "ENSG00000234745",
    "ENSG00000237541",
    "ENSG00000241106",
    "ENSG00000242574",
]

hist_genes = []
