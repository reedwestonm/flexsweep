import glob
import gzip
import heapq
import os
import pickle
import re
from collections import OrderedDict, defaultdict, namedtuple
from contextlib import contextmanager
from copy import deepcopy
from functools import lru_cache, partial, reduce
from itertools import product
from math import ceil, comb
from multiprocessing.pool import ThreadPool
from warnings import filterwarnings, warn

from allel import nsl
from allel.compat import memoryview_safe
from allel.opt.stats import ihh01_scan
from allel.stats.selection import compute_ihh_gaps
from allel.util import asarray_ndim, check_dim0_aligned, check_integer_dtype

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.expanduser("~/.cache/numba/flexsweep"))
from numba import float32, float64, int32, int64, njit, prange, uint64
from numba.typed import Dict

from . import np, pl

filterwarnings("ignore", message="invalid INFO header", module="allel.io.vcf_read")
filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in scalar divide",
)
np.seterr(divide="ignore", invalid="ignore")

# Define the inner namedtuple structure
summaries = namedtuple("summaries", ["stats", "parameters"])
binned_stats = namedtuple("binned_stats", ["mean", "std"])


################## Utils


def load_pickle(f):
    with open(f, "rb") as handle:
        return pickle.load(handle)


def save_pickle(f, data):
    with open(f, "wb") as handle:
        pickle.dump(data, handle)


def reset_sims_bins(results, r_bins=None, nthreads=1):
    from . import Parallel, delayed

    def make_r_bins(params):
        if r_bins is None:
            return None
        return (
            pl.DataFrame({"r": params[:, -1] * 1e8})
            .with_row_index(name="iter", offset=1)
            .with_columns(pl.col("r").cut(breaks=r_bins).alias("r_bins"))
            .select(["iter", "r_bins"])
        )

    def join_stat_frames(frames):
        return (
            reduce(
                lambda left, right: left.join(
                    right, on=["iter", "positions", "daf"], how="full", coalesce=True
                ),
                frames,
            )
            .sort("positions")
            .filter(pl.col("daf").is_not_null())
            .collect()
        )

    def process_stat(i):
        # New format: i is {"snps": snps_joined, "window": window_df}
        return {
            "snps": i["snps"],
            "windows": i["window"],
        }

    def process_batch(batch):
        return [process_stat(i) for i in batch]

    def chunked(iterable, size):
        for i in range(0, len(iterable), size):
            yield iterable[i : i + size]

    df_r = {sim_type: make_r_bins(v.parameters) for sim_type, v in results.items()}

    neutral = results["neutral"]
    batches = list(chunked(neutral.stats, 50))
    norm_stats = [
        item
        for batch_result in Parallel(n_jobs=nthreads)(
            delayed(process_batch)(batch) for batch in batches
        )
        for item in batch_result
    ]

    neutral_binned = binned_stats(
        *normalize_neutral(norm_stats, df_r_bins=df_r["neutral"])
    )
    return neutral_binned, df_r


def reset_empirical_bins(results, r_bins):
    def make_r_bins(iter_ids, params):
        if r_bins is None:
            return None
        return (
            pl.DataFrame({"iter": iter_ids, "cm_mb": params[:, -1]})
            .with_columns(pl.col("cm_mb").cut(breaks=r_bins).alias("r_bins"))
            .select(["iter", "cm_mb", "r_bins"])
        )

    def join_stat_frames(stats):
        stat_keys = [
            "nsl",
            "ihs",
            "isafe",
            "dind_high_low",
            "s_ratio",
            "hapdaf_o",
            "hapdaf_s",
        ]
        return (
            reduce(
                lambda left, right: left.join(
                    right, on=["iter", "positions", "daf"], how="full", coalesce=True
                ),
                [stats[k].lazy() for k in stat_keys],
            )
            .sort("positions")
            .collect()
        )

    norm_stats = []
    df_r_l = []
    regions = {}

    for k, v in results.items():
        print(k)
        iter_ids = v.stats["window"]["iter"].unique(maintain_order=True)
        regions[k] = iter_ids
        df_r_l.append(make_r_bins(iter_ids, v.parameters))
        norm_stats.append(
            {
                "snps": join_stat_frames(v.stats),
                "windows": v.stats.get("window"),
            }
        )

    try:
        df_r = pl.concat(df_r_l)
    except Exception as e:
        print(f"_process_vcf: failed to concat r_bins table: {e}")
        df_r = None

    empirical_bins = binned_stats(
        *normalize_neutral(norm_stats, vcf=True, df_r_bins=df_r)
    )
    return empirical_bins, df_r, regions


################## Utils


def open_tree(ts, seq_len=1.2e6):
    """Read a tree sequence file and return outputs matching parse_ms_numpy format.

    Returns
    -------
    hap_bi : np.ndarray, shape (n_biallelic_sites, n_samples), dtype int8
    rec_bi : np.ndarray, shape (n_biallelic_sites, 4), dtype int64
    ac_bi  : np.ndarray, shape (n_biallelic_sites, 2), dtype int64
    biallelic_mask : np.ndarray, shape (n_sites,), dtype bool
    position_masked : np.ndarray, shape (n_biallelic_sites,), dtype int64
    genetic_position_masked : np.ndarray, shape (n_biallelic_sites,), dtype int64
    """
    from allel import HaplotypeArray

    try:
        if isinstance(ts, str):
            ts = tskit.load(ts)
        G = ts.genotype_matrix()
        positions_raw = np.array([v.position for v in ts.variants()])
    except Exception as e:
        raise ValueError(f"Could not load tree sequence: {e}")

    n_sites, n_samples = G.shape

    if positions_raw.max() <= 1.0:
        # fractional [0, 1] coordinates — scale to bp
        positions_bp = np.round(positions_raw * seq_len).astype(np.int64)
    else:
        positions_bp = positions_raw.astype(np.int64)
    positions_bp = np.clip(positions_bp, 1, int(seq_len))

    rec_map = np.column_stack(
        (
            np.ones(n_sites, dtype=np.int64),
            np.arange(n_sites, dtype=np.int64),
            positions_bp,
            positions_bp,
        )
    )

    hap = HaplotypeArray(G.astype(np.int8), copy=False)
    ac = hap.count_alleles()
    biallelic_mask = ac.is_biallelic_01()

    hap_bi = hap.compress(biallelic_mask, axis=0)
    ac_bi = ac.compress(biallelic_mask, axis=0)
    rec_bi = rec_map[biallelic_mask]

    position_masked = rec_bi[:, 3].astype(np.int64, copy=False)
    genetic_position_masked = rec_bi[:, 2]

    return (
        hap_bi.view(np.ndarray),
        rec_bi,
        ac_bi.view(np.ndarray),
        biallelic_mask,
        position_masked,
        genetic_position_masked,
    )


def best_window_idx_per_position_cm(
    pos_sorted,
    w_start,
    w_end,
    w_cm,
    mode="max",
):
    """
    For each SNP position, pick the overlapping window with max/min cm_mb.
    Tie-breaker: larger end wins.
    Returns -1 if no window covers the position.
    """
    out = np.full(pos_sorted.size, -1, dtype=np.int32)
    heap = []
    i = 0
    nw = w_start.size

    if mode == "max":

        def key(k):  # (primary, secondary, idx)
            return (-w_cm[k], -w_end[k], k)

    elif mode == "min":

        def key(k):
            return (w_cm[k], -w_end[k], k)

    else:
        raise ValueError("mode must be 'max' or 'min'")

    for j, p in enumerate(pos_sorted):
        while i < nw and w_start[i] <= p:
            heapq.heappush(heap, key(i))
            i += 1

        while heap and (-heap[0][1]) < p:
            heapq.heappop(heap)

        if heap:
            out[j] = heap[0][2]

    return out


@contextmanager
def omp_num_threads(n_threads: int):
    """
    Context manager to temporarily set the OMP_NUM_THREADS environment variable.

    Args:
        n_threads (int): Number of OpenMP threads to expose inside the context.

    Usage:
        with omp_num_threads(10):
            # Inside this block, os.environ['OMP_NUM_THREADS'] == "10"
            heavy_compute()
        # On exit, the previous value (or absence) is restored.

    Notes:
        - Only affects libraries honoring OMP_NUM_THREADS (e.g., numexpr, MKL-backed numpy).
        - This modifies process environment for the duration of the context only.
    """
    key = "OMP_NUM_THREADS"
    old_val = os.environ.get(key, None)
    os.environ[key] = str(n_threads)
    try:
        yield
    finally:
        # restore original state
        if old_val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_val


def _ms_float_to_int(positions, total_phys_len):
    """
    Pure-Python/numpy equivalent of msPositionsToIntegerPositions (diploshic msTools.py).
    Converts float ms positions in [0,1) to unique integer bp positions in [1, total_phys_len].
    """
    n = len(positions)
    new_positions = np.empty(n, dtype=np.int64)
    prev_pos = -1.0
    prev_int_pos = -1

    for i in range(n):
        pos = float(positions[i])
        orig_pos = pos
        if pos == prev_pos:
            pos += 1e-6
        prev_pos = orig_pos

        int_pos = int(total_phys_len * pos)
        if int_pos == 0:
            int_pos = 1
        if int_pos <= prev_int_pos:
            int_pos = prev_int_pos + 1
        prev_int_pos = int_pos
        new_positions[i] = int_pos

    # Handle positions that overflow total_phys_len (mirrors fillInSnpSlotsWithOverflowers)
    overflow = new_positions > total_phys_len
    n_over = int(overflow.sum())
    if n_over > 0:
        kept = new_positions[~overflow]
        kept_set = set(kept.tolist())
        extras = []
        for p in range(int(total_phys_len), 0, -1):
            if p not in kept_set:
                extras.append(p)
                if len(extras) == n_over:
                    break
        new_positions = np.sort(
            np.concatenate([kept, np.array(extras, dtype=np.int64)])
        )

    return new_positions


def parse_and_filter_ms(
    ms_file: str,
    seq_len: float = 1.2e6,
    discretize_positions: bool = True,
):
    from allel import HaplotypeArray

    if not ms_file.endswith((".out", ".out.gz", ".ms", ".ms.gz")):
        warn(f"File {ms_file} has an unexpected extension.")

    open_function = gzip.open if ms_file.endswith(".gz") else open

    in_rep = False
    num_segsites = None
    pos_arr = None
    hap_rows = []

    with open_function(ms_file, "rt") as fh:
        for raw in fh:
            line = raw.strip()

            if line.startswith("//"):
                if in_rep:
                    break
                in_rep = True
                num_segsites = None
                pos_arr = None
                hap_rows.clear()
                continue

            if not in_rep or not line:
                continue

            if line.startswith("segsites"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        num_segsites = int(parts[1])
                        if num_segsites == 0:
                            # match parse_ms_numpy: 6-tuple with empty arrays
                            empty_hap = np.empty((0, 1), dtype=np.int8)
                            empty_rec = np.empty((0, 4), dtype=np.int64)
                            empty_ac = np.empty((0, 2), dtype=np.int64)
                            empty_mask = np.empty(0, dtype=bool)
                            empty_pos = np.empty(0, dtype=np.int64)
                            empty_gpos = np.empty(0, dtype=np.float64)
                            return (
                                empty_hap,
                                empty_rec,
                                empty_ac,
                                empty_mask,
                                empty_pos,
                                empty_gpos,
                            )
                    except ValueError:
                        warn(f"File {ms_file} is malformed.")
                        return None
                else:
                    warn(f"File {ms_file} is malformed.")
                    return None
                continue

            if line.startswith("positions"):
                try:
                    _, values = line.split(":", 1)
                    positions = np.fromstring(values, sep=" ", dtype=np.float64)
                except Exception:
                    warn(f"File {ms_file} is malformed.")
                    return None

                if discretize_positions:
                    new_positions = _ms_float_to_int(positions, seq_len)
                    new_positions[new_positions > seq_len] = int(seq_len)
                    pos_arr = new_positions
                else:
                    pos_arr = positions * seq_len
                continue

            if line[0] in "01":
                hap_rows.append(line)

    if not hap_rows or pos_arr is None or num_segsites is None:
        warn(f"File {ms_file} is malformed.")
        return None

    n_samples = len(hap_rows)
    n_sites = len(hap_rows[0])
    H = np.empty((n_samples, n_sites), dtype=np.int8)
    for i, s in enumerate(hap_rows):
        H[i, :] = np.frombuffer(s.encode("ascii"), dtype=np.uint8) - 48
    H = H.T  # (num_segsites, num_samples)

    rec_map = np.column_stack(
        (
            np.ones(n_sites, dtype=np.int64),
            np.arange(n_sites, dtype=np.int64),
            pos_arr,
            pos_arr,
        )
    )

    hap = HaplotypeArray(H, copy=False)
    ac = hap.count_alleles()
    biallelic_mask = ac.is_biallelic_01()

    hap_bi = hap.compress(biallelic_mask, axis=0)
    ac_bi = ac.compress(biallelic_mask, axis=0)
    rec_bi = rec_map[biallelic_mask]

    position_masked = rec_bi[:, 3]
    genetic_position_masked = rec_bi[:, 2]

    return (
        np.ascontiguousarray(hap_bi.view(np.ndarray), dtype=np.int8),
        rec_bi,
        ac_bi.view(np.ndarray),
        biallelic_mask,
        np.ascontiguousarray(position_masked, dtype=np.int64),
        genetic_position_masked,
    )


def parse_ms_numpy(
    ms_file: str,
    seq_len: float = 1.2e6,
    discretize_positions: bool = True,
):
    """
    Vectorized ms parser. Eliminates line-by-line Python loops.
    Optimized for high-core count scaling (128+ threads).
    """
    open_func = gzip.open if ms_file.endswith(".gz") else open

    # Use 'rb' (binary) to avoid the overhead of UTF-8 decoding
    try:
        with open_func(ms_file, "rb") as fh:
            content = fh.read()
    except Exception:
        warn(f"Could not read file {ms_file}")
        return None

    # 1. Fast-find headers using byte-searches
    # find() is implemented in C and much faster than iterating lines in Python
    try:
        sep = b"//"
        start_rep = content.find(sep)
        if start_rep == -1:
            return None

        # Locate segsites
        seg_idx = content.find(b"segsites:", start_rep)
        line_end = content.find(b"\n", seg_idx)
        num_segsites = int(content[seg_idx + 10 : line_end])

        if num_segsites == 0:
            # Return identical empty signature as original
            return (
                np.empty((0, 1), "i1"),
                np.empty((0, 4), "i8"),
                np.empty((0, 2), "i4"),
                np.empty(0, bool),
                np.empty(0, "i8"),
                np.empty(0, "f8"),
            )

        # Locate positions
        pos_idx = content.find(b"positions:", line_end)
        pos_end = content.find(b"\n", pos_idx)
        pos_vals = content[pos_idx + 11 : pos_end].split()
        positions = np.array(pos_vals, dtype=np.float64)

        # 2. Extract Haplotype Matrix Vectorially
        # The matrix starts exactly 1 byte after pos_end
        data_block = content[pos_end + 1 :].strip()

        # Fast-convert ASCII '0'/'1' to integers (0/1)
        # We ignore newlines by filtering the buffer
        raw_haps = (
            np.frombuffer(
                data_block.replace(b"\n", b"").replace(b"\r", b""), dtype=np.uint8
            )
            - 48
        )

        n_samples = len(raw_haps) // num_segsites
        H = raw_haps.reshape((n_samples, num_segsites)).T  # (n_sites, n_samples)

    except (ValueError, IndexError):
        warn(f"File {ms_file} is malformed.")
        return None

    # 3. Position Discretization
    if discretize_positions:
        # Use vectorized floor or round as per your _ms_float_to_int logic
        pos_arr = np.floor(positions * seq_len).astype(np.int64)
        pos_arr[pos_arr > seq_len] = int(seq_len)
    else:
        pos_arr = positions * seq_len

    # 4. Filter Biallelic Sites (Vectorized)
    alt_counts = H.sum(axis=1).astype(np.int32)
    ref_counts = np.int32(n_samples) - alt_counts
    biallelic_mask = (alt_counts > 0) & (ref_counts > 0)

    H_bi = np.ascontiguousarray(H[biallelic_mask], dtype=np.int8)
    ac_bi = np.column_stack((ref_counts[biallelic_mask], alt_counts[biallelic_mask]))
    pos_bi = pos_arr[biallelic_mask]

    # 5. Build Rec Matrix
    n_bi = H_bi.shape[0]
    rec_bi = np.column_stack(
        (
            np.ones(n_bi, dtype=np.int64),
            np.arange(n_bi, dtype=np.int64),
            pos_bi,
            pos_bi,
        )
    )

    return (
        np.ascontiguousarray(H_bi, dtype=np.int8),
        rec_bi,
        ac_bi,
        biallelic_mask,
        np.ascontiguousarray(rec_bi[:, 3], dtype=np.int64),
        rec_bi[:, 2],
    )


def cleaning_summaries(summ_stats, params, model):
    """
    Cleans summary statistics by removing entries where either list in summ_stats has None.

    When an entire axis is consistently None (e.g., all SNPs are None for window-only
    stats), those entries are kept as-is rather than discarding everything.

    Args:
        data: Unused input (kept for compatibility).
        summ_stats (list of 2 lists): Summary statistics [list1, list2].
        params (np.ndarray): Parameter matrix.
        model (str): Model identifier.

    Returns:
        summ_stats_filtered (list of 2 lists): Cleaned summary statistics.
        params (np.ndarray): Filtered params.
        malformed_files (list of str): Indices removed with reason.
    """
    # Detect if an entire axis is uniformly None
    all_x_none = all(x is None for x in summ_stats[0])
    all_y_none = all(y is None for y in summ_stats[1])

    mask = []
    summ_stats_filtered = [[], []]
    malformed_files = []

    for i, (x, y) in enumerate(zip(summ_stats[0], summ_stats[1])):
        # Logic: An entry is "bad" ONLY if it is None while other entries
        # in that same list are NOT None.
        x_bad = x is None and not all_x_none
        y_bad = y is None and not all_y_none

        if x_bad or y_bad:
            mask.append(i)
            malformed_files.append(f"Model {model}, index {i} is malformed.")
        else:
            # If we reach here, either the data is present,
            # or the entire axis was intentionally None.
            summ_stats_filtered[0].append(x)
            summ_stats_filtered[1].append(y)

    # Use the mask to clean the parameter matrix
    if mask and params is not None:
        params = np.delete(params, mask, axis=0)

    return summ_stats_filtered, params, malformed_files


def genome_reader(hap_data, recombination_map=None, region=None, samples=None):
    """
    Read a VCF/BCF region and return haplotypes, recombination map, allel count array, biallelic masking, physical and genetic positions arrays.

    Args:
        hap_data (str): Path to VCF/BCF file.
        recombination_map (str | None, default=None):
            Optional TSV map with columns: chr, start, end, cm_mb, cm.
        region (str | None, default=None): Region string 'CHR:START-END' for subsetting.
        samples (list[str] | np.ndarray | None, default=None): Optional sample subset.

    Returns:
        dict[str, tuple]:
            {region: (hap_int, rec_map, ac.values, biallelic_filter, position_masked, genetic_position_masked)}
            or {region: None} if no biallelic sites are present.

        Where:
            - hap_int: (S x N) np.int8 haplotypes.
            - rec_map: array with columns [chrom, idx, pos, cm].
            - ac.values: allele counts (scikit-allel).
            - biallelic_filter: boolean mask on original sites.
            - position_masked: np.int64 physical positions after biallelic filtering.
            - genetic_position_masked: last column of rec_map.

    Notes:
        - If `recombination_map` is None, genetic distance defaults to physical positions.
    """
    from allel import GenotypeArray, read_vcf

    filterwarnings("ignore", message="invalid INFO header", module="allel.io.vcf_read")

    raw_data = read_vcf(hap_data, region=region, samples=samples, fields="*")

    try:
        gt = GenotypeArray(raw_data["calldata/GT"])
    except Exception:
        return {region: None}

    pos = raw_data["variants/POS"]
    np_chrom = np.char.replace(raw_data["variants/CHROM"].astype(str), "chr", "")
    try:
        np_chrom = np_chrom.astype(int)
    except Exception:
        pass
    _ac = gt.count_alleles()

    # Filtering monomorphic just in case
    biallelic_filter = _ac.is_biallelic_01()
    ac = _ac[biallelic_filter]
    hap_int = gt.to_haplotypes().values[biallelic_filter].astype(np.int8)
    position_masked = pos[biallelic_filter].astype(np.int64)
    np_chrom = np_chrom[biallelic_filter]

    if hap_int.shape[0] == 0:
        return {region: None}

    # if region is None:
    #     d_pos = dict(zip(np.arange(position_masked.size + 1), position_masked))
    # else:
    #     tmp = list(map(int, region.split(":")[-1].split("-")))
    #     d_pos = dict(zip(np.arange(tmp[0], tmp[1] + 1), np.arange(int(5e5)) + 1))

    if recombination_map is None:
        # if CM in field in VCF
        # target_key = next((k for k in raw_data.keys() if k.lower() == 'variants/cm'), None)
        rec_map = pl.DataFrame(
            {
                "chrom": np_chrom,
                "idx": np.arange(position_masked.size),
                "pos": position_masked,
                "cm": position_masked,
                # "cm": raw_data[target_key][:,0] if target_key else position_masked,
            }
        ).to_numpy()

    else:
        df_recombination_map = (
            pl.read_csv(
                recombination_map,
                separator="\t",
                comment_prefix="#",
                schema=pl.Schema(
                    [
                        ("chr", pl.String),
                        ("start", pl.Int64),
                        ("end", pl.Int64),
                        ("cm_mb", pl.Float64),
                        ("cm", pl.Float64),
                    ]
                ),
            )
            .filter(pl.col("chr") == str(np_chrom[0]))
            .sort("start")
        )
        print(np_chrom[0])
        genetic_distance = get_cm(df_recombination_map, position_masked)

        rec_map = pl.DataFrame(
            [
                np_chrom,
                np.arange(position_masked.size),
                position_masked,
                genetic_distance,
            ]
        ).to_numpy()

        if np.all(rec_map[:, -1] == 0):
            rec_map[:, -1] = rec_map[:, -2]

    genetic_position_masked = rec_map[:, -1]

    return (
        np.ascontiguousarray(hap_int, dtype=np.int8),
        rec_map,
        asarray_ndim(ac.values, 2),
        biallelic_filter,
        np.ascontiguousarray(position_masked, dtype=np.int32),
        genetic_position_masked,
    )


def get_cm(df_rec_map, positions, cm_mb=False):
    """
    Interpolate cumulative genetic distance (cM) at given physical positions.

    Args:
        df_rec_map (polars.DataFrame): Map where column 1 is physical position (bp),
            and last column is cumulative cM (monotonic expected).
        positions (np.ndarray): 1D array of physical positions (bp) to interpolate.

    Returns:
        np.ndarray: Interpolated cumulative cM (negative values clamped to 0).

    Notes:
        - Uses linear interpolation with extrapolation at ends.
    """
    from scipy.interpolate import interp1d

    interp_func = interp1d(
        df_rec_map.select("end").to_numpy().flatten(),
        df_rec_map.select("cm").to_numpy().flatten(),
        kind="linear",
        fill_value="extrapolate",
        # fill_value=(0,df_rec_map.select("cm").max().item()),
        # bounds_error=False
    )

    if cm_mb:
        rr1 = interp_func(positions[:, 0])
        rr2 = interp_func(positions[:, 1])

        rr1[rr1 < 0] = 0
        rr2[rr2 < 0] = 0

        rate = (rr2 - rr1) / ((positions[:, 1] - positions[:, 0]) / 1e6)

        return pl.DataFrame(
            {
                "chr": df_rec_map.select("chr").unique().item(),
                "start": positions[:, 0],
                "end": positions[:, 1],
                "cm_mb": rate,
            }
        )

    else:
        # Interpolate the cM values at the interval positions
        rr1 = interp_func(positions)
        # rr2 = interp_func(positions[:, 1])
        rr1[rr1 < 0] = 0

        return rr1


def center_window_cols(df, _iter=1):
    """
    Add iter to statistic dataframe to ensure proper stats/replica combination.

    Args:
        df (polars.DataFrame): Input feature rows for a single window/region.
        _iter (int, default=1): Iteration identifier to add as an 'iter' column.

    Returns:
        polars.DataFrame:
            - If `df` is empty: returns a single-row DF with just 'iter' plus `df` columns (empty).
            - Otherwise: returns `df` with an added 'iter' column (Int64) and with columns ordered as:
                ['iter', 'positions', <all other columns excluding 'iter' and 'positions'>]

    """
    if df.is_empty():
        # Return a dataframe with one row of the specified default values
        return pl.concat(
            [pl.DataFrame({"iter": _iter}), df],
            how="horizontal",
        )

    df = (
        df.with_columns(
            [
                pl.lit(_iter).alias("iter"),
            ]
        )
        .with_columns(pl.col(["iter"]).cast(pl.Int64))
        .select(
            pl.col(["iter", "positions"]),
            pl.all().exclude(["iter", "positions"]),
        )
    )
    return df


def pivot_feature_vectors(df_fv, vcf=False):
    """
    Categorizes genomic sweep data into different models based on timing and fixation status,
    then pivots the data for analysis.


    Args:
        df_fv (polars.DataFrame): Feature vectors with columns including
            't', 'f_t', 'f_i', 's', 'iter', 'window', 'center', and metrics.
        vcf (bool, default=False): Whether the input comes from VCF processing (special handling).

    Returns:
        polars.DataFrame: Wide/pivoted feature table with cleaned column names.
    Notes:
        - When `vcf=True`, constructs 'iter' from 'nchr' and ±600kb window around center.
    """

    # Categorize sweeps based on age and completeness
    df_fv = df_fv.with_columns(
        pl.when((pl.col("t") >= 2000) & (pl.col("f_t") >= 0.9))
        .then(pl.lit("hard_old_complete"))
        .when((pl.col("t") >= 2000) & (pl.col("f_t") < 0.9))
        .then(pl.lit("hard_old_incomplete"))
        .when((pl.col("t") < 2000) & (pl.col("f_t") >= 0.9))
        .then(pl.lit("hard_young_complete"))
        .otherwise(pl.lit("hard_young_incomplete"))
        .alias("model")
    )

    # Further categorize as soft or hard sweep based on initial frequency
    df_fv = df_fv.with_columns(
        pl.when(pl.col("f_i") != df_fv["f_i"].min())
        .then(pl.col("model").str.replace("hard", "soft"))
        .otherwise(pl.col("model"))
        .alias("model")
    )

    # Handle the case where all selection coefficients are zero (neutral model)
    if (df_fv["s"] == 0).all():
        df_fv = df_fv.with_columns(pl.lit("neutral").alias("model"))

    # Determine sorting method based on iter column type
    # sort_multi = True if df_fv["iter"].dtype == pl.Utf8 else False

    # Pivot the data
    # Assuming columns 7 to end-1 are the values to pivot
    # value_columns = df_fv.columns[7:-1]
    value_columns = df_fv.columns[9:-1]

    if vcf:
        if df_fv["iter"].dtype == pl.Int64:
            # remove nchr
            value_columns = value_columns[:-1]

            fv_center = np.linspace(6e5 - 1e5, 6e5 + 1e5, 21).astype(int)
            fv_center = df_fv["center"].unique().to_numpy()
            rows_per_center = df_fv["window"].unique().len()
            n_rows = df_fv.height
            full_center = np.tile(
                np.repeat(fv_center, rows_per_center),
                n_rows // (len(fv_center) * rows_per_center) + 1,
            )[:n_rows]

            df_fv = df_fv.with_columns(
                (
                    pl.col("nchr").cast(pl.String)
                    + ":"
                    + (pl.col("iter").cast(pl.Int64) - int(6e5)).cast(pl.String)
                    + "-"
                    + (pl.col("iter").cast(pl.Int64) + int(6e5)).cast(pl.String)
                ).alias("iter"),
                pl.lit(full_center).alias("center"),
            ).select(pl.exclude("nchr"))

    df_fv_w = df_fv.pivot(
        values=value_columns,
        index=["iter", "s", "t", "f_i", "f_t", "mu", "r", "model"],
        on=["window", "center"],
    )

    # Clean up column names
    df_fv_w = df_fv_w.rename(
        {
            col: col.replace("{", "").replace("}", "").replace(",", "_")
            for col in df_fv_w.columns
        }
    )

    return df_fv_w


def get_closest_snps(position_array, center, N):
    """
    Given a list of SNP positions and a center position, return the indices of the N closest SNPs.

    Args:
        position_array (np.ndarray): 1D array of SNP positions (bp).
        center (int | float): Central genomic coordinate.
        N (int): Number of SNPs to select. Must be <= len(position_array).

    Returns:
        np.ndarray: Indices of the N closest SNPs (sorted by increasing distance, then by position).

    Raises:
        AssertionError: If `position_array` is not 1D or if `N` exceeds array length.

    Notes:
        - Ties are resolved by `np.argsort` stability on the distance array; if exact distances tie,relative order follows input order.
    """
    position_array = np.asarray(position_array)
    assert position_array.ndim == 1, "position_array must be a 1D array"
    assert N <= len(position_array), "N exceeds the number of SNPs in the array"

    distances = np.abs(position_array - center)
    closest_indices = np.argsort(distances)[:N]
    return np.sort(closest_indices)


################## Summaries


def _process_vcf(
    data_dir,
    nthreads,
    windows=None,
    step=1e5,
    step_vcf=int(1e4),
    locus_length=int(1.2e6),
    recombination_map=None,
    r_bins=None,
    min_rate=None,
    suffix=None,
    func=None,
    save_stats=False,
    stats=None,
    centers=None,
):
    from . import Parallel, delayed
    from .data import Data

    """
    Process VCF/BCF files to compute, normalize, and estimate feature vectors.

    This function scans a directory for bgzipped VCFs (``*.vcf.gz``), extracts variant
    information, computes per-window summary statistics, normalizes them using
    empirical distributions estimated from files, and returns feature vectors
    to train/predict the CNN.

    :param str data_dir:
        Directory containing bgzipped VCF or BCF files (pattern ``*vcf.gz`` and ``*bcf.gz``).
        Output Parquet and pickle files are written under the same directory.
    :param int nthreads:
        Number of threads.
    :param list windows:
        List of window sizes (in base pairs) to compute summary statistics.
    :param int step:
        Step size (in base pairs) for sliding windows.
    :param str recombination_map:
        Optional path to a recombination map. If ``None``, physical distances are
        used as a proxy for genetic distances.
    :param str suffix:
        Optional suffix appended to output file names.
    :param callable func:
        Function to  estimate summary statistics. See ``calculate_stats_vcf_flat``.

    :returns:
        A pair of Polars DataFrames:
          - **df_pred**: normalized feature vectors used for model training or prediction.
          - **df_pred_raw**: corresponding raw feature vectors before normalization.
    :rtype: tuple[polars.DataFrame, polars.DataFrame]

    :raises FileNotFoundError:
        If no VCF/BCF files matching ``*vcf.gz`` are found.
    :raises ValueError:
        If an input file cannot be parsed correctly.

    :notes:
      - Writes the following files to ``data_dir``:
          * ``fvs{suffix}.parquet`` – normalized feature vectors
          * ``fvs_raw{suffix}.parquet`` – raw feature vectors
          * ``empirical_bins{suffix}.pickle`` – empirical normalization bins
      - Each VCF is processed independently to reduce memory load.
      - The function assumes variants follow standard diploid encoding and are
        suitable for per-window summary statistic computation.
    """

    assert r_bins is None or (
        min_rate is not None and isinstance(min_rate, float)
    ), "If r_bins is not None, min_rate must be a float (minimum recombination rate simulated)."

    if func is None:
        func = calculate_stats_vcf_flat

    center = [int(step // 2), int(locus_length - step // 2)]
    if windows is None:
        windows = [100000]

    suffix_str = f"_{suffix}" if suffix is not None else ""

    # Paths and containers
    fvs_file = {}
    sims = {}
    regions = {}
    df_params_l = []

    vcf_files = sorted(
        glob.glob(os.path.join(data_dir, "*vcf.gz"))
        + glob.glob(os.path.join(data_dir, "*bcf.gz"))
    )

    if not vcf_files:
        raise FileNotFoundError(f"No VCF/BCF files found in directory: {data_dir}")

    for vcf_path in vcf_files[:]:
        # if 'chr22_' not in vcf_path:
        #     continue
        basename = os.path.basename(vcf_path)
        key = basename.replace(".vcf", "").replace(".bcf", "").replace(".gz", "")
        key = key.replace(".", "_").lower()

        fs_data = Data(
            vcf_path, nthreads=nthreads, window_size=locus_length, step=step_vcf
        )
        sim_dict = fs_data.read_vcf()

        # build parameter DataFrame
        n = len(sim_dict["region"])
        df_params_l.append(
            pl.DataFrame(
                {
                    "model": sim_dict["region"],
                    "s": np.zeros(n),
                    "t": np.zeros(n),
                    "saf": np.zeros(n),
                    "eaf": np.zeros(n),
                    "mu": np.zeros(n),
                    "r": np.zeros(n),
                }
            )
        )

        sims[key] = sim_dict["sweep"]
        regions[key] = sim_dict["region"]
        fvs_file[key] = os.path.join(data_dir, "vcfs", f"fvs_{key}.parquet")

    df_params = pl.concat(df_params_l)

    if recombination_map is not None:
        df_recombination_map = pl.read_csv(
            recombination_map,
            separator="\t",
            comment_prefix="#",
            schema=pl.Schema(
                [
                    ("chr", pl.String),
                    ("start", pl.Int64),
                    ("end", pl.Int64),
                    ("cm_mb", pl.Float64),
                    ("cm", pl.Float64),
                ]
            ),
        )
    else:
        df_recombination_map = None

    results = {}
    tmp_bins = []
    df_r_l = []

    # Pool 1: summary statistics
    # Single pool shared across all VCF files — workers stay warm,
    # numba caches persist across chromosomes.
    with Parallel(n_jobs=nthreads, backend="loky", verbose=2) as stats_pool:
        for k, vcf_file in sims.items():
            print(k)

            # compute center from region strings "chr: start-end"
            center_coords = [
                tuple(map(int, r.split(":")[-1].split("-"))) for r in regions[k]
            ]
            nchr = regions[k][0].split(":")[0]
            params = df_params.filter(pl.col("model").str.contains(f"{nchr}:"))

            if recombination_map is not None:
                cm_mb = get_cm(
                    df_recombination_map.filter(pl.col("chr") == nchr),
                    np.asarray(center_coords),
                    cm_mb=True,
                )
                print(f"cm_mb: {cm_mb}")
            if r_bins is not None:
                tmp_r = cm_mb.with_columns(
                    [
                        pl.col("cm_mb").cut(breaks=r_bins).alias("r_bins"),
                        pl.format(
                            "{}:{}-{}", pl.col("chr"), pl.col("start"), pl.col("end")
                        )
                        .alias("region")
                        .alias("iter"),
                    ]
                ).select("iter", "cm_mb", "r_bins")

                params = params.with_columns((tmp_r["cm_mb"]).alias("r"))

                mask = (tmp_r["cm_mb"] >= min_rate).to_numpy()
                exclude_r = tmp_r.filter(~mask)["iter"].to_numpy()

                # remove excluded regions from regions[k]
                exclude_set = set(exclude_r)
                regions[k] = np.array([r for r in regions[k] if r not in exclude_set])

                # filter tmp_r
                tmp_r = tmp_r.filter(mask)
                params = params.filter(mask).to_numpy()[:, 1:].astype(float)

                df_r_l.append(tmp_r)
            else:
                tmp_r = None

                if recombination_map is not None:
                    params = params.with_columns(cm_mb["cm_mb"].alias("r"))
                params = (
                    params.select(["s", "t", "saf", "eaf", "mu", "r"])
                    .to_numpy()
                    .astype(float)
                )

            _tmp_stats = func(
                vcf_file,
                regions[k],
                center=center,
                windows=windows,
                step=step,
                recombination_map=recombination_map,
                locus_length=locus_length,
                stats=stats,
                nthreads=nthreads,
                parallel_manager=stats_pool,
                centers=centers,
            )
            
            snps_df, window_df = _tmp_stats
            tmp_bins.append({"snps": snps_df, "windows": window_df})

            if not np.all(params[:, 3] == 0):
                params[:, 0] = -np.log(params[:, 0])
            results[k] = summaries({"snps": snps_df, "window": window_df}, params)

    if save_stats:
        save_pickle(f"{data_dir}/raw_statistics{suffix_str}.pickle", results)

    try:
        df_r = pl.concat(df_r_l)
    except Exception as e:
        if r_bins is not None:
            print(f"_process_vcf: failed to concat r_bins table: {e}")
        df_r = None

    empirical_bins = binned_stats(
        *normalize_neutral(tmp_bins, vcf=True, df_r_bins=df_r)
    )
    df_fv_cnn = {}
    df_fv_cnn_raw = {}

    # Pool 2: normalization
    with Parallel(n_jobs=nthreads, backend="loky", verbose=2) as norm_pool:
        for k, stats_values in results.items():
            print(k)
            df_w, df_w_raw = normalize_stats(
                stats_values,
                empirical_bins,
                region=regions[k],
                center=center,
                windows=windows,
                step=step,
                parallel_manager=norm_pool,
                nthreads=nthreads,
                vcf=True,
                df_r_bins=df_r,
                locus_length=locus_length,
                centers=centers,
            )
            df_fv_cnn[k] = df_w
            df_fv_cnn_raw[k] = df_w_raw

    df_pred = pl.concat(df_fv_cnn.values(), how="vertical")
    df_pred_raw = pl.concat(df_fv_cnn_raw.values(), how="vertical")

    df_pred = (
        df_pred.with_columns(
            pl.col("iter")
            .str.extract_groups(r"chr(\d+):(\d+)-(\d+)")
            .struct.rename_fields(["chrom", "start", "end"])
            .alias("g")
        )
        .unnest("g")
        .with_columns(pl.col(["chrom", "start", "end"]).cast(pl.Int64))
        .sort(["chrom", "start", "end"])
        .select(pl.exclude(["chrom", "start", "end"]))
    )
    df_pred_raw = (
        df_pred_raw.with_columns(
            pl.col("iter")
            .str.extract_groups(r"chr(\d+):(\d+)-(\d+)")
            .struct.rename_fields(["chrom", "start", "end"])
            .alias("g")
        )
        .unnest("g")
        .with_columns(pl.col(["chrom", "start", "end"]).cast(pl.Int64))
        .sort(["chrom", "start", "end"])
        .select(pl.exclude(["chrom", "start", "end"]))
    )

    with open(os.path.join(data_dir, f"empirical_bins{suffix_str}.pickle"), "wb") as f:
        pickle.dump(empirical_bins, f)

    df_pred.write_parquet(f"{data_dir}/fvs{suffix_str}.parquet")
    df_pred_raw.write_parquet(f"{data_dir}/fvs_raw{suffix_str}.parquet")

    return df_pred, df_pred_raw


def _process_sims(
    data_dir,
    nthreads,
    windows=None,
    step=1e5,
    r_bins=None,
    suffix=None,
    func=None,
    save_stats=False,
    locus_length=int(1.2e6),
    stats=None,
    centers=None,
):
    """
    Process ms files from simulation to compute, normalize, and estimate feature vectors.

    It scans a directory for simulation outputs (neutral and sweep), computes
    summary statistics for each replicate, normalizes the results between classes,
    and exports Parquet feature vectors to traing the CNN.

    :param str data_dir:
        Directory containing simulation output files (neutral and sweep). Expected
        substructure and file naming conventions follow the Flexsweep.Simulator class
        output (e.g., ``data_dir/sweeps/`` and ``data_dir/neutral/``).
    :param int nthreads:
        Number of threads.
    :param list windows:
        List of window sizes (in base pairs) to compute summary statistics.
    :param int step:
        Step size (in base pairs) for sliding windows.
    :param str suffix:
        Optional suffix appended to output file names.
    :param callable func:
        Function to  estimate summary statistics. See ``calculate_stats_simulations``.


    :returns:
        A pair of Polars DataFrames:
          - **df_pred**: normalized feature vectors for CNN training.
          - **df_pred_raw**: raw feature vectors.
    :rtype: tuple[polars.DataFrame, polars.DataFrame]

    :raises FileNotFoundError:
        If no simulation files are found under ``data_dir``.
    :raises ValueError:
        If simulation data are malformed or incompatible with the expected format.

    :notes:
      - Writes the following files to ``data_dir``:
          * ``fvs{suffix}.parquet`` – normalized feature vectors.
          * ``fvs_raw{suffix}.parquet`` – raw feature vectors.
          * ``empirical_bins{suffix}.pickle`` – empirical normalization bins.
      - Neutral and sweep simulations are processed jointly to derive shared
        empirical normalization bins.
      - Designed for Flexsweep.Simulators class simulations folders but can handle
        compatible structures with proper naming.
    """
    from . import Parallel, delayed
    from .data import Data

    center = [int(step // 2), int(locus_length - step // 2)]
    if windows is None:
        windows = [100000]

    suffix_str = f"_{suffix}" if suffix is not None else ""

    for folder in ("neutral", "sweep"):
        path = os.path.join(data_dir, folder)
        if not os.path.isdir(path):
            raise ValueError(f"Missing folder: {path}")
        if not glob.glob(os.path.join(path, "*")):
            raise ValueError(f"No files in folder: {path}")

    fs_data = Data(data_dir)
    sims, df_params = fs_data.read_simulations()

    results = {}
    malformed_files = {}
    d_centers = {}
    binned_data = {}
    df_r = {}

    with Parallel(n_jobs=nthreads, backend="loky", verbose=2) as parallel:
        for sim_type, sim_list in sims.items():
            # if len(sim_list) >= 5000:
            # mask = np.random.choice(mask_full, 1000,replace=False)

            if len(sim_list) > 250000:
                mask = np.random.choice(np.arange(len(sim_list)), 250000)
            else:
                mask = np.arange(0, len(sim_list))

            params = df_params.filter(pl.col("model") == sim_type)[mask, 1:].to_numpy()
            d_centers[sim_type] = np.array(center).astype(int)

            if r_bins is not None:
                tmp_r = (
                    pl.DataFrame({"r": params[:, -1] * 1e8})
                    .with_row_index(name="iter", offset=1)
                    .with_columns(pl.col("r").cut(breaks=r_bins).alias("r_bins"))
                    .select(["iter", "r_bins"])
                )
            else:
                tmp_r = None

            df_r[sim_type] = tmp_r

            # Small-batch dispatch: ~50 files per task for dynamic load balancing
            # while keeping IPC overhead low (stacked numpy per batch).
            BATCH_SIZE = 50
            n_sims = len(sim_list[mask])
            batches = []
            for i in range(0, n_sims, BATCH_SIZE):
                batch_end = min(i + BATCH_SIZE, n_sims)
                batches.append((sim_list[mask][i:batch_end], i + 1))

            _tmp_results = parallel(
                delayed(batch_simulations)(
                    batch,
                    start_idx,
                    func,
                    center,
                    windows,
                    step,
                    locus_length,
                    stats=stats,
                    centers=centers,
                )
                for batch, start_idx in batches
            )

            # Reconstruct per-file Polars DataFrames from batched numpy results
            window_cols_resolved, _, _, _ = resolve_stats(stats)
            win_schema = (
                ["iter", "center", "window"] + window_cols_resolved
                if window_cols_resolved
                else None
            )

            flat = []
            for win_stacked, snp_list, nwpf in _tmp_results:
                for i in range(len(snp_list)):
                    # Reconstruct window DataFrame from stacked numpy slice
                    win_df = None
                    if win_stacked is not None and nwpf > 0:
                        w = win_stacked[i * nwpf : (i + 1) * nwpf]
                        if not np.isnan(w[0, 0]):  # iter col set → valid file
                            win_df = pl.from_numpy(w, schema=win_schema).with_columns(
                                [
                                    pl.col("iter").cast(pl.Int64),
                                    pl.col("center").cast(pl.Int64),
                                    pl.col("window").cast(pl.Int64),
                                ]
                            )

                    # Reconstruct SNP DataFrame from dict of numpy arrays
                    snp_raw = snp_list[i]
                    snp_df = None
                    if snp_raw is not None:
                        snp_df = pl.DataFrame(snp_raw)
                        if "iter" in snp_df.columns:
                            snp_df = snp_df.with_columns(pl.col("iter").cast(pl.Int64))
                        if "positions" in snp_df.columns:
                            snp_df = snp_df.with_columns(
                                pl.col("positions").fill_nan(None).cast(pl.Int64)
                            )

                    flat.append((snp_df, win_df))

            _tmp_stats = tuple(zip(*flat))

            stats_values, params, malformed = cleaning_summaries(
                _tmp_stats, params, sim_type
            )
            malformed_files[sim_type] = malformed

            snps_list, windows_list = stats_values

            if sim_type == "neutral":
                norm_stats = [
                    {"snps": s, "windows": w} for s, w in zip(snps_list, windows_list)
                ]
                binned_data["neutral"] = binned_stats(
                    *normalize_neutral(norm_stats, df_r_bins=tmp_r)
                )

            # normalize_cut_raw expects {"snps": ..., "window": ...} per item
            raw_stats = [
                {"snps": s, "window": w} for s, w in zip(snps_list, windows_list)
            ]

            if not np.all(params[:, 0] == 0):
                params[:, 0] = -np.log(params[:, 0])
            results[sim_type] = summaries(raw_stats, params)

    if save_stats:
        save_pickle(f"{data_dir}/raw_statistics{suffix_str}.pickle", results)

    df_fv_cnn = {}
    df_fv_cnn_raw = {}
    with Parallel(n_jobs=nthreads, backend="loky", verbose=2) as parallel:
        for sim_type, stats_values in results.items():
            df_w, df_w_raw = normalize_stats(
                stats_values,
                bins=binned_data["neutral"],
                region=None,
                center=center,
                windows=windows,
                step=step,
                parallel_manager=None,
                nthreads=nthreads,
                vcf=False,
                df_r_bins=df_r[sim_type],
                locus_length=locus_length,
                centers=centers,
            )
            df_fv_cnn[sim_type] = df_w
            df_fv_cnn_raw[sim_type] = df_w_raw

    df_train = pl.concat(df_fv_cnn.values(), how="vertical")
    df_train_raw = pl.concat(df_fv_cnn_raw.values(), how="vertical")

    out_base = os.path.join(data_dir, f"fvs{suffix_str}.parquet")
    df_train.write_parquet(out_base)
    df_train_raw.write_parquet(out_base.replace(".parquet", "_raw.parquet"))

    with open(os.path.join(data_dir, f"neutral_bins{suffix_str}.pickle"), "wb") as f:
        pickle.dump(binned_data["neutral"], f)

    return df_train, df_train_raw


def summary_statistics(
    data_dir,
    vcf=False,
    nthreads=1,
    windows=[100000],
    step=1e5,
    step_vcf=1e4,
    locus_length=int(1.2e6),
    recombination_map=None,
    r_bins=None,
    min_rate=0.0,
    suffix=None,
    func=None,
    save_stats=False,
    only_normalize=False,
    stats=None,
    centers=None,
):
    """
    Compute summary statistics to create needed feature vectors for CNN training/prediction
    from either simulated or VCF/BCF input data.

    This function dispatches automatically to the appropriate backend depending
    on the ``vcf`` flag. When ``vcf=False`` (default) it processes mdiscoal
    simulations; when ``vcf=True`` it processes VCF/BCF files.

    :param str data_dir:
        Input directory. When ``vcf=False``, the root folder of simulation
        outputs containing subdirectories ``neutral/``, ``sweep/``, and
        ``params.txt.gz``. When ``vcf=True``, a directory of bgzipped VCF/BCF
        files.
    :param bool vcf:
        Whether to use the VCF/BCF processing pipeline (``True``) or the simulation
        pipeline (``False``).
        **Default:** ``False``.
    :param int nthreads:
        Number of threads.
        **Default:** ``1``.
    :param list windows:
        List of window sizes (in base pairs) to compute summary statistics.
        Supports multi-scale: e.g. ``[50000, 100000, 500000]``.
        **Default:** ``[100000]``.
    :param int step:
        Step size (in base pairs) for sliding windows. Together with
        ``locus_length``, determines the center positions:
        ``centers = range(step//2, locus_length - step//2 + step, step)``.
        **Default:** ``1e5``.
    :param list centers:
        Optional explicit list of center positions (absolute bp within
        ``[0, locus_length]``), overriding the uniform grid derived from ``step``.
        Enables non-uniform centering (e.g. Tier-A log-distance / radial sampling
        concentrated near the sweep). When ``None`` (default) the behaviour is
        unchanged: centers are derived from ``step``. The SAME ``centers`` is used
        for both statistic computation and normalization, so the ``(center, window)``
        feature layout stays internally consistent. Applies to both the simulation
        and VCF paths; for VCF the list is the sim-relative grid, remapped per region.
        **Default:** ``None``.
    :param int locus_length:
        Total locus length in base pairs.
        **Default:** ``1200000``.
    :param str recombination_map:
        Optional path to a recombination map. If ``None``, physical distances are
        used as a proxy for genetic distances.
        **Default:** ``None``.
    :param str suffix:
        Optional suffix appended to output file names.
        **Default:** ``None``.
    :param callable func:
        Function used internally for computing summary statistics per replicate
        or genomic window. See ``calculate_stats_simulations`` or ``calculate_stats_vcf_flat``

    :returns:
        Feature-vector DataFrame combining all computed summary statistics.
        Downstream pipelines may also write Parquet and normalization artifacts
        (e.g., ``fvs.parquet``, ``empirical_bins.pickle``).
    :rtype: polars.DataFrame

    :raises FileNotFoundError:
        If input files or directories are missing.
    :raises ValueError:
        If an input file or directory is malformed or inconsistent.

    :notes:
      - Internally dispatches to :func:`_process_vcf` or :func:`_process_sims`
        depending on the ``vcf`` flag.
      - Center positions are derived automatically from ``locus_length`` and
        ``step`` as ``[step//2, locus_length - step//2]``. For each center,
        stats are computed at every size in ``windows``, producing
        ``n_centers × len(windows)`` rows per replicate.
      - Supports automatic normalization of features using empirical bins.
      - Designed for use as a top-level wrapper in the Flexsweep feature-vector
        generation pipeline.

    :examples:

        From simulated data (discoal/ms format):

        >>> df = summary_statistics("./simulations", nthreads=8)

        Multi-scale windows:

        >>> df = summary_statistics("./simulations", windows=[50000, 100000, 500000], nthreads=8)

        From VCF data with recombination map:

        >>> df = summary_statistics(
        ...     "./vcf_data",
        ...     vcf=True,
        ...     recombination_map="recomb_map.csv",
        ...     nthreads=8
        ... )
    """

    if only_normalize:
        if vcf:
            return _normalize_vcf_stats(
                data_dir,
                nthreads,
                windows,
                step,
                locus_length,
                r_bins,
                suffix,
            )
        else:
            return _normalize_sims_stats(
                data_dir,
                nthreads,
                windows,
                step,
                locus_length,
                r_bins,
                suffix,
            )
    else:
        if vcf:
            if func is not None and stats is None:
                assert (
                    suffix is not None
                ), "You are using a custom function. Please input a suffix string to avoid feature vectors duplications"

            return _process_vcf(
                data_dir,
                nthreads,
                windows=windows,
                step=step,
                step_vcf=step_vcf,
                locus_length=locus_length,
                recombination_map=recombination_map,
                r_bins=r_bins,
                min_rate=min_rate,
                suffix=suffix,
                func=func,
                save_stats=save_stats,
                stats=stats,
                centers=centers,
            )
        else:
            if func is not None:
                assert (
                    suffix is not None
                ), "You are using a custom function. Please input a suffix string to avoid feature vectors duplications"

            return _process_sims(
                data_dir,
                nthreads,
                windows=windows,
                step=step,
                r_bins=r_bins,
                suffix=suffix,
                func=func if func is not None else calculate_stats_simulations,
                save_stats=save_stats,
                locus_length=locus_length,
                stats=stats,
                centers=centers,
            )


################## Stats


def run_fs_stats(
    hap,
    ac,
    rec_map,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=50000,
    hapdaf_o_max_ancest_freq=0.25,
    hapdaf_o_min_tot_freq=0.25,
    hapdaf_s_max_ancest_freq=0.10,
    hapdaf_s_min_tot_freq=0.10,
    _iter=None,
):
    """

    Wrapper to extracts per-focal-SNP neighbor pairs via
    :func:`fast_sq_freq_pairs`, then estimate DIND, hapDAF-o/s, Sratio, highfreq and lowfreq
    statistics. Results are returned as four Polars DataFrames.

    :param numpy.ndarray hap: Haplotype matrix ``(n_snps, n_samples)`` with 0/1 values.
    :param numpy.ndarray ac: Allele counts ``(n_snps, 2)`` as ``[ancestral, derived]``.
    :param numpy.ndarray rec_map: Map array; penultimate column is the window coordinate.
    :param float min_focal_freq: Minimum focal derived frequency. Default ``0.25``.
    :param float max_focal_freq: Maximum focal derived frequency. Default ``0.95``.
    :param int window_size: Window size in coordinate physicial units extracted from ``rec_map[:, -2]``. Default ``50000``.

    :returns:
        Four DataFrames in order: ``df_dind_high_low``, ``df_s_ratio``,
        ``df_hapdaf_s``, ``df_hapdaf_o``.
    :rtype: tuple[polars.DataFrame, polars.DataFrame, polars.DataFrame, polars.DataFrame]
    """

    sq_freqs, info, snps_indices = fast_sq_freq_pairs(
        hap,
        ac,
        rec_map,
        min_focal_freq=min_focal_freq,
        max_focal_freq=max_focal_freq,
        window_size=window_size,
    )
    if info.shape[0] == 0:
        return fs_stats_dataframe(info, [], [], [], [], [], [])

    results_dind, results_high, results_low = dind_high_low_from_pairs(sq_freqs, info)
    results_s_ratio = s_ratio_from_pairs(sq_freqs)
    results_hapdaf_o = hapdaf_from_pairs(
        sq_freqs, hapdaf_o_max_ancest_freq, hapdaf_o_min_tot_freq
    )
    results_hapdaf_s = hapdaf_from_pairs(
        sq_freqs, hapdaf_s_max_ancest_freq, hapdaf_s_min_tot_freq
    )

    df_dind_high_low, df_s_ratio, df_hapdaf_o, df_hapdaf_s = fs_stats_dataframe(
        info,
        results_dind,
        results_high,
        results_low,
        results_s_ratio,
        results_hapdaf_o,
        results_hapdaf_s,
        _iter=_iter,
    )
    return df_dind_high_low, df_s_ratio, df_hapdaf_o, df_hapdaf_s


def run_windows_stat(_tmp_hap, _positions, _ac, _f, window_size=None, iu=None, ju=None):
    try:
        h12_v, h2_h1, h1_v, _, k_counts = garud_h_numba(_tmp_hap)
        # k_counts = np.unique(_tmp_hap, axis=1).shape[1]
    except Exception:
        h12_v, h2_h1, h1_v, k_counts = np.nan, np.nan, np.nan, np.nan
    zns_v, omega_max = Ld(_tmp_hap)

    (
        _tajima_d,
        _theta_h,  # theta_h (Fay-Wu θH, absolute)
        _h_raw,
        _h_norm,
        _pi,  # pi absolute
        _theta_w,  # theta_w absolute
        _theta_w_pb,  # theta_w per-base
        _pi_pb,  # pi per-base
    ) = neutrality_stats(_ac, _positions)

    # Override per-base values using window_size as denominator (matching fv_workstation)
    if window_size is not None and window_size > 0:
        _pi_pb = _pi / window_size
        _theta_w_pb = _theta_w / window_size

    # haf_v = haf_top(_tmp_hap.astype(np.int8), _positions)
    haf_v = haf_top(_tmp_hap, _positions)

    max_fda = _f.max()

    dists = (
        pairwise_diffs_precomp(_tmp_hap, iu, ju, use_float32=False)
        if iu is not None
        else pairwise_diffs(_tmp_hap)
    )
    if window_size is not None:
        dists = dists / window_size
    dist_var, dist_skew, dist_kurtosis = fast_skew_kurt(dists, bias=True)

    # Schema order: pi, tajima_d, theta_w, theta_h, k_counts, haf,
    #               h1, h12, h2_h1, zns, omega_max,
    #               max_fda, dist_var, dist_skew, dist_kurtosis
    return (
        _pi_pb,
        _tajima_d,
        _theta_w_pb,
        _theta_h,
        k_counts,
        haf_v,
        h1_v,
        h12_v,
        h2_h1,
        zns_v,
        omega_max,
        max_fda,
        dist_var,
        dist_skew,
        dist_kurtosis,
    )


def calculate_stats_simulations(
    hap_data,
    _iter=1,
    center=None,
    windows=[100000],
    step=1e5,
    locus_length=int(1.2e6),
    stats=None,
    centers=None,
):
    if center is None:
        center = [int(step // 2), int(locus_length - step // 2)]

    # ── resolve which stats to compute ──────────────────────────────
    window_cols, snp_groups, compute_isafe, snp_cols = resolve_stats(stats)

    try:
        (
            hap_int,
            rec_map_01,
            ac,
            biallelic_mask,
            position_masked,
            genetic_position_masked,
        ) = parse_ms_numpy(hap_data, seq_len=locus_length)
        freqs = np.ascontiguousarray(ac[:, 1] / ac.sum(axis=1), dtype=np.float64)
        if hap_int.shape[0] != rec_map_01.shape[0]:
            return None, None
    except Exception:
        return None, None

    # SNP-level stats (full chromosome)
    snp_dfs = []

    if snp_groups or compute_isafe:
        _snp_window_size = int(np.asarray(windows).min()) // 2

        if "fs" in snp_groups:
            df_dind_high_low, df_s_ratio, df_hapdaf_s, df_hapdaf_o = run_fs_stats(
                hap_int, ac, rec_map_01, window_size=_snp_window_size
            )
            snp_dfs.extend(
                [
                    center_window_cols(df_dind_high_low, _iter=_iter),
                    center_window_cols(df_s_ratio, _iter=_iter),
                    center_window_cols(df_hapdaf_o, _iter=_iter),
                    center_window_cols(df_hapdaf_s, _iter=_iter),
                ]
            )

        if "ihs" in snp_groups:
            df_ihs = ihs_ihh(
                hap_int,
                position_masked,
                map_pos=genetic_position_masked,
                min_ehh=0.05 if locus_length > 1e6 else 0.1,
                min_maf=0.05,
                include_edges=False if locus_length > 1e6 else True,
            )
            snp_dfs.append(center_window_cols(df_ihs, _iter=_iter))

        if "nsl" in snp_groups:
            nsl_v = nsl(hap_int[freqs >= 0.05], use_threads=False)
            df_nsl = pl.DataFrame(
                {
                    "positions": position_masked[freqs >= 0.05],
                    "daf": freqs[freqs >= 0.05],
                    "nsl": nsl_v,
                }
            ).fill_nan(None)
            snp_dfs.append(center_window_cols(df_nsl, _iter=_iter))

        if "hscan" in snp_groups:
            pos_hscan, h_scores = hscan(hap_int, position_masked)
            df_hscan = pl.DataFrame(
                {
                    "positions": pos_hscan.astype(np.int64),
                    "daf": freqs,
                    "hscan": h_scores,
                }
            ).fill_nan(None)
            snp_dfs.append(center_window_cols(df_hscan, _iter=_iter))

        if compute_isafe:
            df_isafe = run_isafe(hap_int, position_masked)
            snp_dfs.append(center_window_cols(df_isafe, _iter=_iter))

        if "beta" in snp_groups:
            df_beta = run_beta_window(ac, position_masked, w=_snp_window_size)
            df_beta = df_beta.rename({"t": "beta_t"})
            daf_lookup = pl.DataFrame(
                {
                    "positions": position_masked.astype(np.int64),
                    "daf": freqs,
                }
            )
            df_beta = df_beta.join(daf_lookup, on="positions", how="left")
            snp_dfs.append(center_window_cols(df_beta, _iter=_iter))

    if snp_dfs:
        snps_joined = (
            reduce(
                lambda left, right: left.join(
                    right,
                    on=["iter", "positions", "daf"],
                    how="full",
                    coalesce=True,
                ),
                [d.lazy() for d in snp_dfs],
            )
            .sort("positions")
            .collect()
        )
        # Filter to only requested SNP columns
        if snp_cols is not None:
            keep = {"iter", "positions", "daf"} | snp_cols
            snps_joined = snps_joined.select(
                [c for c in snps_joined.columns if c in keep]
            )
        # 1a: convert to dict of numpy arrays for lightweight IPC
        snp_data = {col: snps_joined[col].to_numpy() for col in snps_joined.columns}
    else:
        snp_data = None

    # Window-level stats — return raw numpy array (no DataFrame in worker)
    if not window_cols:
        window_data = None
    else:
        if centers is not None:
            centers = np.asarray(sorted(int(c) for c in centers)).astype(int)
        elif len(center) == 1:
            centers = np.arange(center[0], center[0] + step, step).astype(int)
        else:
            centers = np.arange(center[0], center[1] + step, step).astype(int)

        all_combos = list(product(centers, windows))
        lowers = np.array([c - w // 2 for c, w in all_combos])
        uppers = np.array([c + w // 2 for c, w in all_combos])

        left_idxs = np.searchsorted(position_masked, lowers)
        right_idxs = np.searchsorted(position_masked, uppers, side="right")

        num_windows = len(all_combos)
        n_stat_cols = len(window_cols)
        iu, ju = np.triu_indices(hap_int.shape[1], k=1)
        hap_f = hap_int.astype(np.float64, copy=False)

        # When stats is None (default), use run_windows_stat directly
        if stats is None:
            results = np.full((num_windows, 3 + n_stat_cols), np.nan)
            for idx in range(num_windows):
                c, w = all_combos[idx]
                left = left_idxs[idx]
                right = right_idxs[idx]

                _tmp_hap = hap_f[left:right]
                if _tmp_hap.size == 0:
                    results[idx, :3] = [_iter, c, w]
                    continue

                _tmp_pos = position_masked[left:right]
                _ac = ac[left:right]
                _f = freqs[left:right]
                _windowed_stats = run_windows_stat(
                    _tmp_hap,
                    _tmp_pos,
                    _ac,
                    _f,
                    window_size=w,
                    iu=iu,
                    ju=ju,
                )

                results[idx, :3] = [int(_iter), c, w]
                results[idx, 3:] = _windowed_stats
        else:
            # Registry-driven: use compute_window_stats
            _stat_func = partial(
                compute_window_stats,
                stat_cols=window_cols,
                groups_needed=frozenset(
                    WINDOW_STAT_REGISTRY[c][0] for c in window_cols
                ),
            )
            results = np.full((num_windows, 3 + n_stat_cols), np.nan)
            for idx in range(num_windows):
                c, w = all_combos[idx]
                left = left_idxs[idx]
                right = right_idxs[idx]

                _tmp_hap = hap_f[left:right]
                if _tmp_hap.size == 0:
                    results[idx, :3] = [_iter, c, w]
                    continue

                _tmp_pos = position_masked[left:right]
                _ac = ac[left:right]
                _f = freqs[left:right]
                _windowed_stats = _stat_func(
                    _tmp_hap,
                    _tmp_pos,
                    _ac,
                    _f,
                    window_size=w,
                    iu=iu,
                    ju=ju,
                )

                results[idx, :3] = [int(_iter), c, w]
                results[idx, 3:] = _windowed_stats

        # 1a: return raw numpy — DataFrame reconstruction happens in _process_sims
        window_data = results

    return snp_data, window_data


def batch_simulations(
    batch_data,
    start_idx,
    func,
    center,
    windows,
    step,
    locus_length=int(1.2e6),
    stats=None,
    centers=None,
):
    """1a+1b: workers return numpy; window arrays stacked per batch for one IPC transfer."""
    snp_results = []  # list of (dict | None), one per file
    win_results = []  # list of (np.ndarray | None), one per file

    for i, hap_data in enumerate(batch_data, start=start_idx):
        try:
            out = func(
                hap_data,
                i,
                center=center,
                windows=windows,
                step=step,
                locus_length=locus_length,
                stats=stats,
                centers=centers,
            )
            if out is None or not isinstance(out, (tuple, list)) or len(out) < 2:
                snp_results.append(None)
                win_results.append(None)
            else:
                snp_results.append(out[0])
                win_results.append(out[1])
        except Exception:
            snp_results.append(None)
            win_results.append(None)

    # 1b: stack window arrays into one contiguous block for efficient IPC
    valid_wins = [w for w in win_results if w is not None]
    if valid_wins:
        nwpf = valid_wins[0].shape[0]  # num windows per file (constant)
        n_cols = valid_wins[0].shape[1]
        n_files = len(win_results)
        win_stacked = np.full((n_files * nwpf, n_cols), np.nan)
        for i, w in enumerate(win_results):
            if w is not None:
                win_stacked[i * nwpf : (i + 1) * nwpf] = w
    else:
        win_stacked = None
        nwpf = 0

    return win_stacked, snp_results, nwpf


# Schema constants — used by calculate_stats_vcf_flat.
# To add a custom stat variant: define a new list here + a leaf run_windows_stat_* function.

_WINDOW_STAT_COLS = [
    "pi",
    "tajima_d",
    "theta_w",
    "theta_h",
    "k_counts",
    "haf",
    "h1",
    "h12",
    "h2_h1",
    "zns",
    "omega_max",
    "max_fda",
    "dist_var",
    "dist_skew",
    "dist_kurtosis",
]

# Stat registries — map user-facing names to computation groups
# Window stats: (group_key, index_in_group_return_tuple)
WINDOW_STAT_REGISTRY = {
    "pi": ("neutrality", 7),  # pi_per_base
    "tajima_d": ("neutrality", 0),
    "theta_w": ("neutrality", 6),  # theta_w_per_base
    "theta_h": ("neutrality", 1),  # Fay-Wu θH (absolute)
    "fay_wu_h": ("neutrality", 3),  # normalized Fay & Wu's H (h_norm)
    "k_counts": ("garud", 4),
    "h1": ("garud", 2),
    "h12": ("garud", 0),
    "h2_h1": ("garud", 1),
    "zns": ("ld", 0),
    "omega_max": ("ld", 1),
    "haf": ("haf", 0),
    "max_fda": ("max_fda", 0),
    "dist_var": ("pairwise", 0),
    "dist_skew": ("pairwise", 1),
    "dist_kurtosis": ("pairwise", 2),
    # Extended SFS-based neutrality tests — computed per window via ext_neutrality group
    "achaz_y": ("ext_neutrality", 0),
    "zeng_e": ("ext_neutrality", 1),
    "fuli_f_star": ("ext_neutrality", 2),
    "fuli_f": ("ext_neutrality", 3),
    "fuli_d_star": ("ext_neutrality", 4),
    "fuli_d": ("ext_neutrality", 5),
    "achaz_y_star": ("ext_neutrality", 6),
    "achaz_t": ("ext_neutrality", 7),
    # Balancing selection — sliding sub-window mean per fixed window
    "ncd1": ("ncd1_win", 0),
}

# SNP stats: stat_name -> (group_key, actual_column_name)
# The column name is what appears in the output DataFrame.
SNP_STAT_REGISTRY = {
    "ihs": ("ihs", "ihs"),
    "delta_ihh": ("ihs", "delta_ihh"),
    "nsl": ("nsl", "nsl"),
    "isafe": ("isafe", "isafe"),
    "dind": ("fs", "dind"),
    "dind_high_low": ("fs", "dind"),  # alias for dind
    "highfreq": ("fs", "high_freq"),
    "high_freq": ("fs", "high_freq"),
    "lowfreq": ("fs", "low_freq"),
    "low_freq": ("fs", "low_freq"),
    "s_ratio": ("fs", "s_ratio"),
    "hapdaf_o": ("fs", "hapdaf_o"),
    "hapdaf_s": ("fs", "hapdaf_s"),
    # Balancing selection — per-SNP beta statistic (Siewert & Voight 2020)
    "beta": ("beta", "beta"),
    "beta_t": ("beta", "beta_t"),
    # Haplotype homozygosity scan (Enard et al.)
    "hscan": ("hscan", "hscan"),
}

_ALL_VALID_STATS = sorted(set(WINDOW_STAT_REGISTRY) | set(SNP_STAT_REGISTRY))


def resolve_stats(stats):
    """Partition a user stat list into window_cols, snp_groups, compute_isafe, and snp_cols.

    Returns ``(window_cols, snp_groups, compute_isafe, snp_cols)``.

    ``snp_cols`` is the set of actual DataFrame column names to keep in SNP
    output, or ``None`` when stats is ``None`` (default = keep all).

    When stats is ``None``, returns defaults matching the full flexsweep pipeline.
    """
    if stats is None:
        return list(_WINDOW_STAT_COLS), {"ihs", "nsl", "fs"}, True, None

    unknown = [
        s for s in stats if s not in WINDOW_STAT_REGISTRY and s not in SNP_STAT_REGISTRY
    ]
    if unknown:
        raise ValueError(f"Unknown stat(s): {unknown}. Valid: {_ALL_VALID_STATS}")

    window_cols = [s for s in stats if s in WINDOW_STAT_REGISTRY]
    snp_groups = set()
    snp_cols = set()
    compute_isafe = False
    for s in stats:
        if s in SNP_STAT_REGISTRY:
            group, col = SNP_STAT_REGISTRY[s]
            if group == "isafe":
                compute_isafe = True
            else:
                snp_groups.add(group)
            snp_cols.add(col)

    return window_cols, snp_groups, compute_isafe, snp_cols


def compute_window_stats(
    hap,
    pos,
    ac,
    freqs,
    window_size=None,
    iu=None,
    ju=None,
    stat_cols=None,
    groups_needed=None,
):
    """Registry-driven window stat computation.

    Called in the inner center×window loop via functools.partial that bakes in
    stat_cols and groups_needed (both picklable → joblib-safe).
    """
    group_results = {}

    if "neutrality" in groups_needed:
        _neut = list(neutrality_stats(ac, pos))
        if window_size is not None and window_size > 0:
            _neut[6] = _neut[5] / window_size  # theta_w_per_base
            _neut[7] = _neut[4] / window_size  # pi_per_base
        group_results["neutrality"] = tuple(_neut)

    if "garud" in groups_needed:
        try:
            h12_v, h2_h1, h1_v, _, k_counts = garud_h_numba(hap)
            group_results["garud"] = (h12_v, h2_h1, h1_v, None, k_counts)
        except Exception:
            group_results["garud"] = (np.nan, np.nan, np.nan, None, np.nan)

    if "ld" in groups_needed:
        group_results["ld"] = Ld(hap)

    if "haf" in groups_needed:
        group_results["haf"] = (haf_top(hap, pos),)

    if "max_fda" in groups_needed:
        group_results["max_fda"] = (freqs.max(),)

    if "pairwise" in groups_needed:
        dists = (
            pairwise_diffs_precomp(hap, iu, ju, use_float32=False)
            if iu is not None
            else pairwise_diffs(hap)
        )
        if window_size is not None:
            dists = dists / window_size
        group_results["pairwise"] = fast_skew_kurt(dists, bias=True)

    if "ext_neutrality" in groups_needed:
        # All functions take ac (S×2 derived allele counts); SFS built internally.
        # fuli_f and fuli_d require polarized ac (derived allele in ac[:,1]).
        group_results["ext_neutrality"] = (
            achaz_y(ac),  # [0] Achaz's Y       (polarized, excludes ξ₁)
            zeng_e(ac),  # [1] Zeng's E
            fuli_f_star(ac),  # [2] Fu & Li F* (no outgroup)
            fuli_f(ac),  # [3] Fu & Li F  (polarized)
            fuli_d_star(ac),  # [4] Fu & Li D* (no outgroup)
            fuli_d(ac),  # [5] Fu & Li D  (polarized)
            achaz_y_star(ac),  # [6] Achaz's Y*      (folded, excludes η₁)
            achaz_t(ac),  # [7] Achaz's T_Ω     (exponential weights, bottleneck)
        )

    if "ncd1_win" in groups_needed:
        # NCD1 computed on the window's SNP positions/frequencies; summarised as mean.
        if len(pos) >= 2:
            _ncd1 = ncd1(pos, freqs)
            group_results["ncd1_win"] = (
                float(np.nanmean(_ncd1)) if len(_ncd1) > 0 else np.nan,
            )
        else:
            group_results["ncd1_win"] = (np.nan,)

    out = []
    for col in stat_cols:
        group_key, idx = WINDOW_STAT_REGISTRY[col]
        grp = group_results.get(group_key)
        out.append(grp[idx] if grp is not None else np.nan)
    return tuple(out)


def run_isafe_region(hap_int, position_masked, start, end):
    """Compute iSAFE/SAFE scores for one non-overlapping chromosome region.

    Worker for calculate_stats_vcf_flat.  Slices hap_int and position_masked
    to [start, end], calls run_isafe, and returns a Polars DataFrame with
    columns (positions, daf, isafe) carrying absolute physical positions.
    Returns None if fewer than 10 SNPs are present in the region.
    """
    left = int(np.searchsorted(position_masked, start, side="left"))
    right = int(np.searchsorted(position_masked, end, side="right"))
    if right - left < 10:
        return None
    df = run_isafe(
        hap_int[left:right].astype(np.int8, copy=False),
        position_masked[left:right],
    ).fill_nan(None)
    if df.is_empty():
        return None
    return df


def batch_windowed_stats_flat(
    hap_f,
    ac,
    position_masked,
    combos,
    stat_func,
    stat_cols,
):
    """Worker for calculate_stats_vcf_flat.

    Computes stats for a batch of unique absolute (center, window_size) pairs
    directly on the chromosome hap matrix — no locus-window framing, no
    relative_position remapping.  Each (abs_center, window_size) pair is
    computed exactly once regardless of how many locus windows it belongs to.

    Parameters
    ----------
    hap_f : (S, N) float64 ndarray
        Full-chromosome haplotype matrix, pre-cast before joblib dispatch.
    ac : (S, 2) ndarray
        Allele counts array.
    position_masked : (S,) int32 ndarray
        Absolute physical positions after biallelic filtering.
    combos : list of (int, int)
        (abs_center, window_size) pairs to compute in this batch.
    stat_func : callable
        run_windows_stat or partial(compute_window_stats, ...).
    stat_cols : list of str
        Column names matching stat_func output length.

    Returns
    -------
    list of (abs_center, window_size, stats_tuple)
    """
    iu, ju = np.triu_indices(hap_f.shape[1], k=1)
    results = []
    for abs_center, w in combos:
        left = np.searchsorted(position_masked, abs_center - w // 2, side="left")
        right = np.searchsorted(position_masked, abs_center + w // 2, side="right")
        _hap = hap_f[left:right]
        if _hap.size == 0:
            results.append((abs_center, w, (np.nan,) * len(stat_cols)))
            continue
        _ac = ac[left:right]
        _pos = position_masked[left:right]
        _f = _hap.mean(axis=1)
        sv = stat_func(_hap, _pos, _ac, _f, window_size=w, iu=iu, ju=ju)
        results.append((abs_center, w, sv))
    return results


def calculate_stats_vcf_flat(
    vcf_file,
    region,
    center=None,
    windows=[100000],
    step=1e5,
    _iter=1,
    recombination_map=None,
    nthreads=1,
    locus_length=int(1.2e6),
    stats=None,
    # Legacy params — used only when stats is None
    stat_func=None,
    stat_cols=None,
    compute_snp_stats=True,
    parallel_manager=None,
    isafe_region_size=int(2e6),
    centers=None,
):
    """Compute per-locus-window summary statistics from a VCF with O(N) window work.

    Instead of computing stats for every (locus_window × center) pair — which
    would recompute each overlapping sub-window up to numSubWins times — this
    function:

    1. Enumerates the unique set of absolute (center, window_size) positions
       across all locus windows.
    2. Dispatches one joblib task per unique pair (via batch_windowed_stats_flat).
    3. iSAFE is fully supported via non-overlapping region tiling. It divides
       the chromosome into non-overlapping regions of isafe_region_size bp
       (default 2 Mb) and runs run_isafe once per region. SNP scores are
       absolute-position keyed so they join correctly to any locus window
       that contains them.
    4. Assembles output rows by looking up cached results per locus window.

    Parameters
    ----------
    isafe_region_size : int
        Size in bp of non-overlapping regions used to compute iSAFE.
        Valid range 1e6–5e6.  Default 2e6.
    """
    from . import Parallel, delayed

    if center is None:
        center = [int(step // 2), int(locus_length - step // 2)]

    filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in scalar divide",
    )
    np.seterr(divide="ignore", invalid="ignore")

    # resolve which stats to compute
    if stats is not None:
        window_cols, snp_groups, compute_isafe, snp_cols = resolve_stats(stats)
        if window_cols:
            stat_func_inner = partial(
                compute_window_stats,
                stat_cols=window_cols,
                groups_needed=frozenset(
                    WINDOW_STAT_REGISTRY[c][0] for c in window_cols
                ),
            )
        else:
            stat_func_inner = run_windows_stat
            window_cols = list(_WINDOW_STAT_COLS)
        stat_cols_inner = window_cols
    else:
        stat_func_inner = stat_func if stat_func is not None else run_windows_stat
        stat_cols_inner = (
            stat_cols if stat_cols is not None else list(_WINDOW_STAT_COLS)
        )
        snp_groups = {"ihs", "nsl", "fs"} if compute_snp_stats else set()
        compute_isafe = compute_snp_stats
        snp_cols = None

    # read VCF
    try:
        (
            hap_int,
            rec_map_01,
            ac,
            biallelic_mask,
            position_masked,
            genetic_position_masked,
        ) = genome_reader(vcf_file, recombination_map=recombination_map, region=None)
        freqs = np.ascontiguousarray(ac[:, 1] / ac.sum(axis=1), dtype=np.float64)
    except Exception:
        return None

    if recombination_map is None:
        genetic_position_masked = None

    genomic_windows = np.asarray(
        [tuple(map(int, r.split(":")[-1].split("-"))) for r in region]
    )
    nchr = region[0].split(":")[0]

    if centers is not None:
        centers = np.asarray(sorted(int(c) for c in centers)).astype(int)
    elif len(center) == 1:
        centers = np.arange(center[0], center[0] + step, step).astype(int)
    else:
        centers = np.arange(center[0], center[1] + step, step).astype(int)

    # build joblib tasks
    tasks = []
    snp_task_keys = []

    if "fs" in snp_groups:
        tasks.append(
            delayed(run_fs_stats)(
                hap_int,
                ac,
                rec_map_01,
                window_size=int(np.asarray(windows).min()) // 2,
            )
        )
        snp_task_keys.append("fs")

    if "ihs" in snp_groups:
        tasks.append(
            delayed(ihs_ihh)(
                hap_int,
                position_masked,
                map_pos=genetic_position_masked,
                min_ehh=0.05 if locus_length > 1e6 else 0.1,
                min_maf=0.05,
                include_edges=False if locus_length > 1e6 else True,
                use_threads=True,
            )
        )
        snp_task_keys.append("ihs")

    if "nsl" in snp_groups:
        tasks.append(delayed(nsl)(hap_int[freqs >= 0.05], use_threads=True))
        snp_task_keys.append("nsl")

    if "hscan" in snp_groups:
        tasks.append(delayed(hscan)(hap_int, position_masked))
        snp_task_keys.append("hscan")

    if "beta" in snp_groups:
        _snp_window_size = int(np.asarray(windows).min()) // 2
        tasks.append(delayed(run_beta_window)(ac, position_masked, w=_snp_window_size))
        snp_task_keys.append("beta")

    # iSAFE: non-overlapping region tiles
    # Each tile is one joblib task; run_isafe_region returns (positions, daf,
    # isafe) with absolute coordinates.  Tiles cover the full chromosome so
    # every SNP in any locus window gets a score.
    n_isafe_tasks = 0
    if compute_isafe:
        chrom_start = int(position_masked[0])
        chrom_end = int(position_masked[-1])
        isafe_starts = list(range(chrom_start, chrom_end, int(isafe_region_size)))
        isafe_ends = [s + int(isafe_region_size) for s in isafe_starts]
        isafe_ends[-1] = chrom_end + 1  # ensure last SNP is included
        n_isafe_tasks = len(isafe_starts)
        for s, e in zip(isafe_starts, isafe_ends):
            tasks.append(delayed(run_isafe_region)(hap_int, position_masked, s, e))

    # flat unique absolute (center, window_size) pairs
    # For locus window starting at locus_start, relative center c maps to
    # absolute position: locus_start + c - 1
    unique_abs_centers = sorted(
        {int(lw[0]) + int(c) - 1 for lw in genomic_windows for c in centers}
    )
    unique_combos = [(abs_c, w) for abs_c in unique_abs_centers for w in windows]

    hap_f = np.ascontiguousarray(hap_int.astype(np.float64))
    chunk_size = max(1, ceil(len(unique_combos) / (nthreads * 2)))
    tasks.extend(
        delayed(batch_windowed_stats_flat)(
            hap_f,
            ac,
            position_masked,
            unique_combos[i : i + chunk_size],
            stat_func_inner,
            stat_cols_inner,
        )
        for i in range(0, len(unique_combos), chunk_size)
    )

    # execute
    if parallel_manager is not None:
        results = parallel_manager(tasks)
    else:
        with Parallel(n_jobs=nthreads, backend="loky", verbose=2) as parallel:
            results = parallel(tasks)

    # build stat cache from flat window results
    n_snp_tasks = len(snp_task_keys)
    n_pre_window_tasks = n_snp_tasks + n_isafe_tasks
    stat_cache = {}
    for batch in results[n_pre_window_tasks:]:
        for abs_c, w, sv in batch:
            stat_cache[(abs_c, w)] = sv

    # assemble output rows per locus window
    # Output layout: center (relative), window, stat_cols..., iter ("start-end")
    n_stat_cols = len(stat_cols_inner)
    all_combos = list(product(centers, windows))
    num_combos = len(all_combos)
    out_window = []
    for lw in genomic_windows:
        locus_start = int(lw[0])
        iter_str = f"{lw[0]}-{lw[1]}"
        arr = np.full((num_combos, 2 + n_stat_cols), np.nan)
        for idx, (c, w) in enumerate(all_combos):
            abs_c = locus_start + int(c) - 1
            arr[idx, 0] = c
            arr[idx, 1] = w
            sv = stat_cache.get((abs_c, w))
            if sv is not None:
                arr[idx, 2:] = sv
        df = (
            pl.from_numpy(
                arr,
                schema=["center", "window"] + list(stat_cols_inner),
            )
            .with_columns(
                [
                    pl.col("center").cast(pl.Int64),
                    pl.col("window").cast(pl.Int64),
                ]
            )
            .with_columns(pl.lit(iter_str).alias("iter"))
        )
        out_window.append(df)

    df_window_new = pl.concat(out_window, how="vertical")
    df_window_new = df_window_new.with_columns(
        (nchr + ":" + pl.col("iter")).alias("iter")
    )

    # unpack SNP results
    snp_results = {key: results[i] for i, key in enumerate(snp_task_keys)}
    snp_dfs = []

    if "fs" in snp_results:
        df_dind_high_low, df_s_ratio, df_hapdaf_s, df_hapdaf_o = snp_results["fs"]
        df_dind_high_low = center_window_cols(df_dind_high_low, _iter=_iter)
        df_s_ratio = center_window_cols(df_s_ratio, _iter=_iter)
        df_hapdaf_o = center_window_cols(df_hapdaf_o, _iter=_iter)
        df_hapdaf_s = center_window_cols(df_hapdaf_s, _iter=_iter)
        snp_dfs.extend([df_dind_high_low, df_s_ratio, df_hapdaf_o, df_hapdaf_s])

    if "ihs" in snp_results:
        snp_dfs.append(center_window_cols(snp_results["ihs"], _iter=_iter))

    if "nsl" in snp_results:
        nsl_v = snp_results["nsl"]
        df_nsl = pl.DataFrame(
            {
                "positions": position_masked[freqs >= 0.05],
                "daf": freqs[freqs >= 0.05],
                "nsl": nsl_v,
            }
        ).fill_nan(None)
        snp_dfs.append(center_window_cols(df_nsl, _iter=_iter))

    if "hscan" in snp_results:
        pos_hscan, h_scores = snp_results["hscan"]
        df_hscan = pl.DataFrame(
            {
                "positions": pos_hscan.astype(np.int64),
                "daf": freqs,
                "hscan": h_scores,
            }
        ).fill_nan(None)
        snp_dfs.append(center_window_cols(df_hscan, _iter=_iter))

    if "beta" in snp_results:
        df_beta = snp_results["beta"].rename({"t": "beta_t"})
        daf_lookup = pl.DataFrame(
            {
                "positions": position_masked.astype(np.int64),
                "daf": freqs,
            }
        )
        df_beta = df_beta.join(daf_lookup, on="positions", how="left")
        snp_dfs.append(center_window_cols(df_beta, _iter=_iter))

    # unpack iSAFE results
    # Tile results carry absolute positions; concat directly — no mean
    # aggregation needed since each SNP appears in exactly one tile.
    # iSAFE output schema: (iter, positions, daf, isafe).
    if compute_isafe and n_isafe_tasks > 0:
        isafe_tile_results = results[n_snp_tasks : n_snp_tasks + n_isafe_tasks]
        valid_tiles = [r for r in isafe_tile_results if r is not None]
        if valid_tiles:
            df_isafe = (
                pl.concat(valid_tiles, how="vertical")
                .with_columns(pl.col("positions").cast(pl.Int64))
                .sort("positions")
            )
            df_isafe = df_isafe.with_columns(pl.lit(_iter).cast(pl.Int64).alias("iter"))
            snp_dfs.append(df_isafe)

    if snp_dfs:
        df_stats_norm = (
            reduce(
                lambda left, right: left.join(
                    right,
                    on=["iter", "positions", "daf"],
                    how="full",
                    coalesce=True,
                ),
                [d.lazy() for d in snp_dfs],
            )
            .sort("positions")
            .collect()
        ).with_columns(pl.lit(nchr).cast(pl.Utf8).alias("iter"))

        _to_drop = [c for c in ("h12", "haf") if c in df_stats_norm.columns]
        if _to_drop:
            df_stats_norm = df_stats_norm.drop(_to_drop)

        if snp_cols is not None:
            keep = {"iter", "positions", "daf"} | snp_cols
            df_stats_norm = df_stats_norm.select(
                [c for c in df_stats_norm.columns if c in keep]
            )
    else:
        df_stats_norm = None

    return df_stats_norm, df_window_new


################## Normalization


@njit(cache=True)
def relative_position(positions, window):
    return positions - window[0] + 1


def cut_snps(df, centers, windows, stats_names, fixed_center=None, iter_value=1):
    """
    Processes data within windows across multiple centers and window sizes.

    Parameters
    ----------
    normalized_df : polars.DataFrame
        DataFrame containing the positions and statistics.
    iter_value : int
        Iteration or replicate number.
    centers : list
        List of center positions to analyze.
    windows : list
        List of window sizes to use.
    stats_names : list, optional
        Names of statistical columns to compute means for.
        If None, all columns except position-related ones will be used.
    position_col : str, optional
        Name of the column containing position values.
    center_col : str, optional
        Name of the column containing center values.
    fixed_center : int, optional
        If provided, use this fixed center value instead of the ones in centers list.

    Returns
    -------
    polars.DataFrame
        DataFrame with aggregated statistics for each center and window.
    """
    # If stats_names not provided, use all appropriate columns

    # reset centers

    if centers is None:
        centers = np.arange(5e5, 7e5 + 1e4, 1e4).astype(int)
        sim_mid = 6e5
    else:
        sim_mid = (centers[0] + centers[-1]) // 2

    centers = np.asarray(centers).astype(int)

    if fixed_center is not None:
        centers_abs = np.array([fixed_center + c - sim_mid for c in centers]).astype(
            int
        )
    else:
        centers_abs = centers

    results = []
    # out = []
    for c, w in list(product(centers_abs, windows)):
        query = df.lazy()

        # HUGE BUG, REPEATING ACTUAL CENTER/WINDOW VALUES BASED ON ALL CENTERS SIZE
        # 1.2MB simulations derives into 21 center/windows combinations
        # if fixed_center is not None:
        #     c_fix = fixed_center - c
        # else:
        #     c_fix = c
        if fixed_center is not None:
            c_sim = int(c - fixed_center + sim_mid)
        else:
            c_sim = int(c)

        # Filter data by center and window boundaries
        # Define window boundaries
        lower = c - (w // 2)
        upper = c + (w // 2)

        window_data = query.filter(
            (pl.col("positions") >= lower) & (pl.col("positions") <= upper)
        )

        # Calculate mean statistics for window
        window_stats = window_data.select(stats_names).fill_nan(None).mean().collect()

        # Add metadata columns
        metadata_cols = [
            pl.lit(iter_value).alias("iter"),
            pl.lit(c_sim).cast(pl.Int64).alias("center"),
            pl.lit(int(w)).cast(pl.Int64).alias("window"),
        ]

        results.append(window_stats.with_columns(metadata_cols))

    return (
        pl.concat(results, how="vertical").select(
            ["iter", "center", "window"] + stats_names
        )
        if results
        else None
    )


def bin_values(values, freq=0.02):
    """
    Bins allele frequency data into discrete frequency intervals (bins) for further analysis.

    This function takes a DataFrame containing a column of derived allele frequencies ("daf")
    and bins these values into specified frequency intervals. The resulting DataFrame will
    contain a new column, "freq_bins", which indicates the frequency bin for each data point.

    Parameters
    ----------
    values : pandas.DataFrame
        A DataFrame containing at least a column labeled "daf", which represents the derived
        allele frequency for each variant.

    freq : float, optional (default=0.02)
        The width of the frequency bins. This value determines how the frequency range (0, 1)
        is divided into discrete bins. For example, a value of 0.02 will create bins
        such as [0, 0.02], (0.02, 0.04], ..., [0.98, 1.0].

    Returns
    -------
    values_copy : pandas.DataFrame
        A copy of the original DataFrame, with an additional column "freq_bins" that contains
        the frequency bin label for each variant. The "freq_bins" are categorical values based
        on the derived allele frequencies.

    Notes
    -----
    - The `pd.cut` function is used to bin the derived allele frequencies into intervals.
    - The bins are inclusive of the lowest boundary (`include_lowest=True`) to ensure that
      values exactly at the boundary are included in the corresponding bin.
    - The resulting bins are labeled as strings with a precision of two decimal places.
    """
    # Modify the copy
    values_copy = pl.concat(
        [
            values,
            values["daf"]
            .cut(np.arange(0, 1 + freq, freq))
            .to_frame()
            .rename({"daf": "freq_bins"}),
        ],
        how="horizontal",
    )

    try:
        return values_copy.sort("iter", "positions")
    except Exception:
        return values_copy.sort("chr", "positions")


def _parse_breaks_from_rbins(df_r_bins):
    # expects categorical interval strings like "(a, b]"
    uniq = df_r_bins["r_bins"].unique().sort()
    breaks = [float(re.search(r",\s*([0-9.]+)\]$", s).group(1)) for s in uniq]
    return breaks


def snps_to_r_bins(
    snps_df,
    df_r_bins_windows,
    mode="nearest_center",
):
    """
    Returns snps_df with added columns: cm_mb, r_bins (then you can drop cm_mb if you want).
    mode:
      - 'nearest_center': pick window whose center is closest to SNP position
      - 'mean_overlap': mean cm_mb across all overlapping windows
      - 'max_overlap': max cm_mb across all overlapping windows
    """
    breaks = _parse_breaks_from_rbins(df_r_bins_windows)

    snps = snps_df.with_columns(
        [pl.col("chr").cast(pl.Categorical), pl.col("positions").cast(pl.Int64)]
    )

    windows = df_r_bins_windows.with_columns(
        [
            pl.col("chr").cast(pl.Categorical),
            pl.col("start").cast(pl.Int64),
            pl.col("end").cast(pl.Int64),
            pl.col("cm_mb").cast(pl.Float32),
        ]
    ).sort(["chr", "start"])

    snps_parts = snps.partition_by("chr", as_dict=True)
    win_parts = windows.partition_by("chr", as_dict=True)

    annotated_parts = []

    for chrom, snps_chrom in snps_parts.items():
        w = win_parts.get(chrom)
        if w is None or w.height == 0:
            continue

        pos = (
            snps_chrom.select(pl.col("positions").unique().sort())
            .to_series()
            .to_numpy()
        )
        pos = pos.astype(np.int64, copy=False)

        w_start = w["start"].to_numpy().astype(np.int64, copy=False)
        w_end = w["end"].to_numpy().astype(np.int64, copy=False)
        w_cm = w["cm_mb"].to_numpy().astype(np.float32, copy=False)

        if mode == "nearest_center":
            w_center = (w_start + w_end) // 2
            idx = np.searchsorted(w_center, pos, side="left")
            idx = np.clip(idx, 1, len(w_center) - 1)
            left = idx - 1
            right = idx
            choose_right = np.abs(w_center[right] - pos) < np.abs(w_center[left] - pos)
            best = np.where(choose_right, right, left)
            cm_assigned = w_cm[best]

        elif mode in ("mean_overlap", "max_overlap"):
            # windows with start <= p are [0:right)
            right = np.searchsorted(w_start, pos, side="right")
            # overlapping also needs end > p -> left boundary in w_end
            left = np.searchsorted(w_end, pos, side="left")

            cm_assigned = np.full(pos.shape[0], np.nan, dtype=np.float32)
            for i in range(pos.shape[0]):
                l_idx = left[i]
                r_idx = right[i]
                if l_idx >= r_idx:
                    continue
                vals = w_cm[l_idx:r_idx]
                cm_assigned[i] = (
                    float(vals.mean()) if mode == "mean_overlap" else float(vals.max())
                )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        tmp = pl.DataFrame(
            {
                "chr": np.repeat(
                    chrom[0] if isinstance(chrom, tuple) else chrom, pos.size
                ),
                "positions": pos,
                "cm_mb": cm_assigned,
            },
            schema_overrides={
                "chr": pl.Categorical,
                "positions": pl.Int64,
                "cm_mb": pl.Float32,
            },
        ).with_columns(pl.col("cm_mb").cut(breaks=breaks).alias("r_bins"))

        annotated_parts.append(tmp)

    matches = pl.concat(annotated_parts, rechunk=False) if annotated_parts else None

    if matches is None:
        return snps_df

    return snps.join(matches, on=["chr", "positions"], how="left")


def normalize_neutral(d_stats_neutral, vcf=False, df_r_bins=None):
    """
    Calculates the expected mean and standard deviation of summary statistics
    from neutral simulations, used for normalization in downstream analyses.

    This function processes a DataFrame of neutral simulation statistics, bins the
    values based on frequency, and computes the mean (expected) and standard deviation
    for each bin. These statistics are used as a baseline to normalize sweep or neutral simulations

    Parameters
    ----------
    df_stats_neutral : list or pandas.DataFrame
        A list or concatenated pandas DataFrame containing the neutral simulation statistics.
        The DataFrame should contain frequency data and various summary statistics,
        including H12 and HAF, across multiple windows and bins.

    Returns
    -------
    expected : pandas.DataFrame
        A DataFrame containing the mean (expected) values of the summary statistics
        for each frequency bin. The frequency bins are the index, and the columns
        are the summary statistics.

    stdev : pandas.DataFrame
        A DataFrame containing the standard deviation of the summary statistics
        for each frequency bin. The frequency bins are the index, and the columns
        are the summary statistics.

    Notes
    -----
    - The function first concatenates the neutral statistics, if provided as a list,
      and bins the values by frequency using the `bin_values` function.
    - It computes both the mean and standard deviation for each frequency bin, which
      can later be used to normalize observed statistics (e.g., from sweeps).
    - The summary statistics processed exclude window-specific statistics such as "h12" and "haf."

    """

    snps_list = [d["snps"] for d in d_stats_neutral if d["snps"] is not None]
    windows_list = [d["windows"] for d in d_stats_neutral if d["windows"] is not None]

    try:
        if not snps_list:
            raise ValueError("No SNP stats available (window-only mode)")
        tmp_neutral_snps = pl.concat(snps_list, how="vertical", rechunk=False)

        # For VCF, associate each SNP to the nearest window center (default) to assign r_bins
        if df_r_bins is not None:
            if vcf:
                tmp_neutral_snps = tmp_neutral_snps.rename({"iter": "chr"})

                df_r_bins_w = (
                    df_r_bins.with_columns(
                        [
                            pl.col("iter").str.extract(r"^([^:]+)", 1).alias("chr"),
                            pl.col("iter")
                            .str.extract(r":(\d+)-", 1)
                            .cast(pl.Int64)
                            .alias("start"),
                            pl.col("iter")
                            .str.extract(r"-(\d+)$", 1)
                            .cast(pl.Int64)
                            .alias("end"),
                        ]
                    )
                    .select(["chr", "start", "end", "cm_mb", "r_bins", "iter"])
                    .sort(["chr", "start"])
                )

                tmp_neutral_snps = snps_to_r_bins(tmp_neutral_snps, df_r_bins_w)

            else:
                tmp_neutral_snps = tmp_neutral_snps.join(
                    df_r_bins, on="iter", how="left"
                )

        df_binned = bin_values(tmp_neutral_snps).fill_nan(None)

        group_keys = ["freq_bins"] + (
            ["r_bins"]
            if (df_r_bins is not None and "r_bins" in df_binned.columns)
            else []
        )

        stat_cols = [
            c
            for c in df_binned.columns
            if c
            not in ("iter", "chr", "positions", "daf", "freq_bins", "r_bins", "cm_mb")
        ]

        expected = (
            df_binned.group_by(group_keys)
            .agg(pl.col(stat_cols).mean())
            .sort(group_keys)
            .fill_nan(None)
        )

        stdev = (
            df_binned.group_by(group_keys)
            .agg(pl.col(stat_cols).std())
            .sort(group_keys)
            .fill_nan(None)
        )

    except Exception as e:
        print(f"normalize_neutral SNP bins failed: {e}")
        expected, stdev = None, None

    try:
        if not windows_list:
            raise ValueError("No window stats available (SNP-only mode)")
        df_window = pl.concat(windows_list, rechunk=False).fill_nan(None)

        if df_r_bins is not None:
            df_window = df_window.join(
                df_r_bins.select(pl.exclude("cm_mb")), on="iter", how="left"
            )

        group_w = ["center", "window"] + (
            ["r_bins"]
            if (df_r_bins is not None and "r_bins" in df_window.columns)
            else []
        )

        # exclude iter + keys from aggregation
        win_stat_cols = [
            c
            for c in df_window.columns
            if c not in ("iter", "center", "window", "r_bins")
        ]

        df_window_mean = (
            df_window.group_by(group_w).agg(pl.col(win_stat_cols).mean()).sort(group_w)
        )

        df_window_std = (
            df_window.group_by(group_w).agg(pl.col(win_stat_cols).std()).sort(group_w)
        )
    except Exception as e:
        print(f"normalize_neutral window bins failed: {e}")
        df_window_mean = None
        df_window_std = None

    return ([expected, df_window_mean], [stdev, df_window_std])


def normalize_stats(
    stats_values,
    bins,
    region=None,
    center=[5e5, 7e5],
    windows=[50000, 100000, 200000, 500000, 1000000],
    step=1e4,
    parallel_manager=None,
    nthreads=1,
    vcf=False,
    df_r_bins=None,
    locus_length=int(1.2e6),
    centers=None,
):
    df_fv, df_fv_raw = normalization_raw(
        deepcopy(stats_values),
        bins,
        region=region,
        center=center,
        windows=windows,
        step=step,
        parallel_manager=parallel_manager,
        nthreads=nthreads,
        vcf=vcf,
        df_r_bins=df_r_bins,
        locus_length=locus_length,
        centers=centers,
    )

    df_fv_w = pivot_feature_vectors(df_fv, vcf=vcf)
    df_fv_w_raw = pivot_feature_vectors(df_fv_raw, vcf=vcf)

    df_fv_w = df_fv_w.fill_nan(None)
    num_nans = (
        df_fv_w.select(pl.exclude(["iter", "s", "t", "f_i", "f_t", "mu", "r", "model"]))
        .transpose()
        .null_count()
        .to_numpy()
        .flatten()
    )
    df_fv_w = df_fv_w.filter(
        num_nans
        < int(
            df_fv_w.select(
                pl.exclude(["iter", "s", "t", "f_i", "f_t", "r", "mu", "model"])
            ).shape[1]
            * 0.1
        )
    ).fill_null(0)

    if not vcf:
        df_fv_w.sort(["iter", "model"])

    df_fv_w_raw = df_fv_w_raw.fill_nan(None)
    return df_fv_w, df_fv_w_raw


def batch_normalize_cut_raw(
    batch_data, bins, center, windows, step, df_r_bins, centers=None
):
    """Process a batch of normalize_cut_raw calls."""
    results_norm = []
    results_raw = []

    for snps_values in batch_data:
        try:
            df_norm, df_raw = normalize_cut_raw(
                snps_values, bins, center, windows, step, df_r_bins, centers=centers
            )
            results_norm.append(df_norm)
            results_raw.append(df_raw)
        except Exception as e:
            print(f"Error in normalize_cut_raw: {e}")
            results_norm.append(None)
            results_raw.append(None)

    return results_norm, results_raw


def batch_cut_snps(batch_data, centers, windows, stats_names):
    """
    CHANGE:
    - Preserve output length and ordering.
    - Fail fast if anything goes wrong (to avoid silent normalized/raw scrambling).
    """
    results = []
    for df, coord, iter_val in batch_data:
        # let exceptions propagate (or raise a clearer one)
        results.append(
            cut_snps(
                df,
                centers,
                windows,
                stats_names,
                fixed_center=coord,
                iter_value=iter_val,
            )
        )
    return results


def normalization_raw(
    stats_values,
    bins,
    region=None,
    center=[5e5, 7e5],
    step=1e4,
    windows=[50000, 100000, 200000, 500000, 1000000],
    vcf=False,
    df_r_bins=None,
    nthreads=1,
    parallel_manager=None,
    locus_length=int(1.2e6),
    centers=None,
):
    from . import Parallel, delayed

    df_stats, params = stats_values

    center = np.asarray(center).astype(int)
    windows = np.asarray(windows).astype(int)

    if vcf:
        nchr = region[0].split(":")[0]
        center_coords = [tuple(map(int, r.split(":")[-1].split("-"))) for r in region]
        center_g = np.array([(a + b) // 2 for a, b in center_coords])

        df_window = df_stats.get("window")
        has_snps = df_stats.get("snps") is not None

        if has_snps:
            snps_values = df_stats["snps"].sort(["iter", "positions"])
            stats_names = [
                c for c in snps_values.columns if c not in ("iter", "positions", "daf")
            ]

            try:
                if df_r_bins is not None:
                    breaks = [
                        float(re.search(r",\s*([0-9.]+)\]$", s).group(1))
                        for s in df_r_bins["r_bins"].unique().sort()
                    ]

                    tmp_neutral_snps = snps_values.rename({"iter": "chr"}).with_columns(
                        pl.col("chr").cast(pl.Categorical)
                    )
                    nchr = tmp_neutral_snps["chr"].unique().item()
                    df_r_bins_w = (
                        (
                            df_r_bins.with_columns(
                                [
                                    pl.col("iter")
                                    .str.extract(r"^([^:]+)", 1)
                                    .alias("chr"),
                                    pl.col("iter")
                                    .str.extract(r":(\d+)-", 1)
                                    .cast(pl.Int64)
                                    .alias("start"),
                                    pl.col("iter")
                                    .str.extract(r"-(\d+)$", 1)
                                    .cast(pl.Int64)
                                    .alias("end"),
                                ]
                            )
                            .select(["chr", "start", "end", "cm_mb", "r_bins", "iter"])
                            .sort(["chr", "start"])
                        )
                        .filter(pl.col("chr") == nchr)
                        .sort("start")
                    )

                    # unique positions reduces work; join back later
                    pos = (
                        tmp_neutral_snps.select(pl.col("positions").unique().sort())
                        .to_series()
                        .to_numpy()
                    )
                    pos = pos.astype(np.int64, copy=False)

                    # w = df_r_bins_w.sort("start")
                    w_start = (
                        df_r_bins_w["start"].to_numpy().astype(np.int64, copy=False)
                    )
                    w_end = df_r_bins_w["end"].to_numpy().astype(np.int64, copy=False)
                    w_cm = (
                        df_r_bins_w["cm_mb"].to_numpy().astype(np.float32, copy=False)
                    )

                    w_center = (w_start + w_end) // 2
                    idx = np.searchsorted(w_center, pos, side="left")
                    idx = np.clip(idx, 1, len(w_center) - 1)
                    left = idx - 1
                    right = idx
                    choose_right = np.abs(w_center[right] - pos) < np.abs(
                        w_center[left] - pos
                    )
                    best = np.where(choose_right, right, left)
                    cm_assigned = w_cm[best]

                    tmp = pl.DataFrame(
                        {
                            "chr": np.repeat(
                                nchr if isinstance(nchr, tuple) else nchr, pos.size
                            ),
                            "positions": pos,
                            "cm_mb": cm_assigned,
                        },
                        schema_overrides={
                            "chr": pl.Categorical,
                            "positions": pl.Int64,
                            "cm_mb": pl.Float32,
                        },
                    ).with_columns(pl.col("cm_mb").cut(breaks=breaks).alias("r_bins"))

                    snps_values = (
                        tmp_neutral_snps.join(tmp, on=["chr", "positions"], how="left")
                        if tmp is not None
                        else tmp_neutral_snps
                    ).select(pl.exclude("cm_mb"))

                    binned_values = (
                        bin_values(snps_values)
                        .sort("positions")
                        .select(pl.exclude("chr"))
                    )
                    df_window = df_window.join(df_r_bins, on=["iter"]).select(
                        pl.exclude("cm_mb")
                    )

                else:
                    binned_values = bin_values(snps_values)

            except Exception:
                stats_names = None
                binned_values = None

            normalized_df, normalized_window = normalize_snps_statistics(
                binned_values, df_window, bins, stats_names
            )

            # centers range
            if centers is not None:
                centers = np.asarray(sorted(int(c) for c in centers)).astype(int)
            elif len(center) == 2:
                centers = np.arange(center[0], center[1] + step, step).astype(int)
            else:
                centers = center

            left_idxs = np.searchsorted(
                normalized_df["positions"].to_numpy(),
                center_g - (center[0] + center[-1]) // 2,
                side="left",
            )
            right_idxs = np.searchsorted(
                normalized_df["positions"].to_numpy(),
                center_g + (center[0] + center[-1]) // 2,
                side="right",
            )

            tmp_normalized = [
                normalized_df.slice(start, end - start)
                for start, end in zip(left_idxs, right_idxs)
            ]
            tmp_raw = [
                binned_values.slice(start, end - start)
                for start, end in zip(left_idxs, right_idxs)
            ]

            # CHANGE: keep deterministic ordering: [normalized..., raw...]
            all_data = [
                (df, coord, coord) for df, coord in zip(tmp_normalized, center_g)
            ] + [(df, coord, coord) for df, coord in zip(tmp_raw, center_g)]

            batch_size = max(100, len(all_data) // (max(nthreads, 1) * 2))
            batches = [
                all_data[i : i + batch_size]
                for i in range(0, len(all_data), batch_size)
            ]

            _cut_tasks = (
                delayed(batch_cut_snps)(batch, centers, windows, stats_names)
                for batch in batches
            )
            if parallel_manager is not None:
                batch_results = parallel_manager(_cut_tasks)
            else:
                with Parallel(n_jobs=nthreads, backend="loky", verbose=2) as parallel:
                    batch_results = parallel(_cut_tasks)

            # CHANGE: flatten WITHOUT dropping anything
            out_cut = [item for batch in batch_results for item in batch]

            # CHANGE: enforce invariant (prevents silent scrambling)
            expected_len = 2 * center_g.size
            if len(out_cut) != expected_len:
                raise RuntimeError(
                    f"cut_snps produced {len(out_cut)} results; expected {expected_len}. "
                    "Do not drop/skip items before splitting normalized/raw."
                )
            if any(x is None for x in out_cut):
                raise RuntimeError(
                    "cut_snps returned None for at least one element; "
                    "refuse to continue because it scrambles normalized/raw alignment."
                )

            df_fv_n = pl.concat(out_cut[: center_g.size])
            df_fv_n_raw = pl.concat(out_cut[center_g.size :])

            _half = locus_length // 2
            df_fv_n = df_fv_n.with_columns(
                (
                    f"{nchr}:"
                    + (pl.col("iter") - _half + 1).cast(int).cast(str)
                    + "-"
                    + (pl.col("iter") + _half).cast(int).cast(str)
                ).alias("iter")
            )
            df_fv_n_raw = df_fv_n_raw.with_columns(
                (
                    f"{nchr}:"
                    + (pl.col("iter") - _half + 1).cast(int).cast(str)
                    + "-"
                    + (pl.col("iter") + _half).cast(int).cast(str)
                ).alias("iter")
            )

            # window joins unchanged
            if normalized_window is not None:
                df_fv_n = df_fv_n.join(
                    normalized_window,
                    on=["iter", "center", "window"],
                    how="full",
                    coalesce=True,
                )

            if df_window is not None:
                df_fv_n_raw = df_fv_n_raw.join(
                    df_window,
                    on=["iter", "center", "window"],
                    how="full",
                    coalesce=True,
                )

        else:
            # No SNP stats — window-only path
            normalized_window = None
            if df_window is not None and bins is not None:
                # Mirror the has_snps path: join r_bins onto df_window before
                # normalization so window bins are conditioned on r correctly.
                _df_window_norm = df_window
                if df_r_bins is not None:
                    _df_window_norm = _df_window_norm.join(
                        df_r_bins.select(pl.exclude("cm_mb")), on="iter", how="left"
                    )
                normalized_window = normalize_snps_statistics(
                    None, _df_window_norm, bins, None
                )[1]

            if normalized_window is not None:
                df_fv_n = normalized_window
            elif df_window is not None:
                df_fv_n = df_window
            else:
                _half = locus_length // 2
                iter_labels = [
                    f"{nchr}:{int(cg - _half + 1)}-{int(cg + _half)}" for cg in center_g
                ]
                df_fv_n = pl.DataFrame({"iter": iter_labels})

            df_fv_n_raw = df_window if df_window is not None else df_fv_n

        df_params_unpack = pl.DataFrame(
            np.repeat(
                params,
                df_fv_n.select(["center", "window"])
                .unique()
                .sort(["center", "window"])
                .shape[0],
                axis=0,
            ),
            schema=["s", "t", "f_i", "f_t", "mu", "r"],
        )

        df_fv_n = pl.concat([df_params_unpack, df_fv_n], how="horizontal")
        df_fv_n_raw = pl.concat([df_params_unpack, df_fv_n_raw], how="horizontal")

        force_order = ["iter"] + [col for col in df_fv_n.columns if col != "iter"]
        df_fv_n = df_fv_n.select(force_order)

        force_order_raw = ["iter"] + [
            col for col in df_fv_n_raw.columns if col != "iter"
        ]
        df_fv_n_raw = df_fv_n_raw.select(force_order_raw)

        return df_fv_n, df_fv_n_raw

    else:
        batch_size = max(100, len(df_stats) // nthreads)
        batches = [
            df_stats[i : i + batch_size] for i in range(0, len(df_stats), batch_size)
        ]

        if parallel_manager is None:
            batch_results = Parallel(n_jobs=nthreads, verbose=10)(
                delayed(batch_normalize_cut_raw)(
                    batch, bins, center, windows, step, df_r_bins, centers
                )
                for batch in batches
            )
        else:
            batch_results = parallel_manager(
                delayed(batch_normalize_cut_raw)(
                    batch, bins, center, windows, step, df_r_bins, centers
                )
                for batch in batches
            )
            # Flatten the batched results
        df_fv_n_l = [
            item
            for batch_norm, _ in batch_results
            for item in batch_norm
            if item is not None
        ]
        df_fv_n_l_raw = [
            item
            for _, batch_raw in batch_results
            for item in batch_raw
            if item is not None
        ]

        df_fv_n = pl.concat(df_fv_n_l).with_columns(
            pl.col(["iter", "window", "center"]).cast(pl.Int64)
        )
        df_fv_n_raw = pl.concat(df_fv_n_l_raw).with_columns(
            pl.col(["iter", "window", "center"]).cast(pl.Int64)
        )

        df_params_unpack = pl.DataFrame(
            np.repeat(
                params,
                df_fv_n.select(["center", "window"])
                .unique()
                .sort(["center", "window"])
                .shape[0],
                axis=0,
            ),
            schema=["s", "t", "f_i", "f_t", "mu", "r"],
        )

    df_fv_n = pl.concat(
        [df_params_unpack, df_fv_n],
        how="horizontal",
    )
    df_fv_n_raw = pl.concat(
        [df_params_unpack, df_fv_n_raw],
        how="horizontal",
    )

    force_order = ["iter"] + [col for col in df_fv_n.columns if col != "iter"]
    df_fv_n = df_fv_n.select(force_order)
    force_order_raw = ["iter"] + [col for col in df_fv_n_raw.columns if col != "iter"]
    df_fv_n_raw = df_fv_n_raw.select(force_order_raw)

    return df_fv_n, df_fv_n_raw


def normalize_cut_raw(
    snps_values,
    bins,
    center=[5e5, 7e5],
    windows=[50000, 100000, 200000, 500000, 1000000],
    step=int(1e4),
    df_r_bins=None,
    centers=None,
):
    """
    Sims-only refactor:
    - join df_r onto SNP and window tables (by iter) BEFORE binning/normalization
    - normalize using neutral bins conditional on (freq_bins, r_bins) when available
    - drop r_bins from outputs so feature dimension stays unchanged
    """
    # Get _iter from whichever data is non-None
    if snps_values["snps"] is not None:
        _iter = snps_values["snps"]["iter"].unique().to_numpy()
    elif snps_values["window"] is not None:
        _iter = snps_values["window"]["iter"].unique().to_numpy()
    else:
        return None, None

    if centers is not None:
        centers = np.asarray(sorted(int(c) for c in centers)).astype(int)
    elif len(center) == 2:
        centers = np.arange(center[0], center[1] + step, step).astype(int)
    else:
        centers = center

    # Merge SNP-level stats (already pre-joined in calculate_stats_simulations)
    if snps_values["snps"] is not None:
        try:
            df = snps_values["snps"]

            # attach r_bins per replicate
            if df_r_bins is not None:
                df = df.join(df_r_bins, on="iter", how="left")

            # stats columns only (exclude keys + bins)
            stats_names = [
                c
                for c in df.columns
                if c not in ("iter", "positions", "daf", "freq_bins", "r_bins")
            ]

            binned_values = bin_values(df)
        except Exception as e:
            print(f"normalize_cut_raw SNP merge/bin failed: {e}")
            df, binned_values, stats_names = None, None, None
    else:
        df, binned_values, stats_names = None, None, None

    # Window-level stats
    if snps_values["window"] is not None:
        try:
            df_window = snps_values["window"].select(pl.exclude("positions"))
            if df_r_bins is not None:
                df_window = df_window.join(df_r_bins, on="iter", how="left")
        except Exception as e:
            print(f"normalize_cut_raw window join failed: {e}")
            df_window = None
    else:
        df_window = None

    normalized_df, normalized_window = normalize_snps_statistics(
        binned_values, df_window, bins, stats_names
    )

    # Drop r_bins from normalized artifacts to keep downstream feature schema unchanged
    if normalized_df is not None and "r_bins" in normalized_df.columns:
        normalized_df = normalized_df.drop("r_bins")
    if normalized_window is not None and "r_bins" in normalized_window.columns:
        normalized_window = normalized_window.drop("r_bins")
    if df_window is not None and "r_bins" in df_window.columns:
        df_window = df_window.drop("r_bins")

    if normalized_df is not None and normalized_window is not None:
        df_out = cut_snps(
            normalized_df,
            centers,
            windows,
            stats_names,
            fixed_center=None,
            iter_value=_iter,
        )
        df_out_raw = cut_snps(
            df, centers, windows, stats_names, fixed_center=None, iter_value=_iter
        )
        df_out = df_out.join(
            normalized_window,
            on=["iter", "center", "window"],
            how="full",
            coalesce=True,
        )
        df_out_raw = df_out_raw.join(
            df_window,
            on=["iter", "center", "window"],
            how="full",
            coalesce=True,
        )
    elif normalized_df is not None and normalized_window is None:
        df_out = cut_snps(
            normalized_df,
            centers,
            windows,
            stats_names,
            fixed_center=None,
            iter_value=_iter,
        )
        df_out_raw = cut_snps(
            df, centers, windows, stats_names, fixed_center=None, iter_value=_iter
        )
    elif normalized_df is None and normalized_window is not None:
        df_out = normalized_window
        df_out_raw = df_window
    else:
        df_out, df_out_raw = None, None

    _CANONICAL_STAT_ORDER = [
        "iter",
        "center",
        "window",
        "dind",
        "high_freq",
        "low_freq",
        "s_ratio",
        "hapdaf_s",
        "hapdaf_o",
        "ihs",
        "delta_ihh",
        "isafe",
        "nsl",
        "pi",
        "tajima_d",
        "theta_w",
        "theta_h",
        "k_counts",
        "haf",
        "h1",
        "h12",
        "h2_h1",
        "zns",
        "omega_max",
        "max_fda",
        "dist_var",
        "dist_skew",
        "dist_kurtosis",
    ]

    def _reorder(df):
        if df is None:
            return df
        present = [c for c in _CANONICAL_STAT_ORDER if c in df.columns]
        return df.select(present)

    return _reorder(df_out), _reorder(df_out_raw)


def normalize_snps_statistics(df_snps, df_window, bins, stats_names, dps_shape=False):
    # SNP normalization
    if df_snps is not None:
        snp_join = ["freq_bins"]
        # keep r_bins logic; for your target case r_bins absent
        if (
            "r_bins" in df_snps.columns
            and bins.mean[0] is not None
            and "r_bins" in bins.mean[0].columns
        ):
            snp_join.append("r_bins")

        neutral_means = bins.mean[0].select(snp_join + stats_names)
        neutral_stds = bins.std[0].select(snp_join + stats_names)

        normalized_df = (
            df_snps.join(
                neutral_means,
                on=snp_join,
                how="left",
                coalesce=True,
                suffix="_mean_neutral",
            )
            .join(
                neutral_stds,
                on=snp_join,
                how="left",
                coalesce=True,
                suffix="_std_neutral",
            )
            .fill_nan(None)
        )

        normalized_df = normalized_df.with_columns(
            [
                # pl.when(
                #     pl.col(f"{s}_std_neutral").is_null()
                #     | (pl.col(f"{s}_std_neutral") == 0)
                # )
                # .then(pl.lit(0.0))
                # .otherwise
                (
                    (pl.col(s) - pl.col(f"{s}_mean_neutral"))
                    / pl.col(f"{s}_std_neutral")
                ).alias(s)
                for s in stats_names
            ]
        ).select(["positions"] + stats_names)
    else:
        normalized_df = None

    # window normalization: keep your join-based approach if you want,
    # but to match original results best, also remove the std==0 clamp here.
    if df_window is None:
        return (normalized_df, None)

    win_join = ["center", "window"]
    if (
        "r_bins" in df_window.columns
        and bins.mean[1] is not None
        and "r_bins" in bins.mean[1].columns
    ):
        win_join.append("r_bins")

    exclude_cols = {"iter", "center", "window"}
    if "r_bins" in df_window.columns:
        exclude_cols.add("r_bins")
    stats_windowed_all = [c for c in df_window.columns if c not in exclude_cols]

    df_window_z = (
        df_window.join(
            bins.mean[1].select([c for c in bins.mean[1].columns if c != "iter"]),
            on=win_join,
            how="left",
            suffix="_mean",
        ).join(
            bins.std[1].select([c for c in bins.std[1].columns if c != "iter"]),
            on=win_join,
            how="left",
            suffix="_std",
        )
    ).with_columns(
        [
            # pl.when(pl.col(f"{c}_std").is_null() | (pl.col(f"{c}_std") == 0))
            # .then(pl.lit(0.0))
            # .otherwise(
            ((pl.col(c) - pl.col(f"{c}_mean")) / pl.col(f"{c}_std")).alias(c)
            for c in stats_windowed_all
        ]
    )

    keep_base = ["iter", "center", "window"] + (
        ["r_bins"] if "r_bins" in df_window.columns else []
    )
    df_window_z = df_window_z.select(keep_base + stats_windowed_all)

    return (normalized_df, df_window_z.select(pl.exclude("r_bins")))


################## Haplotype structure stats


def ihs_ihh(
    h,
    pos,
    map_pos=None,
    min_ehh=0.05,
    min_maf=0.05,
    include_edges=False,
    gap_scale=20000,
    max_gap=200000,
    is_accessible=None,
    use_threads=False,
):
    """
    Compute iHS (integrated Haplotype Score) and delta iHH from haplotypes.

    The routine integrates EHH (extended haplotype homozygosity) on both sides
    of each focal SNP to obtain iHH for ancestral and derived alleles, and then
    reports iHS (log ratio) and the absolute difference in iHH (``delta_ihh``).

    :param numpy.ndarray h:
        Haplotype matrix of shape ``(n_snps, n_haplotypes)`` with 0/1 values,
        where rows are SNPs and columns are haplotypes.
    :param numpy.ndarray pos:
        Physical positions for SNPs (length ``n_snps``). Used for gap handling
        and, when ``map_pos`` is ``None``, for integration spacing.
    :param numpy.ndarray map_pos:
        Optional genetic map positions (same length as ``pos``). If provided,
        integration uses these coordinates instead of ``pos``. Default ``None``.
    :param float min_ehh:
        Minimum EHH value to include in the integration. Default ``0.05``.
    :param float min_maf:
        Minimum minor-allele frequency required to compute iHS at a SNP.
        Default ``0.05``.
    :param bool include_edges:
        If ``True``, permit edge SNPs to contribute even when EHH dips below
        ``min_ehh``. Default ``False``.
    :param int gap_scale:
        Scaling used for gaps between consecutive SNPs when integrating over
        physical distance (ignored if ``map_pos`` is provided). Default ``20000``.
    :param int max_gap:
        Maximum gap allowed when integrating; larger gaps are capped to
        ``max_gap`` to avoid overweighting sparse regions. Default ``200000``.
    :param numpy.ndarray is_accessible:
        Optional boolean mask (length ``n_snps``) indicating accessible SNPs.
        If ``None``, all SNPs are considered accessible. Default ``None``.
    :param bool use_threads:
        Enable threaded computation in downstream primitives when available.
        Default ``False``.

    :returns:
        Polars DataFrame with columns: ``positions`` (physical position),
        ``daf`` (derived allele frequency), ``ihs`` (log iHH ratio), and
        ``delta_ihh`` (absolute difference between derived and ancestral iHH).
    :rtype: polars.DataFrame

    :raises ValueError:
        Propagated if inputs are inconsistent in length or malformed.

    .. note::
       SNPs that fail the MAF threshold or have invalid iHS values are omitted
       from the returned table.
    """
    # check inputs
    h = asarray_ndim(h, 2)
    check_integer_dtype(h)
    pos = asarray_ndim(pos, 1)
    check_dim0_aligned(h, pos)
    h = memoryview_safe(h)
    pos = memoryview_safe(pos)

    # compute gaps between variants for integration
    gaps = compute_ihh_gaps(pos, map_pos, gap_scale, max_gap, is_accessible)

    # setup kwargs
    kwargs = dict(min_ehh=min_ehh, min_maf=min_maf, include_edges=include_edges)

    if use_threads:
        # run with threads

        # create pool
        pool = ThreadPool(2)

        # scan forward
        result_fwd = pool.apply_async(ihh01_scan, (h, gaps), kwargs)

        # scan backward
        result_rev = pool.apply_async(ihh01_scan, (h[::-1], gaps[::-1]), kwargs)

        # wait for both to finish
        pool.close()
        pool.join()

        # obtain results
        ihh0_fwd, ihh1_fwd = result_fwd.get()
        ihh0_rev, ihh1_rev = result_rev.get()

        # cleanup
        pool.terminate()

    else:
        # run without threads

        # scan forward
        ihh0_fwd, ihh1_fwd = ihh01_scan(h, gaps, **kwargs)

        # scan backward
        ihh0_rev, ihh1_rev = ihh01_scan(h[::-1], gaps[::-1], **kwargs)

    # compute unstandardized score
    ihh0 = ihh0_fwd + ihh0_rev
    ihh1 = ihh1_fwd + ihh1_rev

    # og estimation
    with np.errstate(divide="ignore", invalid="ignore"):
        ihs = np.log(ihh0 / ihh1)

        # a = np.log(ihh0 / ihh1)
        # b = np.log(ihh1 / ihh0)
        # print(b,np.nanmax(a[~np.isinf(a)]),np.nanmax(a[~np.isinf(a)]))
    # mask = (ihh1 != 0) & (ihh0 > 0) & (ihh1 > 0)
    # ihs = np.full_like(ihh0, np.nan, dtype=float)
    # ihs[mask] = np.log(ihh0[mask] / ihh1[mask])

    delta_ihh = np.abs(ihh1 - ihh0)

    df_ihs = (
        pl.DataFrame(
            {
                "positions": pos,
                "daf": h.sum(axis=1) / h.shape[1],
                "ihs": ihs,
                "delta_ihh": delta_ihh,
            }
        )
        .fill_nan(None)
        .drop_nulls()
    )

    df_ihs = df_ihs.filter(~pl.col("ihs").is_infinite())
    return df_ihs


def haf_top(hap, pos, cutoff=0.1, start=None, stop=None, window_size=None, n_snps=None):
    """
    Compute the upper-tail HAF (Haplotype Allele Frequency) summary in a region.

    Rows of ``hap`` are SNPs, columns are haplotypes. HAF values are computed
    per SNP from haplotypes, then restricted to the specified genomic region
    (``start``/``stop`` or ``window_size``) if given. The HAF values are sorted
    and the top portion after trimming by ``cutoff`` is summed.

    :param numpy.ndarray hap:
        Haplotype matrix of shape ``(n_snps, n_haplotypes)`` with 0/1 values.
    :param numpy.ndarray pos:
        Physical positions for SNPs (length ``n_snps``).
    :param float cutoff:
        Proportion used for tail trimming. For example, ``0.1`` trims the lowest
        10% and the highest 10% before summing the remaining HAF values.
        Default ``0.1``.
    :param float start:
        Optional region start position (inclusive). Default ``None``.
    :param float stop:
        Optional region end position (inclusive). Default ``None``.
    :param int window_size:
        Optional window size in base pairs centered by the caller’s convention.
        If provided, it can be used to define the region when ``start``/``stop``
        are not specified. Default ``None``.
    :param int n_snps:
        Optional limit on the number of SNPs considered by certain strategies
        (implementation-dependent). Default ``1001``.

    :returns:
        Upper-tail HAF summary as a single float after trimming by ``cutoff``.
    :rtype: float

    :raises ValueError:
        Propagated if inputs are malformed or if no SNPs fall within the region.

    .. note::
       If neither ``start``/``stop`` nor ``window_size`` is provided, the
       computation uses all SNPs in ``hap``/``pos``.
    """
    if start is not None or stop is not None:
        loc = (pos >= start) & (pos <= stop)
        pos = pos[loc]
        hap = hap[loc, :]
    elif window_size is not None:
        loc = (pos >= (6e5 - window_size // 2)) & (pos <= (6e5 + window_size // 2))
        hap = hap[loc, :]
    elif n_snps is not None:
        S, N = hap.shape

        # if (N >= 50 and N < 100):
        #     n_snps = 401
        # elif N < 50:
        #     n_snps = 201

        closer_center_snp = np.argmin(np.abs(pos - 6e5))
        loc = np.arange(
            max(closer_center_snp - n_snps // 2, 0),
            min(closer_center_snp + n_snps // 2 + 1, pos.size),
        )
        hap = hap[loc, :]

    freqs = hap.sum(axis=1) / hap.shape[1]
    hap_tmp = hap.astype(np.float64, copy=False)[(freqs > 0) & (freqs < 1)]
    haf_num = (np.dot(hap_tmp.T, hap_tmp) / hap.shape[1]).sum(axis=1)
    # haf_num = (jax_dot(hap_tmp.T) / hap.shape[1]).sum(axis=1)
    haf_den = hap_tmp.sum(axis=0)
    # haf = np.sort(haf_num / haf_den)

    if 0 in haf_den:
        mask_zeros = haf_den != 0
        haf = np.full_like(haf_num, np.nan, dtype=np.float64)
        haf[mask_zeros] = haf_num[mask_zeros] / haf_den[mask_zeros]
        haf = np.sort(haf)
    else:
        haf = np.sort(haf_num / haf_den)

    if cutoff <= 0 or cutoff >= 1:
        cutoff = 1
    # idx_low = int(cutoff * haf.size)
    idx_high = int((1 - cutoff) * haf.size)

    # 10% higher
    return np.nansum(haf[idx_high:])


@njit(cache=True)
def fast_skew_kurt(data, bias=False):
    """Single-pass numba variance/skew/kurtosis.
    bias=False matches scipy.stats defaults. Returns (variance, skewness, kurtosis)."""
    n = len(data)
    if n < 4:
        return 0.0, 0.0, 0.0
    mu = 0.0
    for x in data:
        mu += x
    mu /= n
    m2 = m3 = m4 = 0.0
    for x in data:
        diff = x - mu
        d2 = diff * diff
        m2 += d2
        m3 += d2 * diff
        m4 += d2 * d2
    m2 /= n
    m3 /= n
    m4 /= n
    if m2 <= (1.1e-16 * abs(mu)) ** 2 or m2 < 1e-20:
        return 0.0, np.nan, np.nan
    g1 = m3 / (m2**1.5)
    g2 = (m4 / (m2**2)) - 3.0
    if bias:
        return (m2 * n / (n - 1.0)), g1, g2
    skew_u = (np.sqrt(n * (n - 1.0)) / (n - 2.0)) * g1
    kurt_u = (n - 1.0) / ((n - 2.0) * (n - 3.0)) * ((n + 1.0) * g2 + 6.0)
    var_u = m2 * n / (n - 1.0)
    return var_u, skew_u, kurt_u


# @njit(cache=True)
# def garud_h_numba(h):
#     """
#     Compute Garud’s haplotype homozygosity statistics in Numba.

#     The input is a binary haplotype matrix with shape ``(L, n)``, where ``L`` is
#     the number of variant sites (rows) and ``n`` is the number of haplotypes
#     (columns). The function counts distinct haplotypes (columns), converts those
#     counts to frequencies :math:`p_i`, sorts them descending to obtain
#     :math:`p_1 \\ge p_2 \\ge p_3 \\ge \\dots`, and computes:

#     - :math:`H1 = \\sum_i p_i^2`
#     - :math:`H12 = (p_1 + p_2)^2 + \\sum_{i\\ge 3} p_i^2`
#     - :math:`H123 = (p_1 + p_2 + p_3)^2 + \\sum_{i\\ge 4} p_i^2`
#     - :math:`H2/H1 = (H1 - p_1^2) / H1`

#     :param numpy.ndarray h:
#         2D array of dtype ``uint8`` with values in ``{0, 1}`` and shape
#         ``(n_variants, n_haplotypes)``.
#     :returns:
#         Tuple ``(H12, H2_H1, H1, H123)`` as floats.
#     :rtype: tuple[float, float, float, float]

#     """
#     L, n = h.shape

#     # rolling uint64 hash to count distinct columns
#     counts = Dict.empty(key_type=uint64, value_type=int64)
#     for j in range(n):
#         hsh = np.uint64(146527)
#         for i in range(L):
#             hsh = (hsh * np.uint64(1000003)) ^ np.uint64(np.int64(h[i, j]))
#         counts[hsh] = counts.get(hsh, 0) + 1

#     # collect counts into an array
#     m = len(counts)
#     cnts = np.empty(m, np.int64)
#     idx = 0
#     for k in counts:
#         cnts[idx] = counts[k]
#         idx += 1

#     # 3) to frequencies & sort descending
#     freqs = cnts.astype(np.float64) / n
#     freqs = np.sort(freqs)[::-1]

#     # pad top‐3
#     p1 = freqs[0] if freqs.size > 0 else 0.0
#     p2 = freqs[1] if freqs.size > 1 else 0.0
#     p3 = freqs[2] if freqs.size > 2 else 0.0

#     # compute H1, H12, H123, H2/H1
#     H1 = 0.0
#     for i in range(freqs.size):
#         H1 += freqs[i] * freqs[i]

#     H12 = (p1 + p2) ** 2
#     for i in range(2, freqs.size):
#         H12 += freqs[i] * freqs[i]

#     H123 = (p1 + p2 + p3) ** 2
#     for i in range(3, freqs.size):
#         H123 += freqs[i] * freqs[i]

#     H2 = H1 - p1**2
#     H2_H1 = H2 / H1

#     return H12, H2_H1, H1, H123, m


@njit(cache=True)
def garud_h_numba(h):
    L, n = h.shape
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # 1. Compute rolling hashes into a flat array
    # This avoids the Dict.empty() allocation bottleneck
    hashes = np.empty(n, dtype=np.uint64)
    for j in range(n):
        hsh = np.uint64(146527)
        for i in range(L):
            # Using bit-shifting for hash is often faster than multiplication on EPYC
            hsh = (hsh ^ np.uint64(h[i, j])) * np.uint64(1000003)
        hashes[j] = hsh

    # 2. Sort hashes to group identical haplotypes
    hashes.sort()

    # 3. Count frequencies in a single pass over sorted hashes
    # This replaces the need for a Dictionary
    freq_list = []
    current_count = 1
    for i in range(1, n):
        if hashes[i] == hashes[i - 1]:
            current_count += 1
        else:
            freq_list.append(current_count / float(n))
            current_count = 1
    freq_list.append(current_count / float(n))

    # Convert to array and sort descending
    freqs = np.sort(np.array(freq_list))[::-1]
    m = float(len(freqs))

    p1 = freqs[0] if freqs.size > 0 else 0.0
    p2 = freqs[1] if freqs.size > 1 else 0.0
    p3 = freqs[2] if freqs.size > 2 else 0.0

    # H1, H12, H123 in one pass
    H1 = 0.0
    sq_sum_from_3 = 0.0
    sq_sum_from_4 = 0.0

    for i in range(freqs.size):
        f_sq = freqs[i] * freqs[i]
        H1 += f_sq
        if i >= 2:
            sq_sum_from_3 += f_sq
        if i >= 3:
            sq_sum_from_4 += f_sq

    H12 = (p1 + p2) ** 2 + sq_sum_from_3
    H123 = (p1 + p2 + p3) ** 2 + sq_sum_from_4

    H2 = H1 - p1**2
    H2_H1 = H2 / H1 if H1 > 0 else 0.0

    return H12, H2_H1, H1, H123, m


def garud_h(h):
    """Compute the H1, H12, H123 and H2/H1 statistics for detecting signatures
    of soft sweeps, as defined in Garud et al. (2015).

    Parameters
    ----------
    h : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array.

    Returns
    -------
    h1 : float
        H1 statistic (sum of squares of haplotype frequencies).
    h12 : float
        H12 statistic (sum of squares of haplotype frequencies, combining
        the two most common haplotypes into a single frequency).
    h123 : float
        H123 statistic (sum of squares of haplotype frequencies, combining
        the three most common haplotypes into a single frequency).
    h2_h1 : float
        H2/H1 statistic, indicating the "softness" of a sweep.

    """
    from allel import HaplotypeArray

    # check inputs
    h = HaplotypeArray(h, copy=False)

    # compute haplotype frequencies
    f = h.distinct_frequencies()

    # compute H1
    h1 = np.sum(f**2)

    # compute H12
    h12 = np.sum(f[:2]) ** 2 + np.sum(f[2:] ** 2)

    # compute H123
    h123 = np.sum(f[:3]) ** 2 + np.sum(f[3:] ** 2)

    # compute H2/H1
    h2 = h1 - f[0] ** 2
    h2_h1 = h2 / h1

    return h12, h2_h1, h1, h123, f.size


@njit(cache=True)
def comparen_haplos_optimized(haplo1, haplo2):
    identical = 0
    different = 0
    for i in range(len(haplo1)):
        h1 = haplo1[i]
        h2 = haplo2[i]

        if (h1 == 1) and (h2 == 1):
            identical += 1
        elif h1 != h2:
            different += 1

    total = identical + different
    return identical, different, total


@njit(cache=True)
def _compare_hap_cols(H, col_a, col_b):
    """
    identical = #(1,1)
    different = #(mismatch: 1,0 or 0,1)
    total     = identical + different
    """
    L = H.shape[0]
    identical = 0
    different = 0
    for i in range(L):
        a = H[i, col_a]
        b = H[i, col_b]
        if (a == 1) and (b == 1):
            identical += 1
        elif a != b:
            different += 1
    total = identical + different
    return identical, different, total


@njit(cache=True)
def _legacy_row_order_indices(hap, positions, focal_coord, window_size, min_freq):
    """
    Returns row indices exactly in the order legacy visits sites:
      - 100bp bins: left first (-1,-2,...) then right (+1,+2,...)
      - exclude focal bin (step==0)
      - freq filter in [min_freq, 1.0]
      - right cap: sup_i < 1_200_000
    Implementation is O(#rows_in_window): two passes + bucketize by step.
    """
    L_total, n = hap.shape
    int_coord = (focal_coord // 100) * 100
    half = window_size // 2
    low = int_coord - half
    high = int_coord + half
    max_steps = half // 100

    # pass 1: collect "ok" rows and their step
    # store into temporary arrays
    ok_idx = np.empty(L_total, dtype=int64)
    ok_step = np.empty(L_total, dtype=int64)
    k = 0
    for i in range(L_total):
        pos = positions[i]
        if pos == focal_coord:
            continue
        if (pos < low) or (pos > high):
            continue
        # derived freq
        s = 0
        for j in range(n):
            s += hap[i, j]
        f = s / n
        if (f < min_freq) or (f > 1.0):
            continue
        # step in 100bp units
        bin_i = (pos // 100) * 100
        step = (bin_i - int_coord) // 100
        if step == 0:
            continue  # legacy skips focal bin
        ok_idx[k] = i
        ok_step[k] = step
        k += 1

    if k == 0:
        return np.empty(0, dtype=int64)

    # pass 2: bucketize by step with stability
    # LEFT
    left_limit = (int_coord - 1) // 100  # require inf_i > 0
    left_counts = np.zeros(max_steps, dtype=int64)  # index = abs(step)-1
    # RIGHT (with cap sup_i < 1_200_000)
    right_cap = (1_200_000 - int_coord - 1) // 100
    right_counts = np.zeros(max_steps, dtype=int64)  # index = step-1

    # count per step
    for t in range(k):
        step = ok_step[t]
        if step < 0:
            st = -step
            if (st <= max_steps) and (st <= left_limit):
                left_counts[st - 1] += 1
        else:  # step > 0
            st = step
            if (st <= max_steps) and (st <= right_cap):
                right_counts[st - 1] += 1

    left_total = int(left_counts.sum())
    right_total = int(right_counts.sum())
    out = np.empty(left_total + right_total, dtype=int64)

    # prefix sums to place indices in bin order (preserve ok order within step)
    left_off = np.zeros(max_steps, dtype=int64)
    right_off = np.zeros(max_steps, dtype=int64)
    # compute running offsets
    acc = 0
    for s in range(max_steps):
        left_off[s] = acc
        acc += left_counts[s]
    left_end = acc
    acc = 0
    for s in range(max_steps):
        right_off[s] = acc
        acc += right_counts[s]
    # pass 3: fill left, then right
    # LEFT fill in step order 1..max_steps
    for t in range(k):
        step = ok_step[t]
        if step < 0:
            st = -step
            if (st <= max_steps) and (st <= left_limit):
                pos = left_off[st - 1]
                out[pos] = ok_idx[t]
                left_off[st - 1] = pos + 1

    # RIGHT fill appended after left block, in step order 1..max_steps
    base = left_end
    for t in range(k):
        step = ok_step[t]
        if step > 0:
            st = step
            if (st <= max_steps) and (st <= right_cap):
                pos = base + right_off[st - 1]
                out[pos] = ok_idx[t]
                right_off[st - 1] = right_off[st - 1] + 1

    return out


@njit(cache=True)
def _unique_hash_counts_reprs_and_assign(H):
    """
    H: (L, n) uint8
    Returns:
      cnts:   (m,) int64        counts per unique haplotype
      reprj:  (m,) int64        representative column index for each unique hap
      assign: (n,) int64        sample -> uid
    """
    L, n = H.shape
    counts = Dict.empty(key_type=uint64, value_type=int64)
    reprs = Dict.empty(key_type=uint64, value_type=int64)
    hashes = np.empty(n, dtype=uint64)

    for j in range(n):
        hsh = np.uint64(146527)
        for i in range(L):
            hsh = (hsh * np.uint64(1000003)) ^ np.uint64(H[i, j])
        hashes[j] = hsh
        counts[hsh] = counts.get(hsh, 0) + 1
        if hsh not in reprs:
            reprs[hsh] = j

    m = len(counts)
    cnts = np.empty(m, dtype=int64)
    reprj = np.empty(m, dtype=int64)
    key2id = Dict.empty(key_type=uint64, value_type=int64)

    k = 0
    for hsh in counts:
        cnts[k] = counts[hsh]
        reprj[k] = reprs[hsh]
        key2id[hsh] = k
        k += 1

    assign = np.empty(n, dtype=int64)
    for j in range(n):
        assign[j] = key2id[hashes[j]]
    return cnts, reprj, assign


@njit(cache=True, inline="always")
def _lex_less_cols(H, reprj, uid_a, uid_b):
    """
    True if col(uid_a) < col(uid_b) lexicographically
    Equivalent to comparing legacy "0 1 1 ..." strings (spaces don't affect order).
    """
    L = H.shape[0]
    ca = reprj[uid_a]
    cb = reprj[uid_b]
    for i in range(L):
        va = H[i, ca]
        vb = H[i, cb]
        if va < vb:
            return True
        elif va > vb:
            return False
    return False  # equal


@njit(cache=True)
def _argsort_by_count_then_lex(cnts, H, reprj):
    """
    Return indices 0..m-1 sorted by (count desc, lex asc on H[:, reprj[uid]]).
    Implemented via two-stage: count sort (argsort desc) then in-place
    segment lex insertion-sort per count tier.
    """
    m = cnts.size
    order = np.argsort(cnts)  # ascending
    # reverse to descending
    for i in range(m // 2):
        tmp = order[i]
        order[i] = order[m - 1 - i]
        order[m - 1 - i] = tmp

    # walk segments of equal count and lex-sort within each
    i = 0
    while i < m:
        c = cnts[order[i]]
        j = i + 1
        while j < m and cnts[order[j]] == c:
            j += 1
        # insertion sort order[i:j] by lex
        k = i + 1
        while k < j:
            x = order[k]
            p = k - 1
            while (p >= i) and _lex_less_cols(H, reprj, x, order[p]):
                order[p + 1] = order[p]
                p -= 1
            order[p + 1] = x
            k += 1
        i = j
    return order


@njit(cache=True)
def h12_enard(
    hap,
    positions,
    focal_coord=600000,
    n_snps=None,
    window_size=int(5e5),
    min_derived_freq=0.05,
    similarity_threshold=0.8,
    top_k=10,
):
    """
    Estimate Garud's  ``H12, H2/H1, H1`` around a focal coordinate,
    grouping haplotypes that are at least a given identity threshold (default 80%).

    The method builds a count-based, symmetric SNP window centered at ``focal_coord``,
    constructs a haplotype matrix ``H`` for the selected SNPs, collapses identical
    haplotypes (columns), orders the unique haplotypes by frequency (descending) and
    lexicographic order (ascending), selects a set of representative haplotypes, and
    then **merges representatives into similarity groups** whenever the **column-wise
    identity** meets or exceeds ``similarity_threshold`` (``0.8`` by default). Haplotype
    group frequencies are then used to compute the H12 family of statistics.

    Identity between two haplotype columns is defined as:

    .. math::

       \\text{identity} = \\frac{\\#(1,1)}{\\#(1,1) + \\#(1,0) + \\#(0,1)}

    (i.e., matches on the derived allele over all non-equal-or-derived comparisons).

    :param numpy.ndarray hap:
        Haplotype matrix of shape ``(L_total, n)`` with 0/1 values (ancestral/derived).
        Rows are SNPs; columns are haplotypes.
    :param numpy.ndarray positions:
        1D array (length ``L_total``) of genomic coordinates (``int64``) aligned to ``hap`` rows.
    :param int focal_coord:
        Genomic coordinate used to center the SNP window. Default ``600000``.
    :param int n_snps:
        Target number of SNPs for the window (the focal SNP, if present, is excluded from
        the returned set). Default ``1001``.
    :param float min_derived_freq:
        Minimum derived-allele frequency required for a SNP to enter the window
        (inclusive; upper bound is ``1.0``). Default ``0.05``.
    :param float similarity_threshold:
        Column similarity threshold for grouping haplotypes. Two representative columns
        are merged if their identity fraction (formula above) is **≥ this value**.
        Default ``0.8`` (80% identity).
    :param int top_k:
        Limit controlling how many unique-haplotype representatives are considered before
        grouping. Default ``10``.

    :returns:
        Tuple ``(H12, H2_H1, H2)`` as floats. If no usable SNPs or groups are found,
        returns ``(0.0, 0.0, 0.0)``.
    :rtype: tuple[float, float, float]

    .. math::

       H1 = \\sum_g p_g^2,\\qquad
       H12 = (p_1 + p_2)^2,\\qquad
       H2 = H1 - p_1^2,\\qquad
       \\frac{H2}{H1} = \\begin{cases}
           (H1 - p_1^2)/H1, & H1 \\ne 0,\\\\
           0, & H1 = 0~.
       \\end{cases}

    Notes
    -----
    - The SNP window is built by balancing sites to the left and right of ``focal_coord``
      by proximity, after applying the derived-frequency filter ``[min_derived_freq, 1.0]``,
      and excluding the focal position itself.
    - Unique haplotypes are detected via hashing of ``H`` columns; sample-to-group
      frequencies :math:`p_g` are computed after the identity-based grouping step.
    - The default behavior corresponds to **H12 with 80% identity grouping** in the
      haplotype matrix, which can increase robustness by merging highly similar haplotypes.
    """

    L_total, n = hap.shape

    # Change n_snps dinamically maximizing power based on Zhao et al. 2024
    # if (n >= 50 and n < 100):
    #     n_snps = 401
    # elif n < 50:
    #     n_snps = 201

    # 1) legacy row order
    rows = _legacy_row_order_indices(
        hap,
        positions,
        np.int64(focal_coord),
        # np.int64(n_snps),
        np.int64(window_size),
        np.float64(min_derived_freq),
    )
    if rows.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    # 2) window matrix in exact order
    L = rows.size
    H = np.empty((L, n), dtype=np.uint8)
    for r in range(L):
        i = rows[r]
        for j in range(n):
            H[r, j] = hap[i, j]

    # 3) unique haplotypes + per-sample assignment
    cnts, reprj, assign = _unique_hash_counts_reprs_and_assign(H)
    m = cnts.size
    if m == 0:
        return 0.0, 0.0, 0.0, 0.0

    # 4) global order by (count desc, lex asc) and apply legacy accumulator
    order = _argsort_by_count_then_lex(cnts, H, reprj)  # length m
    # accumulator quirk
    done_rev = 0
    counter_rev = 0
    # we don't know K a priori due to accumulator → collect into temporary array
    best_mask = np.zeros(m, dtype=np.uint8)
    # iterate by count tiers (segments of same count)
    i = 0
    while i < m:
        c = cnts[order[i]]
        j = i + 1
        while (j < m) and (cnts[order[j]] == c):
            j += 1
        # add whole tier (i..j-1) in lex order
        for t in range(i, j):
            best_mask[order[t]] = 1
            done_rev += 1
        counter_rev += done_rev
        if counter_rev >= top_k:
            break
        i = j

    # collect selected uids in the *selection order* induced by `order`
    keep_m = 0
    for t in range(order.size):
        u = order[t]
        if best_mask[u] == 1:
            keep_m += 1
    sel = np.empty(keep_m, dtype=int64)
    w = 0
    for t in range(order.size):
        u = order[t]
        if best_mask[u] == 1:
            sel[w] = u
            w += 1

    # 5) forward-only similarity on selected; exclude-as-you-go groups
    # group_of[a] = group id; insertion order defines p1, p2,...
    group_of = np.full(keep_m, -1, dtype=int64)
    groups = 0
    thr = similarity_threshold
    for ai in range(keep_m):
        if group_of[ai] != -1:
            continue
        g = groups
        groups += 1
        group_of[ai] = g
        ua_col = reprj[sel[ai]]
        for bi in range(ai + 1, keep_m):
            if group_of[bi] != -1:
                continue
            ub_col = reprj[sel[bi]]
            ident, diff, tot = _compare_hap_cols(H, ua_col, ub_col)
            if tot == 0:
                continue
            if (ident / tot) >= thr:
                group_of[bi] = g

    # 6) uid -> group id map (only for selected)
    uid_to_group = np.full(m, -1, dtype=int64)
    for ai in range(keep_m):
        uid_to_group[sel[ai]] = group_of[ai]

    # 7) count only selected uids; denominator is n
    if groups == 0:
        return 0.0, 0.0, 0.0, 0.0
    freq_counts = np.zeros(groups, dtype=int64)
    for j in range(n):
        uid = assign[j]
        g = uid_to_group[uid]
        if g != -1:
            freq_counts[g] += 1

    # 8) legacy stats (NO resort)
    toto = float(n)
    # insertion order is 0..groups-1 by construction
    H1 = 0.0
    p1 = freq_counts[0] / toto if groups > 0 else 0.0
    p2 = freq_counts[1] / toto if groups > 1 else 0.0
    # accumulate H1
    for g in range(groups):
        f = freq_counts[g] / toto
        H1 += f * f
    H12 = (p1 + p2) * (p1 + p2)
    H2 = H1 - p1 * p1
    H2_H1 = (H2 / H1) if H1 != 0.0 else 0.0
    return H12, H2_H1, H1, H2


def pairwise_diffs(hap, missing=False):
    """
    Pairwise mismatch counts between samples (columns) for a site x sample matrix.

    Parameters
    ----------
    hap : (S, n) array-like, integer/boolean
        Haplotype/allele matrix with samples in columns.
        - If missing=False: entries must be exactly {0,1}.
        - If missing=True:  entries may be {-1,0,1}, where -1 is treated as missing.
    missing : bool, default False
        If False, assumes data are strictly 0/1 and uses a single dot product.
        If True, treats -1 as missing and uses a mask for two dot product.

    Returns
    -------
    diff_ls : (n*(n-1)//2,) float64
        Pairwise mismatch counts in i<j order (same as the C implementation).
    """

    S, n = hap.shape
    Y = hap.astype(np.float64, copy=False)
    s = Y.sum(axis=0)
    # (n, n) dot products
    G = Y.T @ Y
    # (n, n) Hamming counts
    D = s[None, :] + s[:, None] - 2.0 * G

    # Missing data
    # Let M = 1 on valid (0/1), Y = 1 on allele==1.
    # D = (Y^T M) + (Y^T M)^T - 2*(Y^T Y) # counts 10 + 01 only on jointly valid sites.
    # M = ((hap == 0) | (hap == 1)).astype(ftype)
    # Y = (hap == 1).astype(ftype)
    # A = Y.T @ M
    # C = Y.T @ Y
    # D = A + A.T - 2.0 * C

    # Extract upper triangle (i<j)
    iu, ju = np.triu_indices(n, k=1)
    return D[iu, ju].astype(np.float64, copy=False)


def pairwise_diffs_precomp(hap, iu, ju, use_float32=False):
    """Pairwise mismatch counts reusing pre-computed triu indices.
    Call np.triu_indices once before the window loop, pass here each iteration.
    For best performance, pre-cast hap to float64 (or float32 if use_float32=True)
    before the window loop — the astype(copy=False) call becomes a no-op on slices."""
    dtype = np.float32 if use_float32 else np.float64
    Y = hap.astype(dtype, copy=False)
    s = Y.sum(axis=0)
    G = Y.T @ Y
    D = s[None, :] + s[:, None] - 2.0 * G
    return D[iu, ju].astype(np.float64, copy=False)


@njit(parallel=False, cache=True)
def _hscan_single(hap, pos, x, max_gap, dist_mode):
    S, N = hap.shape
    h_sum = float64(0.0)

    for i in prange(N - 1):
        for j in range(i + 1, N):
            # Extend RIGHT
            x_r = x
            while (
                x_r + 1 < S
                and (pos[x_r + 1] - pos[x_r]) < max_gap
                and hap[x_r + 1, i] == hap[x_r + 1, j]
            ):
                x_r += 1

            # Extend LEFT
            x_l = x
            while (
                x_l - 1 >= 0
                and hap[x_l - 1, i] == hap[x_l - 1, j]
                and (pos[x_l] - pos[x_l - 1]) < max_gap
            ):
                x_l -= 1

            if x_r != x_l:
                if dist_mode == 0:
                    h_sum += float64(pos[x_r] - pos[x_l] - 1)
                elif dist_mode == 1:
                    h_sum += float64(x_r - x_l - 1)
                else:
                    h_sum += float64(pos[x_r] - pos[x_l] - 1) * float64(x_r - x_l - 1)

    return 2.0 * h_sum / (float64(N) * float64(N - 1))


@njit(cache=True)
def _hscan_all(hap, pos, indices, max_gap, dist_mode):
    n_focal = len(indices)
    h_means = np.empty(n_focal, dtype=float64)
    for k in range(n_focal):
        h_means[k] = _hscan_single(hap, pos, indices[k], max_gap, dist_mode)
    return h_means


def hscan(
    hap,
    pos,
    focal_pos=None,
    max_gap=int(1e9),
    dist_mode=0,
    step=1,
    left_bound=0,
    right_bound=int(1e9),
    return_pairs=False,
):
    # F-order: haplotype columns contiguous
    hap_f = np.asfortranarray(hap)
    S = hap.shape[0]

    # Single focal SNP
    if focal_pos is not None:
        focal_idx = int(np.argmin(np.abs(pos - int(focal_pos))))
        h = _hscan_single(hap_f, pos, int64(focal_idx), int64(max_gap), dist_mode)
        return float(h)

    # Full scan
    all_indices = np.arange(0, S, step, dtype=np.int64)
    pos_at_idx = pos[all_indices]
    mask = (pos_at_idx >= left_bound) & (pos_at_idx <= right_bound)
    indices = all_indices[mask]

    h_means = _hscan_all(hap, pos, indices, int64(max_gap), dist_mode)
    return pos[indices], h_means


################## FS stats


@njit(cache=True, parallel=False)
def fast_sq_freq_pairs(
    hap,
    ac,
    rec_map,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=50000,
):
    n_snps, n_samples = hap.shape
    rec_pos = rec_map[:, -2]
    half_window = window_size * 0.5

    # ── single pass: compute freqs + collect focal indices ──────────────
    # Eliminates total_count array and separate focal-detection loop.
    freqs = np.empty(n_snps, dtype=np.float64)
    focal_indices = np.empty(n_snps, dtype=np.int64)
    n_focal = 0
    for i in range(n_snps):
        f = ac[i, 1] / (ac[i, 0] + ac[i, 1])
        freqs[i] = f
        if min_focal_freq <= f <= max_focal_freq:
            focal_indices[n_focal] = i
            n_focal += 1
    focal_indices = focal_indices[:n_focal]

    # ── output containers ────────────────────────────────────────────────
    # Sentinel placeholders [np.empty((1,3))...] removed — always overwritten.
    sq_out_list = [np.empty((0, 3), dtype=np.float64) for _ in range(n_focal)]
    snp_indices_list = [np.empty(0, dtype=np.int64) for _ in range(n_focal)]
    info = np.empty((n_focal, 4), dtype=np.float64)

    # ── main loop ────────────────────────────────────────────────────────
    for j in prange(n_focal):
        focal_idx = focal_indices[j]
        center = rec_pos[focal_idx]

        # Window bounds computed inline — eliminates window_bounds (n_focal,2) array.
        x_l = np.searchsorted(rec_pos, center - half_window, side="left")
        y_r = np.searchsorted(rec_pos, center + half_window, side="right") - 1
        y_l = focal_idx - 1
        x_r = focal_idx + 1

        # info written unconditionally — eliminates duplicated assignment in empty branch.
        focal_d_count = ac[focal_idx, 1]
        focal_a_count = ac[focal_idx, 0]
        info[j, 0] = center
        info[j, 1] = freqs[focal_idx]
        info[j, 2] = np.float64(focal_d_count)
        info[j, 3] = np.float64(focal_a_count)

        len_l = max(0, y_l - x_l + 1)
        len_r = max(0, y_r - x_r + 1)
        total_len = len_l + len_r

        if total_len == 0:
            continue  # sentinels already set above

        out = np.empty((total_len, 3), dtype=np.float64)
        indices_out = np.empty(total_len, dtype=np.int64)
        inv_d = 1.0 / focal_d_count if focal_d_count > 0 else 0.0
        inv_a = 1.0 / focal_a_count if focal_a_count > 0 else 0.0
        hap_f = hap[focal_idx]
        out_idx = 0

        # LEFT window (reverse) — plain 0 inits, Numba infers int64
        for k in range(y_l, x_l - 1, -1):
            hap_k = hap[k]
            overlap_d = 0
            sum_k = 0
            for m in range(n_samples):
                hk = hap_k[m]
                overlap_d += hap_f[m] * hk
                sum_k += hk
            out[out_idx, 0] = overlap_d * inv_d
            out[out_idx, 1] = (sum_k - overlap_d) * inv_a
            out[out_idx, 2] = freqs[k]
            indices_out[out_idx] = k
            out_idx += 1

        # RIGHT window (forward)
        for k in range(x_r, y_r + 1):
            hap_k = hap[k]
            overlap_d = 0
            sum_k = 0
            for m in range(n_samples):
                hk = hap_k[m]
                overlap_d += hap_f[m] * hk
                sum_k += hk
            out[out_idx, 0] = overlap_d * inv_d
            out[out_idx, 1] = (sum_k - overlap_d) * inv_a
            out[out_idx, 2] = freqs[k]
            indices_out[out_idx] = k
            out_idx += 1

        sq_out_list[j] = out
        snp_indices_list[j] = indices_out

    return sq_out_list, info, snp_indices_list


def s_ratio(
    hap,
    ac,
    rec_map,
    max_ancest_freq=1,
    min_tot_freq=0,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=50000,
    genetic_distance=False,
):
    """
    Compute the S-ratio statistic for each focal SNP.

    For each focal SNP (derived frequency in ``[min_focal_freq, max_focal_freq]``),
    neighbors within ``window_size`` are summarized by indicators of intermediate
    frequency on the derived and ancestral partitions, and the ratio of their
    counts is reported.

    :param numpy.ndarray hap: Haplotype matrix ``(n_snps, n_samples)`` with 0/1 values.
    :param numpy.ndarray ac: Allele counts ``(n_snps, 2)`` as ``[ancestral, derived]``.
    :param numpy.ndarray rec_map: Map array; penultimate column is the window coordinate.
    :param float max_ancest_freq: Maximum ancestral-partition frequency threshold. Default ``1``.
    :param float min_tot_freq: Minimum neighbor total derived frequency. Default ``0``.
    :param float min_focal_freq: Minimum focal derived frequency. Default ``0.25``.
    :param float max_focal_freq: Maximum focal derived frequency. Default ``0.95``.
    :param int window_size: Window size in coordinate units of ``rec_map[:, -2]``. Default ``50000``.
    :param bool genetic_distance: Unused here (kept for API symmetry).

    :returns: DataFrame with columns ``positions``, ``daf``, ``s_ratio``.
    :rtype: polars.DataFrame
    """
    sq_freqs, info, snps_indices = fast_sq_freq_pairs(
        hap, ac, rec_map, min_focal_freq, max_focal_freq, window_size
    )

    n_rows = len(sq_freqs)
    results = np.empty((n_rows, 1), dtype=np.float64)

    for i, v in enumerate(sq_freqs):
        f_d = v[:, 0]
        f_a = v[:, 1]

        f_d2 = np.zeros(f_d.shape)
        f_a2 = np.zeros(f_a.shape)

        f_d2[(f_d > 0.0000001) & (f_d < 1)] = 1
        f_a2[(f_a > 0.0000001) & (f_a < 1)] = 1

        num = (f_d2 - f_d2 + f_a2 + 1).sum()
        den = (f_a2 - f_a2 + f_d2 + 1).sum()
        # redefine to add one to get rid of blowup issue introduced by adding 0.001 to denominator

        s_ratio_v = num / den
        # s_ratio_v_flip = den / num
        # results.append((s_ratio_v, s_ratio_v_flip))
        results[i] = s_ratio_v

    tmp_schema = {
        "positions": pl.Int64,
        "daf": pl.Float64,
        "s_ratio": pl.Float64,
        # "s_ratio_flip": pl.Float64,
    }

    try:
        out = np.hstack([info[:, :2], results])
        # out = np.hstack([info[:,:2], np.array(results)])
        df_out = pl.DataFrame(out, schema=tmp_schema)
        # df_out = pl.DataFrame([out[:, 0], out[:, 1], out[:, 4], out[:, 5]], schema=tmp_schema)
    except Exception:
        df_out = pl.DataFrame([[], [], []], schema=tmp_schema)

    return df_out


def hapdaf_o(
    hap,
    ac,
    rec_map,
    max_ancest_freq=0.25,
    min_tot_freq=0.25,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=50000,
    genetic_distance=False,
):
    """
    Compute hapDAF-o for each focal SNP.

    hapDAF-o averages ``f_d^2 - f_a^2`` over neighbors that satisfy
    ``(f_d > f_a) & (f_a <= max_ancest_freq) & (f_tot >= min_tot_freq)``.

    :param numpy.ndarray hap: Haplotype matrix ``(n_snps, n_samples)`` with 0/1 values.
    :param numpy.ndarray ac: Allele counts ``(n_snps, 2)`` as ``[ancestral, derived]``.
    :param numpy.ndarray rec_map: Map array; penultimate column is the window coordinate.
    :param float max_ancest_freq: Ancestral partition frequency threshold. Default ``0.25``.
    :param float min_tot_freq: Minimum neighbor total derived frequency. Default ``0.25``.
    :param float min_focal_freq: Minimum focal derived frequency. Default ``0.25``.
    :param float max_focal_freq: Maximum focal derived frequency. Default ``0.95``.
    :param int window_size: Window size in coordinate units of ``rec_map[:, -2]``. Default ``50000``.
    :param bool genetic_distance: Unused here (kept for API symmetry).

    :returns: DataFrame with columns ``positions``, ``daf``, ``hapdaf_o``.
    :rtype: polars.DataFrame
    """
    sq_freqs, info, snps_indices = fast_sq_freq_pairs(
        hap, ac, rec_map, min_focal_freq, max_focal_freq, window_size
    )

    n_rows = len(sq_freqs)
    results = np.empty((n_rows, 1), dtype=np.float64)

    # nan_index = []

    for i, v in enumerate(sq_freqs):
        f_d = v[:, 0]
        f_a = v[:, 1]
        f_tot = v[:, 2]

        f_d2 = (
            f_d[(f_d > f_a) & (f_a <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )
        f_a2 = (
            f_a[(f_d > f_a) & (f_a <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )

        if len(f_d2) != 0 and len(f_a2) != 0:
            hapdaf = (f_d2 - f_a2).sum() / f_d2.shape[0]
        else:
            hapdaf = np.nan

        # # Flipping derived to ancestral, ancestral to derived
        # f_d2f = (
        #     f_a[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        # )
        # f_a2f = (
        #     f_d[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        # )
        # if len(f_d2f) != 0 and len(f_a2f) != 0:
        #     hapdaf_flip = (f_d2f - f_a2f).sum() / f_d2f.shape[0]
        # else:
        #     hapdaf_flip = np.nan

        # results.append((hapdaf, hapdaf_flip))
        results[i] = hapdaf

    tmp_schema = {
        "positions": pl.Int64,
        "daf": pl.Float64,
        "hapdaf_o": pl.Float64,
        # "hapdaf_o_flip": pl.Float64,
    }

    try:
        out = np.hstack(
            [
                info[:, :2],
                # np.array(results),
                results,
            ]
        )
        df_out = pl.DataFrame(out, schema=tmp_schema)
        # df_out = pl.DataFrame([out[:, 0], out[:, 1], out[:, 4], out[:, 5]], schema=tmp_schema)
    except Exception:
        df_out = pl.DataFrame([[], [], []], schema=tmp_schema)

    return df_out


def hapdaf_s(
    hap,
    ac,
    rec_map,
    max_ancest_freq=0.1,
    min_tot_freq=0.1,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=50000,
    genetic_distance=False,
):
    """
    Compute hapDAF-s for each focal SNP.

    hapDAF-s is the same construction as hapDAF-o but uses more stringent
    thresholds (e.g., smaller ``max_ancest_freq`` and ``min_tot_freq``).

    :param numpy.ndarray hap: Haplotype matrix ``(n_snps, n_samples)`` with 0/1 values.
    :param numpy.ndarray ac: Allele counts ``(n_snps, 2)`` as ``[ancestral, derived]``.
    :param numpy.ndarray rec_map: Map array; penultimate column is the window coordinate.
    :param float max_ancest_freq: Ancestral partition frequency threshold. Default ``0.1``.
    :param float min_tot_freq: Minimum neighbor total derived frequency. Default ``0.1``.
    :param float min_focal_freq: Minimum focal derived frequency. Default ``0.25``.
    :param float max_focal_freq: Maximum focal derived frequency. Default ``0.95``.
    :param int window_size: Window size in coordinate units of ``rec_map[:, -2]``. Default ``50000``.
    :param bool genetic_distance: Unused here (kept for API symmetry).

    :returns: DataFrame with columns ``positions``, ``daf``, ``hapdaf_s``.
    :rtype: polars.DataFrame
    """
    sq_freqs, info, snps_indices = fast_sq_freq_pairs(
        hap, ac, rec_map, min_focal_freq, max_focal_freq, window_size
    )

    n_rows = len(sq_freqs)
    results = np.empty((n_rows, 1), dtype=np.float64)
    # nan_index = []
    for i, v in enumerate(sq_freqs):
        f_d = v[:, 0]
        f_a = v[:, 1]
        f_tot = v[:, 2]

        f_d2 = (
            f_d[(f_d > f_a) & (f_a <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )
        f_a2 = (
            f_a[(f_d > f_a) & (f_a <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        )

        if len(f_d2) != 0 and len(f_a2) != 0:
            hapdaf = (f_d2 - f_a2).sum() / f_d2.shape[0]
        else:
            hapdaf = np.nan

        # # Flipping derived to ancestral, ancestral to derived
        # f_d2f = (
        #     f_a[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        # )
        # f_a2f = (
        #     f_d[(f_a > f_d) & (f_d <= max_ancest_freq) & (f_tot >= min_tot_freq)] ** 2
        # )
        # if len(f_d2f) != 0 and len(f_a2f) != 0:
        #     hapdaf_flip = (f_d2f - f_a2f).sum() / f_d2f.shape[0]
        # else:
        #     hapdaf_flip = np.nan

        # results.append((hapdaf, hapdaf_flip))

        results[i] = hapdaf

    tmp_schema = {
        "positions": pl.Int64,
        "daf": pl.Float64,
        "hapdaf_s": pl.Float64,
        # "hapdaf_s_flip": pl.Float64,
    }

    try:
        out = np.hstack(
            [
                info[:, :2],
                # np.array(results),
                results,
            ]
        )
        df_out = pl.DataFrame(out, schema=tmp_schema)

        # df_out = pl.DataFrame([out[:, 0], out[:, 1], out[:, 4], out[:, 5]], schema=tmp_schema)
    except Exception:
        df_out = pl.DataFrame([[], [], [], []], schema=tmp_schema)

    return df_out


def dind_high_low(
    hap,
    ac,
    rec_map,
    max_ancest_freq=0.25,
    min_tot_freq=0,
    min_focal_freq=0.25,
    max_focal_freq=0.95,
    window_size=50000,
    genetic_distance=False,
):
    """
    Compute DIND, highfreq, and lowfreq statistics per focal SNP.

    :param numpy.ndarray hap: Haplotype matrix ``(n_snps, n_samples)`` with 0/1 values.
    :param numpy.ndarray ac: Allele counts ``(n_snps, 2)`` as ``[ancestral, derived]``.
    :param numpy.ndarray rec_map: Map array; penultimate column is the window coordinate.
    :param float max_ancest_freq: Threshold used in high/low frequency components. Default ``0.25``.
    :param float min_tot_freq: Unused here (kept for API symmetry). Default ``0``.
    :param float min_focal_freq: Minimum focal derived frequency. Default ``0.25``.
    :param float max_focal_freq: Maximum focal derived frequency. Default ``0.95``.
    :param int window_size: Window size in coordinate physical units from ``rec_map[:, -2]``. Default ``50000``.
    :param bool genetic_distance: Unused here (kept for API symmetry).

    :returns:
        DataFrame with columns ``positions``, ``daf``, ``dind``, ``high_freq``,
        ``low_freq``.
    :rtype: polars.DataFrame
    """

    sq_freqs, info, snps_indices = fast_sq_freq_pairs(
        hap, ac, rec_map, min_focal_freq, max_focal_freq, window_size
    )

    focal_counts = info[:, 2:]

    # Pre-allocate arrays for results to avoid growing lists
    n_rows = len(sq_freqs)
    results_dind = np.empty((n_rows, 1), dtype=np.float64)
    results_high = np.empty((n_rows, 1), dtype=np.float64)
    results_low = np.empty((n_rows, 1), dtype=np.float64)

    # Main computation loop
    for i, v in enumerate(sq_freqs):
        f_d = v[:, 0]
        f_a = v[:, 1]

        focal_derived_count = focal_counts[i][0]
        focal_ancestral_count = focal_counts[i][1]

        # Calculate derived and ancestral components with in-place operations
        f_d2 = f_d * (1 - f_d) * focal_derived_count / (focal_derived_count - 1)
        f_a2 = (
            f_a
            * (1 - f_a)
            * focal_ancestral_count
            / (focal_ancestral_count - 1 + 0.001)
        )

        # Calculate dind values
        num = (f_d2 - f_d2 + f_a2).sum()
        den = (f_a2 - f_a2 + f_d2).sum() + 0.001
        dind_v = num / den if not np.isinf(num / den) else np.nan
        # dind_v_flip = den / num if not np.isinf(den / num) else np.nan

        # results_dind[i] = [dind_v, dind_v_flip]
        results_dind[i] = dind_v

        # Calculate high and low frequency values
        hf_v = (f_d[f_d > max_ancest_freq] ** 2).sum() / max(
            len(f_d[f_d > max_ancest_freq]), 1
        )
        # hf_v_flip = (f_a[f_a > max_ancest_freq] ** 2).sum() / max(
        #     len(f_a[f_a > max_ancest_freq]), 1
        # )
        # results_high[i] = [hf_v, hf_v_flip]
        results_high[i] = hf_v

        lf_v = ((1 - f_d[f_d < max_ancest_freq]) ** 2).sum() / max(
            len(f_d[f_d < max_ancest_freq]), 1
        )

        # lf_v_flip = ((1 - f_a[f_a < max_ancest_freq]) ** 2).sum() / max(
        #     len(f_a[f_a < max_ancest_freq]), 1
        # )
        # results_low[i] = [lf_v, lf_v_flip]
        results_low[i] = lf_v

        # Free memory explicitly for large arrays
        del f_d, f_a, f_d2, f_a2

    tmp_schema = {
        "positions": pl.Int64,
        "daf": pl.Float64,
        "dind": pl.Float64,
        # "dind_flip": pl.Float64,
        "high_freq": pl.Float64,
        # "high_freq_flip": pl.Float64,
        "low_freq": pl.Float64,
        # "low_freq_flip": pl.Float64,
    }

    # Final DataFrame creation
    try:
        out = np.hstack([info[:, :2], results_dind, results_high, results_low])
        df_out = pl.DataFrame(out, schema=tmp_schema)
        # df_out = pl.DataFrame([out[:, 0],out[:, 1],out[:, 4],out[:, 5],out[:, 6],out[:, 7],out[:, 8],out[:, 9],],schema=tmp_schema,)

    except Exception:
        df_out = pl.DataFrame([[], [], [], [], [], [], [], []], schema=tmp_schema)

    return df_out


@njit(parallel=False, cache=True)
def s_ratio_from_pairs(sq_freqs, max_ancest_freq=1, min_tot_freq=0):
    n_rows = len(sq_freqs)
    results = np.zeros((n_rows, 1))

    for i in prange(len(sq_freqs)):
        f_d = sq_freqs[i][:, 0]
        f_a = sq_freqs[i][:, 1]

        f_d2 = np.zeros(f_d.shape)
        f_a2 = np.zeros(f_a.shape)

        f_d2[(f_d > 0.0000001) & (f_d < 1)] = 1
        f_a2[(f_a > 0.0000001) & (f_a < 1)] = 1

        num = (f_d2 - f_d2 + f_a2 + 1).sum()
        den = (f_a2 - f_a2 + f_d2 + 1).sum()
        # redefine to add one to get rid of blowup issue introduced by adding 0.001 to denominator

        # Add error checking before division
        if den == 0:
            s_ratio_v = np.nan
        else:
            s_ratio_v = num / den

        # if num == 0:
        #     s_ratio_v_flip = np.nan
        # else:
        #     s_ratio_v_flip = den / num
        # s_ratio_v = num / den
        # s_ratio_v_flip = den / num
        # results[i] = s_ratio_v, s_ratio_v_flip
        results[i] = s_ratio_v

    return results


@njit(cache=True, parallel=False)
def hapdaf_from_pairs(sq_freqs, max_ancest_freq, min_tot_freq):
    """
    Unified hapdaf_o and hapdaf_s — bodies were identical, only defaults differ.

    Call as::

        hapdaf_o = hapdaf_from_pairs(sq_freqs, max_ancest_freq=0.25, min_tot_freq=0.25)
        hapdaf_s = hapdaf_from_pairs(sq_freqs, max_ancest_freq=0.10, min_tot_freq=0.10)

    Removed dead args: hap, snps_indices (were passed to hapdaf_o but never used).
    """
    n_rows = len(sq_freqs)
    results = np.zeros((n_rows, 1), dtype=np.float64)

    for i in prange(n_rows):
        f_d = sq_freqs[i][:, 0]
        f_a = sq_freqs[i][:, 1]
        f_tot = sq_freqs[i][:, 2]

        mask = (f_d > f_a) & (f_a <= max_ancest_freq) & (f_tot >= min_tot_freq)
        f_d2 = f_d[mask] ** 2
        f_a2 = f_a[mask] ** 2

        if len(f_d2) > 0:
            results[i, 0] = (f_d2 - f_a2).sum() / float64(len(f_d2))
        else:
            results[i, 0] = np.nan

    return results


@njit(parallel=False, cache=True)
def dind_high_low_from_pairs(sq_freqs, info, max_ancest_freq=0.25, min_tot_freq=0):
    # Pre-allocate arrays for results to avoid growing lists
    n_rows = len(sq_freqs)
    results_dind = np.zeros((n_rows, 1))
    results_high = np.zeros((n_rows, 1))
    results_low = np.zeros((n_rows, 1))
    # results_dind = np.zeros((n_rows, 2))
    # results_high = np.zeros((n_rows, 2))
    # results_low = np.zeros((n_rows, 2))

    # Main computation loop
    for i in prange(len(sq_freqs)):
        f_d = sq_freqs[i][:, 0]
        f_a = sq_freqs[i][:, 1]
        f_tot = sq_freqs[i][:, 2]

        focal_derived_count = info[i][-2]
        focal_ancestral_count = info[i][-1]

        # Calculate derived and ancestral components
        f_d2 = f_d * (1 - f_d) * focal_derived_count / (focal_derived_count - 1)
        f_a2 = (
            f_a
            * (1 - f_a)
            * focal_ancestral_count
            / (focal_ancestral_count - 1 + 0.001)
        )

        # Calculate dind values
        num = (f_d2 - f_d2 + f_a2).sum()
        den = (f_a2 - f_a2 + f_d2).sum() + 0.001
        if den != 0.0:
            dind_v = num / den
        else:
            dind_v = np.nan

        # if num != 0.0:
        #     dind_v_flip = den / num
        # else:
        #     dind_v_flip = np.nan

        # results_dind[i] = [dind_v, dind_v_flip]
        results_dind[i] = dind_v

        # Calculate high and low frequency values
        fd_h_mask = (f_d > max_ancest_freq) & (f_tot >= min_tot_freq)
        # fa_h_mask = (f_a > max_ancest_freq) & (f_tot >= min_tot_freq)
        fd_l_mask = (f_d < max_ancest_freq) & (f_tot >= min_tot_freq)
        # fa_l_mask = (f_a < max_ancest_freq) & (f_tot >= min_tot_freq)

        fd_l_mask = ((f_d > max_ancest_freq) & (f_d < 0.8)) & (f_tot >= min_tot_freq)
        fd_l_mask = (f_d > max_ancest_freq) & (f_tot >= min_tot_freq)
        # fa_l_mask = ((f_a > 0.25) & (f_a < 0.8)) & (f_tot >= min_tot_freq)

        hf_v = (f_d[fd_h_mask] ** 2).sum() / max(len(f_d[fd_h_mask]), 1)
        # hf_v_flip = (f_a[fa_h_mask] ** 2).sum() / max(len(f_a[fa_h_mask]), 1)
        # results_high[i] = [hf_v, hf_v_flip]
        results_high[i] = hf_v

        lf_v = ((1 - f_d[fd_l_mask]) ** 2).sum() / max(len(f_d[fd_l_mask]), 1)
        # lf_v_flip = ((1 - f_a[fa_l_mask]) ** 2).sum() / max(len(f_a[fa_l_mask]), 1)
        # results_low[i] = [lf_v, lf_v_flip]
        results_low[i] = lf_v

    return results_dind, results_high, results_low


def fs_stats_dataframe(
    info,
    results_dind,
    results_high,
    results_low,
    results_s_ratio,
    results_hapdaf_o,
    results_hapdaf_s,
    _iter=None,
):
    try:
        out_dind_high_low = np.hstack(
            [info[:, :2], results_dind, results_high, results_low]
        )
        df_dind_high_low = pl.DataFrame(
            out_dind_high_low,
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "dind": pl.Float64,
                # "dind_flip": pl.Float64,
                "high_freq": pl.Float64,
                # "high_freq_flip": pl.Float64,
                "low_freq": pl.Float64,
                # "low_freq_flip": pl.Float64,
            },
        )

    except Exception:
        df_dind_high_low = pl.DataFrame(
            # [[], [], [], [], [], [], [], []],
            [[], [], [], [], []],
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "dind": pl.Float64,
                # "dind_flip": pl.Float64,
                "high_freq": pl.Float64,
                # "high_freq_flip": pl.Float64,
                "low_freq": pl.Float64,
                # "low_freq_flip": pl.Float64,
            },
        )

    try:
        out_s_ratio = np.hstack([info[:, :2], results_s_ratio])
        df_s_ratio = pl.DataFrame(
            out_s_ratio,
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "s_ratio": pl.Float64,
                # "s_ratio_flip": pl.Float64,
            },
        )
    except Exception:
        df_s_ratio = pl.DataFrame(
            # [[], [], [], []],
            [[], [], []],
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "s_ratio": pl.Float64,
                # "s_ratio_flip": pl.Float64,
            },
        )

    try:
        out_hapdaf_s = np.hstack([info[:, :2], np.array(results_hapdaf_s)])
        df_hapdaf_s = pl.DataFrame(
            out_hapdaf_s,
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "hapdaf_s": pl.Float64,
                # "hapdaf_s_flip": pl.Float64,
            },
        )
    except Exception:
        df_hapdaf_s = pl.DataFrame(
            [[], [], []],
            # [[], [], [], []],
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "hapdaf_s": pl.Float64,
                # "hapdaf_s_flip": pl.Float64,
            },
        )

    try:
        out_hapdaf_o = np.hstack([info[:, :2], np.array(results_hapdaf_o)])
        df_hapdaf_o = pl.DataFrame(
            out_hapdaf_o,
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "hapdaf_o": pl.Float64,
                # "hapdaf_o_flip": pl.Float64,
                # "omega_diff": pl.Float64,
            },
        )
    except Exception:
        df_hapdaf_o = pl.DataFrame(
            [[], [], []],
            # [[], [], [], []],
            schema={
                "positions": pl.Int64,
                "daf": pl.Float64,
                "hapdaf_o": pl.Float64,
                # "hapdaf_o_flip": pl.Float64,
                # "omega_diff": pl.Float64,
            },
        )

    if _iter is not None:
        df_dind_high_low.with_columns(pl.lit(_iter).alias("iter"))
        df_s_ratio.with_columns(pl.lit(_iter).alias("iter"))
        df_hapdaf_o.with_columns(pl.lit(_iter).alias("iter"))
        df_hapdaf_s.with_columns(pl.lit(_iter).alias("iter"))

    return (
        df_dind_high_low.fill_nan(None),
        df_s_ratio.fill_nan(None),
        df_hapdaf_o.fill_nan(None),
        df_hapdaf_s.fill_nan(None),
    )


################## iSAFE


@njit(cache=True)
def rank_with_duplicates(x):
    # sorted_arr = sorted(x, reverse=True)
    sorted_arr = np.sort(x)[::-1]
    rank_dict = {}
    rank = 1
    prev_value = -1

    for value in sorted_arr:
        if value != prev_value:
            rank_dict[value] = rank
        rank += 1
        prev_value = value

    return np.array([rank_dict[value] for value in x])


# @njit("float64[:,:](float64[:,:])", cache=True)


@njit(parallel=False, cache=True)
def dot_nb(hap):
    return np.dot(hap.T, hap)


@njit(cache=True)
def neutrality_divergence_proxy(kappa, phi, freq, method=3):
    sigma1 = (kappa) * (1 - kappa)
    sigma1[sigma1 == 0] = 1.0
    sigma1 = sigma1**0.5
    p1 = (phi - kappa) / sigma1
    sigma2 = (freq) * (1 - freq)
    sigma2[sigma2 == 0] = 1.0
    sigma2 = sigma2**0.5
    p2 = (phi - kappa) / sigma2
    nu = freq[np.argmax(p1)]
    p = p1 * (1 - nu) + p2 * nu

    if method == 1:
        return p1
    elif method == 2:
        return p2
    elif method == 3:
        return p


@njit(cache=True)
def calc_H_K(hap, haf):
    """
    :param snp_matrix: Binary SNP Matrix
    :return: H: Sum of HAF-score of carriers of each mutation.
    :return: N: Number of distinct carrier haplotypes of each mutation.

    """
    num_snps, num_haplotypes = hap.shape

    haf_matrix = haf * hap

    K = np.zeros((num_snps))

    for j in range(num_snps):
        ar = haf_matrix[j, :]
        K[j] = len(np.unique(ar[ar > 0]))
    H = np.sum(haf_matrix, 1)
    return (H, K)


def safe(hap):
    num_snps, num_haplotypes = hap.shape

    haf = dot_nb(hap.astype(np.float64)).sum(1)
    # haf = np.dot(hap.T, hap).sum(1)
    H, K = calc_H_K(hap, haf)

    phi = 1.0 * H / haf.sum()
    kappa = 1.0 * K / (np.unique(haf).shape[0])
    freq = hap.sum(1) / num_haplotypes
    safe_values = neutrality_divergence_proxy(kappa, phi, freq)

    # rank = np.zeros(safe_values.size)
    # rank = rank_with_duplicates(safe_values)
    # rank = (
    #     pd.DataFrame(safe_values).rank(method="min", ascending=False).values.flatten()
    # )
    rank = (
        pl.DataFrame({"safe": safe_values})
        .select(pl.col("safe").rank(method="min", descending=True))
        .to_numpy()
        .flatten()
    )

    return haf, safe_values, rank, phi, kappa, freq


def creat_windows_summary_stats_nb(hap, pos, w_size=300, w_step=150):
    num_snps, num_haplotypes = hap.shape
    rolling_indices = create_rolling_indices_nb(num_snps, w_size, w_step)
    windows_stats = {}
    windows_haf = []
    snp_summary = []

    for i, I_rolling in enumerate(rolling_indices):
        window_i_stats = {}
        haf, safe_values, rank, phi, kappa, freq = safe(
            hap[I_rolling[0] : I_rolling[1], :]
        )

        tmp = pl.DataFrame(
            {
                "safe": safe_values,
                "rank": rank,
                "phi": phi,
                "kappa": kappa,
                "freq": freq,
                "pos": pos[I_rolling[0] : I_rolling[1]],
                "ordinal_pos": np.arange(I_rolling[0], I_rolling[1]),
                "window": np.repeat(i, I_rolling[1] - I_rolling[0]),
            }
        )

        window_i_stats["safe"] = tmp
        windows_haf.append(haf)
        windows_stats[i] = window_i_stats
        snp_summary.append(tmp)

    combined_df = pl.concat(snp_summary).with_columns(
        pl.col("ordinal_pos").cast(pl.Float64)
    )
    # combined_df = combined_df.with_row_count(name="index")
    # snps_summary.select(snps_summary.columns[1:])

    return windows_stats, windows_haf, combined_df


@njit(cache=True)
def create_rolling_indices_nb(total_variant_count, w_size, w_step):
    assert total_variant_count < w_size or w_size > 0

    rolling_indices = []
    w_start = 0
    while True:
        w_end = min(w_start + w_size, total_variant_count)
        if w_end >= total_variant_count:
            break
        rolling_indices.append([w_start, w_end])
        # rolling_indices += [range(int(w_start), int(w_end))]
        w_start += w_step

    return rolling_indices


def run_isafe(
    hap,
    positions,
    max_freq=1,
    min_region_size_bp=49000,
    min_region_size_ps=300,
    ignore_gaps=True,
    window=300,
    step=150,
    top_k=1,
    max_rank=15,
):
    """
    Estimate iSAFE or SAFE on a genomic region following Flex-sweep default values.

    The function removes monomorphic SNPs, then checks region size. If
    ``num_snps <= min_region_size_ps`` or ``positions.max() - positions.min() < min_region_size_bp``,
    it computes **SAFE**; otherwise it computes **iSAFE** using the provided sliding-window
    settings. Results are returned as a Polars DataFrame with columns ``positions`` (bp),
    ``daf`` (derived allele frequency), and ``isafe`` (score). Variants with
    ``daf >= max_freq`` are filtered out.

    :param numpy.ndarray hap:
        Haplotype matrix of shape ``(n_snps, n_haplotypes)`` with 0/1 values
        (ancestral/derived).
    :param numpy.ndarray positions:
        1D array of physical coordinates (length ``n_snps``) aligned to ``hap`` rows.
    :param float max_freq:
        Maximum allowed derived allele frequency in the output (``daf < max_freq``).
        Default ``1`` (no filter).
    :param int min_region_size_bp:
        Minimum region span in base pairs required to run iSAFE. Default ``49000``.
    :param int min_region_size_ps:
        Minimum number of polymorphic SNPs required to run iSAFE. Default ``300``.
    :param bool ignore_gaps:
        Reserved for gap handling; currently not used. Default ``True``.
    :param int window:
        iSAFE sliding window size (number of SNPs or bp, depending on the
        downstream implementation). Default ``300``.
    :param int step:
        iSAFE step between windows. Default ``150``.
    :param int top_k:
        iSAFE parameter controlling the number of top candidates per window.
        Default ``1``.
    :param int max_rank:
        iSAFE parameter controlling the maximum rank to track. Default ``15``.

    :returns:
        Polars DataFrame with columns ``positions`` (int), ``daf`` (float),
        and ``isafe`` (float), sorted by position and filtered to ``daf < max_freq``.
        If the region is small, the ``isafe`` column contains SAFE scores.
    :rtype: polars.DataFrame

    .. note::
       Monomorphic sites are removed using ``(1 - f) * f > 0``, where ``f`` is the
       derived allele frequency per SNP. When computing iSAFE, the function passes
       ``window``, ``step``, ``top_k``, and ``max_rank`` to the underlying implementation.
    """

    total_window_size = positions.max() - positions.min()

    # dp = np.diff(positions)
    # num_gaps = sum(dp > 6000000)
    f = hap.mean(1)
    freq_filter = ((1 - f) * f) > 0
    hap_filtered = hap[freq_filter, :]
    positions_filtered = positions[freq_filter]
    num_snps = hap_filtered.shape[0]

    if (num_snps <= min_region_size_ps) | (total_window_size < min_region_size_bp):
        haf, safe_values, rank, phi, kappa, freq = safe(hap_filtered)

        df_safe = pl.DataFrame(
            {
                "isafe": safe_values,
                "rank": rank,
                "phi": phi,
                "kappa": kappa,
                "daf": freq,
                "positions": positions_filtered,
            }
        )

        return (
            df_safe.select(["positions", "daf", "isafe"])
            .sort("positions")
            .with_columns(pl.col("positions").cast(pl.Int64))
        )

    else:
        df_isafe = isafe(
            hap_filtered, positions_filtered, window, step, top_k, max_rank
        )
        df_isafe = (
            df_isafe.filter(pl.col("freq") < max_freq)
            .sort("ordinal_pos")
            .rename({"id": "positions", "isafe": "isafe", "freq": "daf"})
            .filter(pl.col("daf") < max_freq)
            .with_columns(pl.col("positions").cast(pl.Int64))
            .select(["positions", "daf", "isafe"])
        )

    return df_isafe


def isafe(hap, pos, w_size=300, w_step=150, top_k=1, max_rank=15):
    windows_summaries, windows_haf, snps_summary = creat_windows_summary_stats_nb(
        hap, pos, w_size, w_step
    )
    df_top_k1 = get_top_k_snps_in_each_window(snps_summary, k=top_k)

    ordinal_pos_snps_k1 = np.sort(df_top_k1["ordinal_pos"].unique()).astype(np.int64)

    psi_k1 = step_function(creat_matrix_Psi_k_nb(hap, windows_haf, ordinal_pos_snps_k1))

    df_top_k2 = get_top_k_snps_in_each_window(snps_summary, k=max_rank)
    temp = np.sort(df_top_k2["ordinal_pos"].unique())

    ordinal_pos_snps_k2 = np.sort(np.setdiff1d(temp, ordinal_pos_snps_k1)).astype(
        np.int64
    )

    psi_k2 = step_function(creat_matrix_Psi_k_nb(hap, windows_haf, ordinal_pos_snps_k2))

    alpha = psi_k1.sum(0) / psi_k1.sum()

    iSAFE1 = pl.DataFrame(
        data={
            "ordinal_pos": ordinal_pos_snps_k1,
            "isafe": np.dot(psi_k1, alpha),
            "tier": np.repeat(1, ordinal_pos_snps_k1.size),
        }
    )

    iSAFE2 = pl.DataFrame(
        {
            "ordinal_pos": ordinal_pos_snps_k2,
            "isafe": np.dot(psi_k2, alpha),
            "tier": np.repeat(2, ordinal_pos_snps_k2.size),
        }
    )

    # Concatenate the DataFrames and reset the index
    iSAFE = pl.concat([iSAFE1, iSAFE2])

    # Add the "id" column using values from `pos`

    iSAFE = iSAFE.with_columns(
        pl.col("ordinal_pos")
        .map_elements(lambda x: pos[x], return_dtype=pl.Int64)
        .alias("id")
    )
    # Add the "freq" column using values from `freq`
    freq = hap.mean(1)
    iSAFE = iSAFE.with_columns(
        pl.col("ordinal_pos")
        .map_elements(lambda x: freq[x], return_dtype=pl.Float64)
        .alias("freq")
    )

    # Select the required columns
    df_isafe = iSAFE.select(["ordinal_pos", "id", "isafe", "freq", "tier"])

    return df_isafe


# @njit
# def creat_matrix_Psi_k_nb(hap, hafs, Ifp):
#     P = np.zeros((len(Ifp), len(hafs)))
#     for i in range(len(Ifp)):
#         for j in range(len(hafs)):
#             P[i, j] = isafe_kernel_nb(hafs[j], hap[Ifp[i], :])
#     return P


@njit(cache=True)
def isafe_kernel_nb(haf, snp):
    phi = haf[snp == 1].sum() * 1.0 / haf.sum()
    kappa = len(np.unique(haf[snp == 1])) / (1.0 * len(np.unique(haf)))
    f = np.mean(snp)
    sigma2 = (f) * (1 - f)
    if sigma2 == 0:
        sigma2 = 1.0
    sigma = sigma2**0.5
    p = (phi - kappa) / sigma
    return p


@njit(cache=True)
def creat_matrix_Psi_k_nb(hap, hafs, Ifp):
    """Further optimized version with pre-computed unique values"""
    P = np.zeros((len(Ifp), len(hafs)))

    # Pre-compute for each haf: sum and unique count
    haf_sums = np.zeros(len(hafs))
    haf_unique_counts = np.zeros(len(hafs))

    for j in range(len(hafs)):
        haf_sums[j] = hafs[j].sum()
        haf_unique_counts[j] = len(np.unique(hafs[j]))

    for i in range(len(Ifp)):
        snp = hap[Ifp[i], :]

        # Pre-compute common values for this row
        f = np.mean(snp)
        sigma2 = f * (1 - f)
        if sigma2 == 0:
            sigma2 = 1.0
        sigma = sigma2**0.5

        snp_ones_idx = np.where(snp == 1)[0]

        for j in range(len(hafs)):
            haf = hafs[j]

            # Use pre-computed values
            phi = haf[snp_ones_idx].sum() / haf_sums[j]
            kappa = len(np.unique(haf[snp_ones_idx])) / haf_unique_counts[j]

            p = (phi - kappa) / sigma
            P[i, j] = p

    return P


def step_function(P0):
    P = P0.copy()
    P[P < 0] = 0
    return P


def get_top_k_snps_in_each_window(df_snps, k=1):
    """
    :param df_snps:  this datafram must have following columns: ["safe","ordinal_pos","window"].
    :param k:
    :return: return top k snps in each window.
    """
    return (
        df_snps.group_by("window")
        .agg(pl.all().sort_by("safe", descending=True).head(k))
        .explode(pl.all().exclude("window"))
        .sort("window")
        .select(pl.all().exclude("window"), pl.col("window"))
    )


################## LD stats


@njit(parallel=False, cache=True)
def r2(locus_A: np.ndarray, locus_B: np.ndarray) -> float:
    """
    Compute the squared correlation coefficient :math:`r^2` between two biallelic loci.

    Given two 0/1 vectors of equal length (haplotypes across samples), this function
    computes:

    .. math::

       D = P_{11} - p_A p_B,\\quad
       r^2 = \\frac{D^2}{p_A (1-p_A)\\, p_B (1-p_B)},

    where :math:`p_A` and :math:`p_B` are the allele-1 frequencies at loci A and B,
    and :math:`P_{11}` is the empirical joint frequency that both loci equal 1.

    :param numpy.ndarray locus_A:
        1D array of 0/1 alleles for locus A (dtype ``int8`` expected by Numba signature).
    :param numpy.ndarray locus_B:
        1D array of 0/1 alleles for locus B (same length and dtype as ``locus_A``).

    :returns:
        The :math:`r^2` value as a float.
    :rtype: float

    .. note::
       If either locus is monomorphic (denominator zero), the result may be ``inf`` or
       ``nan`` depending on arithmetic; callers typically filter such sites beforehand.
    """
    n = locus_A.size
    # Frequency of allele 1 in locus A and locus B
    a1 = 0
    b1 = 0
    count_a1b1 = 0

    for i in range(n):
        a1 += locus_A[i]
        b1 += locus_B[i]
        count_a1b1 += locus_A[i] * locus_B[i]

    a1 /= n
    b1 /= n
    a1b1 = count_a1b1 / n
    D = a1b1 - a1 * b1

    r_squared = (D**2) / (a1 * (1 - a1) * b1 * (1 - b1))
    return r_squared


def compute_r2_matrix_upper(hap, as_float32=False):
    """
    r² via pre-scaled BLAS matmul. Avoids the outer-product subtraction step
    of the original compute_r2_matrix_upper, saving one O(S²) allocation.
    """
    if not as_float32:
        if hap.dtype != np.float64:
            raise TypeError(
                f"compute_r2_matrix_upper: hap must be float64, got {hap.dtype}. "
                "Pass as_float32=True to use float32 internally."
            )
    else:
        hap = hap.astype(np.float32, copy=False)

    S, N = hap.shape
    p = hap.mean(axis=1)
    v = p * (1.0 - p)
    v[v == 0] = np.inf
    std = np.sqrt(v)
    hap_scaled = (hap - p[:, None]) / std[:, None]
    r2 = (hap_scaled @ hap_scaled.T) / N
    np.square(r2, out=r2)
    return np.triu(r2, k=1)


@njit(parallel=False, cache=True)
def omega_linear_correct(r2_matrix):
    """
    Compute :math:`\\omega_\\text{max}` (Kim & Nielsen, 2004) from an :math:`r^2` matrix.

    The statistic compares the average LD within two partitions (left/right of a split)
    to the average LD between the partitions. For a split index :math:`\\ell` on a
    sequence of length :math:`S`, define:

    .. math::

       \\begin{aligned}
       &\\text{within-left}   &&= \\sum_{0 \\le i < j < \\ell} r^2_{ij},\\\\
       &\\text{within-right}  &&= \\sum_{\\ell \\le i < j < S} r^2_{ij},\\\\
       &\\text{between}       &&= \\sum_{0 \\le i < \\ell} \\sum_{\\ell \\le j < S} r^2_{ij},
       \\end{aligned}

    and the means are obtained by dividing by the corresponding pair counts
    :math:`\\binom{\\ell}{2}`, :math:`\\binom{S-\\ell}{2}`, and :math:`\\ell(S-\\ell)`.
    The omega score at :math:`\\ell` is:

    .. math::

       \\omega(\\ell) = \\frac{\\dfrac{\\text{within-left}}{\\binom{\\ell}{2}}
                          + \\dfrac{\\text{within-right}}{\\binom{S-\\ell}{2}}}
                         {\\dfrac{\\text{between}}{\\ell(S-\\ell)}}.

    This function scans admissible :math:`\\ell` and returns the maximum value.

    :param numpy.ndarray _r2:
        Square matrix (``S`` × ``S``) of pairwise :math:`r^2` values. Only the
        upper triangle (``i < j``) is required to hold valid values.
    :param numpy.ndarray mask:
        Optional boolean vector selecting a subset of SNP indices to consider.
        Default ``None`` (use all SNPs).

    :returns:
        The maximum omega value over all candidate split points.
    :rtype: float

    :notes:
        Modification from https://github.com/kr-colab/diploSHIC/blob/master/diploshic/utils.c
        taking advantages of numpy vectorized operations. Very small windows (``S < 3``) return ``0.0``.

    """

    S = r2_matrix.shape[0]
    if S < 3:
        # return np.array([0.0,0.0])
        return 0.0, 0.0

    #   Build row_sum[i] = sum_{j>i} r2[i,j]
    #   and       col_sum[j] = sum_{i<j} r2[i,j]
    #   Also accumulate total of all upper‐triangle entries.
    row_sum = np.zeros(S, np.float64)
    col_sum = np.zeros(S, np.float64)
    total = 0.0
    for i in range(S):
        s = 0.0
        for j in range(i + 1, S):
            v = r2_matrix[i, j]
            s += v
            col_sum[j] += v
        row_sum[i] = s
        total += s

    # Kelly's ZnS
    divisor = (S * (S - 1)) / 2.0
    zns = total / divisor if divisor > 0 else 0.0

    # Build prefix_L[_l] = sum_{i<j<_l} r2[i,j]  (prefix_L[0] = 0 sentinel)
    prefix_L = np.zeros(S, np.float64)
    for _l in range(1, S):
        prefix_L[_l] = prefix_L[_l - 1] + col_sum[_l - 1]

    # Build suffix_R[_l] = sum_{_l≤i<j} r2[i,j]  (suffix_R[S] = 0 sentinel)
    suffix_R = np.zeros(S + 1, np.float64)
    # suffix_R[S] = 0.0
    for _l in range(S - 1, -1, -1):
        suffix_R[_l] = suffix_R[_l + 1] + row_sum[_l]

    # Sweep _l = 3..S-3 in O(S), compute _omega and track maximum
    omega_max = 0.0
    # omega_argmax = -1.0
    for _l in range(3, S - 2):
        sum_L = prefix_L[_l]
        sum_R = suffix_R[_l]
        sum_LR = total - sum_L - sum_R
        if sum_LR > 0.0:
            denom_L = (_l * (_l - 1) / 2.0) + ((S - _l) * (S - _l - 1) / 2.0)
            denom_R = _l * (S - _l)
            _omega = ((sum_L + sum_R) / denom_L) / (sum_LR / denom_R)
            if _omega > omega_max:
                omega_max = _omega
                # omega_argmax = _l + 2

    # return np.array([omega_max,omega_argmax])
    return zns, omega_max


def Ld(hap, as_float32=True) -> tuple:
    """
    Compute **Kelly's ZnS** (mean pairwise :math:`r^2`) and **omega\\_max** from an LD matrix.

    The input ``r_2`` is a square matrix of pairwise linkage disequilibrium
    values :math:`r^2` among SNPs within a window. If ``mask`` is provided,
    the computation is restricted to the subset of indices where ``mask`` is
    ``True``.

    ZnS is defined as:

    .. math::

       \\mathrm{ZnS} = \\frac{\\sum_{i<j} r^2_{ij}}{\\binom{S}{2}},

    where :math:`S` is the number of SNPs after masking.

    The function also returns ``omega_max`` (Kim & Nielsen, 2004), computed via
    :func:`omega_linear_correct_mask`, which scans split points and compares the
    average LD within versus between the two partitions.

    :param numpy.ndarray r_2:
        Square matrix (``S`` × ``S``) of pairwise :math:`r^2` values. The routine
        treats it as symmetric; values on and below the diagonal are ignored for ZnS.
    :param numpy.ndarray mask:
        Optional boolean vector of length ``S`` to select a subset of SNPs. Default ``None``.

    :returns:
        Tuple ``(zns, omega_max)`` as floats.
    :rtype: tuple[float, float]
    """

    r_2 = compute_r2_matrix_upper(hap, as_float32=as_float32)

    # S = _r_2.shape[0]
    # zns = _r_2.sum() / comb(S, 2)

    zns, omega_max = omega_linear_correct(r_2)

    # return zns, 0
    return zns, omega_max


################## Site Frequency Spectrum stats


@njit(cache=True)
def _harmonic_sums(n):
    """
    Return harmonic sums up to ``n-1``.

    Computes:
      - ``a1 = sum_{i=1}^{n-1} 1/i``
      - ``a2 = sum_{i=1}^{n-1} 1/i^2``

    :param int n:
        Sample size (number of chromosomes).
    :returns:
        A length-2 array ``[a1, a2]`` as ``float64``.
    :rtype: numpy.ndarray
    """
    a1 = 0.0
    a2 = 0.0
    for i in range(1, int(n)):
        inv = 1.0 / i
        a1 += inv
        a2 += inv * inv
    return np.array((a1, a2), dtype=np.float64)


@njit(cache=True)
def theta_watterson(ac, positions):
    # count segregating variants
    S = ac.shape[0]
    n = ac[0].sum()

    a1 = _harmonic_sums(n)[0]

    # calculate absolute value
    theta_hat_w_abs = S / a1

    # calculate value per base
    if positions.size < 2:
        # not enough positions to estimate per-base value meaningfully
        return theta_hat_w_abs, np.nan

    n_bases = (positions[-1] - positions[0]) + 1
    theta_hat_w = theta_hat_w_abs / n_bases

    return theta_hat_w_abs, theta_hat_w


@njit(cache=True)
def sfs_nb(dac, n):
    """
    Site-frequency spectrum (SFS) from derived-allele counts.

    :param numpy.ndarray dac:
        1D array of derived allele counts per site, values in ``[0..n]``.
    :param int n:
        Total number of chromosomes. If ``n <= 0``, it is inferred as
        ``max(dac)``.
    :returns:
        Integer array of length ``n+1``; ``sfs[k]`` is the number of sites
        with ``k`` derived copies.
    :rtype: numpy.ndarray
    """

    # infer n if not provided or invalid
    if n <= 0:
        maxv = 0
        for i in range(dac.shape[0]):
            if dac[i] > maxv:
                maxv = dac[i]
        n = maxv

    # initialize spectrum
    s = np.zeros(n + 1, dtype=np.int64)

    # counts
    for i in range(dac.shape[0]):
        k = dac[i]
        if 0 <= k <= n:
            s[k] += 1
    return s


@njit(cache=True)
def theta_pi(ac):
    """
    Per-site nucleotide diversity (π) from allele counts.

    For each site ``j``, computes
    ``pi_j = 2 * a_j * (n - a_j) / [n * (n - 1)]``, where ``a_j`` is the
    derived allele count and ``n`` is the total number of chromosomes.

    :param numpy.ndarray ac:
        Allele counts array of shape ``(S, 2)`` (ancestral, derived), with
        constant ``n`` across sites.
    :returns:
        Array of per-site π values of length ``S``.
    :rtype: numpy.ndarray
    """
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()
    denom_pairs = n * (n - 1.0)

    # pi = np.zeros(S)
    pi = np.zeros(S)
    for j in range(S):
        aj = ac[j, 1]
        pi[j] = 2.0 * aj * (n - aj) / denom_pairs
    return pi


@njit(cache=True)
def tajima_d(ac, min_sites=3):
    """
    Tajima’s D from allele counts.

    Compares the mean pairwise difference (sum of per-site π) to the
    Watterson estimator based on the number of segregating sites.
    Returns ``nan`` if the number of segregating sites is below ``min_sites``.

    :param numpy.ndarray ac:
        Allele counts array of shape ``(S, 2)`` (ancestral, derived), with
        constant ``n`` across sites.
    :param int min_sites:
        Minimum required number of segregating sites. Default ``3``.
    :returns:
        Tajima’s D as a float (``nan`` if insufficient sites).
    :rtype: float
    """
    # count segregating variants
    S = ac.shape[0]

    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()
    if S < min_sites:
        return np.nan

    # (n-1)th harmonic number
    an, bn = _harmonic_sums(n)

    # calculate Watterson's theta (absolute value)
    theta_hat_w_abs = S / an

    # calculate mean pairwise difference
    mpd = theta_pi(ac)

    # calculate theta_hat pi (sum differences over variants)
    theta_hat_pi_abs = mpd.sum()

    # N.B., both theta estimates are usually divided by the number of
    # (accessible) bases but here we want the absolute difference
    d = theta_hat_pi_abs - theta_hat_w_abs

    # calculate the denominator (standard deviation)
    a2 = np.sum(1 / (np.arange(1, n) ** 2))
    b1 = (n + 1) / (3 * (n - 1))
    b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))
    c1 = b1 - (1 / an)
    c2 = b2 - ((n + 2) / (an * n)) + (a2 / (an**2))
    e1 = c1 / an
    e2 = c2 / (an**2 + a2)
    d_stdev = np.sqrt((e1 * S) + (e2 * S * (S - 1)))

    # finally calculate Tajima's D
    D = d / d_stdev

    return D


@njit(cache=True)
def achaz_y(ac):
    """
    Achaz’s Y neutrality test (standardized).

    Unfolded/polarized form — requires ancestral-state information (outgroup or
    polarization). Excludes ξ₁ (derived singletons) from both estimators.

    Reference: Achaz 2008, Appendix B, Equations B28–B30.
    f = (n-2) / (n·(a_n-1));  Var[Y] = α_n·θ + β_n·θ²

    :param numpy.ndarray ac:
        Allele counts array of shape ``(S, 2)`` (ancestral, derived), constant ``n``.
        Derived allele counts in ``ac[:,1]`` define the unfolded SFS.
    :returns:
        Standardized Achaz’s Y as a float; returns ``nan`` if ``n < 3`` or
        if there are no segregating sites excluding singletons.
    :rtype: float
    """
    n = int(ac[0, 0] + ac[0, 1])
    if n < 3:
        return np.nan
    fs = sfs_nb(ac[:, 1], n)
    a1, a2 = _harmonic_sums(n)
    a1m1 = a1 - 1.0
    ff = (n - 2.0) / (n * a1m1)
    inv_n = 1.0 / n
    inv_n1 = 1.0 / (n - 1.0)
    inv_n2 = 1.0 / (n - 2.0)
    n2 = n * n
    alpha = (
        ff * ff * a1m1
        + ff
        * (
            a1 * (4.0 * (n + 1.0) * inv_n1 * inv_n1)
            - 2.0 * (n + 1.0) * (n + 2.0) * inv_n * inv_n1
        )
        - a1 * 8.0 * (n + 1.0) * inv_n * inv_n1 * inv_n1
        + (n * n2 + n2 + 60.0 * n + 12.0) * (inv_n * inv_n) * (1.0 / 3.0) * inv_n1
    )
    beta = (
        ff * ff * (a2 + a1 * (4.0 * inv_n1 * inv_n2) - 4.0 * inv_n2)
        + ff
        * (
            -a1 * (4.0 * (n + 2.0) * inv_n * inv_n1 * inv_n2)
            - ((n * n2 - 3.0 * n2 - 16.0 * n + 20.0) * inv_n * inv_n1 * inv_n2)
        )
        + a1 * (8.0 * inv_n * inv_n1 * inv_n2)
        + (2.0 * (2.0 * n2 * n2 - n2 * n - 17.0 * n2 - 42.0 * n + 72.0))
        * (inv_n * inv_n)
        * (inv_n1 * inv_n2)
        * (1.0 / 9.0)
    )
    y = fs.copy()
    y[0] = y[1] = y[n] = 0.0
    S = 0.0
    pi_sum = 0.0
    for i in range(2, n + 1):
        yi = y[i]
        if i > 1 and i < n:
            S += yi
        if i < n:
            pi_sum += yi * i * (n - i)

    # At least 1 seg site
    if S < 1:
        return np.nan

    pi_hat = pi_sum / (n * (n - 1.0) * 0.5)
    that = S / a1m1
    that_sq = S * (S - 1.0) / (a1m1 * a1m1)
    return (pi_hat - ff * S) / np.sqrt(alpha * that + beta * that_sq)


@njit(cache=True)
def achaz_y_star(ac):
    """
    Achaz's Y* neutrality test (standardized).

    Folded form — does not require ancestral-state polarization. Excludes
    η₁ = ξ₁ + ξ_{n-1} (minor-allele singletons at frequency 1/n or (n-1)/n)
    from both the π and S estimators.

    Reference: Achaz 2008, Appendix B, Equations B19–B21.
    f* = (n-3) / (a_n·(n-1) - n);  Var[Y*] = α*_n·θ + β*_n·θ²

    :param numpy.ndarray ac:
        Allele counts array of shape ``(S, 2)`` (ancestral, derived), constant n.
    :returns:
        Standardized Achaz Y* as a float; nan if n < 4 or no valid sites.
    :rtype: float
    """
    n = int(ac[0, 0] + ac[0, 1])
    if n < 4:
        return np.nan

    a1, a2 = _harmonic_sums(n)

    inv_n1 = 1.0 / (n - 1.0)
    # γ_n = E[S_{-η₁}] / θ = a_n - n/(n-1)
    gamma_n = a1 - n * inv_n1
    if gamma_n <= 0.0:
        return np.nan

    # f* = (n-3) / (a_n*(n-1) - n) = (n-3) / ((n-1)*γ_n)  [Achaz 2008 Eq 21]
    fstar = (n - 3.0) / ((n - 1.0) * gamma_n)

    inv_n = 1.0 / n
    n2 = n * n

    # α*_n  (Achaz 2008 Eq B20)
    alpha_star = (
        fstar * fstar * (a1 - n * inv_n1)
        + fstar * (a1 * (4.0 * (n + 1.0) * inv_n1 * inv_n1) - 2.0 * (n + 3.0) * inv_n1)
        - a1 * 8.0 * (n + 1.0) * inv_n * inv_n1 * inv_n1
        + (n2 + n + 60.0) * inv_n * inv_n1 / 3.0
    )

    # β*_n  (Achaz 2008 Eq B21)
    beta_star = (
        fstar * fstar * (a2 - (2.0 * n - 1.0) * inv_n1 * inv_n1)
        + fstar
        * (
            a2 * 8.0 * inv_n1
            - a1 * 4.0 * inv_n * inv_n1
            - (n * n2 + 12.0 * n2 - 35.0 * n + 18.0) * inv_n * inv_n1 * inv_n1
        )
        - a2 * 16.0 * inv_n * inv_n1
        + a1 * 8.0 * inv_n * inv_n * inv_n1
        + (2.0 * (2.0 * n2 * n2 + 110.0 * n2 - 255.0 * n + 126.0))
        * (inv_n * inv_n)
        * (inv_n1 * inv_n1)
        / 9.0
    )

    # Compute S_{-η₁} and π_{-η₁}: sum over sites with 2 ≤ derived count ≤ n-2
    S_total = ac.shape[0]
    S_excl = 0.0
    pi_excl = 0.0

    for j in range(S_total):
        k = int(ac[j, 1])
        if k >= 2 and k <= n - 2:
            S_excl += 1.0
            pi_excl += k * (n - k)

    if S_excl < 1.0:
        return np.nan

    pi_excl /= n * (n - 1.0) * 0.5

    # θ̂ and θ̂² from S_{-η₁} (consistent with achaz_y approximation)
    that = S_excl / gamma_n
    that_sq = S_excl * (S_excl - 1.0) / (gamma_n * gamma_n)

    variance = alpha_star * that + beta_star * that_sq
    if variance <= 0.0:
        return np.nan

    return (pi_excl - fstar * S_excl) / np.sqrt(variance)


@lru_cache(maxsize=16)
def _achaz_t_coeffs(n, decay=0.9):
    """Precompute αₙ and βₙ for T_Ω (Achaz 2009 Eq. 9).

    Cached per (n, decay) — computed once per sample size, O(1) on all
    subsequent calls. Uses the @njit ``sigma`` function for fast batch
    σᵢⱼ computation, then a BLAS quadratic form for βₙ.

    αₙ = Σᵢ i·Ωᵢ²
    βₙ = vᵀ·Σ·v  where vᵢ = i·Ωᵢ  and Σᵢⱼ = σᵢⱼ (Fu 1995)
    """
    k = np.arange(1, n, dtype=np.float64)  # i = 1..n-1
    w1 = np.exp(-decay * k)  # ω₁ᵢ = e^{-decay·i}
    w2 = np.ones(n - 1)  # ω₂ᵢ = 1 (uniform/Watterson)
    omega = w1 / w1.sum() - w2 / w2.sum()  # Ωᵢ (sums to 0)

    alpha_n = float(np.sum(k * omega**2))  # αₙ = Σᵢ i·Ωᵢ²  (O(n))

    # Full σᵢⱼ matrix via @njit sigma (batch call, fast)
    # Note: sigma is defined later in this module; forward ref is fine in Python.
    ki = k.astype(np.int64)
    ii, jj = np.meshgrid(ki, ki, indexing="ij")
    sig_mat = sigma(n, np.column_stack([ii.ravel(), jj.ravel()])).reshape(n - 1, n - 1)

    v = k * omega  # vᵢ = i·Ωᵢ
    beta_n = float(v @ sig_mat @ v)  # βₙ = vᵀΣv  (BLAS DGEMV)

    return alpha_n, beta_n


def achaz_t(ac, decay=0.9):
    """Achaz's T_Ω neutrality test (Achaz 2009 Eq. 9).

    Unfolded/polarized SFS. Uses exponential weight ω₁ᵢ = e^{−0.9·i}
    vs uniform ω₂ᵢ = 1. Sensitive to excess low-frequency polymorphisms
    such as those produced by severe bottlenecks.

    Variance coefficients αₙ/βₙ are precomputed once per sample size via
    ``_achaz_t_coeffs`` (lru_cache), so per-window cost is O(n).

    α = 0.9 is empirical (Achaz 2009 p.254): gives positive Ωᵢ only for
    i/n ≤ 0.13. θ² estimated as S(S-1)/(a₁²+a₂) following Fu (1995).

    :param numpy.ndarray ac:
        Allele counts array of shape ``(S, 2)`` (ancestral, derived), constant n.
        Derived allele counts in ``ac[:,1]`` define the unfolded SFS.
    :param float decay:
        Exponential decay α (default 0.9 per Achaz 2009).
    :returns:
        Standardized T_Ω as a float; nan if n < 3 or no segregating sites.
    :rtype: float
    """
    n = int(ac[0, 0] + ac[0, 1])
    if n < 3:
        return np.nan
    # O(1) after first call
    alpha_n, beta_n = _achaz_t_coeffs(n, decay)

    a1, a2 = _harmonic_sums(n)

    # Build unfolded SFS ξᵢ (i = 1..n-1) from ac
    xi = np.zeros(n - 1, dtype=np.float64)
    for j in range(ac.shape[0]):
        k = int(ac[j, 1])
        if 1 <= k <= n - 1:
            xi[k - 1] += 1.0

    S = xi.sum()
    if S < 1.0:
        return np.nan

    # θ̂ and θ̂² (exact Fu 1995 form; matches pg-gpu)
    that = S / a1
    that_sq = S * (S - 1.0) / (a1 * a1 + a2)

    variance = alpha_n * that + beta_n * that_sq
    if variance <= 0.0:
        return np.nan

    # Numerator: Σᵢ Ωᵢ·i·ξᵢ  (recompute Ω; cost is O(n), negligible)
    k_arr = np.arange(1, n, dtype=np.float64)
    w1 = np.exp(-decay * k_arr)
    w2 = np.ones(n - 1)
    omega = w1 / w1.sum() - w2 / w2.sum()
    numerator = float(np.sum(omega * k_arr * xi))

    return numerator / np.sqrt(variance)


@njit(cache=True)
def fay_wu_h_norm(ac, positions=None):
    """
    Fay & Wu’s H and its normalized form (single-population, infinite sites).

    Computes:
      - ``theta_h``: estimator that upweights high-frequency derived alleles.
      - ``h = pi - theta_h`` (Fay & Wu’s H).
      - ``h_norm``: normalized H using variance terms.

    :param numpy.ndarray ac:
        Allele counts array of shape ``(S, 2)`` (ancestral, derived), constant ``n``.
    :param numpy.ndarray positions:
        Optional positions (length ``S``). If provided, ``theta_h`` is divided
        by the accessible span ``positions[-1] - (positions[0] - 1)``.
    :returns:
        Tuple ``(theta_h, h, h_norm)`` as floats.
    :rtype: tuple[float, float, float]
    """

    # count segregating variants
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()

    fs = sfs_nb(ac[:, 1], n)[1:-1]

    i_arr = np.arange(1, int(n))
    a1 = np.sum(1.0 / i_arr)
    bn = np.sum(1.0 / (i_arr * i_arr)) + 1.0 / (n * n)
    theta_w = S / a1
    pi = 0.0
    theta_h = 0.0
    for k in range(1, int(n)):
        si = fs[k - 1]
        pi += (2 * si * k * (n - k)) / (n * (n - 1.0))
        theta_h += (2 * si * k * k) / (n * (n - 1.0))
    tl = 0.0
    for k in range(1, int(n)):
        tl += k * fs[k - 1]
    tl /= n - 1.0
    var1 = (n - 2.0) / (6.0 * (n - 1.0)) * theta_w
    theta_sq = S * (S - 1.0) / (a1 * a1 + bn)

    var2 = (
        ((18 * n * n * (3 * n + 2) * bn) - (88 * n * n * n + 9 * n * n - 13 * n - 6))
        / (9.0 * n * (n - 1.0) * (n - 1.0))
    ) * theta_sq

    h = pi - theta_h

    if positions is not None:
        theta_h = theta_h / (positions[-1] - (positions[0] - 1))
    return theta_h, h, h / np.sqrt(var1 + var2)


@njit(cache=True)
def zeng_e(ac):
    """
    Zeng’s E statistic (single-population, infinite sites), standardized.

    Contrasts Watterson’s estimator with a linear SFS component related to
    high-frequency derived signal. Useful alongside Tajima’s D and Fay & Wu’s H.

    :param numpy.ndarray ac:
        Allele counts array of shape ``(S, 2)`` (ancestral, derived), constant ``n``.
    :returns:
        Standardized Zeng’s E as a float.
    :rtype: float
    """

    # count segregating variants
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()

    fs = sfs_nb(ac[:, 1], n)[1:-1]

    # i_arr = np.arange(1, int(n))
    a1, bn = _harmonic_sums(n)
    # bn = np.sum(1.0 / (i_arr * i_arr))
    theta_w = S / a1

    tl = 0.0
    for k in range(1, int(n)):
        tl += k * fs[k - 1]
    tl /= n - 1.0
    theta_sq = S * (S - 1.0) / (a1 * a1 + bn)
    var1 = (n / (2.0 * (n - 1.0)) - 1.0 / a1) * theta_w
    var2 = (
        bn / a1 / a1
        + 2 * (n / (n - 1.0)) * (n / (n - 1.0)) * bn
        - 2 * (n * bn - n + 1) / ((n - 1.0) * a1)
        - (3 * n + 1) / (n - 1.0)
    ) * theta_sq
    return (tl - theta_w) / np.sqrt(var1 + var2)


@njit(cache=True)
def fuli_f_star(ac):
    """
    Fu and Li’s F* (starred) statistic (no outgroup required).

    Focuses on deviations in the **singleton** class of the (folded) SFS,
    contrasting singleton abundance with diversity (π). The starred form
    does not require ancestral state polarization.

    :param numpy.ndarray ac:
        Allele counts array of shape ``(S, 2)`` (ancestral, derived), constant ``n``.
    :returns:
        Fu & Li’s F* as a float.
    :rtype: float
    """

    # count segregating variants
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()

    an, bn = _harmonic_sums(n)
    an1 = an + np.true_divide(1, n)

    denom_pairs = n * (n - 1.0)
    pi = 0.0
    for j in range(S):
        aj = ac[j, 1]
        pi += 2.0 * aj * (n - aj) / denom_pairs

    ss = ((ac[:, 1] == 1) | (ac[:, 1] == n - 1)).sum()

    vfs = (
        (
            (2 * (n**3.0) + 110.0 * (n**2.0) - 255.0 * n + 153)
            / (9 * (n**2.0) * (n - 1.0))
        )
        + ((2 * (n - 1.0) * an) / (n**2.0))
        - ((8.0 * bn) / n)
    ) / ((an**2.0) + bn)
    ufs = (
        (
            n / (n + 1.0)
            + (n + 1.0) / (3 * (n - 1.0))
            - 4.0 / (n * (n - 1.0))
            + ((2 * (n + 1.0)) / ((n - 1.0) ** 2)) * (an1 - ((2.0 * n) / (n + 1.0)))
        )
        / an
    ) - vfs

    num = pi - ((n - 1.0) / n) * ss
    den = np.sqrt(ufs * S + vfs * (S * S))
    return num / den


@njit(cache=True)
def fuli_f(ac):
    """
    Fu and Li’s F statistic (polarized).

    Uses singleton counts and diversity (π); typically assumes **derived**
    states are known (e.g., via outgroup).

    :param numpy.ndarray ac:
        Allele counts array of shape ``(S, 2)`` (ancestral, derived), constant ``n``.
    :returns:
        Fu & Li’s F as a float.
    :rtype: float
    """
    # count segregating variants
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()
    an, bn = _harmonic_sums(n)
    an1 = an + 1.0 / n

    ss = (ac[:, 1] == 1).sum()

    denom_pairs = n * (n - 1.0)
    pi = 0.0
    for j in range(S):
        aj = ac[j, 1]
        pi += 2.0 * aj * (n - aj) / denom_pairs

    if n == 2:
        cn = 1
    else:
        cn = 2.0 * (n * an - 2.0 * (n - 1.0)) / ((n - 1.0) * (n - 2.0))

    v = (
        cn + 2.0 * (np.power(n, 2) + n + 3.0) / (9.0 * n * (n - 1.0)) - 2.0 / (n - 1.0)
    ) / (np.power(an, 2) + bn)
    u = (
        1.0
        + (n + 1.0) / (3.0 * (n - 1.0))
        - 4.0 * (n + 1.0) / np.power(n - 1, 2) * (an1 - 2.0 * n / (n + 1.0))
    ) / an - v
    F = (pi - ss) / np.sqrt(u * S + v * np.power(S, 2))

    return F


@njit(cache=True)
def fuli_d_star(ac):
    """
    Fu and Li’s D* (starred) statistic (no outgroup required).

    Compares the number of segregating sites against singleton counts in the
    folded spectrum. The starred form does not require ancestral state
    polarization.

    :param numpy.ndarray ac:
        Allele counts array of shape ``(S, 2)`` (ancestral, derived), constant ``n``.
    :returns:
        Fu & Li’s D* as a float.
    :rtype: float
    """
    # count segregating variants
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()

    an, bn = _harmonic_sums(n)
    an1 = an + np.true_divide(1, n)

    cn = 2 * ((n * an) - 2 * (n - 1)) / ((n - 1) * (n - 2))
    dn = (
        cn
        + np.true_divide((n - 2), ((n - 1) ** 2))
        + np.true_divide(2, (n - 1)) * (3.0 / 2 - (2 * an1 - 3) / (n - 2) - 1.0 / n)
    )

    vds = (
        ((n / (n - 1.0)) ** 2) * bn
        + (an**2) * dn
        - 2 * (n * an * (an + 1)) / ((n - 1.0) ** 2)
    ) / (an**2 + bn)
    uds = ((n / (n - 1.0)) * (an - n / (n - 1.0))) - vds

    ss = ((ac[:, 1] == 1) | (ac[:, 1] == n - 1)).sum()
    Dstar1 = ((n / (n - 1.0)) * S - (an * ss)) / (uds * S + vds * (S**2)) ** 0.5
    return Dstar1


@njit(cache=True)
def fuli_d(ac):
    """
    Fu and Li’s D statistic (polarized form).

    Uses the total number of segregating sites and singletons; typically assumes
    **derived** states are known (e.g., via outgroup) to define singletons.

    :param numpy.ndarray ac:
        Allele counts array of shape ``(S, 2)`` (ancestral, derived), constant ``n``.
    :returns:
        Fu & Li’s D as a float.
    :rtype: float
    """

    # count segregating variants
    S = ac.shape[0]
    # assume number of chromosomes sampled is constant for all variants
    n = ac[0].sum()

    an, bn = _harmonic_sums(n)

    ss = (ac[:, 1] == 1).sum()

    if n == 2:
        cn = 1
    else:
        cn = 2.0 * (n * an - 2.0 * (n - 1.0)) / ((n - 1.0) * (n - 2.0))

    v = 1.0 + (np.power(an, 2) / (bn + np.power(an, 2))) * (cn - (n + 1.0) / (n - 1.0))
    u = an - 1.0 - v
    D = (S - ss * an) / np.sqrt(u * S + v * np.power(S, 2))
    return D


@njit(cache=True)
def neutrality_stats(ac, positions):
    """

    #     Numba unified call to compute SFS neutrality statistics. Trying to avoid numba overhead as much as possible

    Returns a length-12 array:
        [0]  tajima_d
        [1]  theta_h          (scaled by sequence length if positions provided)
        [2]  h_raw            (Fay & Wu's H = pi - theta_h, unscaled)
        [3]  h_norm           (normalized Fay & Wu's H)
        [8]  pi               (mean pairwise diversity, absolute)
        [9]  theta_w          (Watterson's theta, absolute)
        [10] theta_w_per_base (nan if < 2 positions)
        [11] pi_per_base      (nan if < 2 positions)
    """
    out = np.empty(8, dtype=np.float64)

    S = ac.shape[0]
    if S < 3:
        for i in range(8):
            out[i] = np.nan
        return out

    # Constant counts
    dac = ac[:, 1]
    n = int(ac[0, 0] + ac[0, 1])
    n_f = float(n)

    sfs = np.zeros(n - 1, dtype=np.int64)
    ss_derived = 0
    for i in range(S):
        k = int(dac[i])
        if 0 < k < n:
            sfs[k - 1] += 1
        if k == 1:
            ss_derived += 1

    # harmonic sums
    an = 0.0
    bn = 0.0
    for i in range(1, n):
        inv = 1.0 / i
        an += inv
        bn += inv * inv
    an1 = an + 1.0 / n_f

    # pi, theta_h, theta_l

    theta_w = S / an
    pi = 0.0
    theta_h = 0.0
    theta_l = 0.0
    denom = n_f * (n_f - 1.0)
    for k_idx in range(n - 1):
        k = k_idx + 1
        count = sfs[k_idx]
        pi += (2.0 * count * k * (n_f - k)) / denom
        theta_h += (2.0 * count * k * k) / denom
        theta_l += k * count
    theta_l /= n_f - 1.0

    # Per-base values
    if positions.size >= 2:
        n_bases = float(positions[-1] - positions[0] + 1)
        theta_w_per_base = theta_w / n_bases
        pi_per_base = pi / n_bases
        theta_h_final = theta_h / float(positions[-1] - (positions[0] - 1))
    else:
        theta_w_per_base = np.nan
        pi_per_base = np.nan
        theta_h_final = np.nan

    # Tajima's D
    e1_d = ((n_f + 1.0) / (3.0 * (n_f - 1.0)) - (1.0 / an)) / an
    e2_d = (
        (2.0 * (n_f**2 + n_f + 3.0) / (9.0 * n_f * (n_f - 1.0)))
        - ((n_f + 2.0) / (an * n_f))
        + (bn / (an**2))
    ) / (an**2 + bn)
    d_stdev = np.sqrt(e1_d * S + e2_d * S * (S - 1.0))
    tajima_d_val = (pi - theta_w) / d_stdev if d_stdev > 0.0 else np.nan

    # Fay & Wu's H + normalized
    h_raw = pi - theta_h
    bn_h = bn + 1.0 / (n_f * n_f)
    var1_h = (n_f - 2.0) / (6.0 * (n_f - 1.0)) * theta_w
    th_sq = S * (S - 1.0) / (an**2 + bn)
    var2_h = (
        (
            (18.0 * n_f**2 * (3.0 * n_f + 2.0) * bn_h)
            - (88.0 * n_f**3 + 9.0 * n_f**2 - 13.0 * n_f - 6.0)
        )
        / (9.0 * n_f * (n_f - 1.0) ** 2)
        * th_sq
    )
    denom_h = var1_h + var2_h
    h_norm = h_raw / np.sqrt(denom_h) if denom_h > 0.0 else np.nan

    # Output
    out[0] = tajima_d_val
    out[1] = theta_h
    out[2] = h_raw
    out[3] = h_norm
    out[4] = pi
    out[5] = theta_w
    out[6] = theta_w_per_base
    out[7] = pi_per_base

    return out


################## LASSI


def get_empir_freqs_np_fast(hap):
    """
    Optimized version to calculate the empirical frequencies of haplotypes.

    Parameters:
    - hap (numpy.ndarray): Shape (S, n), where S = SNPs, n = individuals.

    Returns:
    - k_counts (numpy.ndarray): Counts of each unique haplotype.
    - h_f (numpy.ndarray): Frequencies of each unique haplotype.
    """
    # Transpose so each haplotype is a row
    hap_t = hap.T  # shape (n, S)

    # Hash each haplotype row into a unique identifier
    hashes = np.ascontiguousarray(hap_t).view(
        np.dtype((np.void, hap_t.dtype.itemsize * hap_t.shape[1]))
    )

    # Use np.unique on 1D hashes
    _, unique_counts = np.unique(hashes, return_counts=True)

    # Sort counts in descending order
    k_counts = np.sort(unique_counts)[::-1]
    h_f = k_counts / hap_t.shape[0]

    return k_counts, h_f


def process_spectra(k: np.ndarray, h_f: np.ndarray, K_truncation: int, n_ind: int):
    """
    Process haplotype count and frequency spectra.

    Parameters:
    - k (numpy.ndarray): Counts of each unique haplotype.
    - h_f (numpy.ndarray): Empirical frequencies of each unique haplotype.
    - K_truncation (int): Number of haplotypes to consider.
    - n_ind (int): Number of individuals.

    Returns:
    - Kcount (numpy.ndarray): Processed haplotype count spectrum.
    - Kspect (numpy.ndarray): Processed haplotype frequency spectrum.
    """
    # Truncate count and frequency spectrum
    Kcount = k[:K_truncation]
    Kspect = h_f[:K_truncation]

    # Normalize count and frequency spectra
    Kcount = Kcount / Kcount.sum() * n_ind
    Kspect = Kspect / Kspect.sum()

    # Pad with zeros if necessary
    if Kcount.size < K_truncation:
        Kcount = np.concatenate([Kcount, np.zeros(K_truncation - Kcount.size)])
        Kspect = np.concatenate([Kspect, np.zeros(K_truncation - Kspect.size)])

    return Kcount, Kspect


def LASSI_spectrum_and_Kspectrum(input_data, K_truncation=10, window=110, step=5):
    """
    Compute haplotype count and frequency spectra within sliding windows.

    Parameters:
    - hap (numpy.ndarray): Array of haplotypes where each column represents an individual and each row represents a SNP.
    - pos (numpy.ndarray): Array of SNP positions.
    - K_truncation (int): Number of haplotypes to consider.
    - window (int): Size of the sliding window.
    - step (int): Step size for sliding the window.

    Returns:
    - K_count (numpy.ndarray): Haplotype count spectra for each window.
    - K_spectrum (numpy.ndarray): Haplotype frequency spectra for each window.
    - windows_centers (numpy.ndarray): Centers of the sliding windows.
    """
    filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message="invalid value encountered in scalar divide",
    )
    np.seterr(divide="ignore", invalid="ignore")

    if isinstance(input_data, list) or isinstance(input_data, tuple):
        hap_int, position_masked = input_data
    elif isinstance(input_data, str):
        try:
            (
                hap_int,
                rec_map_01,
                ac,
                biallelic_mask,
                position_masked,
                genetic_position_masked,
            ) = genome_reader(input_data)
            # freqs = ac[:, 1] / ac.sum(axis=1)
        except Exception:
            try:
                (
                    hap_int,
                    rec_map_01,
                    ac,
                    biallelic_mask,
                    position_masked,
                    genetic_position_masked,
                ) = parse_ms_numpy(input_data)
                # freqs = ac[:, 1] / ac.sum(axis=1)
            except Exception:
                return None

    else:
        return None

    K_count = []
    K_spectrum = []
    windows_centers = []
    S, n = hap_int.shape
    for i in range(0, S, step):
        hap_subset = hap_int[i : i + window, :]

        # Calculate window center based on median SNP position
        windows_centers.append(np.median(position_masked[i : i + window]))

        # Compute empirical frequencies and process spectra for the window
        k, h_f = get_empir_freqs_np_fast(hap_subset)
        K_count_subset, K_spectrum_subset = process_spectra(k, h_f, K_truncation, n)

        K_count.append(K_count_subset)
        K_spectrum.append(K_spectrum_subset)
        if hap_subset.shape[0] < window:
            break

    return np.array(K_count), np.array(K_spectrum), np.array(windows_centers)


def neut_average(K_spectrum: np.ndarray) -> np.ndarray:
    """
    Compute the neutral average of haplotype frequency spectra.

    Parameters:
    - K_spectrum (numpy.ndarray): Haplotype frequency spectra.

    Returns:
    - out (numpy.ndarray): Neutral average haplotype frequency spectrum.
    """
    weights = []
    S, n = K_spectrum.shape
    # Compute mean spectrum
    gwide_K = np.mean(K_spectrum, axis=0)

    # Calculate weights for averaging
    if S % 5e4 == 0:
        weights.append(5e4)
    else:
        small_weight = S % 5e4
        weights.append(small_weight)

    # Compute weighted average
    out = np.average([gwide_K], axis=0, weights=weights)

    return out


@njit(cache=True)
def easy_likelihood(K_neutral, K_count, K_truncation):
    """
    Basic computation of the likelihood function; runs as-is for neutrality, but called as part of a larger process for sweep model
    """

    likelihood_list = []

    for i in range(K_truncation):
        likelihood_list.append(K_count[i] * np.log(K_neutral[i]))

    likelihood = sum(likelihood_list)

    return likelihood


@njit(cache=True)
def sweep_likelihood(
    K_neutral, K_count, K_truncation, m_val, epsilon, epsilon_max, sweep_mode=4
):
    """
    Computes the likelihood of a sweep under optimized parameters.

    sweep_mode controls how frequency is redistributed among the m sweeping haplotype classes:
      1 — Zipf (1/j), normalized
      2 — Zipf squared (1/j²), normalized
      3 — exponential (exp(-j)), normalized
      4 — exponential squared (exp(-j²)), normalized  [default]
      5 — uniform (1/m)
    """

    if m_val != K_truncation:
        altspect = np.zeros(K_truncation)
        tailclasses = np.zeros(K_truncation - m_val)
        neutdiff = np.zeros(K_truncation - m_val)
        tailinds = np.arange(m_val + 1, K_truncation + 1)

        for i in range(len(tailinds)):
            ti = tailinds[i]
            denom = K_truncation - m_val - 1
            if denom != 0:
                the_ns = epsilon_max - ((ti - m_val - 1) / denom) * (
                    epsilon_max - epsilon
                )
            else:
                the_ns = epsilon
            tailclasses[i] = the_ns
            neutdiff[i] = K_neutral[ti - 1] - the_ns

        headinds = np.arange(1, m_val + 1)

        for hd in headinds:
            altspect[hd - 1] = K_neutral[hd - 1]

        neutdiff_all = np.sum(neutdiff)

        # Precompute denominator for normalized modes 1-4
        denom_sum = 0.0
        for x in headinds:
            if sweep_mode == 1:
                denom_sum += 1.0 / float(x)
            elif sweep_mode == 2:
                denom_sum += 1.0 / float(x * x)
            elif sweep_mode == 3:
                denom_sum += np.exp(-float(x))
            elif sweep_mode == 4:
                denom_sum += np.exp(-float(x * x))
            # sweep_mode 5: uniform, no normalization needed

        for ival in headinds:
            if sweep_mode == 1:
                theadd = (1.0 / float(ival) / denom_sum) * neutdiff_all
            elif sweep_mode == 2:
                theadd = (1.0 / float(ival * ival) / denom_sum) * neutdiff_all
            elif sweep_mode == 3:
                theadd = (np.exp(-float(ival)) / denom_sum) * neutdiff_all
            elif sweep_mode == 4:
                theadd = (np.exp(-float(ival * ival)) / denom_sum) * neutdiff_all
            else:  # sweep_mode == 5
                theadd = (1.0 / float(m_val)) * neutdiff_all
            altspect[ival - 1] += theadd

        altspect[m_val:] = tailclasses

        output = easy_likelihood(altspect, K_count, K_truncation)
    else:
        output = easy_likelihood(K_neutral, K_count, K_truncation)

    return output


@njit(cache=True)
def compute_epsilon_values(K_truncation, K_neutral_last):
    epsilon_min = 1 / (K_truncation * 100)
    values = []
    for i in range(1, 101):
        val = i * epsilon_min
        if val <= K_neutral_last:
            values.append(val)
    return np.array(values)


@njit(cache=True)
def T_m_statistic_core(K_counts, K_neutral, windows, K_truncation, sweep_mode=4):
    num_windows = len(windows)
    m_vals = K_truncation + 1
    epsilon_values = compute_epsilon_values(K_truncation, K_neutral[-1])

    # Estimate max rows possible: 1 row per window
    output = np.zeros(
        (num_windows, 6 + len(K_counts[0]))
    )  # 6 meta values + K_iter size

    for j in range(num_windows):
        w = windows[j]
        K_iter = K_counts[j]

        null_likelihood = easy_likelihood(K_neutral, K_iter, K_truncation)

        best_likelihood = -np.inf
        best_m = 0
        best_e = 0.0

        for e in epsilon_values:
            for m in range(1, m_vals):
                alt_like = sweep_likelihood(
                    K_neutral, K_iter, K_truncation, m, e, K_neutral[-1], sweep_mode
                )
                likelihood_diff = 2 * (alt_like - null_likelihood)
                if likelihood_diff > best_likelihood:
                    best_likelihood = likelihood_diff
                    best_m = m
                    best_e = e

        # Build the output row
        output[j, 0] = w
        output[j, 1] = best_likelihood
        output[j, 2] = best_m
        output[j, 3] = best_e
        output[j, 4] = K_neutral[-1]
        output[j, 5] = sweep_mode
        output[j, 6:] = K_iter

    return output


def T_m_statistic_fast(
    K_counts, K_neutral, windows, K_truncation, sweep_mode=4, _iter=0
):
    t_m = T_m_statistic_core(K_counts, K_neutral, windows, K_truncation, sweep_mode)
    stats_schema = {
        "window_lassi": pl.Int64,
        "T": pl.Float64,
        "m": pl.Float64,
        "frequency": pl.Float64,
        "e": pl.Float64,
        "model": pl.Float64,
    }
    k_schema = {"Kcounts_" + str(i): pl.Float64 for i in range(1, K_truncation + 1)}
    output = pl.DataFrame(
        t_m, schema=pl.Schema({**stats_schema, **k_schema})
    ).with_columns(pl.lit(_iter).cast(pl.Int64).alias("iter"))

    return output


def compute_t_m(
    sim_list,
    K_truncation=10,
    w_size=201,
    w_step=10,
    K_neutral=None,
    sweep_mode=4,
    center=[5e4, 1.2e6 - 5e4],
    windows=[100000],
    step=int(1e5),
    nthreads=1,
    params=None,
    parallel_manager=None,
):
    """
    Compute LASSI-style T and m-hat over a set of simulations.

    The function builds truncated haplotype-frequency spectra per window, estimates a neutral
    spectrum if not provided, scores each window with T and m, and then reduces the scan to
    fixed physical windows around the specified centers. If ``params`` are provided, they are
    attached and the result may be pivoted to feature vectors format.

    :param sim_list: Iterable of simulation items consumable by LASSI_spectrum_and_Kspectrum.
    :type sim_list: sequence
    :param K_truncation: Number of top haplotype counts retained in the truncated spectrum. Default 5.
    :type K_truncation: int
    :param w_size: Sliding window size in SNPs used to build K-spectra. Default 110.
    :type w_size: int
    :param step: Step in SNPs between consecutive windows. Default 5.
    :type step: int
    :param K_neutral: Precomputed neutral truncated spectrum; if None, estimated via neut_average. Optional.
    :type K_neutral: array-like or None
    :param windows: Physical window widths (bp) for cut_t_m_argmax. Default [50000, 100000, 200000, 500000, 1000000].
    :type windows: list[int]
    :param center: Inclusive physical range (bp) defining centers. Default [500000, 700000].
    :type center: list[int]
    :param nthreads: Number of joblib workers. Default 1.
    :type nthreads: int
    :param params: Optional parameter matrix aligned to sim_list with columns [s, t, f_i, f_t].
    :type params: array-like or None
    :param parallel_manager: Existing joblib.Parallel to reuse; if None, a new one is created.
    :type parallel_manager: joblib.Parallel or None

    :returns: (t_m_cut, K_neutral)
    :rtype: tuple

    :notes: T is a log-likelihood ratio comparing sweep-distorted vs neutral truncated spectra.
            m is the estimated number of sweeping haplotypes (1 = hard; >1 = soft),
            upper-bounded by ``K_truncation``.
    """
    from . import Parallel, delayed

    if parallel_manager is None:
        parallel_manager = Parallel(n_jobs=nthreads, verbose=1)

    hfs_stats = parallel_manager(
        delayed(LASSI_spectrum_and_Kspectrum)(hap_data, K_truncation, w_size, w_step)
        for _index, (hap_data) in enumerate(sim_list[:], 1)
    )

    K_counts, K_spectrum, windows_lassi = zip(*hfs_stats)

    if K_neutral is None:
        K_neutral = neut_average(np.vstack(K_spectrum))

    t_m = parallel_manager(
        delayed(T_m_statistic_fast)(
            kc,
            K_neutral,
            windows_lassi[_iter - 1],
            K_truncation,
            sweep_mode=sweep_mode,
            _iter=_iter,
        )
        for _iter, (kc) in enumerate(K_counts, 1)
    )

    t_m_cut = parallel_manager(
        delayed(cut_t_m_argmax)(
            t, windows=windows, center=center, step=step, _iter=_iter
        )
        for _iter, t in enumerate(t_m, 1)
    )

    t_m_cut = pl.concat(t_m_cut)
    t_m_cut = t_m_cut.select(
        [
            "iter",
            "window",
            "center",
            *[
                col
                for col in t_m_cut.columns
                if col not in ("iter", "window", "center")
            ],
        ]
    )
    if params is not None:
        t_m_cut = pivot_feature_vectors(
            pl.concat(
                [
                    pl.DataFrame(
                        np.repeat(
                            params,
                            t_m_cut.select(["center", "window"]).unique().shape[0],
                            axis=0,
                        ),
                        schema=["s", "t", "f_i", "f_t"],
                    ),
                    t_m_cut,
                ],
                how="horizontal",
            )
        )

    return t_m_cut, K_neutral


def cut_t_m_argmax(
    df_t_m,
    center=[5e4, 1.2e6 - 5e4],
    windows=[100000],
    step=1e5,
    _iter=1,
):
    K_names_c = df_t_m.select("^Kcounts_.*$").schema
    t_schema = OrderedDict(
        {
            "T": pl.Float64,
            "m": pl.Float64,
            **K_names_c,
            "iter": pl.Int64,
            "window": pl.Int64,
            "center": pl.Int64,
        }
    )
    out = []
    centers = np.arange(center[0], center[1] + step, step).astype(int)
    iter_c_w = list(product(centers, windows))
    for c, w in iter_c_w:
        # for w in [1000000]:
        lower = c - w / 2
        upper = c + w / 2

        df_t_m_subset = df_t_m.filter(
            (pl.col("window_lassi") > lower) & (pl.col("window_lassi") < upper)
        )

        try:
            max_t = df_t_m_subset["T"].arg_max()

            # df_t_m_subset = df_t_m_subset[df_t_m_subset.m > 0]
            # max_t = df_t_m_subset[df_t_m_subset.m > 0].m.argmin()
            df_t_m_subset = df_t_m_subset[max_t : max_t + 1, :]

            df_t_m_subset = df_t_m_subset.select(
                pl.exclude(["frequency", "e", "model", "window_lassi"])
            ).with_columns(
                pl.lit(w).cast(pl.Int64).alias("window"),
                pl.lit(c).cast(pl.Int64).alias("center"),
            )
            out.append(df_t_m_subset)
        except Exception:
            tmp = pl.DataFrame(
                {
                    col: [
                        None
                        if col not in ["iter", "center", "window"]
                        else _iter
                        if col == "iter"
                        else c
                        if col == "center"
                        else w
                    ]
                    for col in t_schema.keys()
                },
                schema=t_schema,
            )
            out.append(tmp)

    out = pl.concat(out)

    return out


def run_lassi(
    hap_data, K_truncation=10, w_size=201, step=10, K_neutral=None, sweep_mode=4
):
    try:
        (
            hap_int,
            rec_map_01,
            ac,
            biallelic_mask,
            position_masked,
            genetic_position_masked,
        ) = genome_reader(hap_data)
        # freqs = ac[:, 1] / ac.sum(axis=1)
    except Exception:
        (
            hap_int,
            rec_map_01,
            ac,
            biallelic_mask,
            position_masked,
            genetic_position_masked,
        ) = parse_ms_numpy(hap_data)
        # freqs = ac[:, 1] / ac.sum(axis=1)

    K_counts, K_spectrum, windows_lassi = LASSI_spectrum_and_Kspectrum(
        [hap_int, position_masked], K_truncation, w_size, int(step)
    )

    if K_neutral is None:
        K_neutral = neut_average(np.vstack(K_spectrum))

    t_m = T_m_statistic_fast(
        K_counts, K_neutral, windows_lassi, K_truncation, sweep_mode=sweep_mode
    )[:, :-1]

    return t_m


################## saltiLASSI


@njit(parallel=False, cache=True)
def _lassip_precompute(K_counts, K_neutral, K_truncation, sweep_mode=4):
    """
    saltiLASSI precomputation: build 3D dF table using geometric mixture.

    Matches the C++ lassip reference implementation (lassip-winstats.cpp L265-267):
      L1 = Σ_i [α_i · sweep_lik_i(m,ε) + (1-α_i) · null_lik_i]
      L1 - L0 = Σ_i α_i · (sweep_lik_i - null_lik_i) = dot(alpha, dF[mi, ei, :])

    dF[mi, ei, i] = sweep_likelihood(i, m, ε) − easy_likelihood(i)

    Memory: K × n_e × n_windows × 8 bytes ≈ 384 KB for typical inputs (fits in L2 cache).
    Returns (dF, null_composite, epsilon_values).
    """
    n_windows = K_counts.shape[0]
    epsilon_values = compute_epsilon_values(K_truncation, K_neutral[-1])
    epsilon_max = K_neutral[-1]
    n_e = len(epsilon_values)

    null_per_window = np.zeros(n_windows)
    for i in range(n_windows):
        null_per_window[i] = easy_likelihood(K_neutral, K_counts[i], K_truncation)
    null_composite = np.sum(null_per_window)

    dF = np.zeros((K_truncation, n_e, n_windows))
    for mi in range(K_truncation):
        m_val = mi + 1
        for ei in range(n_e):
            epsilon = epsilon_values[ei]
            for i in range(n_windows):
                s_lik = sweep_likelihood(
                    K_neutral,
                    K_counts[i],
                    K_truncation,
                    m_val,
                    epsilon,
                    epsilon_max,
                    sweep_mode,
                )
                dF[mi, ei, i] = s_lik - null_per_window[i]

    return dF, null_composite, epsilon_values


@njit(parallel=False, cache=True)
def _lassip_chunk(
    j_start, j_end, positions, K_truncation, A_grid, dF, epsilon_values, max_extend=1e5
):
    """
    saltiLASSI phase 3: optimize (m, A, ε) for target windows j_start..j_end-1.

    Hot path: exp() + multiply-add only — no log() calls.
    L1 - L0 = Σ_i α_i · dF[mi, ei, i]  (dot product with precomputed 3D dF table).

    max_extend gates which source windows contribute: only windows within max_extend bp
    of the target z* are included (alpha=0 beyond). Matches C++ lassip MAX_EXTEND_BP
    (default 100000 bp). Distances are precomputed once per target window and reused
    across all A/m/ε iterations.

    The inner loop `delta += alpha[i] * dF[mi, ei, i]` is a pure multiply-add reduction
    over a contiguous float64 slice — numba auto-vectorizes to AVX2 (4 doubles/cycle).

    Output column 3 stores 1/best_A (C++ convention: `maxA = 1.0/exp(A_loop)`).
    Returns output array of shape (j_end - j_start, 5).
    """
    n_windows = positions.shape[0]
    n_chunk = j_end - j_start
    output = np.zeros((n_chunk, 5))

    for jj in range(n_chunk):
        j = j_start + jj
        z_star = positions[j]
        best_lambda = -np.inf
        best_m = 1
        best_A = A_grid[0]
        best_e = epsilon_values[0]

        # precompute distances once — reused across all A/m/ε iterations
        distances = np.empty(n_windows)
        for i in range(n_windows):
            distances[i] = np.abs(positions[i] - z_star)

        for a_idx in range(len(A_grid)):
            A_val = A_grid[a_idx]
            alpha = np.zeros(n_windows)
            for i in range(n_windows):
                if distances[i] <= max_extend:
                    alpha[i] = np.exp(-A_val * distances[i])
                # else alpha[i] stays 0 (beyond MAX_EXTEND)

            for mi in range(K_truncation):
                for ei in range(len(epsilon_values)):
                    delta = 0.0
                    for i in range(n_windows):
                        delta += alpha[i] * dF[mi, ei, i]
                    lambda_val = 2.0 * delta
                    if lambda_val > best_lambda:
                        best_lambda = lambda_val
                        best_m = mi + 1
                        best_A = A_val
                        best_e = epsilon_values[ei]

        output[jj, 0] = positions[j]
        output[jj, 1] = best_lambda
        output[jj, 2] = best_m
        # LASSIP C++: maxA = 1.0/exp(A_loop)
        output[jj, 3] = 1.0 / best_A
        output[jj, 4] = best_e

    return output


def Lambda_statistic_fast(
    K_counts,
    K_neutral,
    positions,
    K_truncation,
    n_A=101,
    sweep_mode=4,
    nthreads=1,
    max_extend=1e5,
    _iter=0,
):
    """
    Compute saltiLASSI Λ statistics for all windows, returning a Polars DataFrame.

    Uses geometric mixture formula matching C++ lassip (lassip-winstats.cpp L265-267):
    ``L1 - L0 = Σ_i α_i · dF[mi, ei, i]``
    where ``dF[mi, ei, i] = sweep_likelihood(i, m, ε) − easy_likelihood(i)`` is precomputed once.

    Precomputes dF single-threaded (384 KB, fits in L2 cache), then distributes target
    windows across joblib loky workers. dF serializes cheaply (384 KB); each worker
    JITs _lassip_chunk once then processes all its assigned windows. Threading backend
    does not parallelize because numba @njit(parallel=False) does not release the GIL.

    precompute: O(n_windows × K × n_ε) — all log() calls happen here.
    hot path:   O(n_windows² × n_A × K × n_ε) — multiply-adds only, within max_extend.

    Parameters
    ----------
    K_counts : np.ndarray, shape (n_windows, K_truncation)
    K_neutral : np.ndarray, shape (K_truncation,)
    positions : np.ndarray, shape (n_windows,)
    K_truncation : int
    n_A : int
        Number of log-spaced A values. Default 101 (matches C++ lassip 101-point grid).
    sweep_mode : int
        Sweep haplotype redistribution model (1-5). Default 4 (exponential squared).
    nthreads : int
        Number of joblib threads. Default 1 (single-threaded).
    max_extend : float
        Maximum bp distance from target window to include in composite. Default 1e5 (100 kb),
        matching C++ lassip DEFAULT_MAX_EXTEND_BP. Use np.inf to sum all windows.
    _iter : int
        Replicate index attached as 'iter' column.

    Returns
    -------
    pl.DataFrame with columns: window_lassip, Lambda, m, A, frequency, iter. A column stores 1/actual_A (C++ lassip: maxA = 1.0/exp(A_loop)).
    """
    from . import Parallel, delayed

    n_windows = len(positions)
    d_min = float(np.min(np.diff(np.sort(positions))))
    A_min = -np.log(0.99999) / d_min
    A_max = -np.log(0.00001) / d_min
    A_grid = np.geomspace(A_min, A_max, n_A)

    # Precompute 3D dF table — all log() calls happen here
    dF, null_composite, epsilon_values = _lassip_precompute(
        K_counts, K_neutral, K_truncation, sweep_mode
    )

    # Distribute target windows across loky workers.
    # Each worker JITs _lassip_chunk once then processes all its windows.
    # dF (384 KB) serializes cheaply; loky provides real parallelism unlike threading
    chunk_size = max(1, ceil(n_windows / nthreads))
    chunks = [
        (i, min(i + chunk_size, n_windows)) for i in range(0, n_windows, chunk_size)
    ]
    results = Parallel(n_jobs=nthreads, backend="loky")(
        delayed(_lassip_chunk)(
            j_start,
            j_end,
            positions,
            K_truncation,
            A_grid,
            dF,
            epsilon_values,
            max_extend,
        )
        for j_start, j_end in chunks
    )

    result = np.vstack(results)
    return pl.DataFrame(
        result,
        schema={
            "window_lassip": pl.Int64,
            "Lambda": pl.Float64,
            "m": pl.Float64,
            "A": pl.Float64,
            "frequency": pl.Float64,
        },
    ).with_columns(pl.lit(_iter).cast(pl.Int64).alias("iter"))


def run_lassip(
    hap_data,
    K_truncation=10,
    w_size=201,
    step=10,
    K_neutral=None,
    n_A=100,
    sweep_mode=4,
    nthreads=1,
):
    """
    Run saltiLASSI on a single haplotype dataset (VCF or ms format).

    Extends run_lassi by computing the spatially-aware Λ statistic (DeGiorgio &
    Szpiech 2022) instead of the per-window T statistic. Reuses LASSI_spectrum_and_Kspectrum
    and neut_average unchanged.

    **Algorithm and performance vs C++ lassip**

    saltiLASSI composites LASSI per-window log-likelihoods with a spatial decay kernel:

      L1(m,ε,A,z*) = Σ_i  α_i · sweep_lik(i,m,ε)  +  (1−α_i) · null_lik(i)
      L1 − L0      = Σ_i  α_i · dF[mi,ei,i]           ← dot product, no log()

    where α_i = exp(−A · abs(z_i − z*)) and dF[mi,ei,i] = sweep_lik(i,m,ε) − null_lik(i)
    is precomputed once in O(n_windows × K × n_ε) before the hot path.

    +---------------------------+----------------------------------------+----------------------------------+
    | Aspect                    | C++ lassip                             | This implementation              |
    +---------------------------+----------------------------------------+----------------------------------+
    | Mixture formula           | Geometric (paper uses arithmetic)      | Geometric — identical ✓          |
    | Hot-path inner work       | K log() + K mul-add per source window  | 1 mul-add per source window      |
    | log() calls total         | ~23 B for 480 windows                  | ~48 K (precompute only)          |
    | q / dF working set        | n_win × n_ε × m × K × 8B ≈ 38 MB       | K × n_ε × n_win × 8B ≈ 384 KB    |
    | Cache behaviour           | L3 misses (38 MB)                      | L2 resident (384 KB)             |
    | sweep_mode                | Fixed at compile time (oopt flag)      | Runtime parameter 1–5            |
    | Spatial cutoff            | MAX_EXTEND hard distance limit         | Sums all windows (no cutoff)     |
    | Parallelism               | OpenMP (process forks)                 | joblib loky (separate processes) |
    | Observed 1-thread speed   | ~120 s for 480 windows (39K loci)      | ~2–3 s for 480 windows           |
    +---------------------------+----------------------------------------+----------------------------------+

    The ~40–60× single-thread speedup comes from three compounding effects:
    1. dF precomputation collapses K from the hot path → 10× fewer operations per inner step.
    2. 384 KB dF fits in L2 cache; C++ 38 MB q array thrashes L3.
    3. Pure multiply-add hot path is SIMD-vectorizable; log() calls are not.

    **Complexity**: O(n_windows² × n_A × K × n_ε). For n_windows≈480 on a 1.2 Mb locus,
    this is ~2.3 B multiply-adds ≈ 2–3 s single-thread; ~0.3 s with 8 threads.
    Use a larger step (e.g. step=50) to reduce n_windows and quadratically cut runtime.

    Parameters
    ----------
    hap_data : str or tuple
        Path to VCF/ms file, or (hap_int, position_masked) tuple.
    K_truncation : int
        Number of top haplotypes in truncated spectrum. Default 10.
    w_size : int
        Sliding window size in SNPs for building K-spectra. Default 201.
    step : int
        Step in SNPs between consecutive K-spectrum windows. Default 10.
    K_neutral : np.ndarray or None
        Pre-computed neutral spectrum. Estimated via neut_average if None.
    n_A : int
        Number of log-spaced A grid points (spatial decay rates). Default 100.
    sweep_mode : int
        Sweep haplotype redistribution model passed to sweep_likelihood (1–5). Default 4.
    nthreads : int
        Number of joblib threads for the per-target phase. Default 1.

    Returns
    -------
    pl.DataFrame with columns: window_lassip, Lambda, m, A, frequency, iter
    """
    try:
        (
            hap_int,
            rec_map_01,
            ac,
            biallelic_mask,
            position_masked,
            genetic_position_masked,
        ) = genome_reader(hap_data)
    except Exception:
        (
            hap_int,
            rec_map_01,
            ac,
            biallelic_mask,
            position_masked,
            genetic_position_masked,
        ) = parse_ms_numpy(hap_data)

    K_counts, K_spectrum, windows_centers = LASSI_spectrum_and_Kspectrum(
        [hap_int, position_masked], K_truncation, w_size, int(step)
    )

    if K_neutral is None:
        K_neutral = neut_average(np.vstack(K_spectrum))

    return Lambda_statistic_fast(
        K_counts,
        K_neutral,
        windows_centers,
        K_truncation,
        n_A=n_A,
        sweep_mode=sweep_mode,
        nthreads=nthreads,
    )


################## RAISD


@njit(cache=True)
def compute_mu_var(start_idx, end_idx, snp_positions, D_ln, W_sz):
    """Variation component of the RAiSD mu statistic (paper Eq. 1).

    Measures the reduction of nucleotide diversity in a SNP window.  A hard
    selective sweep purges linked variation, so the physical span covered by
    ``W_sz`` consecutive SNPs shrinks relative to the genome-wide rate.  The
    ratio (window_span / expected_span) rises near the selected site.

    Args:
        start_idx: Index of the first SNP in the window (into snp_positions).
        end_idx:   One-past index of the last SNP in the window.
        snp_positions: Full array of SNP physical positions (bp), shape (S,).
        D_ln:  Physical span of the entire region being scanned (bp).
               Computed once as ``(positions[-1] + 1) - positions[0]``.
        W_sz:  Number of SNPs in the window (== end_idx - start_idx).

    Returns:
        float: mu_var >= 0. Larger values indicate less variation (stronger
        sweep signal). Under neutrality the expectation is ~1.
    """
    l_start = snp_positions[start_idx]
    l_end = snp_positions[end_idx - 1]
    # (window physical span) / (expected span if SNPs were uniformly distributed)
    # multiplied by total SNP count to normalise for region-wide density
    return ((l_end - l_start) / (D_ln * W_sz)) * snp_positions.shape[0]


@njit(cache=True)
def compute_mu_sfs(window, n, theta_W):
    """Site-frequency-spectrum component of the RAiSD mu statistic (paper Eq. 2).

    A selective sweep elevates high-frequency derived alleles (Fay-Wu signal).
    This function approximates that signal without requiring ancestral-state
    polarization by counting "edge" SNPs — those whose derived allele count
    equals 1 (singletons) or n-1 (near-fixation) — and normalising by
    Watterson's theta harmonic sum to account for sample size.

    Args:
        window: Haplotype sub-matrix for the current SNP window, shape
                (W_sz, n), with 0/1 entries (rows = SNPs, cols = haplotypes).
        n:      Number of haplotypes (= window.shape[1]).
        theta_W: Watterson's harmonic sum: sum(1/k, k=1..n-1).
                 Precomputed by ``_harmonic_sums(n)[0]`` in mu_stat to avoid
                 recomputation on every window.

    Returns:
        float: mu_sfs >= 0.  Larger values indicate more edge-frequency SNPs
        (stronger sweep signal). Returns np.nan for empty windows.
    """
    if window.shape[0] == 0:
        return np.nan
    # Count allele-1 copies per SNP (row sums over haplotypes)
    derived_counts = np.sum(window, axis=1)
    # Edge SNPs: singletons (count==1) or near-fixed alleles (count==n-1)
    # These are the tails of the SFS most enriched by a sweep
    edge_mask = (derived_counts == 1) | (derived_counts == n - 1)
    n_edges = np.sum(edge_mask)
    W_sz = window.shape[0]
    # Normalise edge fraction by Watterson's correction for sample size
    return (n_edges / W_sz) * theta_W


@njit(cache=True)
def pack_snp_row(row, n_samples):
    """Pack one SNP's 0/1 haplotype vector into an array of uint64 bit-words.

    Converts a row of haplotype alleles (0 or 1, length n_samples) into a
    compact bit representation stored in ceil(n_samples/64) uint64 words.
    Bits are packed MSB-first within each word (sample 0 → most-significant bit).
    Trailing bits in the last word are zeroed so that ``equal_or_complement``
    and ``hash_pattern`` are not polluted by uninitialized bits.

    This representation enables O(ceil(n/64)) word comparisons instead of
    O(n) element comparisons when testing pattern equality or complement
    equality in ``equal_or_complement``.

    Args:
        row:       1-D array of length n_samples with 0/1 alleles for one SNP.
        n_samples: Number of haplotypes (must equal len(row)).

    Returns:
        np.ndarray: uint64 array of length ceil(n_samples/64) holding the
        packed bit representation.
    """
    words = (n_samples + 63) // 64
    packed = np.zeros(words, dtype=np.uint64)
    word = np.uint64(0)
    lcnt = 0  # bits accumulated in the current word
    w = 0  # index of the current word
    for j in range(n_samples):
        b = np.uint64(row[j] & 1)
        word = (word << np.uint64(1)) | b  # shift left, insert new bit at LSB
        lcnt += 1
        if lcnt == 64:
            packed[w] = word
            word = np.uint64(0)
            lcnt = 0
            w += 1
    if lcnt != 0:
        # Align the partial word to the LSB and zero trailing bits
        shift = np.uint64(64 - lcnt)
        tmp = (word << shift) >> shift
        packed[w] = tmp
    return packed


@njit(cache=True)
def last_word_mask(n_bits):
    """Bitmask for the valid bits in the final packed uint64 word.

    When n_bits is not a multiple of 64, the last uint64 word produced by
    ``pack_snp_row`` has ``r = n_bits % 64`` meaningful bits (at the LSB end)
    and 64-r zero-padding bits.  This function returns the mask with exactly
    those r bits set so that complement operations and hash computations can
    ignore the padding.

    Args:
        n_bits: Total number of bits (== n_samples).

    Returns:
        np.uint64: Mask with the low ``n_bits % 64`` bits set.
        If n_bits is a multiple of 64, all 64 bits are set (full word).
    """
    r = n_bits % 64
    if r == 0:
        # All 64 bits are valid — full all-ones mask
        return np.uint64(~np.uint64(0))
    # Set only the low r bits: (1 << r) - 1
    return (np.uint64(1) << np.uint64(r)) - np.uint64(1)


@njit(cache=True)
def equal_or_complement(a, b, n_samples):
    """Return True if packed patterns a and b are equal or bitwise complements.

    RAiSD treats a SNP pattern as equivalent to its complement because the
    assignment of 0/1 to REF/ALT is arbitrary (polarization ambiguity).  Two
    SNPs that differ only in which allele is labelled ancestral carry the same
    information for the LD component of mu.

    The test is done in two passes over the packed uint64 words:
      Pass 1 — exact equality: a[i] == b[i] for all words.
      Pass 2 — complement equality: a[i] == ~b[i] for full words, and
               (a[lw] ^ ~b[lw]) & last_word_mask == 0 for the partial last word
               (so padding bits don't cause a false mismatch).

    Args:
        a, b:      Packed uint64 arrays from ``pack_snp_row``, same length.
        n_samples: Number of haplotypes (needed to mask the last word).

    Returns:
        bool: True if the two SNP patterns are identical or complementary.
    """
    words = a.shape[0]
    # Pass 1: exact equality (fast path — most pairs differ immediately)
    eq = True
    for i in range(words):
        if a[i] != b[i]:
            eq = False
            break
    if eq:
        return True

    # Pass 2: complement equality — check ~b[i] == a[i] word by word
    full_words = n_samples // 64
    last_mask = last_word_mask(n_samples)
    for i in range(full_words):
        if a[i] != (~b[i]):
            return False
    if (n_samples % 64) != 0:
        # For the partial last word, only compare the valid bits via the mask
        lw = full_words
        if (a[lw] ^ (~b[lw])) & last_mask != 0:
            return False
    return True


@njit(cache=True)
def hash_pattern(packed, n_samples):
    """Compute a complement-symmetric hash key for a packed SNP bit-pattern.

    Returns the same hash value for a pattern and its bitwise complement, so
    that ``get_pattern_ids`` maps polarization-equivalent patterns to the same
    dictionary bucket without needing to normalise patterns upfront.

    Two hashes are computed using the Fibonacci multiplicative hash:
      h1 = hash of the original packed words
      h2 = hash of the bitwise complement (with the last word masked to zero
           padding bits before flipping, so padding never affects h2)

    The canonical key is ``min(h1, h2)``.  Regardless of which form of a
    pattern is encountered first (original or complement), ``min`` always
    selects the same representative → same bucket → correctly identified as the
    same pattern when ``equal_or_complement`` confirms.

    Fibonacci hashing constant: 0x9E3779B97F4A7C15 = floor(2^64 / phi),
    where phi = (1+sqrt(5))/2 (golden ratio).  Multiplying an integer by this
    constant and discarding overflow distributes hash values nearly uniformly
    across the uint64 range (Knuth, TAOCP Vol. 3 §6.4).

    Args:
        packed:    uint64 array from ``pack_snp_row``.
        n_samples: Number of haplotypes (needed to mask padding bits).

    Returns:
        np.uint64: Canonical symmetric hash key.
    """
    # h1: hash of the original pattern
    h1 = np.uint64(0)
    for w in packed:
        # Fibonacci multiplicative hash: XOR-fold each word
        h1 ^= w * np.uint64(11400714819323198485)

    # h2: hash of the bitwise complement (polarization-flipped pattern)
    mask = last_word_mask(n_samples)
    comp = packed.copy()
    for i in range(packed.shape[0] - 1):
        comp[i] = ~packed[i]  # flip all 64 bits of full words
    comp[-1] = (~packed[-1]) & mask  # flip only valid bits of the last word
    h2 = np.uint64(0)
    for w in comp:
        h2 ^= w * np.uint64(11400714819323198485)

    # Symmetric key: min(h1, h2) is the same whichever form is seen first
    return h1 if h1 < h2 else h2


@njit(cache=True)
def get_pattern_ids(hap):
    """Assign a canonical integer ID to each SNP's haplotype pattern.

    Two SNPs receive the same ID if their haplotype patterns are identical or
    bitwise complements (polarization-equivalent), as tested by
    ``equal_or_complement``.  IDs are dense integers starting at 0.

    The deduplication table uses two parallel arrays:
      ``uniq_packed[u]`` — packed bit-pattern of the u-th unique pattern
      ``hashes[u]``      — its symmetric hash (from ``hash_pattern``)

    For each incoming SNP:
      1. Compute the symmetric hash (cheap, O(words)).
      2. Linear-scan ``hashes`` for a hash collision (fast rejection for
         non-matching patterns without an expensive bit comparison).
      3. On a hash match, call ``equal_or_complement`` to confirm.
      4. If confirmed, reuse the existing ID; otherwise register a new pattern.

    Note: Python dicts are not available in Numba nopython mode, so
    deduplication is implemented as a manual linear scan.  In practice the
    number of distinct patterns per window is small (bounded by window_size),
    so this is efficient despite the O(uniq_count) scan.

    Args:
        hap: Haplotype matrix (S, n) with 0/1 entries (rows=SNPs, cols=haps).

    Returns:
        np.ndarray: int32 array of length S.  ``ids[i]`` is the canonical
        pattern ID for SNP i; equal IDs mean the patterns are the same or
        complementary.
    """
    snps, n_samples = hap.shape
    words_per_snp = (n_samples + 63) // 64
    ids = np.empty(snps, dtype=np.int32)

    # Deduplication table: stores one packed pattern per unique ID
    uniq_packed = np.zeros((snps, words_per_snp), dtype=np.uint64)
    hashes = np.zeros(snps, dtype=np.uint64)
    uniq_count = 0

    for i in range(snps):
        packed = pack_snp_row(hap[i], n_samples)
        h = hash_pattern(packed, n_samples)
        found = -1
        # Linear scan: check hash first (fast), then full bit comparison
        for u in range(uniq_count):
            if hashes[u] == h:
                if equal_or_complement(uniq_packed[u], packed, n_samples):
                    found = u
                    break
        if found == -1:
            # New unique pattern: register it
            uniq_packed[uniq_count, :] = packed
            hashes[uniq_count] = h
            ids[i] = uniq_count
            uniq_count += 1
        else:
            ids[i] = found

    return ids


@njit(cache=True)
def compute_mu_ld(start_idx: int, end_idx: int, pattern_ids: np.ndarray) -> float32:
    """LD contrast component of the RAiSD mu statistic (paper Eq. 3).

    A selective sweep creates long haplotype blocks on both sides of the
    selected site.  The two flanking regions each have their own coherent LD
    structure (high internal LD, few distinct patterns) but different patterns
    from each other (high inter-flank exclusivity).

    This function splits the SNP window [start_idx, end_idx) in half and
    counts:
      pcntl      — unique patterns in the left half [p0, p1]
      pcntr      — unique patterns in the right half [p2, p3]
      excl_left  — unique patterns found ONLY in the left half
      excl_right — unique patterns found ONLY in the right half
      excntsnpsl — individual left-half SNPs whose pattern is exclusive to left
      excntsnpsr — individual right-half SNPs whose pattern is exclusive to right

    The statistic is:
      mu_ld = (excl_left * excntsnpsl + excl_right * excntsnpsr) / (pcntl * pcntr)

    Near a sweep center both numerator products are large (each flank is
    internally coherent and mutually exclusive), while the denominator stays
    small (few distinct patterns per half), so mu_ld peaks at the sweep.

    Window pointers (using ``mid = length // 2``):
      p0 = start_idx              (first SNP of left half)
      p1 = start_idx + mid - 1   (last  SNP of left half)
      p2 = start_idx + mid        (first SNP of right half)
      p3 = end_idx - 1            (last  SNP of right half)

    Args:
        start_idx:   First SNP index (into the chromosome-level ``pattern_ids``).
        end_idx:     One-past index of the last SNP in the window.
        pattern_ids: int32 array of pattern IDs for all SNPs on the chromosome,
                     produced by ``get_pattern_ids``.

    Returns:
        float32: mu_ld >= 0.  Returns 1e-10 (not 0) when numerator or
        denominator is zero, so that mu_total = mu_var * mu_sfs * mu_ld is
        never exactly zero (avoids log(0) in downstream scoring).
    """
    if end_idx <= start_idx:
        return float32(0.0)

    length = end_idx - start_idx
    if length == 0:
        return float32(0.0)

    # Split window into left [p0..p1] and right [p2..p3] halves
    mid = length // 2
    p0 = int32(start_idx)
    p1 = int32(start_idx + mid - 1)
    p2 = int32(start_idx + mid)
    p3 = int32(end_idx - 1)

    # Build unique-pattern list for the left half
    # list_left holds distinct pattern IDs; list_left_cnt tracks their counts
    list_left = np.empty(length, dtype=int32)
    list_left_cnt = np.zeros(length, dtype=int32)
    list_left_size = int32(0)

    list_left[0] = int32(pattern_ids[p0])
    list_left_cnt[0] = 1
    list_left_size += 1

    for i in range(p0 + 1, p1 + 1):
        pid = int32(pattern_ids[i])
        match = 0
        for j in range(list_left_size):
            if list_left[j] == pid:
                list_left_cnt[j] += 1
                match = 1
                break
        if match == 0:
            list_left[list_left_size] = pid
            list_left_cnt[list_left_size] = 1
            list_left_size += 1

    pcntl = int32(list_left_size)  # total unique patterns in left half

    # Build unique-pattern list for the right half
    list_right = np.empty(length, dtype=int32)
    list_right_cnt = np.zeros(length, dtype=int32)
    list_right_size = int32(0)

    list_right[0] = int32(pattern_ids[p2])
    list_right_cnt[0] = 1
    list_right_size += 1

    for i in range(p2 + 1, p3 + 1):
        pid = int32(pattern_ids[i])
        match = 0
        for j in range(list_right_size):
            if list_right[j] == pid:
                list_right_cnt[j] += 1
                match = 1
                break
        if match == 0:
            list_right[list_right_size] = pid
            list_right_cnt[list_right_size] = 1
            list_right_size += 1

    pcntr = int32(list_right_size)  # total unique patterns in right half

    # Exclusive unique patterns (pattern-level exclusivity)
    # excl_left: unique patterns in left that do NOT appear in right
    excl_left = int32(list_left_size)
    for i in range(list_left_size):
        for j in range(list_right_size):
            if list_left[i] == list_right[j]:
                excl_left -= 1
                break

    # excl_right: unique patterns in right that do NOT appear in left
    excl_right = int32(list_right_size)
    for i in range(list_right_size):
        for j in range(list_left_size):
            if list_right[i] == list_left[j]:
                excl_right -= 1
                break

    # SNP-level exclusivity
    # excntsnpsl: number of left-half SNPs whose pattern is not in the right half
    excntsnpsl = int32(0)
    for i in range(p0, p1 + 1):
        pid_i = int32(pattern_ids[i])
        match = 0
        for j in range(p2, p3 + 1):
            if pid_i == int32(pattern_ids[j]):
                match = 1
                break
        if match == 0:
            excntsnpsl += 1

    # excntsnpsr: number of right-half SNPs whose pattern is not in the left half
    excntsnpsr = int32(0)
    for i in range(p2, p3 + 1):
        pid_i = int32(pattern_ids[i])
        match = 0
        for j in range(p0, p1 + 1):
            if pid_i == int32(pattern_ids[j]):
                match = 1
                break
        if match == 0:
            excntsnpsr += 1

    # Cross-products: exclusive unique patterns × exclusive SNP count per half
    pcntexll = int32(excl_left * excntsnpsl)
    pcntexlr = int32(excl_right * excntsnpsr)

    denom = int32(pcntl * pcntr)
    if (pcntexll + pcntexlr) == 0 or denom == 0:
        # Return a small non-zero floor so mu_total is never exactly zero
        return float32(1e-10)
    return float32((pcntexll + pcntexlr) / float32(denom))


def mu_stat(hap, snp_positions, window_size=50):
    """
    Compute RAiSD composite sweep score :math:`\\mu` over overlapping SNP windows.

    For each sliding window of ``window_size`` consecutive SNPs (step = 1 SNP),
    this routine evaluates three components and their product:

    * **mu_var** – reduction-of-variation component (computed by
      :func:`compute_mu_var`), scaled by the region length.
    * **mu_sfs** – site-frequency-spectrum skew component (from
      :func:`compute_mu_sfs`), standardized using Watterson’s harmonic correction
      (``_harmonic_sums(n)[0]``).
    * **mu_ld** – linkage-disequilibrium contrast component (from
      :func:`compute_mu_ld`) using the supplied :math:`r^2` matrix.
    * **mu_total** – composite statistic ``mu_var * mu_sfs * mu_ld``.

    The window center coordinate is recorded as the midpoint between the first
    and last SNP positions in the window. Results are returned as a Polars
    DataFrame with one row per window.

    :param numpy.ndarray hap: Haplotype matrix of shape ``(S, n)`` with 0/1 alleles
        (rows = SNPs, columns = haplotypes or chromosomes).
    :param numpy.ndarray snp_positions: Monotonically increasing physical positions
        of length ``S`` (aligned to rows of ``hap``).
    :param int window_size: Number of consecutive SNPs per sliding window;
        defaults to ``50`` (RAiSD’s ``-w`` default).

    :returns: A Polars DataFrame with columns
        * ``positions`` (int): window center (bp, midpoint of first/last SNP in window)
        * ``mu_var`` (float): variation component
        * ``mu_sfs`` (float): SFS component
        * ``mu_ld`` (float): LD component
        * ``mu_total`` (float): composite score ``mu_var * mu_sfs * mu_ld``
    :rtype: polars.DataFrame

    :notes:
        * ``D_ln = (snp_positions[-1] + 1) - snp_positions[0]`` is the physical
          span of the full input region (bp), used to scale mu_var.
        * ``theta_w_correction = _harmonic_sums(n)[0]`` = Watterson’s harmonic
          sum (1 + 1/2 + ... + 1/(n-1)), precomputed once for all windows.
        * ``get_pattern_ids`` is called once on the full haplotype matrix before
          the window loop; pattern IDs are indexed per-window by start/end.
        * Windows advance by one SNP (maximally overlapping). For S SNPs and
          window size W, the output has S - W + 1 rows.

    :see also:
        :func:`compute_mu_var`, :func:`compute_mu_sfs`, :func:`compute_mu_ld`,
        :func:`get_pattern_ids`
    """
    # Physical span of the full region; used to normalise mu_var across regions
    D_ln = (snp_positions[-1] + 1) - snp_positions[0]
    S, n = hap.shape

    # Watterson’s harmonic correction: sum(1/k, k=1..n-1); normalises mu_sfs for sample size
    theta_w_correction = _harmonic_sums(n)[0]
    # Match RAiSD -w option (default: 50)
    _window_size = window_size

    _iter_windows = list(range(S - _window_size + 1))
    mu_var_np = np.zeros(len(_iter_windows))
    mu_sfs_np = np.zeros(len(_iter_windows))
    mu_ld_np = np.zeros(len(_iter_windows))
    mu_total_np = np.zeros(len(_iter_windows))
    center_np = np.zeros(len(_iter_windows))

    # Compute pattern IDs once for the full chromosome; windows index into this array
    pattern_ids = get_pattern_ids(hap)
    for i in _iter_windows:
        start_idx = i
        end_idx = i + _window_size

        # Center = midpoint between first and last SNP positions in this window
        center_pos = (snp_positions[start_idx] + snp_positions[end_idx - 1]) / 2
        window = hap[start_idx:end_idx, :]

        if end_idx <= start_idx or end_idx > hap.shape[0]:
            mu_var_np[i] = np.nan
            mu_sfs_np[i] = np.nan
            mu_ld_np[i] = np.nan
            mu_total_np[i] = np.nan
            continue

        window = hap[start_idx:end_idx]
        mu_var = compute_mu_var(
            start_idx, end_idx, snp_positions, D_ln, end_idx - start_idx
        )
        mu_sfs = compute_mu_sfs(window, n, theta_w_correction)
        mu_ld = compute_mu_ld(start_idx, end_idx, pattern_ids)
        mu_total = mu_var * mu_sfs * mu_ld

        mu_var_np[i] = mu_var
        mu_sfs_np[i] = mu_sfs
        mu_ld_np[i] = mu_ld
        mu_total_np[i] = mu_total
        center_np[i] = center_pos

    df_mu = pl.DataFrame(
        {
            "positions": center_np.astype(int),
            "mu_var": mu_var_np,
            "mu_sfs": mu_sfs_np,
            "mu_ld": mu_ld_np,
            "mu_total": mu_total_np,
        }
    )

    # return mu_var_np,mu_sfs_np,mu_ld_np,mu_total_np
    return df_mu


def run_raisd(hap_data, window_size=50):
    """Public entry point for the RAiSD mu statistic.

    Accepts either a VCF/VCF.gz file path or an ms/discoal simulation text
    file and returns the per-SNP-window mu scores computed by ``mu_stat``.

    Input dispatch:
      1. Tries ``genome_reader`` (allel-based VCF reader).  This is the primary
         path for real data.
      2. On any exception, falls back to ``parse_ms_numpy`` (ms/discoal text
         format).  This allows the same function to be used for both empirical
         and simulated data without separate entry points.

    Args:
        hap_data:    Path to a VCF/VCF.gz file or an ms/discoal output file.
        window_size: Number of consecutive SNPs per sliding window (default 50, matching RAiSD's ``-w`` default).

    Returns:
        polars.DataFrame: Same schema as ``mu_stat`` — columns
        ``[positions, mu_var, mu_sfs, mu_ld, mu_total]``, one row per window.
    """
    try:
        (
            hap_int,
            rec_map_01,
            ac,
            biallelic_mask,
            position_masked,
            genetic_position_masked,
        ) = genome_reader(hap_data)
        # freqs = ac[:, 1] / ac.sum(axis=1)
    except Exception:
        (
            hap_int,
            rec_map_01,
            ac,
            biallelic_mask,
            position_masked,
            genetic_position_masked,
        ) = parse_ms_numpy(hap_data)
        # freqs = ac[:, 1] / ac.sum(axis=1)

    df_mu = mu_stat(hap_int, position_masked, window_size=window_size)

    return df_mu


################## Balancing stats


@njit(cache=True)
def calc_d(freq, core_freq, p):
    """Calculates the value of d, the similarity measure

    Parameters:
        freq: freq of SNP under consideration, ranges from 0 to 1
        core_freq: freq of coresite, ranges from 0 to 1
        p: the p parameter specifying sharpness of peak
    """
    xf = min(core_freq, 1.0 - core_freq)
    f = np.minimum(freq, 1.0 - freq)
    maxdiff = np.maximum(xf, 0.5 - xf)
    corr = ((maxdiff - np.abs(xf - f)) / maxdiff) ** p
    return corr


@njit(cache=True)
def omegai_nb(freqs, core_freq, n, p):
    """Calculates 9a

    Parameters:
        i:freq of SNP under consideration, ranges between 0 and 1
        snp_n: number of chromosomes used to calculate frequency of core SNP
        x: freq of coresite, ranges from 0 to 1
        p: the p parameter specifying sharpness of peak
    """
    n1num = calc_d(freqs, core_freq, p)
    n1denom = np.sum(calc_d(np.arange(1.0, n) / n, core_freq, p))
    n1 = n1num / n1denom
    n2 = (1.0 / (freqs * n)) / (np.sum(1.0 / np.arange(1.0, n)))
    return n1 - n2


@njit(cache=True)
def an_nb(n, core_freq, p):
    """
    Calculates alpha_n from Achaz 2009, eq 9b

        n: Sample size
        x: frequency, ranges from 0 to 1
        p: value of p parameter
    """
    i = np.arange(1, n)
    return np.sum(i * omegai_nb(i / n, core_freq, n, p) ** 2.0)


@njit(cache=True)
def fu_an_vec(n):
    """Calculates a_n from Fu 1995, eq 4 for a single integer value"""
    if n <= 1:
        return 0.0
    return np.sum(1.0 / np.arange(1.0, n))


@njit(cache=True)
def fu_Bn(n, i):
    """Calculates Beta_n(i) from Fu 1995, eq 5"""

    r = 2.0 * n / ((n - i + 1.0) * (n - i)) * (fu_an_vec(n + 1) - fu_an_vec(i)) - (
        2.0 / (n - i)
    )
    return r


@njit(cache=True)
def sigma(n, ij):
    res = np.zeros(ij.shape[0])

    for k in range(ij.shape[0]):
        i = max(ij[k, 0], ij[k, 1])
        j = min(ij[k, 0], ij[k, 1])

        if i == j and 2 * i == n:
            res[k] = 2.0 * ((fu_an_vec(n) - fu_an_vec(i)) / (n - i)) - (1.0 / (i * i))

        elif i == j and i < n / 2.0:  # FIXED: removed 2*
            res[k] = fu_Bn(n, i + 1)

        elif i == j and i > n / 2.0:
            res[k] = fu_Bn(n, i) - (1.0 / (i * i))

        elif i > j and (i + j == n):
            an_n = fu_an_vec(n)
            an_i = fu_an_vec(i)
            an_j = fu_an_vec(j)
            term1 = (an_n - an_i) / (n - i)
            term2 = (an_n - an_j) / (n - j)
            term3 = (fu_Bn(n, i) + fu_Bn(n, j + 1)) / 2.0
            term4 = 1.0 / (i * j)
            res[k] = (term1 + term2) - (term3 + term4)

        elif i > j and (i + j < n):
            res[k] = (fu_Bn(n, i + 1) - fu_Bn(n, i)) / 2.0

        elif i > j and (i + j > n):
            res[k] = (fu_Bn(n, j) - fu_Bn(n, j + 1)) / 2.0 - (1.0 / (i * j))

    return res


@njit(cache=True)
def Bn_nb(n, core_freq, p):
    """
    Returns Beta_N from Achaz 2009, eq 9c

    Parameters:
        n: Sample size
        x: frequency, ranges from 0 to 1
        p: value of p parameter
    """

    i = np.arange(1, n)
    n1 = np.sum(
        i**2.0
        * omegai_nb(i / n, core_freq, n, p) ** 2.0
        * sigma(n, np.column_stack((i, i)))
    )

    # coords = np.asarray([(j, i) for i in range(1, n) for j in range(1, i)])
    m = (n - 1) * (n - 2) // 2
    coords = np.empty((m, 2), dtype=np.int64)
    idx = 0
    for i in range(1, n):
        for j in range(1, i):
            coords[idx, 0] = j
            coords[idx, 1] = i
            idx += 1

    s2 = np.sum(
        coords[:, 0]
        * coords[:, 1]
        * omegai_nb(coords[:, 0] / n, core_freq, n, p)
        * omegai_nb(coords[:, 1] / n, core_freq, n, p)
        * sigma(n, coords)
    )

    n2 = 2.0 * s2
    return n1 + n2


def calc_thetaw_unfolded(snp_freq_list, num_ind):
    """Calculates watterson's theta

    Parameters:
        snp_freq_list: a list of frequencies, one for each SNP in the window,
            first column ranges from 1 to number of individuals, second columns is # individuals
        num_ind: number of individuals used to calculate the core site frequency
    """
    if snp_freq_list.size == 0:
        return 0

    a1 = np.sum(1.0 / np.arange(1, num_ind))

    thetaW = len(snp_freq_list[:, 0]) / a1
    return thetaW


def calc_t_unfolded(freqs, core_freq, n, p, theta, var_dic):
    """
    Using equation 8 from Achaz 2009

    Parameters:
        core_freq: freq of SNP under consideration, ranges from 1 to sample size
        snp_n: sample size of core SNP
        p: the p parameter specifying sharpness of peak
        theta: genome-wide estimate of the mutation rate
    """

    # x = float(core_freq)/snp_n

    num = np.sum(freqs * n * omegai_nb(freqs, core_freq, n, p))
    # if not (n, core_freq, theta) in var_dic:
    if (n, core_freq, theta) not in var_dic:
        denom = np.sqrt(
            an_nb(n, core_freq, p) * theta + Bn_nb(n, core_freq, p) * theta**2.0
        )
        var_dic[(n, core_freq, theta)] = denom
    else:
        denom = var_dic[(n, core_freq, theta)]
    return num / denom


@njit(cache=True)
def calc_t_unfolded_cached(freqs, denom, core_freq, n, p, theta):
    num = np.sum(freqs * n * omegai_nb(freqs, core_freq, n, p))
    return num, num / denom


@njit(cache=True)
def precompute_denoms(n, p, theta, omega_func):
    denom_array = np.zeros(n + 1)

    # Precompute shared structures
    i_vals = np.arange(1, n)
    diag_sigma = sigma(n, np.column_stack((i_vals, i_vals)))

    m = (n - 1) * (n - 2) // 2
    coords = np.empty((m, 2), dtype=np.int64)
    idx = 0
    for i in range(1, n):
        for j in range(1, i):
            coords[idx, 0] = j
            coords[idx, 1] = i
            idx += 1
    coords_i = coords[:, 0]
    coords_j = coords[:, 1]
    off_diag_sigma = sigma(n, coords)

    for cf in range(1, n + 1):
        x = cf / n
        omega = omega_func(i_vals / n, x, n, p)
        an = np.sum(i_vals * omega**2)

        omega_i = omega_func(coords_i / n, x, n, p)
        omega_j = omega_func(coords_j / n, x, n, p)
        s2 = np.sum(coords_i * coords_j * omega_i * omega_j * off_diag_sigma)

        b_n = np.sum(i_vals**2 * omega**2 * diag_sigma) + 2.0 * s2
        denom_array[cf] = np.sqrt(an * theta + b_n * theta**2)

    return denom_array, diag_sigma, off_diag_sigma


@njit(cache=True)
def find_win_indx(prev_start_i, prev_end_i, pos, snp_info, win_size):
    """Takes in the previous indices of the start_ing and end of the window,
    then returns the appropriate start_ing and ending index for the next SNP

    Parameters:
        prev_start_i: start_ing index in the array of SNP for the previous core SNP's window, inclusive
        prev_end_i: ending index in the array for the previous SNP's window, inclusive
        snp_i, the index in the array for the current SNP under consideration
        snp_info: the numpy array of all SNP locations & frequencies
    """

    win_start = pos - win_size / 2

    # array index of start of window, inclusive
    firstI = prev_start_i + np.searchsorted(
        snp_info[prev_start_i:, 0], win_start, side="left"
    )
    winEnd = pos + win_size / 2

    # array index of end of window, exclusive
    endI = (
        prev_end_i - 1 + np.searchsorted(snp_info[prev_end_i:, 0], winEnd, side="right")
    )
    return (firstI, endI)


def run_beta_window(ac, position_masked, p=2, m=0.1, w=None, theta=None):
    _n = ac.sum(axis=1)
    snp_info = np.column_stack([position_masked, ac[:, 1] / _n, ac, _n])

    # S = int(snp_info.shape[0])
    n = int(snp_info[0, -1])

    mask = (snp_info[:, 4] == n) & (snp_info[:, 1] < (1 - m)) & (snp_info[:, 1] > m)
    snp_info_masked = snp_info[mask]

    output = np.zeros((snp_info_masked.shape[0], 3))

    if w is None:
        theta = theta_watterson(snp_info[:, 2:4], snp_info[:, 0])[0]

        denom_array, sigma_term1, sigma_term2 = precompute_denoms(
            n, p, theta, omegai_nb
        )

        for j, snp_i in enumerate(snp_info_masked):
            snp_set = np.concatenate((snp_info[:j], snp_info[j + 1 :]))
            core_freq = snp_i[1]
            denom = denom_array[int(round(core_freq * n))]
            # distances = np.abs(snp_set[:, 0] - snp_i[0])

            B, T = calc_t_unfolded_cached(snp_set[:, 1], denom, core_freq, n, p, theta)

            output[j] = np.array([snp_i[0], B, T])

    else:
        if theta is None:
            theta = theta_watterson(snp_info[:, 2:4], snp_info[:, 0])[1] * w

        denom_array, sigma_term1, sigma_term2 = precompute_denoms(
            n, p, theta, omegai_nb
        )

        prev_start_i = 0
        prev_end_i = 0
        _idx = 0

        for j, snp_i in enumerate(snp_info):
            core_freq = snp_i[1]
            if not mask[j]:
                continue

            # print(prev_start_i, prev_end_i)
            sI, endI = find_win_indx(prev_start_i, prev_end_i, snp_i[0], snp_info, w)
            prev_start_i = sI
            prev_end_i = endI

            if endI == sI:
                # B, T, B_decay, T_decay = 0, 0, 0, 0
                B, T, _, _ = 0, 0, 0, 0
            elif endI > sI:
                snp_set = np.concatenate(
                    (snp_info[sI:j], snp_info[(j + 1) : (endI + 1)])
                )
                denom = denom_array[int(core_freq * n)]
                B, T = calc_t_unfolded_cached(
                    snp_set[:, 1], denom, core_freq, n, p, theta * w
                )
            output[_idx] = np.array([snp_i[0], B, T])
            _idx += 1

    schema = {"positions": pl.Int64, "beta": pl.Float64, "t": pl.Float64}

    return pl.DataFrame(output, schema=schema)


@njit(parallel=False, cache=True)
def ncd1(position_masked, freqs, tf=0.5, w=3000, minIS=2):
    n = len(position_masked)
    maf = np.minimum(freqs, 1 - freqs)
    w1 = w / 2.0
    start_positions = np.arange(position_masked[0], position_masked[-1], w1)
    n_windows = len(start_positions)

    # Preallocate outputs
    results = np.empty(n_windows, dtype=np.float64)
    valid_mask = np.zeros(n_windows, dtype=np.bool_)

    j_start = 0
    j_end = 0

    for widx in range(n_windows):
        start = start_positions[widx]
        end = start + w

        # Advance start pointer
        while j_start < n and position_masked[j_start] < start:
            j_start += 1

        # Advance end pointer
        while j_end < n and position_masked[j_end] <= end:
            j_end += 1

        # Now [j_start:j_end) are the indices within window
        count = j_end - j_start
        if count < minIS:
            continue

        # Compute temp2 = sum((maf - tf)^2)
        tmp = 0.0
        for k in range(j_start, j_end):
            diff = maf[k] - tf
            tmp += diff * diff

        results[widx] = np.sqrt(tmp / count)
        valid_mask[widx] = True

    return results[valid_mask]
