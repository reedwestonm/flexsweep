import glob
import os
import re
from collections import OrderedDict
from itertools import chain

# from subprocess import run
# from typing import Any, cast
from warnings import filterwarnings

from allel import GenotypeArray, index_windows, read_vcf, read_vcf_headers
from scipy.interpolate import interp1d

from . import Parallel, delayed, np, pl

filterwarnings("ignore", message="invalid INFO header", module="allel.io.vcf_read")


def _chrom_id(s):
    s = s.lower()

    # 1. Unify mitochondrial naming
    s = re.sub(r"chr?m(t)?", "mt", s)

    # 2. Extract token
    # We use (?:^|[^a-z0-9]) at the start to ensure we aren't mid-word.
    # Note: We now use m.group(1) for the chromosome because group 0
    # will include the leading separator.
    pattern = (
        r"(?:^|[^a-z0-9])(?:chr|contig|super)?[^a-z0-9]*([0-9]+|x|y|mt)(?=[^a-z0-9]|$)"
    )

    m = re.search(pattern, s)
    if not m:
        return None

    # The capturing group ([0-9]+|x|y|mt) is now the first group
    chrom = m.group(1)

    if chrom.isdigit():
        chrom = str(int(chrom))

    return chrom


def get_contig_from_vcf_filename(path):
    """
    Returns:
        (contig_name, contig_length)
    """

    # Read headers
    headers = read_vcf_headers(path)

    # Parse contigs
    # Changed regex pattern because bat vcfs have assembly name after the contig length
    contig_pattern = re.compile(
    r"##contig=<ID=([^,]+),length=(\d+)(?:,|>)"
)


    contigs = {
        m.group(1): int(m.group(2))
        for line in headers.headers
        if (m := contig_pattern.match(line))
    }

    # Extract chrom from filename
    fname = os.path.basename(path)
    file_chr = _chrom_id(fname)

    if file_chr is None:
        return None, None

    # Match against header contigs
    matches = []

    for contig_name, length in contigs.items():
        contig_chr = _chrom_id(contig_name)

        if contig_chr == file_chr:
            matches.append((contig_name, length))

    # Resolve
    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        raise ValueError(f"Ambiguous contig match: {matches}")

    return None, None


class Data:
    def __init__(
        self,
        data,
        region=None,
        samples=None,
        recombination_map=None,
        window_size=int(1.2e6),
        step=int(1e4),
        nthreads=1,
        sequence_length=int(1.2e6),
        in_memory=False,
    ):
        """
        Utilities for loading VCF/ms-style simulations and producing windowed inputs.

        Args:
            data (str | Any):
                Path to input data:
                  - Path containing VCF/BCF files
                  - Path contiaining discoal simulations
            region (str, default=None):
                Optional genomic region string 'CHR:START-END' to read from VCF/BCF.
            samples (list[str] | np.ndarray | None, default=None):
                Optional subset of samples (names or indices) to read from VCF/BCF.
            recombination_map (str | None, default=None):
                Optional path to recombination map CSV with columns:
                [Chr, Begin, End, cMperMb, cM]. If None, assumes physical distance ~ genetic distance.
            window_size (int, default=1_200_000):
                Genomic window size in base pairs when scanning a VCF.
            step (int, default=10_000):
                Sliding step in base pairs between adjacent windows when scanning a VCF.
            nthreads (int, default=1):
                Number of parallel workers for joblib.
            sequence_length (int, default=1_200_000):
                Sequence length used to scale ms/discoal 'positions' (0..1) into bp.

        Notes:
            - The class handle both functions to read VCF files and discoal simulations.
            - Randomness is not used here; outputs are deterministic given inputs.
        """
        self.data = data
        self.region = region
        self.samples = samples
        self.recombination_map = recombination_map
        self.window_size = window_size
        self.step = step
        self.nthreads = nthreads
        self.sequence_length = sequence_length
        self.in_memory = in_memory

    def genome_reader(self, region, samples):
        """
        Read a genomic region from a VCF/BCF and return haplotypes and rec_map given the region  interval.

        Args:
            region (str): Region string 'CHR:START-END'.
            samples (list[str] | np.ndarray): Sample subset to read.

        Returns:
            dict[str, tuple[np.ndarray, np.ndarray] | None]:
                {region: (hap, rec_map_subset)} or {region: None} if no biallelic sites.
                - hap: (S x N) int8 array of phased haplotypes (derived-allele indicators).
                - rec_map_subset: (S x 4) array [chrom, idx, genetic_pos(cm or proxy), physical_pos].

        Notes:
            - Filters to biallelic variants via scikit-allel's AlleleCounts.is_biallelic_01.
            - If no recombination map is supplied, genetic distance is set to physical pos and
              later remapped to a relative [1..sequence_length] coordinate system.
        """
        filterwarnings(
            "ignore", message="invalid INFO header", module="allel.io.vcf_read"
        )

        raw_data = read_vcf(self.data, region=region, samples=samples)

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
        ac = gt.count_alleles()

        biallelic_filter = ac.is_biallelic_01()

        hap = gt.to_haplotypes()
        hap = hap.values[biallelic_filter]
        pos = pos[biallelic_filter]
        np_chrom = np_chrom[biallelic_filter]

        if hap.shape[0] == 0:
            return {region: None}

        tmp = list(map(int, region.split(":")[-1].split("-")))

        d_pos = dict(
            zip(np.arange(tmp[0], tmp[1] + 1), np.arange(self.sequence_length) + 1)
        )

        if self.recombination_map is None:
            rec_map = pl.DataFrame(
                {"chrom": np_chrom, "idx": np.arange(pos.size), "pos": pos, "cm": pos}
            ).to_numpy()

            for r in rec_map:
                r[-1] = d_pos[r[-1]]
                r[-2] = d_pos[r[-2]]
        else:
            df_recombination_map = pl.read_csv(self.recombination_map, separator=",")
            genetic_distance = self.get_cm(df_recombination_map, pos)
            rec_map = pl.DataFrame(
                [np_chrom, np.arange(pos.size), pos, genetic_distance]
            ).to_numpy()

            for r in rec_map:
                r[-2] = d_pos[r[-2]]

            if np.all(rec_map[:, -1] == 0):
                rec_map[:, -1] = rec_map[:, -2]

            # # physical position to relative physical position (1,1.2e6)
            # # this way we do not perform any change on summary_statistics center/windows combinations
            # f = interp1d(w, [1, int(self.window_size) + 1])
            # rec_map = np.column_stack([rec_map, f(rec_map[:, 2]).astype(int)])

        # return hap
        return {region: (hap, rec_map[:, [0, 1, -1, 2]])}

    def read_vcf(self):
        """
        Slide windows across a single-contig VCF/BCF and collect per-window data haplotypes

        Returns:
            dict | OrderedDict:
                If in_memory=True:
                    {'sweep': list[[hap, rec_map, zeros(4)]], 'region': list[str]}
                    where each element corresponds to a window with data.
                If in_memory=False:
                    OrderedDict({'sweep': <vcf_path>, 'region': list[str]})
                    with region strings to be used downstream for lazy reading.

        Raises:
            IOError: If the file extension is not one of [.vcf, .vcf.gz, .bcf, .bcf.gz].
            FileNotFoundError: If the VCF/BCF is not tabix-indexed (<path>.tbi is missing).

        Notes:
            - Determines contig name/length by reading last line of the vcf file.
            - Generates overlapping windows of size `window_size` and step `step`.
            - Uses multiprocessing when in_memory=True to read per-window haplotypes.
        """
        if (
            not self.data.endswith(".vcf")
            and not self.data.endswith(".vcf.gz")
            and not self.data.endswith(".bcf")
            and not self.data.endswith(".bcf.gz")
        ):
            raise IOError("VCF file must be vcf or bcf format")

        if not os.path.exists(self.data + ".tbi"):
            raise FileNotFoundError(
                f"Please index the vcf/bcf file before continue using: tabix -p vcf {self.data}"
            )
        # check_contig_length = (
        # f"{'zcat' if '.gz' in self.data else 'cat'} {self.data} | tail -n 1"
        # )
        # contig_name, contig_length = run(
        #     check_contig_length, shell=True, capture_output=True, text=True
        # ).stdout.split("\t")[:2]

        contig_name, contig_length = get_contig_from_vcf_filename(self.data)

        # check_contig_start = (
        # f'zgrep -v "#" {self.data} | head -n 1'
        # if self.data.endswith(".gz")
        # else f'fgrep -v "^#" {self.data} | head -n 1'
        # )
        # contig_start = run(
        # check_contig_start, shell=True, capture_output=True, text=True
        # ).stdout.split("\t")[1]

        # if (int(contig_start) - 6e5) < 0:
        # contig_start = 1

        # if self.step is None:
        #     step = None
        # else:
        #     step = int(self.step)

        window_iter = list(
            index_windows(
                np.arange(1, int(contig_length)),
                int(self.window_size - 1),
                1,
                int(contig_length) + int(self.window_size - 1),
                int(self.step),
            )
        )

        if self.in_memory:
            with Parallel(
                n_jobs=self.nthreads, backend="multiprocessing", verbose=5
            ) as parallel:
                region_data = parallel(
                    delayed(self.genome_reader)(
                        contig_name + ":" + str(w[0]) + "-" + str(w[1]), self.samples
                    )
                    for w in window_iter
                )

            out_dict = dict(chain.from_iterable(d.items() for d in region_data))

            sims = {"sweep": [], "region": []}
            for k, v in out_dict.items():
                if v is not None:
                    tmp = list(v)
                    tmp.append(np.zeros(4))
                    sims["sweep"].append(tmp)
                    sims["region"].append(k)
        else:
            sims = OrderedDict({"sweep": [], "region": []})
            sims["sweep"] = self.data

            for w in window_iter:
                # if v is not None:
                sims["region"].append(contig_name + ":" + str(w[0]) + "-" + str(w[1]))

        return sims

    def read_simulations(self):
        """
        Load paths (or parsed data if in_memory=True) for discoal simulations.

        Expects a directory structure:
            <self.data>/neutral/*.ms.gz
            <self.data>/sweep/*.ms.gz
            <self.data>/params.txt.gz

        Returns:
            tuple[OrderedDict, pl.DataFrame]:
                sims: OrderedDict({'neutral': <list|array>, 'sweep': <list|array>})
                params: polars DataFrame with columns ['model', 's', 't', 'saf', 'eaf'].

        Raises:
            AssertionError: If self.data is not a string.
            ValueError: If required 'neutral' or 'sweep' subfolders are missing or empty.
        """
        assert isinstance(self.data, str)

        for folder in ("sweep", "neutral"):
            folder_path = os.path.join(self.data, folder)
            if not os.path.exists(folder_path):
                raise ValueError(f"Required directory not found: {folder_path}")
            if not glob.glob(os.path.join(folder_path, "*")):
                raise ValueError(f"Directory is empty: {folder_path}")

        df_params = pl.read_csv(self.data + "/params.txt.gz")
        params = df_params.select(["model", "s", "t", "saf", "eaf", "mu", "r"])
        df_neutral = df_params.filter(pl.col("model") == "neutral")
        df_sweeps = df_params.filter(pl.col("model") == "sweep")

        sweeps = (
            (self.data + "/sweep/sweep_" + (df_sweeps.select("iter") + ".ms.gz"))
            .to_numpy()
            .flatten()
        )
        neutral = (
            (self.data + "/neutral/neutral_" + (df_neutral.select("iter") + ".ms.gz"))
            .to_numpy()
            .flatten()
        )

        sims = OrderedDict(
            {
                "neutral": neutral,
                "sweep": sweeps,
            }
        )

        return sims, params

    def get_cm(self, df_rec_map, positions):
        """
        Interpolate genetic distances (cM) for given physical positions.

        Args:
            df_rec_map (pl.DataFrame):
                Recombination map with numeric columns where:
                  columns[1] holds physical coordinate (bp) and
                  columns[-1] holds cumulative genetic distance (cM).

            positions (np.ndarray[int]): Physical positions to interpolate.

        Returns:
            np.ndarray[float]: Interpolated cumulative cM values (negative values clamped to 0).

        Notes:
            Uses linear interpolation with extrapolation at ends.
        """
        interp_func = interp1d(
            df_rec_map.select(df_rec_map.columns[1]).to_numpy().flatten(),
            df_rec_map.select(df_rec_map.columns[-1]).to_numpy().flatten(),
            kind="linear",
            fill_value="extrapolate",
        )

        # Interpolate the cM values at the interval positions
        rr1 = interp_func(positions)
        # rr2 = interp_func(positions[:, 1])
        rr1[rr1 < 0] = 0

        # Calculate the recombination rate in cM/Mb
        # rate = (rr2 - rr1) / ((positions[:, 1] - positions[:, 0]) / 1e6)

        return rr1
