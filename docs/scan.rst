Outlier scan
============

``flexsweep scan`` is a standalone positive-selection scan independent of the
CNN/DANN pipeline. It computes one or more selection statistics from VCF data
and assigns each locus an empirical p-value based on the genome-wide
distribution of that statistic. No neutral simulations are required.

How it works
------------

Each statistic is computed at its natural resolution — per-SNP statistics
(iHS, nSL, etc.) produce one value per polymorphic site; sliding-window
statistics (H12, LASSI, etc.) produce one value per window. Once all contigs
are processed, raw values from the entire genome are pooled and ranked
together to produce genome-wide empirical p-values.

Empirical p-values
~~~~~~~~~~~~~~~~~~

For each statistic an empirical p-value is assigned following the
empirical outlier approach (`Akey 2009 <https://doi.org/10.1101/gr.086652.108>`_):

.. math::

   p_i = \frac{\mathrm{rank}(-x_i)}{N_{\mathrm{valid}}}

where rank is computed on the negative value so that the **largest** statistic
gets the **smallest** p-value (rank 1 → p ≈ 0, outlier). :math:`N_{\mathrm{valid}}`
is the count of non-NaN loci. NaN values are excluded from the ranking and do
not contribute to :math:`N_{\mathrm{valid}}`.

For **signed statistics** (iHS, nSL, Tajima's D, Fay-Wu H, Zeng E), ranking
is done on :math:`|x_i|` before convert to negative, so that extreme values at both tails
are flagged as outliers.

The output column is named ``{stat}_pvalue``. A value close to 0 means the
locus is among the most extreme in the genome for that statistic.

.. note::

   The empirical p-value is not an analytical p-value. It reflects the
   position of a locus within *this* genome's distribution; it does not
   correspond to a controlled false-positive rate.


Available statistics
--------------------

Per-SNP statistics
~~~~~~~~~~~~~~~~~~

One score per polymorphic site. No sliding window.

.. list-table::
   :header-rows: 1
   :widths: 15 20 65

   * - Key
     - Rank column
     - Description
   * - ``ihs``
     - ``ihs``
     - Integrated haplotype score (`Voight et al. 2006 <https://doi.org/10.1371/journal.pbio.0040072>`_). Detects incomplete
       hard sweeps via extended haplotype homozygosity. Normalized within
       DAF bins. Configurable: ``min_maf`` (0.05), ``include_edges`` (False),
       ``gap_scale`` (20000), ``max_gap`` (200000).
   * - ``nsl``
     - ``nsl``
     - Number of segregating sites by length (`Ferrer-Admetlla et al. 2014 <https://doi.org/10.1093/molbev/msu077>`_).
       Robust alternative to iHS; no genetic map required. Normalized within
       DAF bins. Configurable: ``min_maf`` (0.05).
   * - ``isafe``
     - ``isafe``
     - Identifying the favored allele in a sweep (`Akbari et al. 2018 <https://doi.org/10.1038/nmeth.4606>`_).
       Pinpoints the causal mutation within a detected sweep region. Runs on
       non-overlapping regions. Configurable: ``region_size_bp`` (1000000),
       ``isafe_window`` (300), ``isafe_step`` (150), ``top_k`` (1),
       ``max_rank`` (15).
   * - ``dind``
     - ``dind``
     - Derived intra-allelic nucleotide diversity ratio (`Barreiro et al.
       2009 <https://doi.org/10.1371/journal.pgen.1000562>`_). Configurable: ``window_size`` (50000), ``min_focal_freq``
       (0.25), ``max_focal_freq`` (0.95).
   * - ``high_freq``
     - ``high_freq``
     - Frequency of high-frequency derived variants in a focal window
       (`Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_). Configurable: ``window_size`` (50000),
       ``min_focal_freq`` (0.25), ``max_focal_freq`` (0.95).
   * - ``low_freq``
     - ``low_freq``
     - Frequency of low-frequency derived variants in a focal window
       (`Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_). Configurable: ``window_size`` (50000),
       ``min_focal_freq`` (0.25), ``max_focal_freq`` (0.95).
   * - ``s_ratio``
     - ``s_ratio``
     - Ratio of segregating sites on derived vs. ancestral haplotypes
       (`Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_). Configurable: ``window_size`` (50000),
       ``min_focal_freq`` (0.25), ``max_focal_freq`` (0.95).
   * - ``hapdaf_o``
     - ``hapdaf_o``
     - Haplotype-derived allele frequency, other background (`Lauterbur et al.
       2023 <https://doi.org/10.1093/molbev/msad139>`_). Configurable: ``window_size`` (50000), ``min_focal_freq``
       (0.25), ``max_focal_freq`` (0.95), ``max_ancest_freq`` (0.25),
       ``min_tot_freq`` (0.25).
   * - ``hapdaf_s``
     - ``hapdaf_s``
     - Haplotype-derived allele frequency, sweep background (`Lauterbur et al.
       2023 <https://doi.org/10.1093/molbev/msad139>`_). Stricter ancestral-frequency thresholds than ``hapdaf_o``.
       Configurable: ``window_size`` (50000), ``min_focal_freq`` (0.25),
       ``max_focal_freq`` (0.95), ``max_ancest_freq`` (0.10),
       ``min_tot_freq`` (0.10).
   * - ``hscan``
     - ``hscan``
     - Average pairwise haplotype homozygosity tract length H(x)
       (`h-scan <https://messerlab.org/resources/>`_, Messer; evaluated in
       `Schlamp et al. 2016 <https://doi.org/10.1111/mec.13485>`_). Measures
       the mean shared haplotype block length across all
       sample pairs; detects hard and soft sweeps. Configurable:
       ``max_gap`` (200000), ``dist_mode`` (0), ``hscan_step`` (1).

       Use ``hscan_step`` (not ``step``) to control scan resolution; ``step``
       is a shared SNP-window parameter and is not used by hscan.

Sliding SNP-window statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One score per window of a fixed number of SNPs.

.. list-table::
   :header-rows: 1
   :widths: 15 12 12 12 49

   * - Key
     - Rank column
     - Default window
     - Default step
     - Description
   * - ``haf``
     - ``haf``
     - 201 SNPs
     - 10 SNPs
     - Haplotype allele frequency (`Ronen et al. 2015 <https://doi.org/10.1371/journal.pgen.1005527>`_). Mean pairwise
       haplotype similarity across a SNP window. Configurable via shared
       ``w_size`` and ``step``.
   * - ``h12``
     - ``h12``
     - 200 SNPs
     - 10 SNPs
     - H12 haplotype homozygosity (`Garud et al. 2015 <https://doi.org/10.1371/journal.pgen.1005004>`_). Combines the two most
       common haplotype frequencies. Configurable via shared ``w_size`` and
       ``step``.
   * - ``garud``
     - ``h12``
     - 200 SNPs
     - 10 SNPs
     - Full Garud statistics: H1, H12, H2/H1 (`Garud et al. 2015 <https://doi.org/10.1371/journal.pgen.1005004>`_).
       Configurable via shared ``w_size`` and ``step``.
   * - ``lassi``
     - ``T_m``
     - 201 SNPs
     - 10 SNPs
     - Composite likelihood sweep scan using the haplotype frequency spectrum
       (`Harris & DeGiorgio 2020 <https://doi.org/10.1093/molbev/msaa115>`_).
       Configurable: ``K_truncation`` (10),
       ``sweep_mode`` (4), and shared ``w_size``, ``step``.
   * - ``raisd``
     - ``mu_total``
     - 50 SNPs
     - 1 SNP
     - RAiSD μ composite statistic combining SFS, SNP density variation, and
       LD (`Alachiotis & Pavlidis 2018 <https://doi.org/10.1038/s42003-018-0085-8>`_). Configurable: ``window_size`` (50).
       Additional output columns: ``mu_var``, ``mu_sfs``, ``mu_ld``.

Sliding bp-window statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One score per physical window. All configurable via shared ``w_size_bp``
(default 1 Mb) and ``step_bp`` (default 10 kb), except ``omega``, ``zns``,
``beta``, and ``ncd`` which have their own narrower defaults.

.. list-table::
   :header-rows: 1
   :widths: 15 10 10 10 55

   * - Key
     - Rank column
     - Default window
     - Default step
     - Description
   * - ``tajima_d``
     - ``tajima_d``
     - 1 Mb
     - 10 kb
     - Tajima's D (`Tajima 1989 <https://doi.org/10.1093/genetics/123.3.585>`_). SFS-based test; negative values signal
       directional selection. Signed stat — ranked by ``abs(value)``.
   * - ``pi``
     - ``pi``
     - 1 Mb
     - 10 kb
     - Nucleotide diversity θ\ :sub:`π`.
   * - ``theta_w``
     - ``theta_w``
     - 1 Mb
     - 10 kb
     - Watterson's θ\ :sub:`W`.
   * - ``fay_wu_h``
     - ``fay_wu_h``
     - 1 Mb
     - 10 kb
     - Fay & Wu's H (`Fay & Wu 2000 <https://doi.org/10.1093/genetics/155.3.1405>`_). Sensitive to high-frequency derived
       alleles. Signed stat — ranked by ``abs(value)``.
   * - ``zeng_e``
     - ``zeng_e``
     - 1 Mb
     - 10 kb
     - Zeng's E (`Zeng et al. 2006 <https://doi.org/10.1534/genetics.106.061432>`_). Signed stat — ranked by ``abs(value)``.
   * - ``achaz_y``
     - ``achaz_y``
     - 1 Mb
     - 10 kb
     - Achaz Y (`Achaz 2009 <https://doi.org/10.1534/genetics.109.104042>`_). Robust to sequencing errors.
   * - ``fuli_d``
     - ``fuli_d``
     - 1 Mb
     - 10 kb
     - Fu & Li's D (`Fu & Li 1993 <https://doi.org/10.1093/genetics/133.3.693>`_).
   * - ``fuli_d_star``
     - ``fuli_d_star``
     - 1 Mb
     - 10 kb
     - Fu & Li's D* (no outgroup required).
   * - ``fuli_f``
     - ``fuli_f``
     - 1 Mb
     - 10 kb
     - Fu & Li's F (`Fu & Li 1993 <https://doi.org/10.1093/genetics/133.3.693>`_).
   * - ``fuli_f_star``
     - ``fuli_f_star``
     - 1 Mb
     - 10 kb
     - Fu & Li's F* (no outgroup required).
   * - ``neutrality``
     - ``tajima_d``
     - 1 Mb
     - 10 kb
     - Composite: Tajima's D, π, θ\ :sub:`W`, Fay-Wu H in one pass.
       Ranked by ``abs(tajima_d)``.
   * - ``omega``
     - ``omega_max``
     - 100 kb
     - 10 kb
     - Kim & Nielsen's ω (`Kim & Nielsen 2004 <https://doi.org/10.1534/genetics.103.025387>`_). LD patterns around a putative
       sweep centre.
   * - ``zns``
     - ``zns``
     - 100 kb
     - 10 kb
     - Kelly's Z\ :sub:`nS` (`Kelly 1997 <https://doi.org/10.1093/genetics/146.3.1197>`_). Mean pairwise r² across all SNP
       pairs in a window.
   * - ``beta``
     - ``beta1``
     - 50 kb
     - 5 kb
     - Beta1 statistic for balancing selection (`Siewert & Voight 2017
       <https://doi.org/10.1093/molbev/msx209>`_; standardised form in
       `Siewert & Voight 2020 <https://doi.org/10.1093/gbe/evaa013>`_).
       Configurable: ``m`` (0.1).
   * - ``ncd``
     - ``ncd1``
     - 3 kb
     - 1.5 kb
     - NCD1 for balancing selection (`Bitarello et al. 2018 <https://doi.org/10.1093/gbe/evy054>`_). Configurable:
       ``tf`` (0.5), ``w`` (3000), ``minIS`` (2).

Window mode
~~~~~~~~~~~

SNP-count windows are required for H12, LASSI, and RAiSD — physical windows
confound SNP density with haplotype diversity for those statistics. SFS-based statistics (Tajima's D, Fay-Wu H, etc.) use physical
bp windows by default.

With the default ``window_mode="auto"``, each statistic uses its built-in
mode. Pass ``window_mode="snp"`` or ``window_mode="bp"`` to force a uniform
mode across all window statistics.


Normalization
-------------

Statistics sensitive to allele frequency (iHS, nSL, DIND, high_freq,
low_freq, s_ratio, hapdaf_o, hapdaf_s) are z-scored within genome-wide DAF
bins before p-values are computed. This removes the frequency-dependent bias
that would otherwise cause high-frequency SNPs to dominate outlier lists.

**DAF-only normalization** (default):

1. Compute 50 equal-frequency DAF bin edges over the genome-wide DAF
   distribution.
2. Assign each SNP to a bin.
3. Within each bin: z-score = (value − mean) / std. Bins with fewer than 2
   SNPs are left as NaN.

**Joint DAF × recombination rate normalization** (when ``recombination_map``
and ``n_r_bins`` are both set):

The genome is additionally stratified by recombination rate. Each SNP is
assigned to a (DAF bin, recomb rate bin) cell and z-scored within that cell.
This further reduces false positives in low-recombination regions (Johnson et
al. approach). To enable it, pass both ``recombination_map`` and
``n_r_bins`` (typically ``n_r_bins=10``):

.. code-block:: python

    results = scan(
        "data/vcf/",
        "results/YRI",
        stats=["ihs", "nsl"],
        recombination_map="data/decode_sexavg_2019.txt.gz",
        n_daf_bins=50,
        n_r_bins=10,
    )

.. note::

   Passing only ``recombination_map`` without ``n_r_bins`` uses DAF-only
   normalization with genetic-distance windows for T3 stats (dind, hapdaf,
   s_ratio). Set ``n_r_bins`` explicitly to enable joint normalization.


Multi-contig usage
------------------

Pass a directory to ``--vcf_path`` to process all ``*.vcf.gz`` / ``*.bcf.gz``
files. The scan uses a two-step approach:

1. Each contig is processed independently; raw unranked values are
produced for every requested statistic.

2. Results from all contigs are concatenated per statistic.
Normalization and empirical p-values are computed across all contigs together,
ensuring p-values reflect the true genome-wide distribution.

.. code-block:: bash

    flexsweep scan \
        --vcf_path data/vcf/ \
        --out_prefix results/YRI \
        --stats ihs,nsl,h12 \
        --recombination_map data/decode_sexavg_2019.txt.gz \
        --nthreads 4


CLI reference
-------------

.. code-block:: text

    flexsweep scan [OPTIONS]

.. list-table::
   :header-rows: 1
   :widths: 25 12 63

   * - Option
     - Default
     - Description
   * - ``--vcf_path PATH``
     - required
     - Directory of ``*.vcf.gz`` files (one per chromosome/contig).
   * - ``--out_prefix PREFIX``
     - required
     - Output prefix. Writes ``{PREFIX}.{stat}.txt`` for each stat.
   * - ``--stats LIST``
     - required
     - Comma-separated stat keys, e.g. ``ihs,nsl,h12,raisd``.
   * - ``--w_size INT``
     - 201
     - SNP-count window size for SNP-mode stats.
   * - ``--step INT``
     - 10
     - SNP step size for SNP-mode stats.
   * - ``--w_size_bp INT``
     - 1000000
     - Physical window size (bp) for bp-mode stats.
   * - ``--step_bp INT``
     - 10000
     - Physical step size (bp) for bp-mode stats.
   * - ``--window_mode``
     - auto
     - ``auto`` uses per-stat defaults; ``snp`` forces SNP-count windows;
       ``bp`` forces physical bp windows for all window stats.
   * - ``--min_maf FLOAT``
     - 0.05
     - Minimum minor allele frequency for iHS and nSL.
   * - ``--window_size INT``
     - 50000
     - Focal window size (bp) for per-SNP stats (dind, hapdaf_o,
       hapdaf_s, s_ratio, high_freq, low_freq).
   * - ``--recombination_map PATH``
     - None
     - TSV recombination map (chr, start, end, cm_mb, cm). Enables
       genetic-distance windows for T3 stats.
   * - ``--n_daf_bins INT``
     - 50
     - Number of equal-frequency DAF bins for normalization.
   * - ``--n_r_bins INT``
     - None
     - Number of recombination rate bins for joint DAF × recomb
       normalization. Set to 10 to match Johnson et al. Requires
       ``--recombination_map``.
   * - ``--K_truncation INT``
     - 10
     - K truncation for LASSI (number of HFS classes).
   * - ``--sweep_mode INT``
     - 4
     - Sweep spectral model for LASSI (1–5; 4 = Gaussian decay).
   * - ``--raisd_window INT``
     - 50
     - SNP window size for RAiSD.
   * - ``--nthreads INT``
     - 1
     - Number of parallel workers.

Examples:

.. code-block:: bash

    # iHS + nSL on a single chromosome
    flexsweep scan \
        --vcf_path YRI.chr22.vcf.gz \
        --out_prefix results/YRI.chr22 \
        --stats ihs,nsl

    # H12 + RAiSD with custom SNP window
    flexsweep scan \
        --vcf_path YRI.chr22.vcf.gz \
        --out_prefix results/YRI.chr22 \
        --stats h12,raisd \
        --w_size 400 \
        --nthreads 4

    # SFS statistics using physical bp windows
    flexsweep scan \
        --vcf_path YRI.chr22.vcf.gz \
        --out_prefix results/YRI.chr22 \
        --stats tajima_d,fay_wu_h,zeng_e,omega

    # DIND + HapDAF with a larger focal window
    flexsweep scan \
        --vcf_path YRI.chr22.vcf.gz \
        --out_prefix results/YRI.chr22 \
        --stats dind,hapdaf_o,hapdaf_s \
        --window_size 100000 \
        --recombination_map data/decode_sexavg_2019.txt.gz

    # Genome-wide scan — joint DAF × recomb normalization
    flexsweep scan \
        --vcf_path data/vcf/ \
        --out_prefix results/YRI \
        --stats ihs,nsl,h12 \
        --recombination_map data/decode_sexavg_2019.txt.gz \
        --n_r_bins 10 \
        --nthreads 4


Python API
----------

.. code-block:: python

    from flexsweep.scan import scan, available_stats, stat_params

    # List all available stat keys
    print(available_stats())

    # Inspect default parameters for all stats
    stat_params()

    # Inspect a single stat — shows rank_col, resolution, window_mode,
    # default_window, default_step, shared_params, and stat_params
    stat_params("raisd")
    # {'raisd': {'rank_col': 'mu_total', 'resolution': 'window',
    #            'window_mode': 'snp', 'default_window': '50 SNPs',
    #            'default_step': '10 SNPs',
    #            'stat_params': {'window_size': 50}, ...}}

    stat_params("hscan")
    # {'hscan': {'rank_col': 'hscan', 'resolution': 'snp',
    #            'window_mode': 'n/a (per-SNP stat)',
    #            'stat_params': {'max_gap': 200000, 'dist_mode': 0,
    #                            'hscan_step': 1}, ...}}

    # Basic scan
    results = scan(
        "data/vcf/",
        "results/YRI",
        stats=["ihs", "nsl", "h12"],
        min_maf=0.05,
        recombination_map="data/decode_sexavg_2019.txt.gz",
        nthreads=4,
    )
    # results["ihs"]   → Polars DataFrame, SNP resolution, ihs_pvalue column
    # results["raisd"] → Polars DataFrame, window resolution, mu_total_pvalue column

    # Joint DAF × recomb normalization
    results = scan(
        "data/vcf/",
        "results/YRI",
        stats=["ihs", "nsl"],
        recombination_map="data/decode_sexavg_2019.txt.gz",
        n_daf_bins=50,
        n_r_bins=10,
        nthreads=4,
    )

    # Per-stat parameter overrides via config dict
    results = scan(
        "data/vcf/",
        "results/YRI",
        stats=["raisd", "hscan"],
        config={
            "raisd":  {"window_size": 100},
            "hscan":  {"hscan_step": 5, "max_gap": 100_000},
        },
        nthreads=4,
    )


Output format
-------------

Each statistic writes one tab-separated file ``{out_prefix}.{stat}.txt``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Column
     - Description
   * - ``chrom``
     - Chromosome/contig name from VCF.
   * - ``pos``
     - Genomic position (bp). SNP stats: SNP position. Window stats: centre
       of window.
   * - ``daf``
     - Derived allele frequency (iHS, nSL, DIND, HapDAF, s_ratio, iSAFE).
   * - ``n_snps``
     - Number of SNPs in window (window stats only).
   * - ``{stat_col}``
     - Main statistic value (after DAF normalization for iHS/nSL/dind/hapdaf/
       s_ratio).
   * - ``{stat_col}_pvalue``
     - Empirical p-value: rank(−value) / N_valid. Range (0, 1]; smaller =
       more extreme. Signed stats (iHS, nSL, Tajima's D, Fay-Wu H, Zeng E)
       are ranked by ``abs(value)``.

Additional columns vary by statistic (e.g., ``h2_h1`` for garud, ``m`` for
lassi, ``mu_var``, ``mu_sfs``, ``mu_ld`` for raisd).


Scan results visualization
---------------------------

Use ``plot_scan`` from ``flexsweep.utils`` to generate Manhattan-style or
regional zoom plots directly from ``scan()`` output or saved TSV files.

**Genome-wide plot:**

.. code-block:: python

    from flexsweep.utils import plot_scan

    # When passing a scan() dict, stat_cols is resolved automatically
    # from STAT_REGISTRY (e.g. "raisd" → "mu_total", "lassi" → "T_m").
    # No need to specify stat_cols.

    # Single statistic — raw values, top 1% highlighted
    plot_scan({"ihs": results["ihs"]}, out="results/YRI.ihs.png")

    # Single statistic — empirical p-values (-log10 scale)
    plot_scan({"ihs": results["ihs"]}, pvalue=True, out="results/YRI.ihs.png")

    # Stacked multi-statistic panels — rank columns resolved automatically
    plot_scan(
        {k: results[k] for k in ["ihs", "h12", "raisd"]},
        pvalue=True,
        out="results/YRI.multi.png",
    )
    # Plots: ihs, h12, mu_total (raisd)

**Regional zoom plot** (raw + p-value side by side):

.. code-block:: python

    plot_scan(
        {k: results[k] for k in ["ihs", "raisd"]},
        chrom="22",
        center=17_000_000,
        window_bp=500_000,
        out="results/YRI.zoom.png",
    )

**From saved TSV files** (stat_cols must be provided explicitly):

.. code-block:: python

    plot_scan(
        ["results/YRI.ihs.txt", "results/YRI.raisd.txt"],
        stat_cols=["ihs", "mu_total"],   # must match column name in file
        pvalue=True,
        out="results/YRI.multi.png",
    )

``plot_scan`` parameters:

.. list-table::
   :header-rows: 1
   :widths: 20 12 68

   * - Parameter
     - Default
     - Description
   * - ``stats``
     - required
     - ``dict`` from ``scan()``, a single TSV path, or a list of TSV paths.
   * - ``stat_cols``
     - None
     - Stat column name(s). Required when ``stats`` is a file path / list.
       Defaults to all keys when ``stats`` is a dict.
   * - ``pvalue``
     - False
     - If True, plot :math:`-\log_{10}(p_{\mathrm{emp}})` with threshold
       lines at p = 0.01 and p = 0.001. If False, plot raw values with
       outliers highlighted.
   * - ``top_pct``
     - 0.01
     - Fraction of loci highlighted as outliers in raw mode.
   * - ``chrom``
     - None
     - Chromosome for zoom mode. Provide together with ``center``.
   * - ``center``
     - None
     - Centre position (bp) for zoom mode.
   * - ``window_bp``
     - 500000
     - Half-window in bp for zoom mode (±500 kb around ``center``).
   * - ``out``
     - None
     - Save path. If None, shows interactively.
   * - ``figsize``
     - None
     - Figure size tuple. Defaults to (14, 4) genome-wide, (10, 2.5×n) zoom.
   * - ``sharey``
     - False
     - Share y-axis across stacked panels.
   * - ``threshold_lines``
     - None
     - List of ``(y_value, linestyle, label)`` for horizontal lines in
       p-value mode. Pass ``[]`` to suppress defaults.


References
----------

**iHS**
  Voight, B.F., Kudaravalli, S., Wen, X. and Pritchard, J.K. (2006) A map of recent
  positive selection in the human genome. *PLOS Biology*, 4, e72.

**nSL**
  Ferrer-Admetlla, A., Liang, M., Korneliussen, T. and Nielsen, R. (2014) On detecting
  incomplete soft or hard selective sweeps using haplotype structure. *Molecular Biology
  and Evolution*, 31, 1275–1286.

**iSAFE**
  Akbari, A., Vitti, J.J., Iranmehr, A., Bakhtiari, M., Sabeti, P.C., Mirarab, S. and
  Bafna, V. (2018) Identifying the favored mutation in a positive selective sweep.
  *Nature Methods*, 15, 183–185.

**DIND**
  Barreiro, L.B., Henriques, R., Soares, M.J., Oliveira, J., Gasche, C., … and
  Quintana-Murci, L. (2009) Evolutionary dynamics of human Toll-like receptors and their
  different contributions to host defense. *PLOS Genetics*, 5, e1000562.

**HapDAF-s/o, s_ratio, high_freq, low_freq**
  Lauterbur, M.E., Munch, K. and Enard, D. (2023) Versatile detection of diverse selective
  sweeps with Flex-sweep. *Molecular Biology and Evolution*, 40, msad139.

**H12, H2/H1**
  Garud, N.R., Messer, P.W., Buzbas, E.O. and Petrov, D.A. (2015) Recent selective sweeps
  in North American *Drosophila melanogaster* show signatures of soft sweeps. *PLOS
  Genetics*, 11, e1005004.

**h-scan**
  Messer, P.W. h-scan: a program for detecting selective sweeps from patterns of
  homozygosity tract lengths. https://messerlab.org/resources/

  Schlamp, F., van der Made, J., Stambler, R., Chesebrough, L., Boyko, A.R. and
  Messer, P.W. (2016) Evaluating the performance of selection scans to detect
  selective sweeps in domestic dogs. *Molecular Ecology*, 25, 342-356.

**LASSI**
  Harris, A. and DeGiorgio. (2020) A Likelihood Approach for Uncovering Selective Sweep Signatures from haplotype Data

**RAiSD**
  Alachiotis, N. and Pavlidis, P. (2018) RAiSD detects positive selection based on multiple
  signatures of a selective sweep and SNP vectors. *Communications Biology*, 1, 79.

**ω (omega)**
  Kim, Y. and Nielsen, R. (2004) Linkage disequilibrium as a signature of selective sweeps.
  *Genetics*, 167, 1513–1524.

**ZnS**
  Kelly, J.K. (1997) A test of neutrality based on interlocus associations. *Genetics*,
  146, 1197–1206.

**Tajima's D**
  Tajima, F. (1989) Statistical method for testing the neutral mutation hypothesis by DNA
  polymorphism. *Genetics*, 123, 585–595.

**Fay-Wu H**
  Fay, J.C. and Wu, C.I. (2000) Hitchhiking under positive Darwinian selection. *Genetics*,
  155, 1405–1413.

**Zeng E**
  Zeng, K., Fu, Y.X., Shi, S. and Wu, C.I. (2006) Statistical tests for detecting positive
  selection by utilizing high-frequency variants. *Genetics*, 174, 1431–1439.

**Fu-Li D, F**
  Fu, Y.X. and Li, W.H. (1993) Statistical tests of neutrality of mutations. *Genetics*,
  133, 693–709.

**Beta (balancing selection)**
  Siewert, K.M. and Voight, B.F. (2020) BetaScan2: Standardized statistics to detect
  balancing selection utilizing substitution data. *Genome Biology and Evolution*, 12, evaa013.

**NCD1**
  Bitarello, B.D., de Filippo, C., Teixeira, J.C., Schmidt, J.M., Kleinert, P., Meyer, D.
  and Andrés, A.M. (2018) Signatures of long-term balancing selection in human genomes.
  *The American Journal of Human Genetics*, 102, 725–742.

**Outlier approach**
  Akey, J.M. (2009) Constructing genomic maps of positive selection in humans: where do
  we go from here? *Genome Research*, 19, 711–722.

