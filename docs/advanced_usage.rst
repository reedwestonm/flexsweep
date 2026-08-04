Advanced usage
==============

Custom summary statistics
-------------------------

Users can combine any included summary statistic over user-defined genomic
intervals, to fit the most informative statistic and region combination for a
given organism. Select any combination of built-in statistics by passing a list
of names to the ``stats`` argument of ``summary_statistics``. The pipeline
handles computation, windowing, and normalisation automatically for both
simulations and VCF data.

By default (``stats=None``) the full Flex-sweep statistic set is used.

Normalisation follows two schemes. SNP-based statistics are normalised by
frequency bin following `Voight et al. 2006
<https://doi.org/10.1371/journal.pbio.0040072>`_. Window-based statistics are
Z-scored per window/centre combination:

.. math::

   Z\text{-score }\pi_i^{(w_1,c_1)}
   \;=\;
   \frac{\pi_i^{(w_1,c_1)} - \mu^{(w_1,c_1)}}{\sigma^{(w_1,c_1)}},
   \qquad \\
   \text{where }\mu^{(w_1,c_1)}=\operatorname{mean}_{i=1}^{n}\!\bigl(\pi_i^{(w_1,c_1)}\bigr)


Available statistics
~~~~~~~~~~~~~~~~~~~~

Statistics fall into two families. **Window-based** statistics are computed for
every centre × window-size combination and summarise a genomic interval.
**SNP-based** statistics are computed per variant and are subsequently
aggregated over the same centre × window grid. Every name below is a valid
entry of the ``stats`` argument; passing an unknown name raises a ``ValueError``
listing all valid names.

**Window-based statistics**

.. list-table::
   :header-rows: 1
   :widths: 16 40 20 24

   * - Name
     - Description
     - Summarises
     - Reference
   * - ``pi``
     - Average number of pairwise nucleotide differences, per base pair
     - Nucleotide diversity
     - `Tajima 1983 <https://doi.org/10.1093/genetics/105.2.437>`_
   * - ``theta_w``
     - Watterson's :math:`\theta_W` based on segregating sites, per base pair
     - Nucleotide diversity
     - `Watterson 1975 <https://doi.org/10.1016/0040-5809(75)90020-9>`_
   * - ``tajima_d``
     - Tajima's :math:`D`; difference between :math:`\theta_\pi` and
       :math:`\theta_W`
     - SFS
     - `Tajima 1989 <https://doi.org/10.1093/genetics/123.3.585>`_
   * - ``theta_h``
     - Fay & Wu's :math:`\theta_H`; excess of high-frequency derived alleles
     - SFS
     - `Fay & Wu 2000 <https://doi.org/10.1093/genetics/155.3.1405>`_
   * - ``fay_wu_h``
     - Normalised Fay & Wu's :math:`H`
     - SFS
     - `Fay & Wu 2000 <https://doi.org/10.1093/genetics/155.3.1405>`_; `Zeng et al. 2006 <https://doi.org/10.1534/genetics.106.061432>`_
   * - ``zeng_e``
     - Zeng's :math:`E`; contrast between :math:`\theta_\pi` and Fay & Wu's
       :math:`H`
     - SFS
     - `Zeng et al. 2006 <https://doi.org/10.1534/genetics.106.061432>`_
   * - ``achaz_y``
     - Achaz's :math:`Y`; Tajima's :math:`D` analogue excluding derived
       singletons, robust to sequencing errors (polarized, requires outgroup)
     - SFS
     - `Achaz 2008 <https://doi.org/10.1534/genetics.107.082198>`_
   * - ``achaz_y_star``
     - Achaz's :math:`Y^*`; folded analogue excluding minor-allele singletons
     - SFS
     - `Achaz 2008 <https://doi.org/10.1534/genetics.107.082198>`_
   * - ``achaz_t``
     - Achaz's :math:`T_\Omega`; generalised neutrality test defined as the
       normalized difference between any two frequency-spectrum-based
       :math:`\hat{\theta}` estimators
     - SFS
     - `Achaz 2009 <https://doi.org/10.1534/genetics.109.104042>`_
   * - ``fuli_d``
     - Fu & Li's :math:`D`; excess of singletons relative to total mutations
       (polarized, requires outgroup)
     - SFS
     - `Fu & Li 1993 <https://doi.org/10.1093/genetics/133.3.693>`_
   * - ``fuli_d_star``
     - Fu & Li's :math:`D^*` (folded)
     - SFS
     - `Fu & Li 1993 <https://doi.org/10.1093/genetics/133.3.693>`_
   * - ``fuli_f``
     - Fu & Li's :math:`F`; excess of singletons relative to average pairwise
       differences (polarized, requires outgroup)
     - SFS
     - `Fu & Li 1993 <https://doi.org/10.1093/genetics/133.3.693>`_
   * - ``fuli_f_star``
     - Fu & Li's :math:`F^*` (folded)
     - SFS
     - `Fu & Li 1993 <https://doi.org/10.1093/genetics/133.3.693>`_
   * - ``max_fda``
     - :math:`\mathrm{DAF}_{\max}`; maximum derived allele frequency in a
       subwindow
     - SFS
     - `Kern & Schrider 2018 <https://doi.org/10.1534/g3.118.200262>`_
   * - ``k_counts``
     - Number of distinct haplotypes
     - Haplotype structure
     - `Kern & Schrider 2018 <https://doi.org/10.1534/g3.118.200262>`_
   * - ``h1``
     - Garud's :math:`H_1`; haplotype homozygosity
     - Haplotype structure
     - `Garud et al. 2015 <https://doi.org/10.1371/journal.pgen.1005004>`_
   * - ``h12``
     - Garud's :math:`H_{12}`; frequencies of the first and second most common
       haplotypes, modified to use an 80% identity threshold
     - Haplotype structure
     - `Garud et al. 2015 <https://doi.org/10.1371/journal.pgen.1005004>`_; `Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_
   * - ``h2_h1``
     - Garud's :math:`H_2/H_1`
     - Haplotype structure
     - `Garud et al. 2015 <https://doi.org/10.1371/journal.pgen.1005004>`_
   * - ``haf``
     - Haplotype allele frequency, reporting only the summed upper 10% of the
       sorted scores (HAF-top)
     - Haplotype structure
     - `Ronen et al. 2015 <https://doi.org/10.1371/journal.pgen.1005527>`_; `Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_
   * - ``dist_var``
     - :math:`\mathrm{Var}(d_{ij})`; variance of the distribution of pairwise
       haplotype mismatch distances within a subwindow
     - Haplotype structure
     - `Kern & Schrider 2018 <https://doi.org/10.1534/g3.118.200262>`_
   * - ``dist_skew``
     - :math:`\mathrm{Skew}(d_{ij})`; skewness of the same distribution
     - Haplotype structure
     - `Kern & Schrider 2018 <https://doi.org/10.1534/g3.118.200262>`_
   * - ``dist_kurtosis``
     - :math:`\mathrm{Kurt}(d_{ij})`; excess kurtosis of the same distribution
     - Haplotype structure
     - `Kern & Schrider 2018 <https://doi.org/10.1534/g3.118.200262>`_
   * - ``zns``
     - Kelly's :math:`Z_{nS}`; average linkage disequilibrium (:math:`r^2`)
       between segregating sites
     - Linkage disequilibrium
     - `Kelly 1997 <https://doi.org/10.1093/genetics/146.3.1197>`_
   * - ``omega_max``
     - :math:`\omega_{\max}`; maximum LD between selected and flanking regions
     - Linkage disequilibrium
     - `Kim & Nielsen 2004 <https://doi.org/10.1534/genetics.103.025387>`_
   * - ``ncd1``
     - NCD1; non-central deviation of the SFS from neutrality
     - SFS (balancing selection)
     - `Bitarello et al. 2018 <https://doi.org/10.1093/gbe/evy054>`_

**SNP-based statistics**

.. list-table::
   :header-rows: 1
   :widths: 16 40 20 24

   * - Name
     - Description
     - Summarises
     - Reference
   * - ``ihs``
     - Integrated haplotype score
     - Haplotype structure
     - `Voight et al. 2006 <https://doi.org/10.1371/journal.pbio.0040072>`_
   * - ``delta_ihh``
     - :math:`\Delta\text{-}iHH`; absolute iHH difference between ancestral and
       derived alleles
     - Haplotype structure
     - `Grossman et al. 2010 <https://doi.org/10.1126/science.1183863>`_
   * - ``nsl``
     - :math:`nS_L`; number of segregating sites by length
     - Haplotype structure
     - `Ferrer-Admetlla et al. 2014 <https://doi.org/10.1093/molbev/msu077>`_
   * - ``isafe``
     - Integrated selection of allele favoured by evolution
     - Haplotype structure
     - `Akbari et al. 2018 <https://doi.org/10.1038/nmeth.4606>`_
   * - ``hscan``
     - Average pairwise haplotype homozygosity tract length
     - Haplotype structure
     - `Schlamp et al. 2016 <https://doi.org/10.1111/mec.13485>`_
   * - ``dind`` / ``dind_high_low``
     - Derived intra-allelic nucleotide diversity
     - Diversity on derived background
     - `Barreiro et al. 2009 <https://doi.org/10.1371/journal.pgen.1000562>`_
   * - ``s_ratio``
     - Segregating sites ratio
     - Diversity on derived background
     - `Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_
   * - ``low_freq`` / ``lowfreq``
     - Low-frequency alleles on derived background
     - Diversity on derived background
     - `Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_
   * - ``high_freq`` / ``highfreq``
     - High-frequency alleles on derived background
     - Diversity on derived background
     - `Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_
   * - ``hapdaf_o``
     - Haplotype-derived allele frequency (old)
     - SFS
     - `Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_
   * - ``hapdaf_s``
     - Haplotype-derived allele frequency (standing)
     - SFS
     - `Lauterbur et al. 2023 <https://doi.org/10.1093/molbev/msad139>`_
   * - ``beta``
     - :math:`\beta^{(1)}`; correlation of allele frequency with local
       polymorphism
     - SFS (balancing selection)
     - `Siewert & Voight 2020 <https://doi.org/10.1093/gbe/evaa013>`_
   * - ``beta_t``
     - :math:`\beta^{(1)*}_{(std)}`; standardised form of the above
     - SFS (balancing selection)
     - `Siewert & Voight 2020 <https://doi.org/10.1093/gbe/evaa013>`_

.. note::

   ``ihs`` and ``nsl`` are computed with `scikit-allel
   <https://scikit-allel.readthedocs.io/>`_; every other statistic is a Numba
   or NumPy vectorised implementation. ``delta_ihh`` is computed together with
   ``ihs`` and is present in the output, but it is excluded from the default
   CNN feature set.

Statistics available through other modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The composite sweep statistics below are not selectable through the ``stats``
argument of the feature-vector pipeline. They are available through the
standalone outlier scan (see :doc:`scan`) and can be called directly from the
Python API (see :doc:`api` for full signatures).

.. list-table::
   :header-rows: 1
   :widths: 16 24 36 24

   * - Statistic
     - ``scan`` key / function
     - Description
     - Reference
   * - LASSI :math:`T`, :math:`\hat{m}`
     - ``lassi`` / ``compute_t_m``
     - Likelihood-based detection of selective sweeps using haplotype structure
     - `Harris & DeGiorgio 2020 <https://doi.org/10.1093/molbev/msaa115>`_
   * - RAiSD :math:`\mu`
     - ``raisd`` / ``mu_stat``
     - Composite detection of selective sweeps using SFS, LD, and diversity
     - `Alachiotis & Pavlidis 2018 <https://doi.org/10.1038/s42003-018-0085-8>`_


Selecting statistics
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import flexsweep as fs

    # A subset of window and SNP statistics
    df = fs.summary_statistics(
        "./simulations",
        stats=["pi", "h12", "ihs", "nsl"],
        nthreads=8,
    )

The same ``stats`` argument applies to VCF data:

.. code-block:: python

    df_vcf = fs.summary_statistics(
        "./vcf_data",
        vcf=True,
        stats=["pi", "h12", "ihs", "nsl"],
        recombination_map="recomb_map.csv",
        nthreads=8,
    )

Passing an invalid stat name raises a ``ValueError`` listing all valid names.


diploS/HIC-like feature vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To estimate the same statistics as `diploS/HIC
<https://doi.org/10.1534/g3.118.200262>`_ — :math:`\pi`, :math:`\theta_W`,
:math:`\theta_H`, Fay & Wu's :math:`H`, Tajima's :math:`D`, distinct
haplotypes, :math:`H_1`, :math:`H_{12}`, :math:`H_2/H_1`, :math:`Z_{nS}`,
:math:`\omega_{max}`, maximum derived allele frequency, and pairwise distance
moments — use the parameters below. The locus is divided into contiguous
sub-windows matching the diploS/HIC subwindow approach.

From simulations:

.. code-block:: python

    import flexsweep as fs

    diploshic_stats = [
        "pi", "fay_wu_h", "theta_h", "max_fda", "theta_w", "tajima_d",
        "k_counts", "h1", "h12", "h2_h1", "zns", "omega_max",
        "dist_var", "dist_skew", "dist_kurtosis",
    ]

    df = fs.summary_statistics(
        "./simulations",
        stats=diploshic_stats,
        locus_length=1100000,
        step=100000,
        windows=[100000],
        nthreads=8,
    )

From VCF data:

.. code-block:: python

    df_vcf = fs.summary_statistics(
        "./vcf_data",
        vcf=True,
        stats=diploshic_stats,
        locus_length=1100000,
        step=100000,
        step_vcf=1000000,
        windows=[100000],
        nthreads=8,
    )


Locus and window settings
~~~~~~~~~~~~~~~~~~~~~~~~~

Three arguments control the locus geometry. They must be consistent between
the simulation and VCF runs so that normalisation bins match.

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Argument
     - Default
     - Description
   * - ``locus_length``
     - ``1200000``
     - Total locus length in base pairs. Together with ``step``, determines
       the centre grid: one centre every ``step`` bp from ``step//2`` to
       ``locus_length − step//2``.
   * - ``windows``
     - ``[100000]``
     - List of window sizes (bp). Multiple sizes produce a multi-scale
       feature vector. Each centre is evaluated at every window size, giving
       ``n_centres × len(windows)`` rows per replicate.
   * - ``step``
     - ``100000``
     - Sliding-window step size (bp) for simulations and the within-locus
       centre grid for VCF.
   * - ``step_vcf``
     - ``10000``
     - Step size (bp) for tiling VCF contigs into genomic windows. Independent
       of ``step``.

.. code-block:: python

    # Multi-scale windows over the default 1.2 Mb locus
    df = fs.summary_statistics(
        "./simulations",
        windows=[50000, 100000, 500000],
        nthreads=8,
    )

    # Larger locus with a finer step
    df = fs.summary_statistics(
        "./simulations",
        locus_length=2000000,
        step=50000,
        windows=[100000],
        nthreads=8,
    )


Recombination-rate stratified normalisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, SNP statistics are normalised within frequency bins only. When
``r_bins`` is provided, normalisation is additionally stratified by
recombination rate — a separate mean and standard deviation is computed for
each recombination-rate stratum defined by the breakpoints. Stratifying jointly
by allele frequency and recombination rate addresses the spurious correlation
of haplotype-based statistics with the local recombination rate
(`Johnson & Voight 2018 <https://doi.org/10.1038/s41559-018-0478-6>`_), making
predictions more robust to recombination-rate discrepancies between the
training and the tested sets.

``r_bins`` accepts a list of breakpoints in cM/Mb; ``n`` breakpoints define
``n + 1`` strata. A recombination map must also be supplied. The same units
apply to both paths — per-base-pair rates from simulations are converted to
cM/Mb internally. The ``r_bins`` column is dropped from the output feature
matrix, so the feature dimension stays unchanged regardless of whether
stratification is used.

Deriving the breakpoints
^^^^^^^^^^^^^^^^^^^^^^^^

Boundaries should be defined from the empirical recombination-rate
distribution of the organism under study. ``flexsweep recombination-bins``
slides the analysis windows across the VCFs, interpolates the local rate from
the recombination map, discards windows below ``--min_rate``, and reports the
quantile breakpoints:

.. code-block:: bash

    flexsweep recombination-bins \
        --vcf_path yri_vcfs \
        --recombination_map decode_sexavg_2019.txt \
        --bins 10 \
        --min_rate 0.01

With ``--bins 10`` the command returns nine decile breakpoints plus the
rounded-up maximum rate.

Human recombination bins
^^^^^^^^^^^^^^^^^^^^^^^^

For human data we derived the boundaries from the `deCODE recombination map
<https://doi.org/10.1126/science.aau1043>`_, interpolating genome-wide 1.2 Mb
regions to obtain 10 recombination bins, filtering regions below 0.01 cM/Mb
before the binning process:

.. code-block:: python

    import flexsweep as fs

    # deCODE-derived boundaries used for the 1000GP populations
    decode_r_bins = [0.37, 0.55, 0.71, 0.87, 1.05, 1.26, 1.53, 1.88, 2.46, 6.1]

    df = fs.summary_statistics(
        "./simulations",
        recombination_map=fs.DECODE_MAP,
        r_bins=decode_r_bins,
        min_rate=0.01,
        nthreads=8,
    )

The identical ``r_bins`` and ``min_rate`` values must be passed to the VCF run,
so that simulated and empirical statistics are standardised within matching
strata:

.. code-block:: python

    df_vcf = fs.summary_statistics(
        "./vcf_data",
        vcf=True,
        recombination_map=fs.DECODE_MAP,
        r_bins=decode_r_bins,
        min_rate=0.01,
        nthreads=8,
    )

.. .. note::

..    ``r_bins`` stratification is not recommended for domain-adaptive (DANN)
..    training — it has been shown to increase domain shift in that context.
..    Use it only with the standard CNN.


Custom CNN
----------

Flex-sweep is now able to work with custom CNN architectures. The API includes a ``CNN`` class able to pre-process the feature vectors while being ready to use with custom CNN implementations. By default ``CNN`` class will work with the default Flex-sweep architecture. Nonetheless, we changed the old 2D CNN behaviour so we now input statistics as channels into the 2D CNN: ``(batch, windows, centers, stats)``. If you are planning to use custom CNN architectures, please be extremely careful, you must pay attention to feature vector reshaping as needed.

.. code-block:: python

    import flexsweep as fs
    from flexsweep.cnn import *

    def cnn_finer(model_input, num_classes=1):
        """
        Changing filter and kernels sizes to look for finer summary statistics dimensions.
        Includes bach normalization, global pooling (no flatten), and dropout.
        """

        initializer = tf.keras.initializers.HeNormal()

        x = tf.keras.layers.Conv2D(
            64, (2, 1), padding="same", kernel_initializer=initializer, name="conv_2x1"
        )(model_input)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            64, (1, 2), padding="same", kernel_initializer=initializer, name="conv_1x2"
        )(model_input)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.Conv2D(
            128, (2, 2), padding="same", kernel_initializer=initializer, name="conv_2x2"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        # Optional deeper conv layer
        x = tf.keras.layers.Conv2D(
            128, (1, 1), padding="same", kernel_initializer=initializer, name="conv_1x1"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        # x = tf.keras.layers.MaxPooling2D(pool_size=(2, 1), name="gentle_pool")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(1, 1), name="gentle_pool")(x)
        x = tf.keras.layers.Dropout(0.15, name="dropout_1")(x)

        # GlobalAveragePooling2D
        x = tf.keras.layers.Flatten()(x)
        # x = self.attention_pool_2d(x, name="attn_pool")

        x = tf.keras.layers.Dense(128, activation="relu", name="dense_1")(x)
        x = tf.keras.layers.Dropout(0.2, name="dropout_2")(x)

        x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(x)
        x = tf.keras.layers.Dropout(0.1, name="dropout_3")(x)

        output = tf.keras.layers.Dense(
            num_classes, activation="sigmoid", name="output"
        )(x)

        return output

    fs_cnn = fs.CNN(
            train_data = "yri_test/fvs.parquet",
            predict_data = "yri_vcf/fvs_yri.parquet",
            output_folder = "yri_vcf",
    )
    fs_cnn.train(cnn = cnn_finer)


Because we're providing new custom feature vectors (e.g, new genomic center and window size ranges), we're also providing an interface to train and predict using a 1D CNN. **When using 1D CNN, you must input your own CNN architecture**. We're providing a 1D CNN example with channel attention (Squeeze-and-Excitation) that learns local patterns across genomic positions and reweights features before classification. You can easily provide your own CNN similar to the example above:

.. code-block:: python

    def cnn_flexsweep_conv1d(model_input, num_classes=1):
        """
        Conv1D over spatial positions (steps) with stats as channels,
        followed by channel-wise (per-stat) attention.
        Expects model_input shape: (batch, positions=105, stats=11)
        """

        x = model_input

        # Conv1D over positions (channels_last): output (batch, positions, filters)
        x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv1D(256, 2, padding="same", activation="relu")(x)

        # Channel attention (Squeeze-and-Excitation)
        se = tf.keras.layers.GlobalAveragePooling1D()(x)
        se = tf.keras.layers.Dense(256, activation="sigmoid")(se)
        se = tf.keras.layers.Reshape((1, 256))(se)
        x  = tf.keras.layers.Multiply()([x, se])

        # Head
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        output = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

        return output

    fs_cnn_1d = CNN(
            train_data = "yri_test/fvs.parquet",
            predict_data = "yri_vcf/fvs_yri.parquet",
            output_folder = "yri_vcf",
    )
    fs_cnn_1d.train(cnn = cnn_flexsweep_conv1d, one_dim = True)


Haplotype sorting
~~~~~~~~~~~~~~~~~

Before feeding a raw haplotype matrix into a custom CNN you may want to
rearrange rows (haplotypes) or columns (SNPs) so that similar haplotypes are
placed adjacently — this can improve the spatial patterns a 2D CNN learns.
All functions below accept a binary ``(samples × sites)`` NumPy array. Please read for further information `Zhao et al. 2023 <https://doi.org/10.1093/bioinformatics/btad265>`_ and `Tran et al. 2025 <https://doi.org/10.1093/molbev/msaf250>`_ for further information.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - ``daf_sorting(matrix)``
     - Sort **columns** (SNPs) by descending derived allele frequency. Most
       common derived alleles appear first.
   * - ``freq_sorting(matrix)``
     - Sort **rows** (haplotypes) by descending number of derived alleles
       (Hamming weight). Most derived haplotypes appear first.
   * - ``corr_sorting(matrix)``
     - Sort **rows** by Pearson correlation coefficient with the most
       correlated haplotype. Groups similar haplotypes together.
   * - ``pcc_column_sort_numba(matrix)``
     - Sort **columns** by total PCC score — SNPs most correlated with the
       rest of the matrix appear first.
   * - ``haplotype_freq_sorting(matrix)``
     - Sort **columns** by haplotype frequency (most common haplotype group
       first). Returns ``(reordered, col_order, groups, freqs)``.
   * - ``haplotype_freq_sorting_hamming(matrix)``
     - Same as above but within each frequency group, haplotypes are
       additionally ordered by Hamming distance to the most frequent
       haplotype. Returns ``(reordered, freqs)``.

``daf_sorting``, ``freq_sorting``, ``corr_sorting``, and
``pcc_column_sort_numba`` are Numba-compiled and operate on integer arrays.
The two ``haplotype_freq_sorting`` variants operate on general NumPy arrays.

.. code-block:: python

    import numpy as np
    from flexsweep.utils import (
        daf_sorting, freq_sorting, haplotype_freq_sorting
    )

    # random hap: binary (n_haplotypes × n_sites) array from your VCF window
    hap = np.random.randint(0, 2, (1000, 216), dtype=np.int32)

    # Sort SNPs by DAF, then haplotypes by frequency
    hap_daf = daf_sorting(hap.copy())
    hap_sorted, col_order, groups, freqs = haplotype_freq_sorting(hap_daf)

    # hap_sorted is now ready for a (batch, haplotypes, sites, 1) CNN input as needed

Demography mis-specification
----------------------------

Flex-sweep is now more versatile to analyse non-model organisms where the quality or availability of simulated parameters, such as the mutation rate, recombination rate, and demography, is limited. We extend the CNN with the Domain Adaptive model proposed by `Mo, Z. and Siepel A. 2023 <https://doi.org/10.1371/journal.pgen.1011032>`_. If you plan to use Flex-sweep DA, please cite `Mo, Z. and Siepel A. 2023 <https://doi.org/10.1371/journal.pgen.1011032>`_. We highly recommend to read deep the paper along with the code `source code <https://github.com/ziyimo/popgen-dom-adapt>`_ provided by the authors.

Flex-sweep-DA trainining takes into account not only labelled simulated data (source domain) as expected for a CNN, but also incorporates empirical unlabelled data (target domain) during the training. The goal then is to generalise the classification task across any feature distorting simulated feature vectors distribution from real data by learning a shared representation that is highly predictive for the CNN classifier but uninformative about the domain (whether the input is simulated or real).

Demography is known to highly shift summary statistics toward values that can mimic sweep signals, even assuming strict neutrality. It's been a matter of debate how realistic demographic, along with BGS, could explain most sweep signals. In the case of ML approaches like CNN, trained models under unrealistic demographies easily confound sweep prediction due to the overfitting of demographic artifacts. The DA model implemented is explicitly designed to account for and mitigate such a mismatch between simulated and real data. Note that when working with extremely out-of-range demographies (e.g, training over constant population sizes) or simulated parameters, DA implementation may still perform worse than the original CNN, so to work safer, the simulations should span plausible demographic scenarios.

Flex-sweep DA will subset the exact same number of ``source_data`` (labelled simulations) from ``target_data`` (empirical data) to balance the discriminator during training. Once the model is trained, the software will use the entire ``target_data`` dataset to make the predictions.

.. code-block:: python

    import flexsweep as fs
    fs_cnn = fs.CNN(
        source_data="yri_test/fvs.parquet",
        target_data="yri_vcfs/fvs_yri.parquet",
        output_folder="yri_vcfs",
    )

    fs_cnn.train_da(ramp_epochs=30,max_lambda=1)
    df_prediction = fs_cnn.predict_da()

