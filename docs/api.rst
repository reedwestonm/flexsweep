=============
API Reference
=============

.. contents:: Contents
   :local:
   :depth: 2

Simulation
----------


.. autoclass:: flexsweep.Simulator
    :members:
    :undoc-members:
    :show-inheritance:


Data I/O
--------

.. autoclass:: flexsweep.data.Data
    :members:
    :undoc-members:
    :show-inheritance:


Feature vectors
---------------

.. autofunction:: flexsweep.fv.summary_statistics
.. autofunction:: flexsweep.fv._process_sims
.. autofunction:: flexsweep.fv._process_vcf
.. autofunction:: flexsweep.fv.calculate_stats_simulations
.. autofunction:: flexsweep.fv.calculate_stats_vcf_flat
.. autofunction:: flexsweep.fv.normalize_neutral
.. autofunction:: flexsweep.fv.normalize_stats
.. autofunction:: flexsweep.fv.normalize_cut_raw
.. autofunction:: flexsweep.fv.resolve_stats


Outlier scan
------------

.. autofunction:: flexsweep.scan.scan
.. autofunction:: flexsweep.scan.available_stats
.. autofunction:: flexsweep.scan.stat_params
.. autofunction:: flexsweep.scan.empirical_pvalues


CNN / DANN
----------

.. autoclass:: flexsweep.cnn.CNN
    :members:
    :undoc-members:
    :show-inheritance:


Enrichment
----------

.. autofunction:: flexsweep.enrichment.run_enrichment
.. autofunction:: flexsweep.enrichment.run_bootstrap_nomatchomega
.. autofunction:: flexsweep.enrichment.estimate_fdr
.. autofunction:: flexsweep.enrichment.count_sweeps_singlepop
.. autofunction:: flexsweep.enrichment.count_sweeps_multipop


Ancestral polarization
----------------------

Flexsweep annotates the ancestral allele in a VCF from a multi-species
alignment (MAF) through a compiled Rust extension, ``flexsweep.polarize``. The
extension is bundled with the published wheels and requires no separate
installation step; a source build compiles it as part of ``uv build``, as
described in :doc:`installation`. The functions below are its Python entry
points, and are also exposed on the command line as ``flexsweep split-maf`` and
``flexsweep polarize``.

**Workflow**:

1. Split the MAF by reference contig and sort it by position (``maf_split``,
   ``maf_sort``).
2. Polarize the VCF using the sorted MAF and one or more outgroup species
   (``polarize``). The extension streams both files together and writes a
   bgzipped VCF with REF/ALT swapped where the outgroup consensus supports the
   alternate allele as ancestral.

**Polarization methods** (``--method``):

- ``parsimony`` — majority vote over outgroup alleles; no model fitting.
- ``jc`` — Jukes–Cantor substitution model.
- ``kimura`` — Kimura two-parameter model (default).
- ``r6`` — General reversible model (6 free parameters); slowest, most accurate.

.. autofunction:: flexsweep.polarize.maf_sort
.. autofunction:: flexsweep.polarize.maf_split
.. autofunction:: flexsweep.polarize.polarize


Visualization
-------------

.. autofunction:: flexsweep.utils.plot_scan
.. autofunction:: flexsweep.utils.plot_manhattan
.. autofunction:: flexsweep.utils.plot_diversity
.. autofunction:: flexsweep.utils.plot_sweep_density
.. autofunction:: flexsweep.utils.plot_fv_pca
.. autofunction:: flexsweep.utils.plot_stat_distributions
.. autofunction:: flexsweep.utils.rank_probabilities


Population genetics statistics
-------------------------------

These are the low-level functions called internally by the feature-vector
pipeline and the scan module. They can also be called directly on numpy arrays.

Site Frequency Spectrum (SFS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.fv.neutrality_stats
.. autofunction:: flexsweep.fv.theta_watterson
.. autofunction:: flexsweep.fv.sfs_nb
.. autofunction:: flexsweep.fv.theta_pi
.. autofunction:: flexsweep.fv.tajima_d
.. autofunction:: flexsweep.fv.achaz_y
.. autofunction:: flexsweep.fv.achaz_y_star
.. autofunction:: flexsweep.fv.achaz_t
.. autofunction:: flexsweep.fv.fay_wu_h_norm
.. autofunction:: flexsweep.fv.zeng_e
.. autofunction:: flexsweep.fv.fuli_f_star
.. autofunction:: flexsweep.fv.fuli_f
.. autofunction:: flexsweep.fv.fuli_d_star
.. autofunction:: flexsweep.fv.fuli_d


Haplotype-based
~~~~~~~~~~~~~~~
.. autofunction:: flexsweep.fv.ihs_ihh
.. autofunction:: flexsweep.fv.haf_top
.. autofunction:: flexsweep.fv.garud_h
.. autofunction:: flexsweep.fv.h12_enard
.. autofunction:: flexsweep.fv.hscan
.. autofunction:: flexsweep.fv.run_isafe
.. autofunction:: flexsweep.fv.isafe

Derived-background diversity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.fv.dind_high_low
.. autofunction:: flexsweep.fv.s_ratio
.. autofunction:: flexsweep.fv.hapdaf_o
.. autofunction:: flexsweep.fv.hapdaf_s
.. autofunction:: flexsweep.fv.fast_sq_freq_pairs
.. autofunction:: flexsweep.fv.dind_high_low_from_pairs
.. autofunction:: flexsweep.fv.s_ratio_from_pairs
.. autofunction:: flexsweep.fv.hapdaf_from_pairs

Linkage disequilibrium
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.fv.Ld
.. autofunction:: flexsweep.fv.r2
.. autofunction:: flexsweep.fv.compute_r2_matrix_upper
.. autofunction:: flexsweep.fv.omega_linear_correct

Composite sweep statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: flexsweep.fv.LASSI_spectrum_and_Kspectrum
.. autofunction:: flexsweep.fv.T_m_statistic_fast
.. autofunction:: flexsweep.fv.compute_t_m
.. autofunction:: flexsweep.fv.mu_stat
.. autofunction:: flexsweep.fv.run_raisd

Balancing selection
~~~~~~~~~~~~~~~~~~~
.. autofunction:: flexsweep.fv.ncd1
.. autofunction:: flexsweep.fv.run_beta_window
