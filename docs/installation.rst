
Installation
============

Flex-sweep is distributed as a Python package with a Command Line Interface,
avoiding non-standard input files and the manual installation of external
dependencies. The `discoal <https://github.com/kr-colab/discoal>`_ binary
is bundled with the package, and the software compiles it from source on first
use if no executable binary is found, so no external simulator installation is
required.

Flex-sweep requires Python ``>=3.12,<3.13``. We recommend installing it into a
virtual environment to improve reproducibility.

pip
---

.. code-block:: bash

    pip install flexsweep

On some systems you will need to use ``python3 -m pip`` rather than ``pip``.
This installs the package into your local user Python packages; if the
``flexsweep`` command is not found afterwards, the Python scripts directory is
not on your ``PATH``.

uv
--

`uv <https://docs.astral.sh/uv/>`_ resolves and installs the dependency tree
considerably faster than pip, and can manage the virtual environment for you:

.. code-block:: bash

    uv venv --python 3.12
    source .venv/bin/activate
    uv pip install flexsweep

To install the CLI as a standalone tool, without adding Flex-sweep to a
project environment:

.. code-block:: bash

    uv tool install flexsweep

Building from source
--------------------

Building from source is only necessary to run an unreleased revision, or to
build for a platform without a published wheel. The published wheels already
contain the compiled extension described below.

Flex-sweep ships the ancestral-state polarization module
(``flexsweep.polarize``) as a compiled Rust extension, refactored and extended
from `est-sfs <https://doi.org/10.1534/genetics.118.301120>`_. The Rust sources
are bound to Python with `maturin <https://www.maturin.rs/>`_, which is
resolved automatically as a build dependency. The Rust toolchain itself is not,
so a source build additionally requires ``cargo`` and ``rustc``:

.. code-block:: bash

    # Rust toolchain (skip if `cargo --version` already works)
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

    # uv (skip if `uv --version` already works)
    curl -LsSf https://astral.sh/uv/install.sh | sh

Clone the repository and build the wheel:

.. code-block:: bash

    git clone https://github.com/jmurga/flexsweep.git
    cd flexsweep
    uv build

``uv build`` compiles the Rust extension and writes both a wheel and an sdist
to ``dist/``. Install the wheel into the environment of your choice:

.. code-block:: bash

    uv venv --python 3.12
    source .venv/bin/activate
    uv pip install dist/flexsweep-*.whl

The same wheel can be installed with pip if you prefer:

.. code-block:: bash

    pip install dist/flexsweep-*.whl

.. note::

   ``uv build`` tags the wheel for the glibc of the machine that built it, which
   is what a local installation needs. Wheels meant for redistribution must
   carry a ``manylinux`` tag instead, and therefore must be linked against an
   older glibc than the build host provides. Flex-sweep does this with
   `ziglang <https://pypi.org/project/ziglang/>`_, a package that vendors the
   Zig compiler and is declared as a build dependency, so it is resolved
   automatically rather than installed by hand; ``cargo`` still compiles the
   extension and Zig only acts as the cross-linker. The published wheels are
   built with:

   .. code-block:: bash

       uv run --with 'maturin>=1.5,<2.0' --with ziglang \
           maturin build --release --zig --compatibility manylinux_2_28 --out dist

   The target platform and the Zig linker are already configured under
   ``[tool.maturin]`` in ``pyproject.toml``, but ``uv build`` overrides them, so
   redistributable wheels must be built by calling maturin directly.

Testing the installation
------------------------

You can easily test flexsweep by running the CLI in your terminal

.. code-block:: bash

    flexsweep --help

.. code-block:: console

    Usage: flexsweep [OPTIONS] COMMAND [ARGS]...

      CLI for Simulator and CNN.

    Options:
      --help  Show this message and exit.

    Commands:
      cnn                 Run the Flex-sweep CNN for training or prediction.
      dann                Run the Flex-sweep DANN for training or prediction.
      enrichment          Gene-set sweep enrichment and FDR analysis.
      fvs-discoal         Estimate summary statistics from discoal...
      fvs-vcf             Estimate summary statistics from VCF files and...
      polarize            Polarize VCF using rust est-sfs refactor.
      rank                Rank genomic features by their maximum nearby sweep...
      recombination-bins  Output recombination bins from empirical...
      scan                Standalone outlier scan from a directory of...
      scan-plot           Visualize outlier scan results.
      simulator           Run the discoal simulator with user-specified...
      split-maf           Split a MAF by reference contig and sort it by...

You can also try to import the package into your python enviroment

.. code-block:: bash

    python -c "import flexsweep"

Each command is documented in its own section: :doc:`basic_usage` for the
simulation, feature-vector and training workflow, :doc:`scan` for the
standalone outlier scan, :doc:`polarize` for ancestral-state polarization, and
:doc:`enrichment` for the gene-set enrichment pipeline.
