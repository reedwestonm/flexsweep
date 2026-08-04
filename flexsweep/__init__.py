import importlib
import os
import warnings
from typing import TYPE_CHECKING

# Suppress polars warnings. Force 1 thread polars shouldn't cause deadlocks
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="joblib.externals.loky.backend.fork_exec"
)

# Force libraries to single thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
os.environ["POLARS_MAX_THREADS"] = "1"
# os.environ["JOBLIB_TEMP_FOLDER"] = "/labstorage/jmurgamoreno/"
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

_CONFIGURED = False


def _configure_runtime():
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already set; keep current behavior.
        pass

    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(1)
    except Exception:
        # If threadpoolctl isn't available, just continue.
        pass


# Version
try:
    from . import _version

    __version__ = _version.version
except ImportError:
    __version__ = "2.0"


# Lazy access to cnn module
# Avoid loading tensorflow till the fs.CNN is call
class _LazyModule:
    """Proxy for a submodule; loads on first real use."""

    __slots__ = ("_fqname", "_pkg", "_mod")

    def __init__(self, fqname: str, pkg: str):
        self._fqname = fqname
        self._pkg = pkg
        self._mod = None  # real module once loaded

    def _load(self):
        if self._mod is None:
            self._mod = importlib.import_module(self._fqname, self._pkg)
        return self._mod

    def __getattr__(self, name):
        return getattr(self._load(), name)

    def __dir__(self):
        # Don’t trigger import during tab; show minimal names
        return [] if self._mod is None else dir(self._mod)

    def __repr__(self):
        suffix = "unloaded" if self._mod is None else "loaded"
        return f"<lazy module {self._fqname!r} ({suffix})>"


class _LazyAttr:
    """Proxy for an attribute in a (lazy) module; loads on first real use."""

    __slots__ = ("_mod_proxy", "_attr")

    def __init__(self, mod_proxy: _LazyModule, attr: str):
        self._mod_proxy = mod_proxy
        self._attr = attr

    def _target(self):
        return getattr(self._mod_proxy._load(), self._attr)

    def __call__(self, *a, **kw):
        return self._target()(*a, **kw)

    def __getattr__(self, name):
        return getattr(self._target(), name)

    def __repr__(self):
        return f"<lazy attr {self._mod_proxy._fqname}.{self._attr} (unloaded)>"


_cnn_module_proxy = _LazyModule(".cnn", __name__)
cnn = _cnn_module_proxy
CNN = _LazyAttr(_cnn_module_proxy, "CNN")


_LAZY_ATTRS = {
    # Heavy numeric libs
    "np": ("numpy", None),
    "pl": ("polars", None),
    "Parallel": ("joblib", "Parallel"),
    "delayed": ("joblib", "delayed"),
    # Project modules
    "fv": (".fv", None),
    "cnn_regressor": (".cnn_regressor", None),
    "CNNRegressor": (".cnn_regressor", "CNNRegressor"),
    "enrichment": (".enrichment", None),
    "run_enrichment": (".enrichment", "run_enrichment"),
    "scan": (".scan", None),
    "npe": (".npe", None),
    "fit_npe": (".npe", "fit_npe"),
    "predict_npe": (".npe", "predict_npe"),
    "analyze_npe": (".npe", "analyze_npe"),
    "plot_evaluation": (".npe", "plot_evaluation"),
    "polarize": (".polarize", None),
    "simulate_discoal": (".simulate_discoal", None),
    "utils": (".utils", None),
    "Data": (".data", "Data"),
    "summary_statistics": (".fv", "summary_statistics"),
    "Simulator": (".simulate_discoal", "Simulator"),
    "DISCOAL": (".simulate_discoal", "DISCOAL"),
    "DECODE_MAP": (".simulate_discoal", "DECODE_MAP"),
    "DEMES_EXAMPLES": (".simulate_discoal", "DEMES_EXAMPLES"),
    "rank_probabilities": (".utils", "rank_probabilities"),
    "plot_sfs": (".utils", "plot_sfs"),
    "plot_diversity": (".utils", "plot_diversity"),
    "threadpoolctl": ("threadpoolctl", None),
    "multiprocessing": ("multiprocessing", None),
}


def __getattr__(name):
    if name in _LAZY_ATTRS:
        _configure_runtime()
        module_name, attr_name = _LAZY_ATTRS[name]
        if module_name.startswith("."):
            module = importlib.import_module(module_name, __name__)
        else:
            module = importlib.import_module(module_name)
        value = module if attr_name is None else getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# What the package exports (also helps tab completion)
__all__ = [
    "utils",
    "Simulator",
    "DISCOAL",
    "DECODE_MAP",
    "DEMES_EXAMPLES",
    "summary_statistics",
    "Data",
    "np",
    "pl",
    "Parallel",
    "delayed",
    "os",
    "warnings",
    "importlib",
    "threadpoolctl",
    "multiprocessing",
    "cnn",
    "CNN",
    "cnn_regressor",
    "CNNRegressor",
    "rank_probabilities",
    "plot_sfs",
    "plot_diversity",
    "fv",
    "npe",
    "fit_npe",
    "predict_npe",
    "analyze_npe",
    "plot_evaluation",
    "polarize",
    "simulate_discoal",
    "enrichment",
    "__version__",
]


def __dir__():
    return sorted(set(list(globals().keys()) + list(__all__) + ["cnn", "CNN"]))


# if TYPE_CHECKING:
#     from .cnn import CNN as _CNNType
#     from .fv import summary_statistics as _summary_statistics


if TYPE_CHECKING:
    from .cnn import CNN as CNN
    from .cnn_regressor import CNNRegressor as CNNRegressor
    from .fv import summary_statistics as summary_statistics
else:
    _cnn_module_proxy = _LazyModule(".cnn", __name__)
    cnn = _cnn_module_proxy
    CNN = _LazyAttr(_cnn_module_proxy, "CNN")
