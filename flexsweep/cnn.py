import importlib
import json
import math
import os
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.special import expit, logit
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# from tensorflow.keras import layers
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.utils import Sequence

from . import np, pl


@register_keras_serializable(package="fs", name="masked_bce")
def masked_bce(y_true, y_pred):
    y_pred = tf.boolean_mask(
        y_pred, tf.not_equal(y_true, -1)
    )  # -1 will be masked/ y_true or y_pred?
    y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))

    return tf.keras.losses.binary_crossentropy(y_true, y_pred)


@register_keras_serializable(package="fs", name="masked_binary_accuracy")
def masked_binary_accuracy(y_true, y_pred):
    y_pred = tf.boolean_mask(y_pred, tf.not_equal(y_true, -1))
    y_true = tf.boolean_mask(y_true, tf.not_equal(y_true, -1))
    return tf.keras.metrics.binary_accuracy(y_true, y_pred)


@register_keras_serializable(package="fs", name="GradReverse")
class GradReverse(tf.keras.layers.Layer):
    """
    Gradient Reversal Layer (GRL) with tunable strength ``λ``.

    Forward pass: identity (returns the input unchanged).
    Backward pass: multiplies the incoming gradient by ``-λ``, which
    *reverses* (and scales) gradients flowing into the shared feature extractor.
    This encourages the extractor to learn **domain-invariant** features when
    the GRL feeds a domain classifier.

    Parameters
    ----------
    lambd : float, default=0.0
        Initial GRL strength ``λ``. The effective gradient multiplier is ``-λ``.
        Can be updated during training (e.g., via :class:`GRLRamp`).
    **kw : Any
        Passed to :class:`tf.keras.layers.Layer`.

    Attributes
    ----------
    lambd : tf.Variable
        Non-trainable scalar variable storing the current ``λ`` value. It can be
        modified by callbacks to schedule warm-up or annealing.

    Notes
    -----
    - Serialization: the layer is Keras-serializable and preserves the initial
      ``λ`` in configs. At runtime, the **variable** value may be updated.
    - Typical schedules **warm up** ``λ`` from 0 → 0.4–1.0 over several epochs.

    References
    ----------
    Ganin & Lempitsky (2015), "Unsupervised Domain Adaptation by
    Backpropagation" (DANN/GRL).
    """

    @staticmethod
    @tf.custom_gradient
    def _grl_with_lambda(x, lambd):
        y = tf.identity(x)

        def grad(dy):
            # grad wrt x is -λ * dy; no grad wrt λ
            return -lambd * dy, tf.zeros_like(lambd)

        return y, grad

    def __init__(self, lambd=0.0, **kw):
        super().__init__(**kw)
        # Keep JSON-safe init value for serialization
        self._lambd_init = float(lambd)
        # Non-trainable so you can control it via callback
        self.lambd = tf.Variable(
            self._lambd_init, trainable=False, dtype=tf.float32, name="grl_lambda"
        )

    def call(self, x):
        # Use the staticmethod custom op
        return GradReverse._grl_with_lambda(x, self.lambd)

    # ---- Keras serialization ----
    def get_config(self):
        cfg = super().get_config()
        cfg.update({"lambd": float(self._lambd_init)})
        return cfg

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)


class GRLRamp(tf.keras.callbacks.Callback):
    """
    Linear warm-up schedule for GRL strength ``λ``.

    Increases the GRL factor linearly from 0 to ``max_lambda`` over
    ``epochs`` calls to :meth:`on_epoch_begin`. After warm-up, ``λ`` is held
    constant at ``max_lambda``.

    Parameters
    ----------
    grl_layer : GradReverse
        The GRL layer instance whose ``lambd`` variable will be updated.
    max_lambda : float, default=0.5
        Target value for ``λ`` at the end of the warm-up.
    epochs : int, default=50
        Number of warm-up epochs. If total training epochs exceed this value,
        ``λ`` remains fixed thereafter.

    Notes
    -----
    - Warm-up helps stabilize training by letting the classifier learn a useful
      decision surface **before** strong domain-adversarial pressure is applied.
    - Consider tuning ``max_lambda`` and warm-up length based on how quickly the
      domain accuracy approaches ~0.5 (a sign of domain invariance).
    """

    def __init__(self, grl_layer, max_lambda=0.5, epochs=50):
        """
        epochs = number of ramp epochs (not total training epochs).
        After this many epochs, λ will be held at max_lambda.
        """
        super().__init__()
        self.grl_layer = grl_layer
        self.max_lambda = float(max_lambda)
        self.ramp_epochs = int(max(1, epochs))

    def on_epoch_begin(self, epoch, logs=None):
        # linear warmup 0 → max_lambda over `ramp_epochs`, then hold
        if epoch < self.ramp_epochs:
            t = epoch / max(1, self.ramp_epochs - 1)
            lam = self.max_lambda * t
        else:
            lam = self.max_lambda
        self.grl_layer.lambd.assign(lam)


class LogGRLLambda(tf.keras.callbacks.Callback):
    def __init__(self, grl_layer, key="grl_lambda"):
        super().__init__()
        self.grl = grl_layer
        self.key = key

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs[self.key] = float(self.grl.lambd.numpy())


class LossWeightsScheduler(tf.keras.callbacks.Callback):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def on_epoch_end(self, epoch, logs={}):
        gamma = 10  # 10, 5
        p = epoch / 30
        lambda_new = 2 / (1 + math.exp(-gamma * p)) - 1
        K.set_value(self.beta, lambda_new)


class LossWeightsLogger(tf.keras.callbacks.Callback):
    def __init__(self, loss_weights):
        super().__init__()
        self.loss_weights = loss_weights  # e.g., [alpha, beta]

    def on_epoch_end(self, epoch, logs=None):
        aw = float(K.get_value(self.loss_weights[0]))
        bw = float(K.get_value(self.loss_weights[1]))
        print(f"Loss Weights @ epoch {epoch + 1}: alpha={aw:.4f}, beta={bw:.4f}")
        if logs is not None:
            logs["alpha"] = aw
            logs["beta"] = bw


class CNN:
    """
    Class to build and train a Convolutional Neural Network (CNN) for Flex-sweep.
    It loads/reshapes Flex-sweep feature vectors, trains, evaluates and predicts, including
    domain-adaptation extension.

    Attributes
    ----------
    train_data : str | pl.DataFrame | None
        Path to training parquet/CSV (or a Polars DataFrame).
    source_data : str | None
        Path to *source* (labeled) parquet for domain adaptation.
    target_data : str | None
        Path to *target/empirical* parquet for domain adaptation (unlabeled).
    predict_data : str | pl.DataFrame | None
        Path/DataFrame with samples to predict (standard supervised path).
    valid_data : Any
        (Reserved) Optional separate validation set path/DF (unused).
    output_folder : str | None
        Directory where models, figures and predictions are written.
    normalize : bool
        If True, apply a Keras `Normalization` layer (fit on train only).
    model : tf.keras.Model | str | None
        A compiled Keras model or a path to a saved model.
    num_stats : int
        Number of per-window statistics used as channels. Default 11.
    center : np.ndarray[int]
        Center coordinates (bp) used to index columns; defaults to 500k..700k step 10k.
    windows : np.ndarray[int]
        Window sizes used to index columns; default [50k, 100k, 200k, 500k, 1M].
    train_split : float
        Fraction of data used for training (rest split equally into val/test).
    gpu : bool
        If False, disable CUDA via `CUDA_VISIBLE_DEVICES=-1`.
    tf : module | None
        TensorFlow module, set by :meth:`check_tf`.
    history : pl.DataFrame | None
        Training history after :meth:`train` / :meth:`train_da`.
    prediction : pl.DataFrame | None
        Latest prediction table produced by :meth:`train` or :meth:`predict*`.
    """

    def __init__(
        self,
        train_data=None,
        source_data=None,
        target_data=None,
        predict_data=None,
        valid_data=None,
        output_folder=None,
        normalize=False,
        model=None,
        num_stats=24,
        center=[0, 1.2e6],
        step=1e5,
        windows=np.array([100000]),
    ):
        """
        Initialize a CNN runner.

        Parameters
        ----------
        train_data : str | pl.DataFrame | None
            Path to training data (`.parquet`, `.csv[.gz]`) or Polars DataFrame.
        source_data : str | None
            Path to labeled source parquet for domain adaptation.
        target_data : str | None
            Path to unlabeled empirical/target parquet for domain adaptation.
        predict_data : str | pl.DataFrame | None
            Path/DataFrame for inference in :meth:`predict`.
        valid_data : Any, optional
            Reserved for a future explicit validation split (unused).
        output_folder : str | None
            Output directory for artifacts (models, plots, CSVs).
        normalize : bool, default=False
            If True, fit a `Normalization` layer on training features.
        model : tf.keras.Model | str | None
            Prebuilt Keras model or path to a saved model.

        Notes
        -----
        Defaults assume 11 statistics × 5 windows × 21 centers
        organized in column names like: ``{stat}_{window}_{center}``.
        """
        # self.sweep_data = sweep_data
        self.normalize = normalize
        self.train_data = train_data
        self.predict_data = predict_data
        self.test_train_data = None
        self.output_folder = output_folder
        self.output_prediction = "predictions.txt"
        self.num_stats = 24
        self.center = np.arange(center[0] + step // 2, center[1], step)
        self.windows = np.asarray(windows)
        self.step = step
        self.train_split = 0.8
        self.prediction = None
        self.history = None
        self.model = model
        self.gpu = True
        self.tf = None
        self.source_data = source_data
        self.target_data = target_data
        self.mean = None
        self.std = None
        self.scores = None

    def check_tf(self):
        """
        Import TensorFlow (optionally forcing CPU).

        Returns
        -------
        module
            Imported ``tensorflow`` module.

        Notes
        -----
        If ``self.gpu`` is ``False``, the environment variable
        ``CUDA_VISIBLE_DEVICES`` is set to ``-1`` **before** importing TF.
        """
        if self.gpu is False:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf = importlib.import_module("tensorflow")
        return tf

    def preprocess(self, x, y=None, training=False, epsilon=1e-7):
        x = tf.cast(x, tf.float32)

        # mean = tf.cast(self.mean, tf.float32)
        # std = tf.cast(self.std, tf.float32)

        # mean = tf.reshape(mean, (self.num_stats, 1, 1))
        # std = tf.reshape(std, (self.num_stats, 1, 1))

        # Feature-wise normalization using training mean/std
        x = (x - self.mean) / (self.std + epsilon)
        # x = (x - mean) / (std + epsilon)

        if training:
            # # # Optional: small Gaussian noise (try stddev ~0.01-0.05)
            # x = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02, dtype=x.dtype)

            # # Optional: channel/stat dropout (drops whole stats)
            # keep_prob = 0.90
            # if x.shape.rank == 3:  # (S, W*C, 1)
            #     mask = tf.cast(
            #         tf.random.uniform((self.num_stats, 1, 1)) < keep_prob, x.dtype
            #     )
            # else:  # (B, S, W*C, 1)
            #     mask = tf.cast(
            #         tf.random.uniform((1, self.num_stats, 1, 1)) < keep_prob, x.dtype
            #     )
            # x = x * mask / keep_prob

            # Horizontal flip augmentation
            x = tf.image.random_flip_left_right(x)

        if y is not None:
            return x, y
        else:
            return x

    def cnn_flexsweep_feature(self, model_input):
        """
        Shared 2D-CNN feature extractor (three branches).

        Parameters
        ----------
        model_input : tf.keras.layers.Input
            Input tensor of shape ``(W, C, S)``.

        Returns
        -------
        tf.Tensor
            Flattened concatenated features from the three branches.

        See Also
        --------
        cnn_flexsweep : Similar branch structure followed by classification head.
        """
        tf = self.check_tf()

        He = tf.keras.initializers.HeNormal()

        # --- Branch 1: 3x3 convs ---
        b1 = tf.keras.layers.Conv2D(
            64, 3, padding="same", kernel_initializer=He, name="fx_b1_c1"
        )(model_input)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.Conv2D(
            128, 3, padding="same", kernel_initializer=He, name="fx_b1_c2"
        )(b1)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.Conv2D(
            256, 3, padding="same", kernel_initializer=He, name="fx_b1_c3"
        )(b1)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.MaxPooling2D(
            pool_size=3, padding="same", name="fx_b1_pool"
        )(b1)
        b1 = tf.keras.layers.Dropout(0.15, name="fx_b1_drop")(b1)
        b1 = tf.keras.layers.Flatten(name="fx_b1_flat")(b1)

        # --- Branch 2: 2x2 convs with dilation (1,3) ---
        b2 = tf.keras.layers.Conv2D(
            64,
            2,
            dilation_rate=[1, 3],
            padding="same",
            kernel_initializer=He,
            name="fx_b2_c1",
        )(model_input)
        b2 = tf.keras.layers.ReLU()(b2)
        b2 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 3],
            padding="same",
            kernel_initializer=He,
            name="fx_b2_c2",
        )(b2)
        b2 = tf.keras.layers.ReLU()(b2)
        b2 = tf.keras.layers.Conv2D(
            256,
            2,
            dilation_rate=[1, 3],
            padding="same",
            kernel_initializer=He,
            name="fx_b2_c3",
        )(b2)
        b2 = tf.keras.layers.ReLU()(b2)
        b2 = tf.keras.layers.MaxPooling2D(pool_size=2, name="fx_b2_pool")(b2)
        b2 = tf.keras.layers.Dropout(0.15, name="fx_b2_drop")(b2)
        b2 = tf.keras.layers.Flatten(name="fx_b2_flat")(b2)

        # --- Branch 3: 2x2 convs with dilation (5,1) then (1,5) ---
        b3 = tf.keras.layers.Conv2D(
            64,
            2,
            dilation_rate=[5, 1],
            padding="same",
            kernel_initializer=He,
            name="fx_b3_c1",
        )(model_input)
        b3 = tf.keras.layers.ReLU()(b3)
        b3 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 5],
            padding="same",
            kernel_initializer=He,
            name="fx_b3_c2",
        )(b3)
        b3 = tf.keras.layers.ReLU()(b3)
        b3 = tf.keras.layers.Conv2D(
            256,
            2,
            dilation_rate=[1, 5],
            padding="same",
            kernel_initializer=He,
            name="fx_b3_c3",
        )(b3)
        b3 = tf.keras.layers.ReLU()(b3)
        b3 = tf.keras.layers.MaxPooling2D(pool_size=2, name="fx_b3_pool")(b3)
        b3 = tf.keras.layers.Dropout(0.15, name="fx_b3_drop")(b3)
        b3 = tf.keras.layers.Flatten(name="fx_b3_flat")(b3)

        feat = tf.keras.layers.Concatenate(name="fx_concat")([b1, b2, b3])

        h = tf.keras.layers.Dense(128, activation="relu")(feat)
        h = tf.keras.layers.Dropout(0.20)(h)
        h = tf.keras.layers.Dense(32, activation="relu")(h)
        h = tf.keras.layers.Dropout(0.10)(h)
        out_cls = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(h)

        return out_cls

    def cnn_flexsweep(self, model_input, num_classes=1):
        """
        Flex-sweep CNN architecture with multiple convolutional and pooling layers.

        Args:
            input_shape (tuple): Shape of the input data, e.g., (224, 224, 3). Default Flex-sweep input statistics, windows and centers
            num_classes (int): Number of output classes in the classification problem. Default: Flex-sweep binary classification

        Returns:
            Model: A Keras model instance representing the Flex-sweep CNN architecture.
        """
        tf = self.check_tf()
        # 3x3 layer
        layer1 = tf.keras.layers.Conv2D(
            64,
            3,
            padding="same",
            name="convlayer1_1",
            kernel_initializer="glorot_uniform",
        )(model_input)
        layer1 = tf.keras.layers.ReLU(negative_slope=0)(layer1)
        layer1 = tf.keras.layers.Conv2D(
            128,
            3,
            padding="same",
            name="convlayer1_2",
            kernel_initializer="glorot_uniform",
        )(layer1)
        layer1 = tf.keras.layers.ReLU(negative_slope=0)(layer1)
        layer1 = tf.keras.layers.Conv2D(256, 3, padding="same", name="convlayer1_3")(
            layer1
        )
        layer1 = tf.keras.layers.ReLU(negative_slope=0)(layer1)
        layer1 = tf.keras.layers.MaxPooling2D(
            pool_size=3, name="poollayer1", padding="same"
        )(layer1)
        layer1 = tf.keras.layers.Dropout(0.15, name="droplayer1")(layer1)
        layer1 = tf.keras.layers.Flatten(name="flatlayer1")(layer1)

        # 2x2 layer with 1x3 dilation
        layer2 = tf.keras.layers.Conv2D(
            64,
            2,
            dilation_rate=[1, 3],
            padding="same",
            name="convlayer2_1",
            kernel_initializer="glorot_uniform",
        )(model_input)
        layer2 = tf.keras.layers.ReLU(negative_slope=0)(layer2)
        layer2 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 3],
            padding="same",
            name="convlayer2_2",
            kernel_initializer="glorot_uniform",
        )(layer2)
        layer2 = tf.keras.layers.ReLU(negative_slope=0)(layer2)
        layer2 = tf.keras.layers.Conv2D(
            256, 2, dilation_rate=[1, 3], padding="same", name="convlayer2_3"
        )(layer2)
        layer2 = tf.keras.layers.ReLU(negative_slope=0)(layer2)
        layer2 = tf.keras.layers.MaxPooling2D(pool_size=2, name="poollayer2")(layer2)
        layer2 = tf.keras.layers.Dropout(0.15, name="droplayer2")(layer2)
        layer2 = tf.keras.layers.Flatten(name="flatlayer2")(layer2)

        # 2x2 with 1x5 dilation
        layer3 = tf.keras.layers.Conv2D(
            64,
            2,
            dilation_rate=[1, 5],
            padding="same",
            name="convlayer4_1",
            kernel_initializer="glorot_uniform",
        )(model_input)
        layer3 = tf.keras.layers.ReLU(negative_slope=0)(layer3)
        layer3 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 5],
            padding="same",
            name="convlayer4_2",
            kernel_initializer="glorot_uniform",
        )(layer3)
        layer3 = tf.keras.layers.ReLU(negative_slope=0)(layer3)
        layer3 = tf.keras.layers.Conv2D(
            256, 2, dilation_rate=[1, 5], padding="same", name="convlayer4_3"
        )(layer3)
        layer3 = tf.keras.layers.ReLU(negative_slope=0)(layer3)
        layer3 = tf.keras.layers.MaxPooling2D(pool_size=2, name="poollayer3")(layer3)
        layer3 = tf.keras.layers.Dropout(0.15, name="droplayer3")(layer3)
        layer3 = tf.keras.layers.Flatten(name="flatlayer3")(layer3)

        # concatenate convolution layers
        concat = tf.keras.layers.concatenate([layer1, layer2, layer3])
        concat = tf.keras.layers.Dense(512, name="512dense", activation="relu")(concat)
        concat = tf.keras.layers.Dropout(0.2, name="dropconcat1")(concat)
        concat = tf.keras.layers.Dense(128, name="last_dense", activation="relu")(
            concat
        )
        concat = tf.keras.layers.Dropout(0.2 / 2, name="dropconcat2")(concat)
        output = tf.keras.layers.Dense(
            num_classes,
            name="out_dense",
            activation="sigmoid",
            kernel_initializer="glorot_uniform",
        )(concat)

        return output

    def load_training_data(self, _stats=None, w=None, n=None, one_dim=False):
        """
        Load and reshape training/validation/test tensors from table-format features.

        Parameters
        ----------
        _stats : list[str] | None
            List of statistic base names to include (e.g., ``["ihs","nsl",...]``).
            If None, you must pass an explicit list later in :meth:`train`.
        w : int | list[int] | None
            Restrict to specific window sizes (e.g., 100000 or [50000,100000]).
            Columns are selected by regex suffix ``_{window}``.
        n : int | None
            Optional number of rows to sample from parquet.
        one_dim : bool, default=False
            If True, flatten spatial grid to ``(W*C, S)`` for 1D models.

        Returns
        -------
        tuple
            ``(X_train, X_test, Y_train, Y_test, X_valid, Y_valid)`` with shapes:

            - if ``one_dim`` is False:
              ``X_*`` → ``(N, W, C, S)``, labels are 0/1.
            - if ``one_dim`` is True:
              ``X_*`` → ``(N, W*C, S)``.

        Raises
        ------
        AssertionError
            If ``train_data`` is missing or has an unsupported extension.

        Notes
        -----
        Any ``model`` value not equal to ``"neutral"`` is coerced to ``"sweep"``.
        """

        assert self.train_data is not None, "Please input training data"

        assert (
            "txt" in self.train_data
            or "csv" in self.train_data
            or self.train_data.endswith(".parquet")
        ), "Please save your dataframe as CSV or parquet"

        if isinstance(self.train_data, pl.DataFrame):
            pass
        elif self.train_data.endswith(".gz"):
            tmp = pl.read_csv(self.train_data, separator=",")
        elif self.train_data.endswith(".parquet"):
            tmp = pl.read_parquet(self.train_data)
            if n is not None:
                tmp = tmp.sample(n)

        # tmp = tmp.with_columns(
        #     pl.when(pl.col("model") != "neutral")
        #     .then(pl.lit("sweep"))
        #     .otherwise(pl.lit("neutral"))
        #     .alias("model")
        # )

        if w is not None:
            try:
                self.center = np.array([int(w)])
                tmp = tmp.select(
                    "iter", "s", "t", "f_i", "f_t", "model", f"^*._{int(w)}$"
                )
            except Exception:
                self.center = np.sort(np.array(w).astype(int))
                _tmp = []
                _h = tmp.select("iter", "s", "t", "f_i", "f_t", "model")
                for window in self.center:
                    _tmp.append(tmp.select(f"^*._{int(window)}$"))
                tmp = pl.concat(_tmp, how="horizontal")
                tmp = pl.concat([_h, tmp], how="horizontal")

        # sweep_parameters = tmp.filter("model" != "neutral").select(tmp.columns[:7])

        stats = []

        if _stats is not None:
            stats = stats + _stats

        train_stats = []
        for i in stats:
            train_stats.append(tmp.select(pl.col(f"^{i}_[0-9]+_[0-9]+$")))

        train_stats = pl.concat(train_stats, how="horizontal")
        train_stats = pl.concat(
            [
                tmp.select("model", "iter", "s", "f_i", "f_t", "t", "mu", "r"),
                train_stats,
            ],
            how="horizontal",
        )

        y = train_stats.select(
            ((~pl.col("model").str.contains("neutral")).cast(pl.Int8)).alias(
                "neutral_flag"
            )
        )["neutral_flag"].to_numpy()

        test_split = round(1 - self.train_split, 2)

        (
            X_train,
            X_test,
            Y_train,
            y_test,
        ) = train_test_split(train_stats, y, test_size=test_split, shuffle=True)

        X_train = (
            X_train.select(X_train.columns[8:])
            .to_numpy()
            .reshape(
                X_train.shape[0],
                self.num_stats,
                self.windows.size * self.center.size,
                1,
            )
        )

        X_valid, X_test, Y_valid, Y_test = train_test_split(
            X_test, y_test, test_size=0.5
        )

        X_test_params = X_test.select(X_test.columns[:6])
        X_test = (
            X_test.select(train_stats.columns[8:])
            .to_numpy()
            .reshape(
                X_test.shape[0],
                self.num_stats,
                self.windows.size * self.center.size,
                1,
            )
        )
        X_valid = (
            X_valid.select(train_stats.columns[8:])
            .to_numpy()
            .reshape(
                X_valid.shape[0],
                self.num_stats,
                self.windows.size * self.center.size,
                1,
            )
        )

        # Normalization on training data
        if self.normalize:
            self.stat_norm = tf.keras.layers.Normalization(axis=-1, name="stat_norm")
            self.stat_norm.adapt(X_train)
            # learns mean/std from training set only

        # Input stats as channel to improve performance
        # Avoiding changes stats order

        X_train = X_train.reshape(
            X_train.shape[0], self.windows.size, self.center.size, self.num_stats
        )
        X_test = X_test.reshape(
            X_test.shape[0], self.windows.size, self.center.size, self.num_stats
        )
        X_valid = X_valid.reshape(
            X_valid.shape[0], self.windows.size, self.center.size, self.num_stats
        )

        X_test = X_test.reshape(
            X_test.shape[0], self.windows.size, self.center.size, self.num_stats
        )

        if one_dim:
            X_train = X_train.reshape(
                -1, self.windows.size * self.center.size, self.num_stats
            )
            X_valid = X_valid.reshape(
                -1, self.windows.size * self.center.size, self.num_stats
            )
            X_test = X_test.reshape(
                -1, self.windows.size * self.center.size, self.num_stats
            )

        self.test_train_data = [X_test, X_test_params, Y_test]

        return (
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_valid,
            Y_valid,
        )

    def train(
        self,
        _iter=1,
        _stats=None,
        w=None,
        cnn=None,
        one_dim=False,
        preprocess=False,
        show_plot=False,
    ):
        """
        Train a CNN on flex-sweep tensors with early stopping and checkpoints.

        Parameters
        ----------
        _iter : int, default=1
            Tag for output naming (kept for backwards compatibility).
        _stats : list[str] | None
            Statistic base names. If None, defaults to the 11 flex-sweep stats.
        w : int | list[int] | None
            Window size(s) to select (see :meth:`load_training_data`).
        cnn : callable | None
            A function mapping a Keras input tensor to an output tensor.
            Defaults to :meth:`cnn_flexsweep`. If ``one_dim=True``, you must
            provide a compatible 1D architecture.
        one_dim : bool, default=False
            If True, uses flattened ``(W*C, S)`` inputs.

        Returns
        -------
        pl.DataFrame
            Predictions on the held-out test set with columns:
            ``['model','f_i','f_t','s','t','predicted_model','prob_sweep','prob_neutral']``.

        Notes
        -----
        - Optimizer: Adam with cosine-restarts schedule.
        - Loss: Binary cross-entropy with label smoothing (0.05).
        - Early stopping monitors validation AUC (restore best weights).
        - Saves ``model.keras`` to ``output_folder`` if provided.
        """

        if one_dim:
            assert cnn is not None, "Please input a 1D CNN architecture"

        # Default stats
        if _stats is None:
            _stats = [
                "dind",
                "dist_kurtosis",
                "dist_skew",
                "dist_var",
                "h1",
                "h12",
                "h2_h1",
                "haf",
                "hapdaf_o",
                "hapdaf_s",
                "high_freq",
                "ihs",
                "isafe",
                "k_counts",
                "low_freq",
                "max_fda",
                "nsl",
                "omega_max",
                "pi",
                "s_ratio",
                "tajima_d",
                "theta_h",
                "theta_w",
                "zns",
            ]
        if one_dim:
            assert cnn is not None, "Please input a 1D CNN architecture"

        self.num_stats = len(_stats)
        self.feature_names = list(_stats)

        # Default CNN
        if cnn is None:
            cnn = self.cnn_flexsweep

        (
            X_train,
            X_test,
            Y_train,
            Y_test,
            X_valid,
            Y_valid,
        ) = self.load_training_data(w=w, _stats=_stats, one_dim=one_dim)

        self.num_stats = len(_stats)
        self.feature_names = list(_stats)

        X_train = X_train.reshape(
            X_train.shape[0], self.num_stats, self.center.size * self.windows.size, 1
        )
        X_test = X_test.reshape(
            X_test.shape[0], self.num_stats, self.center.size * self.windows.size, 1
        )
        X_valid = X_valid.reshape(
            X_valid.shape[0], self.num_stats, self.center.size * self.windows.size, 1
        )

        # put model together
        input_to_model = tf.keras.Input(X_train.shape[1:])
        batch_size = 32

        # norm = tf.keras.layers.Normalization(axis=(0, 1, 2))
        # augment = tf.keras.Sequential(
        #     [tf.keras.layers.RandomFlip("horizontal")],
        #     name="augment",
        # )
        if preprocess:
            self.mean = X_train.mean(axis=(0, 1, 2), keepdims=False)
            self.std = X_train.std(axis=(0, 1, 2), keepdims=False)
            # self.mean = X_train.mean(axis=(0, 2, 3), keepdims=True)
            # self.std = X_train.std(axis=(0, 2, 3), keepdims=True)
            train_dataset = (
                tf.data.Dataset.from_tensor_slices((X_train, Y_train))
                .shuffle(10000)
                .map(lambda x, y: self.preprocess(x, y, training=True))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            valid_dataset = (
                tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
                .map(lambda x, y: self.preprocess(x, y, training=False))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            test_dataset = (
                tf.data.Dataset.from_tensor_slices((X_test, Y_test))
                .map(lambda x, y: self.preprocess(x, y, training=False))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
        else:
            train_dataset = (
                tf.data.Dataset.from_tensor_slices((X_train, Y_train))
                .shuffle(10000)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            valid_dataset = (
                tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )
            test_dataset = (
                tf.data.Dataset.from_tensor_slices((X_test, Y_test))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE)
            )

        model = tf.keras.models.Model(
            inputs=[input_to_model], outputs=[cnn(input_to_model)]
        )

        model_path = f"{self.output_folder}/model.keras"

        metrics_measures = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.AUC(name="roc", curve="ROC"),
        ]

        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=1e-4, first_decay_steps=300
        )
        opt_adam = tf.keras.optimizers.Adam(
            learning_rate=lr_decayed_fn, epsilon=0.0000001, amsgrad=True
        )

        # Keep only one compilation
        model.compile(
            optimizer=opt_adam,
            loss="binary_crossentropy",
            # loss=custom_loss,
            metrics=metrics_measures,
        )
        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            # monitor="val_auc",
            min_delta=0.0001,
            patience=5,
            verbose=2,
            mode="max",
            restore_best_weights=True,
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            # monitor="val_auc",
            verbose=2,
            save_best_only=True,
            mode="max",
        )

        callbacks_list = [checkpoint, earlystop]

        start = time.time()

        history = model.fit(
            train_dataset,
            epochs=1000,
            validation_data=valid_dataset,
            callbacks=callbacks_list,
        )

        val_score = model.evaluate(
            valid_dataset,
            batch_size=32,
            steps=len(Y_valid) // 32,
        )
        test_score = model.evaluate(
            test_dataset,
            batch_size=32,
            steps=len(Y_test) // 32,
        )

        train_score = model.evaluate(
            train_dataset,
            batch_size=32,
            steps=len(Y_train) // 32,
        )
        self.scores = [val_score, test_score, train_score]

        self.model = model

        df_history = pl.DataFrame(history.history)
        self.history = df_history
        print(
            f"Training and testing model took {round(time.time() - start, 3)} seconds"
        )

        if self.output_folder is not None:
            model.save(model_path)

        # ROC curves and confusion matrix
        if self.output_folder is None:
            _output_prediction = self.output_prediction
        else:
            _output_prediction = f"{self.output_folder}/{self.output_prediction}"

        test_X, test_X_params, test_Y = deepcopy(self.test_train_data)

        test_X = test_X.reshape(
            test_X.shape[0], self.num_stats, self.windows.size * self.center.size, 1
        )

        # self.mean = test_X.mean(axis=(0, 1, 2), keepdims=False)
        # self.std = test_X.std(axis=(0, 1, 2), keepdims=False)

        if preprocess:
            preds = model.predict(self.preprocess(test_X))
        else:
            preds = model.predict(test_X)

        preds = np.column_stack([1 - preds, preds])
        predictions = np.argmax(preds, axis=1)
        prediction_dict = {
            0: "neutral",
            1: "sweep",
        }
        predictions_class = np.vectorize(prediction_dict.get)(predictions)
        df_prediction = pl.concat(
            [
                test_X_params.select("model", "f_i", "f_t", "s", "t"),
                pl.DataFrame(
                    {
                        "predicted_model": predictions_class,
                        "prob_sweep": preds[:, 1],
                        "prob_neutral": preds[:, 0],
                    }
                ),
            ],
            how="horizontal",
        )

        self.prediction = df_prediction.with_columns(
            (
                pl.when(pl.col("model").str.contains("neutral"))
                .then(pl.lit("neutral"))
                .otherwise(pl.lit("sweep"))
            ).alias("model")
        )
        # self.prediction.write_csv("train_predictions.txt")

        self.roc_curve(show_plot=show_plot)

        if self.output_folder is not None:
            df_prediction.write_csv(_output_prediction)

        return df_prediction

    def _load_X_y(self):
        """Reload feature tensor and labels from train_data using stored feature_names."""
        if isinstance(self.train_data, pl.DataFrame):
            df = self.train_data
        elif self.train_data.endswith(".parquet"):
            df = pl.read_parquet(self.train_data)
        else:
            df = pl.read_csv(self.train_data, separator=",")

        stat_frames = [
            df.select(pl.col(f"^{name}_[0-9]+_[0-9]+$")) for name in self.feature_names
        ]
        X_df = pl.concat(stat_frames, how="horizontal")
        X = X_df.to_numpy().reshape(
            df.shape[0],
            len(self.feature_names),
            self.center.size * self.windows.size,
            1,
        )
        y = (~df["model"].str.contains("neutral")).cast(pl.Int8).to_numpy()
        return X, y

    def feature_importance(self, X=None, y=None, n_repeats=5, output_folder=None):
        """
        Permutation feature importance over stat channels.

        For each stat (axis=1 of the CNN input), shuffle values across samples
        n_repeats times and measure the mean accuracy drop vs baseline.

        Parameters
        ----------
        X : np.ndarray, shape (N, num_stats, n_positions, 1), optional
            Feature tensor. If None, reloads from self.train_data.
        y : np.ndarray, shape (N,), optional
            Integer labels (0=neutral, 1=sweep). Required when X is provided.
        n_repeats : int
            Shuffle repetitions per stat. Default 5.
        output_folder : str, optional
            If given, saves feature_importance.svg and feature_importance.csv.

        Returns
        -------
        df : pl.DataFrame
            Columns: feature, mean_drop, std_drop — sorted descending by mean_drop.
        fig : matplotlib.figure.Figure
        """
        assert hasattr(
            self, "feature_names"
        ), "Call train() before feature_importance()."

        if X is None:
            X, y = self._load_X_y()

        baseline_pred = self.model.predict(X, verbose=0).argmax(axis=1)
        baseline_acc = (baseline_pred == y).mean()

        rng = np.random.default_rng(42)
        records = []
        for i, name in enumerate(self.feature_names):
            drops = []
            for _ in range(n_repeats):
                X_perm = X.copy()
                perm_idx = rng.permutation(X_perm.shape[0])
                X_perm[:, i, :, :] = X_perm[perm_idx, i, :, :]
                acc = (self.model.predict(X_perm, verbose=0).argmax(axis=1) == y).mean()
                drops.append(baseline_acc - acc)
            records.append(
                {
                    "feature": name,
                    "mean_drop": float(np.mean(drops)),
                    "std_drop": float(np.std(drops)),
                }
            )

        df = pl.DataFrame(records).sort("mean_drop", descending=True)

        names = df["feature"].to_list()
        drops_v = df["mean_drop"].to_list()
        errs_v = df["std_drop"].to_list()
        fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.28)))
        ax.barh(
            names[::-1],
            drops_v[::-1],
            xerr=errs_v[::-1],
            color="steelblue",
            ecolor="gray",
            capsize=3,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Mean accuracy drop (permutation importance)")
        fig.tight_layout()

        if output_folder is not None:
            fig.savefig(
                os.path.join(output_folder, "feature_importance.svg"),
                bbox_inches="tight",
            )
            df.write_csv(os.path.join(output_folder, "feature_importance.csv"))

        return df, fig

    def predict(
        self, _stats=None, w=None, one_dim=False, _iter=1, fname=None, preprocess=False
    ):
        """
        Predict on a feature table using a trained model.

        Parameters
        ----------
        _stats : list[str] | None
            Statistic base names to include; defaults to the 11 flex-sweep stats.
        w : int | list[int] | None
            Window size(s) to select.
        simulations : bool, default=False
            Reserved flag; has no effect here.
        _iter : int, default=1
            Tag for output naming (unused).

        Returns
        -------
        pl.DataFrame
            Sorted predictions per region with columns:
            ``['chr','start','end','f_i','f_t','s','t','predicted_model','prob_sweep','prob_neutral']``.

        Raises
        ------
        AssertionError
            If ``self.model`` is not set or ``predict_data`` is missing.

        Notes
        -----
        If ``self.model`` is a string path, it is loaded via
        ``tf.keras.models.load_model``.
        """

        if _stats is None:
            _stats = [
                "dind",
                "dist_kurtosis",
                "dist_skew",
                "dist_var",
                "h1",
                "h12",
                "h2_h1",
                "haf",
                "hapdaf_o",
                "hapdaf_s",
                "high_freq",
                "ihs",
                "isafe",
                "k_counts",
                "low_freq",
                "max_fda",
                "nsl",
                "omega_max",
                "pi",
                "s_ratio",
                "tajima_d",
                "theta_h",
                "theta_w",
                "zns",
            ]

        self.num_stats = len(_stats)

        assert self.model is not None, "Please input the CNN trained model"

        if isinstance(self.model, str):
            model = tf.keras.models.load_model(self.model)
        else:
            model = self.model

        # import data to predict
        assert self.predict_data is not None, "Please input training data"
        assert (
            isinstance(self.predict_data, pl.DataFrame)
            or "txt" in self.predict_data
            or "csv" in self.predict_data
            or self.predict_data.endswith(".parquet")
        ), "Please input a parquet pl.DataFrame"

        df_test = pl.read_parquet(self.predict_data)

        df_test = df_test.with_columns(
            pl.when(pl.col("model") != "neutral")
            .then(pl.lit("sweep"))
            .otherwise(pl.lit("neutral"))
            .alias("model")
        )
        regions = df_test["iter"].to_numpy()

        stats = []
        if _stats is not None:
            stats = stats + _stats
        test_stats = []

        for i in stats:
            test_stats.append(df_test.select(pl.col(f"^{i}_[0-9]+_[0-9]+$")))

        X_test = pl.concat(test_stats, how="horizontal")

        if w is not None:
            try:
                self.center = np.array([int(w)])
                X_test = X_test.select(f"^*._{int(w)}$")
            except Exception:
                self.center = np.sort(np.array(w).astype(int))
                _X_test = []
                for window in self.center:
                    _X_test.append(X_test.select(f"^*._{int(window)}$"))
                X_test = pl.concat(_X_test, how="horizontal")

        test_X_params = df_test.select(
            "model", "iter", "s", "f_i", "f_t", "t", "mu", "r"
        )

        test_X = X_test.to_numpy().reshape(
            X_test.shape[0], self.num_stats, self.windows.size * self.center.size, 1
        )

        if one_dim:
            test_X = test_X.reshape(
                -1, self.windows.size * self.center.size, self.num_stats
            )

        if preprocess:
            self.mean = test_X.mean(axis=(0, 1, 2), keepdims=False)
            self.std = test_X.std(axis=(0, 1, 2), keepdims=False)

            # self.mean = test_X.mean(axis=(0, 2, 3), keepdims=True)
            # self.std = test_X.std(axis=(0, 2, 3), keepdims=True)

            test_X_ds = (
                tf.data.Dataset.from_tensor_slices(test_X)
                .map(lambda x: self.preprocess(x, training=False))
                .batch(32)
                .prefetch(tf.data.AUTOTUNE)
            )

            preds = model.predict(self.preprocess(test_X, training=False))
        else:
            test_X_ds = (
                tf.data.Dataset.from_tensor_slices(test_X)
                .batch(32)
                .prefetch(tf.data.AUTOTUNE)
            )
            preds = model.predict(test_X_ds)

        preds = np.column_stack([1 - preds, preds])
        predictions = np.argmax(preds, axis=1)
        prediction_dict = {
            0: "neutral",
            1: "sweep",
        }
        predictions_class = np.vectorize(prediction_dict.get)(predictions)
        df_prediction = pl.concat(
            [
                test_X_params.select("model", "f_i", "f_t", "s", "t", "mu", "r"),
                pl.DataFrame(
                    {
                        "predicted_model": predictions_class,
                        "prob_sweep": preds[:, 1],
                        "prob_neutral": preds[:, 0],
                    }
                ),
            ],
            how="horizontal",
        )

        # Same folder custom fvs name based on input VCF.
        # _output_prediction = f"{self.output_folder}/{os.path.basename(self.predict_data).replace("fvs_", "").replace(".parquet", "_predictions.txt")}"
        _output_prediction = f"{self.output_folder}/{os.path.basename(self.predict_data).replace('fvs_', '').replace('.parquet', '_predictions.txt')}"
        df_prediction = df_prediction.with_columns(pl.Series("region", regions))

        try:
            chr_start_end = np.array(
                [item.replace(":", "-").split("-") for item in regions]
            )
            df_prediction = df_prediction.with_columns(
                pl.Series("chr", chr_start_end[:, 0]),
                pl.Series("start", chr_start_end[:, 1], dtype=pl.Int64),
                pl.Series("end", chr_start_end[:, 2], dtype=pl.Int64),
                pl.Series(
                    "nchr",
                    pl.Series(chr_start_end[:, 0]).str.replace("chr", "").cast(int),
                ),
            )
            df_prediction = df_prediction.sort("nchr", "start").select(
                pl.exclude("region", "iter", "model", "nchr")
            )
        except:
            chr_start_end = np.zeros((regions.size, 3))
            df_prediction = df_prediction.with_columns(
                pl.Series("chr", chr_start_end[:, 0].astype(str)),
                pl.Series("start", chr_start_end[:, 1], dtype=pl.Int64),
                pl.Series("end", chr_start_end[:, 2], dtype=pl.Int64),
                pl.Series(
                    "nchr",
                    pl.Series(chr_start_end[:, 0]),
                ),
            )

            df_prediction = df_prediction.sort("nchr", "start").select(
                pl.exclude("region", "iter", "nchr")
            )

        if self.output_folder is not None:
            if fname is not None:
                _output_prediction = f"{self.output_folder}/{fname}"

            df_prediction.write_csv(_output_prediction)

        df_prediction = df_prediction.select(
            [
                "chr",
                "start",
                "end",
                "f_i",
                "f_t",
                "s",
                "t",
                "predicted_model",
                "prob_sweep",
                "prob_neutral",
            ]
        )

        return df_prediction

    def roc_curve(self, _iter=1, show_plot=False):
        """
        Build ROC curve, confusion matrix and training-history plots.

        Parameters
        ----------
        _iter : int, default=1
            Tag for output naming (kept for compatibility).

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.figure.Figure]
            ``(plot_roc, plot_history)`` figures. Confusion matrix is also saved
            to ``confusion_matrix.svg`` when ``output_folder`` is set.

        Notes
        -----
        - AUC is computed treating ``'sweep'`` as the positive class.
        - The method expects :attr:`prediction` to contain the latest
          predictions including ``prob_sweep``.
        """

        import matplotlib.pyplot as plt

        if isinstance(self.prediction, str):
            pred_data = pl.read_csv(self.prediction)
        else:
            pred_data = self.prediction

        pred_data = self.prediction

        # --- Confusion Matrix & Metrics ---
        y_true = pred_data["model"]
        y_pred = pred_data["predicted_model"]

        cm = confusion_matrix(
            y_true, y_pred, labels=["neutral", "sweep"], normalize="true"
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["neutral", "sweep"]
        )
        cm_plot = disp.plot(cmap="Blues")

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label="sweep")

        print("Confusion Matrix:\n", cm)
        print("Accuracy:", accuracy)
        print("Precision:", precision)

        # --- ROC Curve ---
        roc_auc_value = roc_auc_score(
            (y_true == "sweep").cast(int),
            pred_data["prob_sweep"].cast(float),
        )
        fpr, tpr, _ = roc_curve(
            (y_true == "sweep").cast(int),
            pred_data["prob_sweep"].cast(float),
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(
            fpr,
            tpr,
            color="orange",
            linewidth=2,
            label=f"ROC Curve (AUC = {roc_auc_value:.3f})",
        )
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("Sensitivity")
        ax.set_title("ROC Curve")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.legend(loc="lower right")
        fig.tight_layout()
        plot_roc = fig

        # --- Training History ---
        history_data = self.history
        h = history_data.select(
            [
                "loss",
                "val_loss",
                "accuracy",
                "val_accuracy",
            ]
        ).clone()
        h = h.with_columns((pl.arange(0, h.height) + 1).alias("epoch"))

        h_melted = h.unpivot(
            index=["epoch"],
            on=["loss", "val_loss", "accuracy", "val_accuracy"],
            variable_name="metric_name",
            value_name="metric_val",
        )

        line_styles = {
            "loss": "-",
            "val_loss": "--",
            "accuracy": "-",
            "val_accuracy": "--",
        }
        colors = {
            "loss": "orange",
            "val_loss": "orange",
            "accuracy": "blue",
            "val_accuracy": "blue",
        }

        fig, ax = plt.subplots(figsize=(10, 6))
        for group_name, group_df in h_melted.group_by("metric_name"):
            ax.plot(
                group_df["epoch"].to_numpy(),
                group_df["metric_val"].to_numpy(),
                label=group_name[0],
                linestyle=line_styles[group_name[0]],
                color=colors[group_name[0]],
                linewidth=2,
            )
        ax.set_title("History")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True)
        ax.legend(title="", loc="upper right")
        plot_history = fig

        #####################
        y_true = (pred_data["model"] == "sweep").cast(int).to_numpy()
        y_score = pred_data["prob_sweep"].cast(float).to_numpy()

        pr, rc, _ = precision_recall_curve(y_true, y_score)
        auc_pr = auc(rc, pr)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rc, pr, linewidth=2, label=f"AUC-PR = {auc_pr:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall (positive = sweep)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left")
        fig.tight_layout()
        pr_curve = fig

        y_score_clip = np.clip(y_score, 1e-6, 1 - 1e-6)
        prob_true, prob_pred = calibration_curve(
            y_true, y_score_clip, n_bins=10, strategy="quantile"
        )

        brier = brier_score_loss(y_true, y_score_clip)
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.plot([0, 1], [0, 1], "--", linewidth=1.5, label="perfect calibration")
        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            label=f"model (Brier={brier:.3f})",
        )
        plt.xlabel("Mean predicted probability (sweep)")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration (Reliability Diagram)")
        plt.grid(True, alpha=0.4)
        plt.legend(loc="upper left")
        cal = fig

        # --- Save if needed ---
        if self.output_folder is not None:
            plot_roc.savefig(f"{self.output_folder}/roc_curve.svg")
            plot_history.savefig(f"{self.output_folder}/train_history.svg")
            pr_curve.savefig(f"{self.output_folder}/auprc.svg")
            cal.savefig(f"{self.output_folder}/calibration.svg")
            cm_plot.figure_.savefig(f"{self.output_folder}/confusion_matrix.svg")

        if show_plot:
            plt.show()
        else:
            plt.close("all")

        return plot_roc, plot_history, cm_plot

    def _select_stats_matrix(self, df: pl.DataFrame, stats: list[str]):
        # Standardize model: anything not 'neutral' -> 'sweep'
        df = df.with_columns(
            pl.when(pl.col("model") != "neutral")
            .then(pl.lit("sweep"))
            .otherwise(pl.lit("neutral"))
            .alias("model")
        )

        blocks = []
        windows_set = set(self.windows.tolist())
        centers_set = set(self.center.tolist())

        for stat in stats:
            blk = df.select(pl.col(f"^{stat}_[0-9]+_[0-9]+$"))
            cols = blk.columns
            keys = []
            for col in cols:
                _, a, b = col.rsplit("_", 2)
                a, b = int(a), int(b)
                if a in windows_set and b in centers_set:
                    wv, cv = a, b
                elif a in centers_set and b in windows_set:
                    wv, cv = b, a
                else:
                    cv, wv = a, b
                keys.append((wv, cv, col))
            sorted_cols = [col for _, _, col in sorted(keys)]
            blocks.append(blk.select(sorted_cols))

        X = pl.concat(blocks, how="horizontal")
        y = (df["model"] != "neutral").cast(pl.Int8).to_numpy().astype(np.float32)
        params = df.select("iter", "s", "t", "f_i", "f_t", "model")

        N = df.height
        X = (
            X.to_numpy()
            .reshape(N, self.windows.size, self.center.size, len(stats))
            .astype(np.float32)
        )

        return X, y, params

    def load_da_data(self, _stats=None, src_val_frac=0.10):
        """
        Prepares DA inputs for the binary (neutral=0, sweep=1) setup.

        Produces:
          - src_neutral_tr, src_sweep_tr : source train arrays per class
          - neutral_train_idx, sweep_train_idx : counts for generator slicing
          - X_tgt : unlabeled target (domain discriminator pool)
          - val_X, val_Y_class, val_Y_discr : validation set
          - test_data : (X_test, y_test, X_test_params) from held-out source
        """
        # ---------- Load ----------
        df_all = pl.read_parquet(
            self.source_data
        )  # labeled source with 'model' ∈ {'neutral','sweep'}
        tgt_df = pl.read_parquet(self.target_data)  # target (may be unlabeled)

        # Hold-out from source for a final test set (kept as in your original code)
        (src_df, df_test) = train_test_split(
            df_all, test_size=(1 - self.train_split) * 0.5, shuffle=True
        )

        stats = [] if _stats is None else list(_stats)

        # ---------- Source matrices ----------
        X_src, y_src, _src_params = self._select_stats_matrix(src_df, stats)
        X_test, y_test, X_test_params = self._select_stats_matrix(df_test, stats)

        # Map labels to binary {0,1} if needed (accepts strings or ints)
        if y_src.ndim > 1 and y_src.shape[-1] == 2:
            # one-hot -> index
            y_src_bin = np.argmax(y_src, axis=-1).astype(np.int64)
        else:
            # strings or ints
            y_src_bin = np.array(y_src).reshape(-1)
            if y_src_bin.dtype.kind in {"U", "S", "O"}:
                map_dict = {"neutral": 0, "sweep": 1}
                y_src_bin = np.vectorize(map_dict.get)(y_src_bin).astype(np.int64)

        # Source train/val split for early stopping
        Xs_tr, Xs_va, ys_tr, ys_va = train_test_split(
            X_src, y_src_bin, test_size=src_val_frac, stratify=y_src_bin
        )

        # Build class-specific source training arrays for the generator
        src_neutral_tr = Xs_tr[ys_tr == 0]
        src_sweep_tr = Xs_tr[ys_tr == 1]

        # ---------- Target matrix (unlabeled for discriminator) ----------
        X_tgt, _yt_placeholder, tgt_params = self._select_stats_matrix(tgt_df, stats)

        # ---------- Validation set ----------
        # source validation
        val_X = Xs_va
        val_Y_class = ys_va.astype(np.float32)
        val_Y_discr = -1 * np.ones((val_X.shape[0],), dtype=np.float32)

        # ---------- Package ----------
        self.da_data = {
            "stats": stats,
            "src_neutral_tr": src_neutral_tr.astype(np.float32),
            "src_sweep_tr": src_sweep_tr.astype(np.float32),
            "X_tgt": X_tgt.astype(np.float32),  # unlabeled target pool
            "tgt_params": tgt_params,
            # Validation (binary labels 0/1; discriminator masked with -1)
            "val_X": val_X.astype(np.float32),
            "val_Y_class": val_Y_class.astype(np.float32),
            "val_Y_discr": val_Y_discr.astype(np.float32),
            # Kept for downstream evaluation on held-out source
            "test_data": [
                X_test.astype(np.float32),
                (
                    np.argmax(y_test, axis=-1)
                    if (y_test.ndim > 1 and y_test.shape[-1] == 2)
                    else y_test
                ).astype(np.int64),
                X_test_params,
            ],
        }

    def feature_extractor(self, model_input):
        """
        Shared 2D-CNN feature extractor (three branches).

        Parameters
        ----------
        model_input : tf.keras.layers.Input
            Input tensor of shape ``(W, C, S)``.

        Returns
        -------
        tf.Tensor
            Flattened concatenated features from the three branches.

        See Also
        --------
        cnn_flexsweep : Similar branch structure followed by classification head.
        """
        He = tf.keras.initializers.HeNormal()

        # # ---- Channel Dropout on stats (drops whole statistic channels) ----
        x = tf.keras.layers.SpatialDropout2D(0.10, name="fx_input_chdrop")(model_input)
        # x = model_input
        # ---- Stem: 1×1 mixes stats early to avoid single-stat shortcutting ----
        x = tf.keras.layers.Conv2D(
            64, 1, padding="same", kernel_initializer=He, name="fx_stem_conv"
        )(x)
        x = tf.keras.layers.BatchNormalization(name="fx_stem_bn")(x)
        x = tf.keras.layers.ReLU(name="fx_stem_relu")(x)

        # --- Branch 1: 3x3 convs ---
        b1 = tf.keras.layers.Conv2D(
            64, 3, padding="same", kernel_initializer=He, name="fx_b1_c1"
        )(x)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.Conv2D(
            128, 3, padding="same", kernel_initializer=He, name="fx_b1_c2"
        )(b1)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.Conv2D(
            256, 3, padding="same", kernel_initializer=He, name="fx_b1_c3"
        )(b1)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.MaxPooling2D(
            pool_size=3, padding="same", name="fx_b1_pool"
        )(b1)
        b1 = tf.keras.layers.Dropout(0.15, name="fx_b1_drop")(b1)
        b1 = tf.keras.layers.Flatten(name="fx_b1_flat")(b1)

        # --- Branch 2: 2x2 convs with dilation (1,3) ---
        b2 = tf.keras.layers.Conv2D(
            64,
            2,
            dilation_rate=[1, 3],
            padding="same",
            kernel_initializer=He,
            name="fx_b2_c1",
        )(x)
        b2 = tf.keras.layers.ReLU()(b2)
        b2 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 3],
            padding="same",
            kernel_initializer=He,
            name="fx_b2_c2",
        )(b2)
        b2 = tf.keras.layers.ReLU()(b2)
        b2 = tf.keras.layers.Conv2D(
            256,
            2,
            dilation_rate=[1, 3],
            padding="same",
            kernel_initializer=He,
            name="fx_b2_c3",
        )(b2)
        b2 = tf.keras.layers.ReLU()(b2)
        # b2 = tf.keras.layers.MaxPooling2D(pool_size=2, name="fx_b2_pool")(b2)
        b2 = tf.keras.layers.MaxPooling2D(
            pool_size=(1, 2), padding="same", name="fx_b2_pool"
        )(b2)
        b2 = tf.keras.layers.Dropout(0.15, name="fx_b2_drop")(b2)
        b2 = tf.keras.layers.Flatten(name="fx_b2_flat")(b2)

        # --- Branch 3: 2x2 convs with dilation (5,1) then (1,5) ---
        b3 = tf.keras.layers.Conv2D(
            64,
            2,
            dilation_rate=[5, 1],
            padding="same",
            kernel_initializer=He,
            name="fx_b3_c1",
        )(x)
        b3 = tf.keras.layers.ReLU()(b3)
        b3 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 5],
            padding="same",
            kernel_initializer=He,
            name="fx_b3_c2",
        )(b3)
        b3 = tf.keras.layers.ReLU()(b3)
        b3 = tf.keras.layers.Conv2D(
            256,
            2,
            dilation_rate=[1, 5],
            padding="same",
            kernel_initializer=He,
            name="fx_b3_c3",
        )(b3)
        b3 = tf.keras.layers.ReLU()(b3)
        # b3 = tf.keras.layers.MaxPooling2D(pool_size=2, name="fx_b3_pool")(b3)
        b3 = tf.keras.layers.MaxPooling2D(
            pool_size=(1, 2), padding="same", name="fx_b3_pool"
        )(b3)
        b3 = tf.keras.layers.Dropout(0.15, name="fx_b3_drop")(b3)
        b3 = tf.keras.layers.Flatten(name="fx_b3_flat")(b3)

        feat = tf.keras.layers.Concatenate(name="fx_concat")(
            [
                b1,
                b2,
                b3,
            ]
        )  # shared representation

        return feat

    def build_grl_model(self, input_shape):
        """
        Build a two-head domain-adversarial CNN with a Gradient Reversal Layer.

        Architecture
        ------------
        - **Shared feature extractor**: :meth:`feature_extractor` over inputs shaped
          ``(W, C, S)`` (windows × centers × statistics), channels-last.
        - **Classifier head** (task): 2 dense layers + sigmoid output named
          ``"classifier"`` (sweep vs. neutral, BCE).
        - **Domain head**: GRL → 2 dense layers + sigmoid output named
          ``"discriminator"`` (source=0 vs. target=1, BCE).

        Parameters
        ----------
        input_shape : tuple[int, int, int]
            ``(W, C, S)`` defining windows, centers, and number of stats (channels).

        Returns
        -------
        tf.keras.Model
            Uncompiled Keras model with two outputs:
            ``[classifier(sigmoid), discriminator(sigmoid)]``.

        Notes
        -----
        - The GRL instance is stored at ``self.grl`` so a callback (e.g., :class:`GRLRamp`)
          can update its strength during training.
        - Compilation (optimizer, losses, metrics) is performed in
          :meth:`train_da_empirical`.
        """
        inp = tf.keras.Input(shape=input_shape)  # (W, C, S), channels-last

        # x_in = (
        #     self.stat_norm_da(inp)
        #     if hasattr(self, "stat_norm_da") and self.stat_norm_da is not None
        #     else inp
        # )

        feat = self.feature_extractor(inp)

        # classifier head
        h = tf.keras.layers.Dense(128, activation="relu")(feat)
        h = tf.keras.layers.Dropout(0.20)(h)
        h = tf.keras.layers.Dense(32, activation="relu")(h)
        h = tf.keras.layers.Dropout(0.10)(h)
        out_cls = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(h)

        # domain head via GRL (store the layer for ramping)
        self.grl = GradReverse(lambd=0)
        g = self.grl(feat)
        g = tf.keras.layers.Dense(128, activation="relu")(g)
        g = tf.keras.layers.Dropout(0.20)(g)
        g = tf.keras.layers.Dense(32, activation="relu")(g)
        out_dom = tf.keras.layers.Dense(1, activation="sigmoid", name="discriminator")(
            g
        )

        model = tf.keras.Model(inputs=inp, outputs=[out_cls, out_dom])

        return model

    def feature_extractor_f(self, model_input):
        """
        Shared 2D-CNN feature extractor (three branches).

        Parameters
        ----------
        model_input : tf.keras.layers.Input
            Input tensor of shape ``(W, C, S)``.

        Returns
        -------
        tf.Tensor
            Flattened concatenated features from the three branches.

        See Also
        --------
        cnn_flexsweep : Similar branch structure followed by classification head.
        """
        He = tf.keras.initializers.HeNormal()

        # --- Branch 1: 3x3 convs ---
        b1 = tf.keras.layers.Conv2D(
            64, 3, padding="same", kernel_initializer=He, name="fx_b1_c1"
        )(model_input)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.Conv2D(
            128, 3, padding="same", kernel_initializer=He, name="fx_b1_c2"
        )(b1)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.Conv2D(
            256, 3, padding="same", kernel_initializer=He, name="fx_b1_c3"
        )(b1)
        b1 = tf.keras.layers.ReLU()(b1)
        b1 = tf.keras.layers.MaxPooling2D(
            pool_size=3, padding="same", name="fx_b1_pool"
        )(b1)
        b1 = tf.keras.layers.Dropout(0.15, name="fx_b1_drop")(b1)
        b1 = tf.keras.layers.Flatten(name="fx_b1_flat")(b1)

        # --- Branch 2: 2x2 convs with dilation (1,3) ---
        b2 = tf.keras.layers.Conv2D(
            64,
            2,
            dilation_rate=[1, 3],
            padding="same",
            kernel_initializer=He,
            name="fx_b2_c1",
        )(model_input)
        b2 = tf.keras.layers.ReLU()(b2)
        b2 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 3],
            padding="same",
            kernel_initializer=He,
            name="fx_b2_c2",
        )(b2)
        b2 = tf.keras.layers.ReLU()(b2)
        b2 = tf.keras.layers.Conv2D(
            256,
            2,
            dilation_rate=[1, 3],
            padding="same",
            kernel_initializer=He,
            name="fx_b2_c3",
        )(b2)
        b2 = tf.keras.layers.ReLU()(b2)
        b2 = tf.keras.layers.MaxPooling2D(pool_size=2, name="fx_b2_pool")(b2)
        b2 = tf.keras.layers.Dropout(0.15, name="fx_b2_drop")(b2)
        b2 = tf.keras.layers.Flatten(name="fx_b2_flat")(b2)

        # --- Branch 3: 2x2 convs with dilation (5,1) then (1,5) ---
        b3 = tf.keras.layers.Conv2D(
            64,
            2,
            dilation_rate=[5, 1],
            padding="same",
            kernel_initializer=He,
            name="fx_b3_c1",
        )(model_input)
        b3 = tf.keras.layers.ReLU()(b3)
        b3 = tf.keras.layers.Conv2D(
            128,
            2,
            dilation_rate=[1, 5],
            padding="same",
            kernel_initializer=He,
            name="fx_b3_c2",
        )(b3)
        b3 = tf.keras.layers.ReLU()(b3)
        b3 = tf.keras.layers.Conv2D(
            256,
            2,
            dilation_rate=[1, 5],
            padding="same",
            kernel_initializer=He,
            name="fx_b3_c3",
        )(b3)
        b3 = tf.keras.layers.ReLU()(b3)
        b3 = tf.keras.layers.MaxPooling2D(pool_size=2, name="fx_b3_pool")(b3)
        b3 = tf.keras.layers.Dropout(0.15, name="fx_b3_drop")(b3)
        b3 = tf.keras.layers.Flatten(name="fx_b3_flat")(b3)

        feat = tf.keras.layers.Concatenate(name="fx_concat")(
            [
                b1,
                b2,
                b3,
            ]
        )  # shared representation

        return feat

    def build_grl_model_f(self, input_shape):
        inp = tf.keras.Input(shape=input_shape)  # (W, C, S), channels-last

        feat = self.feature_extractor(inp)

        # classifier head
        h = tf.keras.layers.Dense(128, activation="relu")(feat)
        h = tf.keras.layers.Dropout(0.20)(h)
        h = tf.keras.layers.Dense(32, activation="relu")(h)
        h = tf.keras.layers.Dropout(0.10)(h)
        out_cls = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(h)

        # domain head via GRL (store the layer for ramping)
        self.grl = GradReverse(lambd=0.0)
        g = self.grl(feat)
        g = tf.keras.layers.Dense(128, activation="relu")(g)
        g = tf.keras.layers.Dropout(0.20)(g)
        g = tf.keras.layers.Dense(32, activation="relu")(g)
        out_dom = tf.keras.layers.Dense(1, activation="sigmoid", name="discriminator")(
            g
        )

        model = tf.keras.Model(inputs=inp, outputs=[out_cls, out_dom])

        return model

    def build_grl_model_beta(self, input_shape, max_lambda):
        """
        Build a two-head domain-adversarial CNN with a Gradient Reversal Layer.

        Architecture
        ------------
        - **Shared feature extractor**: :meth:`feature_extractor` over inputs shaped
          ``(W, C, S)`` (windows × centers × statistics), channels-last.
        - **Classifier head** (task): 2 dense layers + sigmoid output named
          ``"classifier"`` (sweep vs. neutral, BCE).
        - **Domain head**: GRL → 2 dense layers + sigmoid output named
          ``"discriminator"`` (source=0 vs. target=1, BCE).

        Parameters
        ----------
        input_shape : tuple[int, int, int]
            ``(W, C, S)`` defining windows, centers, and number of stats (channels).

        Returns
        -------
        tf.keras.Model
            Uncompiled Keras model with two outputs:
            ``[classifier(sigmoid), discriminator(sigmoid)]``.

        Notes
        -----
        - The GRL instance is stored at ``self.grl`` so a callback (e.g., :class:`GRLRamp`)
          can update its strength during training.
        - Compilation (optimizer, losses, metrics) is performed in
          :meth:`train_da_empirical`.
        """
        inp = tf.keras.Input(shape=input_shape)  # (W, C, S), channels-last

        # x_in = (
        #     self.stat_norm_da(inp)
        #     if hasattr(self, "stat_norm_da") and self.stat_norm_da is not None
        #     else inp
        # )

        feat = self.feature_extractor(inp)

        # classifier head
        h = tf.keras.layers.Dense(128, activation="relu")(feat)
        h = tf.keras.layers.Dropout(0.20)(h)
        h = tf.keras.layers.Dense(32, activation="relu")(h)
        h = tf.keras.layers.Dropout(0.10)(h)
        out_cls = tf.keras.layers.Dense(1, activation="sigmoid", name="classifier")(h)

        self.grl = GradReverse(lambd=max_lambda)
        g = self.grl(feat)
        g = tf.keras.layers.Dense(128, activation="relu")(g)
        g = tf.keras.layers.Dropout(0.20)(g)
        g = tf.keras.layers.Dense(32, activation="relu")(g)
        out_dom = tf.keras.layers.Dense(1, activation="sigmoid", name="discriminator")(
            g
        )

        model = tf.keras.Model(inputs=inp, outputs=[out_cls, out_dom])

        return model

    def train_da_f(
        self,
        _stats=None,
        max_lambda=1,
        ramp_epochs=20,
        tgt_ratio=1,
        batch_size=32,
        preprocess=True,
    ):
        tf = self.check_tf()
        if _stats is None:
            _stats = [
                "dind",
                "dist_kurtosis",
                "dist_skew",
                "dist_var",
                "h1",
                "h12",
                "h2_h1",
                "haf",
                "hapdaf_o",
                "hapdaf_s",
                "high_freq",
                "ihs",
                "isafe",
                "k_counts",
                "low_freq",
                "max_fda",
                "nsl",
                "omega_max",
                "pi",
                "s_ratio",
                "tajima_d",
                "theta_h",
                "theta_w",
                "zns",
            ]

        self.predict_data = self.target_data
        if not hasattr(self, "da_data") or self.da_data is None:
            self.load_da_data(_stats=_stats)
        dd = self.da_data

        dd["src_neutral_tr"] = dd["src_neutral_tr"].reshape(
            dd["src_neutral_tr"].shape[0],
            self.windows.size * self.center.size,
            self.num_stats,
            1,
        )
        dd["src_sweep_tr"] = dd["src_sweep_tr"].reshape(
            dd["src_sweep_tr"].shape[0],
            self.windows.size * self.center.size,
            self.num_stats,
            1,
        )
        dd["val_X"] = dd["val_X"].reshape(
            dd["val_X"].shape[0],
            self.num_stats,
            self.windows.size * self.center.size,
            1,
        )
        dd["test_data"][0] = dd["test_data"][0].reshape(
            dd["test_data"][0].shape[0],
            self.num_stats,
            self.windows.size * self.center.size,
            1,
        )
        dd["X_tgt"] = dd["X_tgt"].reshape(
            dd["X_tgt"].shape[0],
            self.num_stats,
            self.windows.size * self.center.size,
            1,
        )

        dd["val_X"] = dd["val_X"].reshape(
            dd["val_X"].shape[0],
            self.num_stats,
            self.windows.size * self.center.size,
            1,
        )

        if preprocess:
            X_src_tr = np.vstack([dd["src_neutral_tr"], dd["src_sweep_tr"]])
            self.mean = X_src_tr.mean(axis=(0, 1, 2))
            self.std = X_src_tr.std(axis=(0, 1, 2))

            dd["src_neutral_tr"] = self.preprocess(
                dd["src_neutral_tr"], training=True
            ).numpy()
            dd["src_sweep_tr"] = self.preprocess(
                dd["src_sweep_tr"], training=True
            ).numpy()
            dd["val_X"] = self.preprocess(dd["val_X"]).numpy()

            # normalize target using its own stats (as in your previous code)
            self.mean = dd["X_tgt"].mean(axis=(0, 1, 2))
            self.std = dd["X_tgt"].std(axis=(0, 1, 2))
            dd["X_tgt"] = self.preprocess(dd["X_tgt"], training=True).numpy()

            dd["test_data"][0] = self.preprocess(
                dd["test_data"][0], training=False
            ).numpy()

        val_X = dd["val_X"]
        val_Y_class = dd["val_Y_class"]  # 0/1 (binary)
        val_Y_discr = dd["val_Y_discr"]  # all -1 to mask discriminator on val

        data_gen = DAParquetSequence(
            src_neutral=dd["src_neutral_tr"],
            src_sweep=dd["src_sweep_tr"],
            tar_all=dd["X_tgt"],
            batch_size=batch_size,
        )

        input_shape = (self.num_stats, self.windows.size * self.center.size, 1)
        model = self.build_grl_model_f(input_shape)

        opt = tf.keras.optimizers.AdamW(
            learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(5e-5, 300),
            epsilon=1e-7,
            amsgrad=True,
        )

        model.compile(
            optimizer=opt,
            loss={"classifier": masked_bce, "discriminator": masked_bce},
            loss_weights={"classifier": 1.0, "discriminator": 1.0},
            metrics={
                "classifier": masked_binary_accuracy,
                "discriminator": masked_binary_accuracy,
            },
            jit_compile=False,
        )

        callbacks = [
            GRLRamp(self.grl, max_lambda=max_lambda, epochs=ramp_epochs),
            LogGRLLambda(self.grl),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_classifier_masked_binary_accuracy",
                mode="max",
                patience=20,
                min_delta=1e-4,
                restore_best_weights=True,
            ),
        ]
        if self.output_folder:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    f"{self.output_folder}/model_da.keras",
                    monitor="val_classifier_masked_binary_accuracy",
                    mode="max",
                    save_best_only=True,
                    verbose=1,
                )
            )
        hist = model.fit(
            data_gen,
            epochs=max(1000, ramp_epochs),
            steps_per_epoch=len(data_gen),
            validation_data=(
                val_X,
                {"classifier": val_Y_class, "discriminator": val_Y_discr},
            ),
            callbacks=callbacks,
            verbose=2,
        )

        # Logging with same keys you already read elsewhere

        hh = hist.history

        self.history = pl.DataFrame(
            {
                "loss": hh["loss"],
                "classifier_accuracy": hh["classifier_masked_binary_accuracy"],
                "discriminator_accuracy": hh["discriminator_masked_binary_accuracy"],
                "classifier_loss": hh["classifier_loss"],
                "discriminator_loss": hh["discriminator_loss"],
                "val_classifier_accuracy": hh["val_classifier_masked_binary_accuracy"],
                "val_discriminator_accuracy": hh[
                    "val_discriminator_masked_binary_accuracy"
                ],
                "val_classifier_loss": hh["val_classifier_loss"],
                "val_discriminator_loss": hh["val_discriminator_loss"],
            }
        )

        self.model = model
        if self.output_folder:
            model.save(f"{self.output_folder}/model_da.keras")

        # quick eval on held-out source test for the plots you already have wired
        X_test, Y_test, X_test_params = dd["test_data"]
        out = model.predict(X_test, verbose=0, batch_size=32)
        cls = out[0] if isinstance(out, (list, tuple)) else out
        p = cls.ravel().astype(np.float32)

        # self._fit_platt(Y_test.astype(int), p)
        # self._save_calibration()

        # p_cal = self._apply_calibration(p)
        df_pred = (
            pl.concat(
                [
                    X_test_params,
                    pl.DataFrame(
                        {
                            "predicted_model": np.where(p >= 0.5, "sweep", "neutral"),
                            "prob_sweep": p,
                            "prob_neutral": 1.0 - p,
                        }
                    ),
                ],
                how="horizontal",
            )
            .drop("model")
            .with_columns(pl.Series("model", np.where(Y_test == 1, "sweep", "neutral")))
        )

        self.prediction = df_pred
        return self.history

    def train_da(
        self,
        _stats=None,
        max_lambda=1,
        ramp_epochs=30,
        tgt_ratio=1,
        batch_size=128,
        preprocess=True,
    ):
        tf = self.check_tf()
        if _stats is None:
            _stats = [
                "dind",
                "dist_kurtosis",
                "dist_skew",
                "dist_var",
                "h1",
                "h12",
                "h2_h1",
                "haf",
                "hapdaf_o",
                "hapdaf_s",
                "high_freq",
                "ihs",
                "isafe",
                "k_counts",
                "low_freq",
                "max_fda",
                "nsl",
                "omega_max",
                "pi",
                "s_ratio",
                "tajima_d",
                "theta_h",
                "theta_w",
                "zns",
            ]

        self.predict_data = self.target_data
        if not hasattr(self, "da_data") or self.da_data is None:
            self.load_da_data(_stats=_stats)
        dd = self.da_data

        if preprocess:
            X_src_tr = np.vstack([dd["src_neutral_tr"], dd["src_sweep_tr"]])
            self.mean = X_src_tr.mean(axis=(0, 1, 2))
            self.std = X_src_tr.std(axis=(0, 1, 2))

            dd["src_neutral_tr"] = self.preprocess(
                dd["src_neutral_tr"], training=True
            ).numpy()
            dd["src_sweep_tr"] = self.preprocess(
                dd["src_sweep_tr"], training=True
            ).numpy()
            dd["val_X"] = self.preprocess(dd["val_X"]).numpy()
            dd["test_data"][0] = self.preprocess(dd["test_data"][0]).numpy()

            # normalize target using its own stats (as in your previous code)
            self.mean = dd["X_tgt"].mean(axis=(0, 1, 2))
            self.std = dd["X_tgt"].std(axis=(0, 1, 2))
            dd["X_tgt"] = self.preprocess(dd["X_tgt"], training=True).numpy()

            dd["test_data"][0] = self.preprocess(
                dd["test_data"][0], training=False
            ).numpy()

        data_gen = DAParquetSequence(
            src_neutral=dd["src_neutral_tr"],
            src_sweep=dd["src_sweep_tr"],
            tar_all=dd["X_tgt"],
            batch_size=batch_size,
            tgt_ratio=tgt_ratio,
        )

        val_X = dd["val_X"]
        val_Y_class = dd["val_Y_class"]  # 0/1 (binary)
        val_Y_discr = dd["val_Y_discr"]  # all -1 to mask discriminator on val

        # input_shape = (self.windows.size, self.center.size, self.num_stats)
        input_shape = dd["X_tgt"].shape[1:]
        model = self.build_grl_model(input_shape)  # GRL λ fixed at 1.0

        # {'max_lambda': 0.2843709709293154
        #     'ramp_epochs': 64
        #     'patience': 18
        #     'batch_size': 128
        #     'tgt_ratio': 1.651652308372508
        #     'clip_value': 5.443662527929382
        #     'lr': 0.000989613742860149
        #     'weight_decay': 0.0003667748863292507
        #     'loss_weight_discriminator': 0.5948347312577422}

        opt = tf.keras.optimizers.AdamW(
            learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(5e-5, 300),
            # learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(1e-3, 300),
            weight_decay=1e-4,
            # weight_decay=0.0003667748863292507,
            epsilon=1e-7,
            clipnorm=1.0,
        )

        model.compile(
            optimizer=opt,
            loss={"classifier": masked_bce, "discriminator": masked_bce},
            loss_weights={"classifier": 1.0, "discriminator": 0.75},
            # loss_weights={"classifier": 1.0, "discriminator": 1.0},
            metrics={
                "classifier": masked_binary_accuracy,
                "discriminator": masked_binary_accuracy,
            },
            jit_compile=False,
        )

        callbacks = [
            GRLRamp(self.grl, max_lambda=max_lambda, epochs=ramp_epochs),
            LogGRLLambda(self.grl),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_classifier_masked_binary_accuracy",
                mode="max",
                patience=20,
                min_delta=1e-4,
                restore_best_weights=True,
            ),
        ]
        if self.output_folder:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    f"{self.output_folder}/model_da.keras",
                    monitor="val_classifier_masked_binary_accuracy",
                    mode="max",
                    save_best_only=True,
                    verbose=1,
                )
            )
        hist = model.fit(
            data_gen,
            epochs=max(1000, ramp_epochs),
            # epochs=30,
            steps_per_epoch=len(data_gen),
            validation_data=(
                val_X,
                {"classifier": val_Y_class, "discriminator": val_Y_discr},
            ),
            callbacks=callbacks,
            verbose=2,
        )

        # Logging with same keys you already read elsewhere

        hh = hist.history

        self.history = pl.DataFrame(
            {
                "loss": hh["loss"],
                "classifier_accuracy": hh["classifier_masked_binary_accuracy"],
                "discriminator_accuracy": hh["discriminator_masked_binary_accuracy"],
                "classifier_loss": hh["classifier_loss"],
                "discriminator_loss": hh["discriminator_loss"],
                "val_classifier_accuracy": hh["val_classifier_masked_binary_accuracy"],
                "val_discriminator_accuracy": hh[
                    "val_discriminator_masked_binary_accuracy"
                ],
                "val_classifier_loss": hh["val_classifier_loss"],
                "val_discriminator_loss": hh["val_discriminator_loss"],
            }
        )

        self.model = model
        if self.output_folder:
            model.save(f"{self.output_folder}/model_da.keras")
            self.history.write_csv(f"{self.output_folder}/history_da.txt")

        # quick eval on held-out source test for the plots you already have wired
        X_test, Y_test, X_test_params = dd["test_data"]
        out = model.predict(X_test, verbose=0, batch_size=32)
        cls = out[0] if isinstance(out, (list, tuple)) else out
        p = cls.ravel().astype(np.float32)

        self._fit_platt(Y_test.astype(int), p)
        self._save_calibration()

        # p_cal = p
        p_cal = self._apply_calibration(p)
        df_pred = (
            pl.concat(
                [
                    X_test_params,
                    pl.DataFrame(
                        {
                            "predicted_model": np.where(
                                p_cal >= 0.5, "sweep", "neutral"
                            ),
                            "prob_sweep": p_cal,
                            "prob_neutral": 1.0 - p_cal,
                        }
                    ),
                ],
                how="horizontal",
            )
            .drop("model")
            .with_columns(pl.Series("model", np.where(Y_test == 1, "sweep", "neutral")))
        )

        self.prediction = df_pred

        self.plot_da_curves()

        return self.history

    def train_da_beta(
        self,
        _stats=None,
        max_lambda=1,
        max_beta=1.0,
        ramp_epochs=31,
        tgt_ratio=1,
        batch_size=32,
        preprocess=True,
    ):
        tf = self.check_tf()
        if _stats is None:
            _stats = [
                "dind",
                "dist_kurtosis",
                "dist_skew",
                "dist_var",
                "h1",
                "h12",
                "h2_h1",
                "haf",
                "hapdaf_o",
                "hapdaf_s",
                "high_freq",
                "ihs",
                "isafe",
                "k_counts",
                "low_freq",
                "max_fda",
                "nsl",
                "omega_max",
                "pi",
                "s_ratio",
                "tajima_d",
                "theta_h",
                "theta_w",
                "zns",
            ]

        self.predict_data = self.target_data
        if not hasattr(self, "da_data") or self.da_data is None:
            self.load_da_data(_stats=_stats)
        dd = self.da_data

        if preprocess:
            X_src_tr = np.vstack([dd["src_neutral_tr"], dd["src_sweep_tr"]])
            self.mean = X_src_tr.mean(axis=(0, 1, 2))
            self.std = X_src_tr.std(axis=(0, 1, 2))

            dd["src_neutral_tr"] = self.preprocess(
                dd["src_neutral_tr"], training=True
            ).numpy()
            dd["src_sweep_tr"] = self.preprocess(
                dd["src_sweep_tr"], training=True
            ).numpy()
            dd["val_X"] = self.preprocess(dd["val_X"]).numpy()
            dd["test_data"][0] = self.preprocess(dd["test_data"][0]).numpy()

            # normalize target using its own stats (as in your previous code)
            self.mean = dd["X_tgt"].mean(axis=(0, 1, 2))
            self.std = dd["X_tgt"].std(axis=(0, 1, 2))
            dd["X_tgt"] = self.preprocess(dd["X_tgt"], training=True).numpy()

            dd["test_data"][0] = self.preprocess(
                dd["test_data"][0], training=False
            ).numpy()

        data_gen = DAParquetSequence(
            src_neutral=dd["src_neutral_tr"],
            src_sweep=dd["src_sweep_tr"],
            tar_all=dd["X_tgt"],
            batch_size=batch_size,
            tgt_ratio=tgt_ratio,
        )

        val_X = dd["val_X"]
        val_Y_class = dd["val_Y_class"]  # 0/1 (binary)
        val_Y_discr = dd["val_Y_discr"]  # all -1 to mask discriminator on val

        input_shape = (self.windows.size, self.center.size, self.num_stats)
        model = self.build_grl_model_beta(input_shape, max_lambda=max_lambda)

        opt = tf.keras.optimizers.AdamW(
            learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(5e-5, 300),
            weight_decay=1e-4,
            epsilon=1e-7,
            clipnorm=5.0,
        )

        # Variable loss weights (alpha fixed at 1.0, beta starts at 0.0 then ramp)
        alpha = K.variable(1.0, dtype="float32", name="alpha_cls")
        beta = K.variable(0.0, dtype="float32", name="beta_dom")

        model.compile(
            optimizer=opt,
            loss={"classifier": masked_bce, "discriminator": masked_bce},
            loss_weights={"classifier": 1.0, "discriminator": 1.0},
            metrics={
                "classifier": masked_binary_accuracy,
                "discriminator": masked_binary_accuracy,
            },
            jit_compile=False,
        )

        # callbacks: your beta ramp & logger + early stopping
        callbacks = [
            LossWeightsScheduler(alpha, beta),
            LossWeightsLogger([alpha, beta]),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_classifier_masked_binary_accuracy",
                mode="max",
                patience=25,
                min_delta=1e-4,
                restore_best_weights=True,
            ),
        ]

        if self.output_folder:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    f"{self.output_folder}/model_da.keras",
                    monitor="val_classifier_masked_binary_accuracy",
                    mode="max",
                    save_best_only=True,
                    verbose=1,
                )
            )
        hist = model.fit(
            data_gen,
            epochs=30,
            steps_per_epoch=len(data_gen),
            validation_data=(
                val_X,
                {"classifier": val_Y_class, "discriminator": val_Y_discr},
            ),
            callbacks=callbacks,
            verbose=2,
        )

        hh = hist.history

        self.history = pl.DataFrame(
            {
                "loss": hh["loss"],
                "classifier_accuracy": hh["classifier_masked_binary_accuracy"],
                "discriminator_accuracy": hh["discriminator_masked_binary_accuracy"],
                "classifier_loss": hh["classifier_loss"],
                "discriminator_loss": hh["discriminator_loss"],
                "val_classifier_accuracy": hh["val_classifier_masked_binary_accuracy"],
                "val_discriminator_accuracy": hh[
                    "val_discriminator_masked_binary_accuracy"
                ],
                "val_classifier_loss": hh["val_classifier_loss"],
                "val_discriminator_loss": hh["val_discriminator_loss"],
            }
        )

        self.model = model
        if self.output_folder:
            model.save(f"{self.output_folder}/model_da.keras")

        X_test, Y_test, X_test_params = dd["test_data"]
        out = model.predict(X_test, verbose=0, batch_size=32)
        cls = out[0] if isinstance(out, (list, tuple)) else out
        p = cls.ravel().astype(np.float32)

        # self._fit_platt(Y_test.astype(int), p)
        # self._save_calibration()

        # p_cal = self._apply_calibration(p)
        df_pred = (
            pl.concat(
                [
                    X_test_params,
                    pl.DataFrame(
                        {
                            "predicted_model": np.where(p >= 0.5, "sweep", "neutral"),
                            "prob_sweep": p,
                            "prob_neutral": 1.0 - p,
                        }
                    ),
                ],
                how="horizontal",
            )
            .drop("model")
            .with_columns(pl.Series("model", np.where(Y_test == 1, "sweep", "neutral")))
        )

        self.prediction = df_pred
        return self.history

    def predict_da(self, _stats=None, preprocess=True, fname=None):
        """
        Predict sweep probabilities on empirical (target) data using a DA model.

        Loads a trained two-head model and returns per-region predictions from the
        **classifier** head (sweep vs. neutral). The domain head is unused at inference.

        Parameters
        ----------
        _stats : list[str] | None
            Statistic base names to include (must match training).

        Returns
        -------
        pl.DataFrame
            Table with per-region predictions and metadata, including:
            ``['chr','start','end','f_i','f_t','s','t','predicted_model',
            'prob_sweep','prob_neutral']`` sorted by chromosome and start.

        Raises
        ------
        AssertionError
            If no model is loaded or the test data path is invalid.

        Notes
        -----
        - Expects the same (W, C, S) layout used in training.
        - Output ``prob_sweep`` is the classifier sigmoid; ``prob_neutral=1-prob_sweep``.
        """
        assert self.model is not None, "Call train_da() first"

        tf = self.check_tf()
        if _stats is None:
            _stats = [
                "dind",
                "dist_kurtosis",
                "dist_skew",
                "dist_var",
                "h1",
                "h12",
                "h2_h1",
                "haf",
                "hapdaf_o",
                "hapdaf_s",
                "high_freq",
                "ihs",
                "isafe",
                "k_counts",
                "low_freq",
                "max_fda",
                "nsl",
                "omega_max",
                "pi",
                "s_ratio",
                "tajima_d",
                "theta_h",
                "theta_w",
                "zns",
            ]

        if isinstance(self.model, str):
            model = tf.keras.models.load_model(
                self.model,
                safe_mode=True,
            )
        else:
            model = self.model

        assert (
            isinstance(self.predict_data, pl.DataFrame)
            or "txt" in self.predict_data
            or "csv" in self.predict_data
            or self.predict_data.endswith(".parquet")
        ), "Please input a pl.DataFrame or save it as CSV or parquet"
        try:
            df_test = pl.read_parquet(self.predict_data)
            if "test" in self.predict_data:
                df_test = df_test.sample(
                    with_replacement=False, fraction=1.0, shuffle=True
                )
        except Exception:
            df_test = pl.read_csv(self.predict_data, separator=",")

        df_test = df_test.with_columns(
            pl.when(pl.col("model") != "neutral")
            .then(pl.lit("sweep"))
            .otherwise(pl.lit("neutral"))
            .alias("model")
        )
        regions = df_test["iter"].to_numpy()

        stats = []
        if _stats is not None:
            stats = stats + _stats
        test_stats = []

        self.num_stats = len(stats)

        for i in stats:
            test_stats.append(df_test.select(pl.col(f"^{i}_[0-9]+_[0-9]+$")))

        X_test = pl.concat(test_stats, how="horizontal")

        test_X_params = df_test.select("model", "iter", "s", "f_i", "f_t", "t")

        test_X = (
            X_test.select(X_test)
            .to_numpy()
            .reshape(
                X_test.shape[0],
                self.num_stats,
                self.windows.size * self.center.size,
                1,
            )
        )
        test_X = (
            X_test.select(X_test)
            .to_numpy()
            .reshape(
                X_test.shape[0],
                self.windows.size,
                self.center.size,
                self.num_stats,
            )
        )

        if preprocess:
            self.mean = test_X.mean(axis=(0, 1, 2), keepdims=False)
            self.std = test_X.std(axis=(0, 1, 2), keepdims=False)
            test_X = self.preprocess(test_X)

        out = model.predict(test_X, batch_size=32)
        # two heads → [classifier_probs, discriminator_probs]
        cls = out[0] if isinstance(out, (list, tuple)) else out
        p = cls.ravel().astype(np.float32)

        p_cal = self._apply_calibration(p)
        # p_cal = p
        df_pred = pl.concat(
            [
                test_X_params,
                pl.DataFrame(
                    {
                        "predicted_model": np.where(p_cal >= 0.5, "sweep", "neutral"),
                        "prob_sweep_raw": p,
                        "prob_sweep": p_cal,
                        "prob_neutral": 1.0 - p_cal,
                    }
                ),
            ],
            how="horizontal",
        )

        df_prediction = df_pred.with_columns(pl.Series("region", regions))
        chr_start_end = np.array(
            [item.replace(":", "-").split("-") for item in regions]
        )

        df_prediction = df_prediction.with_columns(
            pl.Series("chr", chr_start_end[:, 0]),
            pl.Series("start", chr_start_end[:, 1], dtype=pl.Int64),
            pl.Series("end", chr_start_end[:, 2], dtype=pl.Int64),
            pl.Series(
                "nchr",
                pl.Series(chr_start_end[:, 0]).str.replace("chr", "").cast(int),
            ),
        )
        df_prediction = df_prediction.sort("nchr", "start").select(
            [
                "chr",
                "start",
                "end",
                "f_i",
                "f_t",
                "s",
                "t",
                "predicted_model",
                "prob_sweep",
                "prob_neutral",
            ]
        )

        self.prediction = df_prediction

        if self.output_folder:
            # Same folder custom fvs name based on input VCF.
            _output_prediction = f"{self.output_folder}/{os.path.basename(self.predict_data).replace('fvs_', '').replace('.parquet', '_da_predictions.txt')}"

            if fname is not None:
                _output_prediction = f"{self.output_folder}/{fname}"

            df_prediction.write_csv(_output_prediction)

        return df_prediction

    def predict_da_f(self, _stats=None, preprocess=True, fname=None):
        """
        Predict sweep probabilities on empirical (target) data using a DA model.

        Loads a trained two-head model and returns per-region predictions from the
        **classifier** head (sweep vs. neutral). The domain head is unused at inference.

        Parameters
        ----------
        _stats : list[str] | None
            Statistic base names to include (must match training).

        Returns
        -------
        pl.DataFrame
            Table with per-region predictions and metadata, including:
            ``['chr','start','end','f_i','f_t','s','t','predicted_model',
            'prob_sweep','prob_neutral']`` sorted by chromosome and start.

        Raises
        ------
        AssertionError
            If no model is loaded or the test data path is invalid.

        Notes
        -----
        - Expects the same (W, C, S) layout used in training.
        - Output ``prob_sweep`` is the classifier sigmoid; ``prob_neutral=1-prob_sweep``.
        """
        assert self.model is not None, "Call train_da() first"

        tf = self.check_tf()
        if _stats is None:
            _stats = [
                "dind",
                "dist_kurtosis",
                "dist_skew",
                "dist_var",
                "h1",
                "h12",
                "h2_h1",
                "haf",
                "hapdaf_o",
                "hapdaf_s",
                "high_freq",
                "ihs",
                "isafe",
                "k_counts",
                "low_freq",
                "max_fda",
                "nsl",
                "omega_max",
                "pi",
                "s_ratio",
                "tajima_d",
                "theta_h",
                "theta_w",
                "zns",
            ]

        if isinstance(self.model, str):
            model = tf.keras.models.load_model(
                self.model,
                safe_mode=True,
            )
        else:
            model = self.model

        assert (
            isinstance(self.predict_data, pl.DataFrame)
            or "txt" in self.predict_data
            or "csv" in self.predict_data
            or self.predict_data.endswith(".parquet")
        ), "Please input a pl.DataFrame or save it as CSV or parquet"
        try:
            df_test = pl.read_parquet(self.predict_data)
            if "test" in self.predict_data:
                df_test = df_test.sample(
                    with_replacement=False, fraction=1.0, shuffle=True
                )
        except Exception:
            df_test = pl.read_csv(self.predict_data, separator=",")

        df_test = df_test.with_columns(
            pl.when(pl.col("model") != "neutral")
            .then(pl.lit("sweep"))
            .otherwise(pl.lit("neutral"))
            .alias("model")
        )
        regions = df_test["iter"].to_numpy()

        stats = []
        if _stats is not None:
            stats = stats + _stats
        test_stats = []

        for i in stats:
            test_stats.append(df_test.select(pl.col(f"^{i}_[0-9]+_[0-9]+$")))

        X_test = pl.concat(test_stats, how="horizontal")

        test_X_params = df_test.select("model", "iter", "s", "f_i", "f_t", "t")

        test_X = (
            X_test.select(X_test)
            .to_numpy()
            .reshape(
                X_test.shape[0],
                self.num_stats,
                self.windows.size * self.center.size,
                1,
            )
        )

        if preprocess:
            self.mean = test_X.mean(axis=(0, 1, 2), keepdims=False)
            self.std = test_X.std(axis=(0, 1, 2), keepdims=False)
            test_X = self.preprocess(test_X)

        out = model.predict(test_X, batch_size=32)
        # two heads → [classifier_probs, discriminator_probs]
        cls = out[0] if isinstance(out, (list, tuple)) else out
        p = cls.ravel().astype(np.float32)

        # p_cal = self._apply_calibration(p)
        p_cal = p
        df_pred = pl.concat(
            [
                test_X_params,
                pl.DataFrame(
                    {
                        "predicted_model": np.where(p_cal >= 0.5, "sweep", "neutral"),
                        "prob_sweep": p_cal,
                        "prob_neutral": 1.0 - p_cal,
                    }
                ),
            ],
            how="horizontal",
        )

        df_prediction = df_pred.with_columns(pl.Series("region", regions))
        chr_start_end = np.array(
            [item.replace(":", "-").split("-") for item in regions]
        )

        df_prediction = df_prediction.with_columns(
            pl.Series("chr", chr_start_end[:, 0]),
            pl.Series("start", chr_start_end[:, 1], dtype=pl.Int64),
            pl.Series("end", chr_start_end[:, 2], dtype=pl.Int64),
            pl.Series(
                "nchr",
                pl.Series(chr_start_end[:, 0]).str.replace("chr", "").cast(int),
            ),
        )
        df_prediction = df_prediction.sort("nchr", "start").select(
            [
                "chr",
                "start",
                "end",
                "f_i",
                "f_t",
                "s",
                "t",
                "predicted_model",
                "prob_sweep",
                "prob_neutral",
            ]
        )

        self.prediction = df_prediction

        if self.output_folder:
            # Same folder custom fvs name based on input VCF.
            _output_prediction = f"{self.output_folder}/{os.path.basename(self.predict_data).replace('fvs_', '').replace('.parquet', '_da_predictions.txt')}"

            if fname is not None:
                _output_prediction = f"{self.output_folder}/{fname}"

            df_prediction.write_csv(_output_prediction)

        return df_prediction

    def plot_da_curves(self):
        """
        Saves:
          - classifier_accuracy_hist.png   (train + val)
          - discriminator_accuracy_hist.png (train only)
          - classifier_loss_hist.png       (train + val)
          - discriminator_loss_hist.png    (train only)
          - classifier_auc_hist.png        (optional ROC/PR AUC, train + val)
          - confusion_matrix.png, auprc.png, calibration_curve.png, probability_hist.png
        """

        H = self.history
        outdir = self.output_folder or "."
        os.makedirs(outdir, exist_ok=True)

        def get(key):
            return H[key].to_numpy() if key in H.columns else np.array([])

        # names aligned to new training logs
        loss = get("loss")

        cls_loss, val_cls_loss = get("classifier_loss"), get("val_classifier_loss")
        cls_acc = get("classifier_accuracy")
        val_cls_acc = get("val_classifier_accuracy")

        disc_loss = get("discriminator_loss")
        disc_acc = get("discriminator_accuracy")

        # def L(*arrs):
        #     return max([len(a) for a in arrs if len(a) > 0] + [0])

        # T = L(loss, cls_loss, disc_loss, cls_acc, val_cls_acc, val_cls_loss)
        epochs = np.arange(1, len(loss) + 1)

        def savefig(name):
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, name), dpi=150)
            plt.close()

        def plot_series(y, label=None, ls="-", lw=2):
            if y.size:
                m = min(len(epochs), len(y))
                yy = y[:m]
                mask = np.isfinite(yy)
                if mask.any():
                    plt.plot(epochs[:m][mask], yy[mask], ls, linewidth=lw, label=label)

        # classifier accuracy
        plt.figure(figsize=(7, 4))
        plot_series(cls_acc, "train")
        plot_series(val_cls_acc, "val", ls="--")
        plt.title("classifier accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="lower right")
        savefig("classifier_accuracy_hist.png")

        # discriminator accuracy (train only)
        plt.figure(figsize=(7, 4))
        plot_series(disc_acc)
        plt.title("discriminator accuracy (train)")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.grid(True, alpha=0.3)
        savefig("discriminator_accuracy_hist.png")

        # classifier loss
        plt.figure(figsize=(7, 4))
        plot_series(cls_loss, "train")
        plot_series(val_cls_loss, "val", ls="--")
        plt.title("classifier loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        savefig("classifier_loss_hist.png")

        # discriminator loss (train only)
        plt.figure(figsize=(7, 4))
        plot_series(disc_loss)
        plt.title("discriminator loss (train)")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid(True, alpha=0.3)
        savefig("discriminator_loss_hist.png")

        # # optional AUCs
        # if any(len(s) > 0 for s in [ ]):
        #     plt.figure(figsize=(7, 4))
        #     plot_series(cls_auc, "ROC AUC (train)")
        #     plot_series(val_cls_auc, "ROC AUC (val)", ls="--")
        #     plot_series(cls_auc_pr, "PR AUC (train)")
        #     plot_series(val_cls_auc_pr, "PR AUC (val)", ls="--")
        #     plt.title("classifier AUCs")
        #     plt.xlabel("epoch")
        #     plt.ylabel("AUC")
        #     plt.ylim(0, 1)
        #     plt.grid(True, alpha=0.3)
        #     plt.legend(loc="lower right")
        #     savefig("classifier_auc_hist.png")

        # --- downstream prediction plots (unchanged) ---
        pred = self.prediction  # Polars DF with: model, predicted_model, prob_sweep

        y_true_labels = pred["model"]
        y_pred_labels = pred["predicted_model"]
        cm = confusion_matrix(
            y_true_labels, y_pred_labels, labels=["neutral", "sweep"], normalize="true"
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["neutral", "sweep"]
        )
        disp.plot(cmap="Blues")
        savefig("confusion_matrix.png")

        y_true = (pred["model"] == "sweep").cast(int).to_numpy()
        y_score = pred["prob_sweep"].cast(float).to_numpy()

        pr, rc, _ = precision_recall_curve(y_true, y_score)
        auc_pr = auc(rc, pr)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rc, pr, linewidth=2, label=f"AUC-PR = {auc_pr:.3f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall (positive = sweep)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower left")
        fig.tight_layout()
        savefig("auprc.png")

        y_score_clip = np.clip(y_score, 1e-6, 1 - 1e-6)
        prob_true, prob_pred = calibration_curve(
            y_true, y_score_clip, n_bins=10, strategy="quantile"
        )

        brier = brier_score_loss(y_true, y_score_clip)
        plt.figure(figsize=(7, 5))
        plt.plot([0, 1], [0, 1], "--", linewidth=1.5, label="perfect calibration")
        plt.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2,
            label=f"model (Brier={brier:.3f})",
        )
        plt.xlabel("Mean predicted probability (sweep)")
        plt.ylabel("Fraction of positives")
        plt.title("Calibration (Reliability Diagram)")
        plt.grid(True, alpha=0.4)
        plt.legend(loc="upper left")
        savefig("calibration_curve.png")

        plt.figure(figsize=(7, 3.2))
        plt.hist(y_score_clip, bins=20, range=(0, 1))
        plt.xlabel("Predicted probability (sweep)")
        plt.ylabel("Count")
        plt.title("Prediction Probability Histogram")
        plt.grid(True, alpha=0.25)
        savefig("probability_hist.png")

    def _fit_platt(self, y, p):
        # Only fit on finite logit values
        mask = (p > 0) & (p < 1)
        X = logit(p[mask]).reshape(-1, 1)
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(X, y[mask].astype(int))
        a = float(lr.coef_[0, 0])
        b = float(lr.intercept_[0])
        self.calibration = {"type": "platt", "a": a, "b": b}

    def _fit_temperature(self, y, p):
        from scipy.optimize import minimize

        # Only fit on finite logit values
        mask = (p > 0) & (p < 1)
        z = logit(p[mask])
        y_fit = y[mask]

        def nll(T):
            q = expit(z / T)
            # q is naturally bounded by expit, no clipping needed
            return -(y_fit * np.log(q) + (1 - y_fit) * np.log(1 - q)).mean()

        res = minimize(lambda t: nll(t[0]), x0=[1.0], bounds=[(0.5, 10.0)])
        T = float(res.x[0])
        self.calibration = {"type": "temperature", "T": T}

    def _apply_calibration(self, p):
        """Apply calibration only where mathematically defined."""
        if getattr(self, "calibration", None) is None:
            return p

        p_cal = p.copy()
        # Only transform where logit is defined
        mask = (p > 0) & (p < 1)

        if not np.any(mask):
            return p

        cal = self.calibration

        if cal["type"] == "platt":
            a, b = cal["a"], cal["b"]
            p_cal[mask] = expit(a * logit(p[mask]) + b)
        elif cal["type"] == "temperature":
            T = cal["T"]
            p_cal[mask] = expit(logit(p[mask]) / T)

        # p=0 and p=1 pass through unchanged (not in mask)
        return p_cal

    def _save_calibration(self):
        if getattr(self, "output_folder", None):
            with open(os.path.join(self.output_folder, "calibration.json"), "w") as f:
                json.dump(self.calibration, f)

    def _load_calibration(self):
        try:
            with open(os.path.join(self.output_folder, "calibration.json")) as f:
                self.calibration = json.load(f)
        except Exception:
            self.calibration = None


class DAParquetSequence(Sequence):
    """
    Domain-adversarial generator (binary: neutral=0, sweep=1) with adjustable target ratio.

    Matches CustomDataGenBinary's contract:
      __getitem__ -> X, {'classifier': y_cls, 'discriminator': y_discr}
      where labels are 1D float32 arrays with -1 as mask sentinel.

    Per step:
      - Classifier chunk (size = batch_size): SOURCE only (true labels 0/1), domain masked (-1).
      - Discriminator chunk (size = batch_size): SOURCE (domain=0) + TARGET (domain=1)
        split by `tgt_ratio` (target:source). With tgt_ratio=1, discriminator is 50/50.

    Total samples per step = 2 * batch_size (same as CustomDataGenBinary).
    """

    def __init__(
        self,
        src_neutral,
        src_sweep,
        tar_all,
        batch_size,
        tgt_ratio=1.0,
        shuffle=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert (
            batch_size % 2 == 0
        ), "batch_size must be even (keeps math tidy; total per step = 2*batch_size)."
        self.shuffle = bool(shuffle)

        # Trim to train set
        self.src_neutral = src_neutral
        self.src_sweep = src_sweep
        self.tar_all = tar_all

        # Basic sizes
        # classifier chunk size
        self.B = int(batch_size)
        # discriminator chunk size per step (keeps total = 2*B)
        self.disc_chunk = self.B

        # --- ratio → discriminator split (target:source = tgt_ratio) ---
        self.tgt_ratio = max(1e-6, float(tgt_ratio))
        # fraction of discriminator chunk to SOURCE
        src_frac = 1.0 / (1.0 + self.tgt_ratio)
        self.dis_src = max(1, int(round(self.disc_chunk * src_frac)))
        self.dis_tgt = max(1, self.disc_chunk - self.dis_src)  # ensure >=1 from both

        # Build flat SOURCE pool (combine neutral + sweep)
        neu_idx = np.arange(self.src_neutral.shape[0], dtype=np.int64)
        swp_idx = np.arange(self.src_sweep.shape[0], dtype=np.int64)
        # 0=neu,1=sweep
        self.src_pool_tag = np.concatenate(
            [
                np.zeros_like(neu_idx),
                np.ones_like(swp_idx),
            ]
        )
        # local indices into arrays
        self.src_pool_lidx = np.concatenate([neu_idx, swp_idx])

        # TARGET pool indices
        self.tgt_idx = np.arange(self.tar_all.shape[0], dtype=np.int64)

        # Build epoch pools (shuffled views)
        self._reset_epoch()

        # Steps/epoch limited by consumptions of each pool per step
        n_cls = len(self.src_pool_cls) // self.B
        n_dis_src = len(self.src_pool_dis) // self.dis_src
        n_dis_tgt = len(self.tgt_pool_dis) // self.dis_tgt
        self.n_batches = int(min(n_cls, n_dis_src, n_dis_tgt))

    def __len__(self):
        return self.n_batches

    def _reset_epoch(self):
        rng = np.random.default_rng()
        # independent shuffles for classifier and discriminator source pools
        base = np.arange(self.src_pool_tag.size, dtype=np.int64)
        self.src_pool_cls = rng.permutation(base)  # for classifier (labels used)
        self.src_pool_dis = rng.permutation(base)  # for discriminator (domain=0)
        self.tgt_pool_dis = rng.permutation(
            self.tgt_idx
        )  # for discriminator (domain=1)

    def on_epoch_end(self):
        if self.shuffle:
            self._reset_epoch()

    def _gather_source_arrays(self, take):
        """Return concatenated X and per-sample class labels (0/1) for given flat-source indices."""
        pools = self.src_pool_tag[take]
        lidx = self.src_pool_lidx[take]
        X_neu = self.src_neutral[lidx[pools == 0]]
        X_swp = self.src_sweep[lidx[pools == 1]]
        # Concatenate in stable order (neutral first then sweep)
        X = np.concatenate([X_neu, X_swp], axis=0)
        y = np.concatenate(
            [
                np.zeros((X_neu.shape[0],), dtype=np.float32),
                np.ones((X_swp.shape[0],), dtype=np.float32),
            ],
            axis=0,
        )
        return X, y

    def __getitem__(self, idx):
        # --- A) Classifier chunk: SOURCE (labels 0/1), domain masked ---
        idxA = self.src_pool_cls[idx * self.B : (idx + 1) * self.B]
        XA, yA = self._gather_source_arrays(idxA)
        yA_cls = yA  # shape (B,)
        yA_dom = -np.ones((XA.shape[0],), dtype=np.float32)  # mask domain

        # --- B) Discriminator chunk (SOURCE): domain=0, classifier masked ---
        idxB = self.src_pool_dis[idx * self.dis_src : (idx + 1) * self.dis_src]
        XB, _ = self._gather_source_arrays(idxB)
        yB_cls = -np.ones((XB.shape[0],), dtype=np.float32)  # mask classifier
        yB_dom = np.zeros((XB.shape[0],), dtype=np.float32)  # source domain=0

        # --- C) Discriminator chunk (TARGET): domain=1, classifier masked ---
        idxC = self.tgt_pool_dis[idx * self.dis_tgt : (idx + 1) * self.dis_tgt]
        XC = self.tar_all[idxC]
        yC_cls = -np.ones((XC.shape[0],), dtype=np.float32)  # mask classifier
        yC_dom = np.ones((XC.shape[0],), dtype=np.float32)  # target domain=1

        # --- Concatenate in the same order as CustomDataGenBinary ---
        X = np.concatenate([XA, XB, XC], axis=0)
        y_cls = np.concatenate([yA_cls, yB_cls, yC_cls], axis=0)
        y_dis = np.concatenate([yA_dom, yB_dom, yC_dom], axis=0)

        # Safety (total = 2*B even with arbitrary tgt_ratio because disc_chunk==B)
        assert X.shape[0] == 2 * self.B
        assert y_cls.shape[0] == y_dis.shape[0] == X.shape[0]

        return X, {"classifier": y_cls, "discriminator": y_dis}


def subset_genomic_windows(
    df: pl.DataFrame,
    centers: list[int],
    metrics: list[str] | None = None,
    window_size: int = 100_000,
    update_iter: bool = True,
) -> pl.DataFrame:
    window_size = int(window_size)
    centers = np.sort(centers).astype(int)
    half = window_size // 2

    # Validate consecutive centers
    if len(centers) > 1:
        diffs = np.diff(centers)
        if not np.all(diffs == window_size):
            raise ValueError("Centers must be consecutive and spaced by window_size")

    left_offset = centers[0] - half
    right_offset = centers[-1] + half

    # Define base columns to keep
    base_keep = ["iter", "s", "t", "f_i", "f_t", "mu", "r", "model"]

    # Resolve metrics if not provided
    if metrics is None:
        # Get columns that aren't in base_keep
        other_cols = [c for c in df.columns if c not in base_keep]
        # Extract base names (dist_var, etc) preserving order
        raw_names = ["_".join(col.split("_")[:-2]) for col in other_cols]
        metrics = list(dict.fromkeys(raw_names))

    # Generate the specific windowed column names we expect
    expected_cols = [f"{m}_{window_size}_{c}" for c in centers for m in metrics]

    # Final column list: intersection of (base + expected) and what actually exists
    all_potential = base_keep + expected_cols
    existing_cols = [c for c in all_potential if c in df.columns]

    out = df.select(existing_cols)

    if update_iter and "iter" in out.columns:
        out = (
            out.with_columns(
                pl.col("iter").str.extract(r"(chr\w+):(\d+)-", 1).alias("_chrom"),
                pl.col("iter")
                .str.extract(r":(\d+)-", 1)
                .cast(pl.Int64)
                .alias("_start"),
            )
            .with_columns(
                pl.format(
                    "{}:{}-{}",
                    pl.col("_chrom"),
                    pl.col("_start") + left_offset,
                    pl.col("_start") + right_offset,
                ).alias("iter")
            )
            .drop(["_chrom", "_start"])
        )

    return out
