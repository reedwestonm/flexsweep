"""
flexsweep/cnn_regressor.py
==========================
CNN **regression** of sweep parameters (``s``, ``t``).

Generic regression derived from :mod:`flexsweep.cnn`.
  * columns selected with ``cnn.py``'s 24-stat list and per-stat regex concat
    (:data:`DEFAULT_STATS`, :meth:`CNNRegressor._feature_tensor`);
  * tensor shaped ``(N, num_stats, W*C, 1)``, the layout ``CNN.train`` feeds
    :meth:`flexsweep.cnn.CNN.cnn_flexsweep`;
  * two-way split via ``sklearn.train_test_split`` as
    :meth:`flexsweep.cnn.CNN.load_training_data` does, optionally stratified by
    subclass (``cnn.py`` does not stratify — enabled by default here, since
    unstratified splitting leaves rare subclasses unevenly represented across
    train/valid/test);
  * optimizer and schedule copied from ``CNN.train`` (Adam + ``CosineDecayRestarts``);
  * targets are the parquet columns **as stored** unless ``s_natural=True``;
  * outputs named ``{target}_pred``.

Two departures from the classification path:

  * **Sweep-only data, unconditionally.** Neutral rows carry ``s = t = 0``
    sentinels, not measurements; training on them teaches the head a point mass at
    zero. Both :meth:`CNNRegressor.load_training_data` and
    :meth:`CNNRegressor.predict` drop ``model == "neutral"`` rows — there is no
    flag to keep them.
  * **Early stopping on ``val_mae``.** ``val_accuracy`` does not exist for a
    regression head, and ``cnn.py``'s patience of 5 is too tight against the
    cosine-restart schedule, whose loss oscillates by construction.

Because the filter also applies at predict time, empirical tables must arrive
pre-filtered: feature vectors built from a VCF carry ``model = "neutral"`` as a
placeholder on **every** window, so passing one straight through would leave no
rows. :meth:`CNNRegressor._drop_neutral` raises rather than returning an empty
frame in that case.

Target units
------------
The reference parquets store ``s`` as ``-ln(s)`` (values ~[3.0, 4.6], i.e. a
linear selection coefficient ~[0.01, 0.05]) and ``t`` in generations. Set
``s_natural=True`` to decode ``s`` to the linear coefficient before fitting; the
flag changes the UNITS of the targets and of every reported error, and is stored
in the scaling sidecar so predictions come back in the units they were fit in.

Target standardization (``scale_targets``, default ``True``)
------------------------------------------------------------
``t`` ~ O(1e3) generations and ``s`` ~ O(1e-2), so in a shared MSE the gradient
ratio is ~1e10 and the ``s`` head never leaves the prior mean. Fitting two targets
of different scale jointly leaves only two options — standardize the targets, or
weight the per-target losses — and standardizing is the conventional one. It is
exposed as a flag so the failure mode can be reproduced: with
``scale_targets=False`` expect ``s`` to converge to its prior mean, which the
reported ``pearson_r ~ 0`` / ``bias ~ 0`` / ``mae ~ baseline_mae`` signature makes
explicit.

Feature vectors are consumed as they are (``preprocess=False`` upstream — the
parquet is already z-scored by ``fv.py``); no re-normalization is applied.

See Also
--------
flexsweep.cnn.CNN : the classifier this module subclasses; supplies the tower,
    the grid attributes (``center``/``step``/``windows``) and ``train_split``.
"""

import json
import os
import time

import tensorflow as tf
from sklearn.model_selection import train_test_split

from . import np, pl
from .cnn import CNN

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: Default statistics — the classifier's list verbatim (``CNN.train``).
#: ``delta_ihh`` is absent here exactly as it is there; it is computed upstream
#: but intentionally excluded from every model in the package.
DEFAULT_STATS: list[str] = [
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

#: Meta columns carried through to the prediction tables when present.
META_COLS: tuple[str, ...] = ("model", "iter", "s", "t", "f_i", "f_t", "mu", "r")

#: Layer of ``cnn_flexsweep``'s graph the regression head replaces the sigmoid on.
#: This is the last dropout before ``out_dense``; renaming it in ``cnn.py`` breaks
#: :meth:`CNNRegressor.cnn_flexsweep_regression` loudly (``get_layer`` raises).
TRUNK_LAYER: str = "dropconcat2"

#: Loss names accepted by :meth:`CNNRegressor.train`.
SUPPORTED_LOSSES: tuple[str, ...] = ("mse", "mae", "huber")


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------


class CNNRegressor(CNN):
    """
    Regress sweep parameters with the Flex-sweep CNN tower.

    Subclasses :class:`flexsweep.cnn.CNN`, so ``center``, ``step``, ``windows``,
    ``train_split`` and ``output_folder`` behave exactly as they do for the
    classifier. The ``center``/``step``/``windows`` passed to the constructor must
    reproduce the grid of the parquet — :meth:`load_training_data` checks the
    implied column count and fails loudly if it does not match.

    Parameters
    ----------
    *args, **kwargs
        Forwarded verbatim to :meth:`flexsweep.cnn.CNN.__init__`.
    targets : sequence of str, default=("s", "t")
        Parquet columns to regress.
    s_natural : bool, default=False
        If True, map the stored ``s`` column through ``exp(-s)`` before training —
        i.e. treat it as ``-ln(s)`` and recover the linear selection coefficient.
        Leave False to regress the column exactly as stored. This changes only the
        UNITS of the targets and of every reported error.
    scale_targets : bool, default=True
        Z-score each target using TRAIN-split statistics before fitting, and invert
        at predict time. See the module docstring; leave enabled unless the
        collapse is being reproduced deliberately.
    stratify : bool, default=True
        Stratify the train/valid/test splits by the ``model`` subclass column when
        it is present.

    Attributes
    ----------
    y_mu, y_sd : np.ndarray | None
        Per-target train-split mean/std used to standardize the targets. Set by
        :meth:`load_training_data`, persisted next to the model, and applied in
        reverse to decode predictions.
    feature_names : list[str] | None
        Statistics the tensor was built from, in column order.
    test_params : pl.DataFrame | None
        Meta columns of the held-out test split.
    metrics_ : pl.DataFrame | None
        Per-target test-split ``mae``/``rmse``/``pearson_r``/``bias`` plus
        ``baseline_mae``, the MAE of a mean-predictor. A head sitting at or above
        ``baseline_mae`` with ``pearson_r ~ 0`` has learned nothing.
    prediction : pl.DataFrame | None
        Test-split predictions from the last :meth:`train` call.

    Notes
    -----
    - Neutral rows are dropped on every path; see :meth:`_drop_neutral`.
    - TensorFlow is imported at module level, matching ``cnn.py``. Importing this
      module therefore costs the same as importing :mod:`flexsweep.cnn`, which it
      pulls in anyway.
    """

    def __init__(
        self,
        *args,
        targets=("s", "t"),
        s_natural=False,
        scale_targets=True,
        stratify=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.targets = list(targets)
        self.s_natural = bool(s_natural)
        self.scale_targets = bool(scale_targets)
        self.stratify = bool(stratify)
        self.y_mu = None
        self.y_sd = None
        self.feature_names = None
        self.test_params = None
        self.metrics_ = None

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------

    def _drop_neutral(self, df: pl.DataFrame, context: str) -> pl.DataFrame:
        """
        Drop ``model == "neutral"`` rows. The regressor never sees neutral data.

        Neutral replicates carry ``s = t = 0`` sentinels rather than measurements,
        so they are removed on every path — training and prediction alike. Tables
        without a ``model`` column are assumed to be pre-filtered and pass through
        untouched.

        Parameters
        ----------
        df : pl.DataFrame
            Table to filter.
        context : str
            Short label naming the caller, used in the log line and the error.

        Returns
        -------
        pl.DataFrame
            ``df`` without its neutral rows.

        Raises
        ------
        ValueError
            If every row is neutral. This is the expected outcome for a raw
            empirical table: ``fv.py`` labels every VCF window ``"neutral"`` as a
            placeholder, so such a table must be reduced to classifier-flagged
            sweeps (or have its ``model`` column dropped) before it gets here.
        """
        if "model" not in df.columns:
            return df

        n0 = df.height
        out = df.filter(~pl.col("model").str.contains("neutral"))
        print(
            f"sweep-only ({context}): {n0} -> {out.height} rows "
            f"({n0 - out.height} neutral dropped)"
        )
        if out.is_empty():
            raise ValueError(
                f'no rows left after dropping `model == "neutral"` ({context}): '
                f"all {n0} rows are labelled neutral. Feature vectors built from a "
                'VCF carry `model = "neutral"` on every window as a placeholder — '
                "subset the table to the windows the classifier flagged as sweeps, "
                "or drop the `model` column, before passing it here."
            )
        return out

    def _targets_from(self, df: pl.DataFrame) -> np.ndarray:
        """
        Build the target matrix in the units the head will learn.

        Parameters
        ----------
        df : pl.DataFrame
            Table holding every column named in :attr:`targets`.

        Returns
        -------
        np.ndarray
            ``(N, n_targets)`` float32 matrix, columns ordered as
            :attr:`targets`. ``s`` is decoded with ``exp(-s)`` when
            :attr:`s_natural` is set.
        """
        cols = []
        for nm in self.targets:
            v = df[nm].to_numpy().astype(np.float64)
            if nm == "s" and self.s_natural:
                v = np.exp(-v)
            cols.append(v)
        return np.column_stack(cols).astype(np.float32)

    def _meta_frame(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Select the meta columns for the output table, in the SAME units as the targets.

        Parameters
        ----------
        df : pl.DataFrame
            Source table; only the columns present in :data:`META_COLS` are kept.

        Returns
        -------
        pl.DataFrame
            Meta columns of ``df``. When :attr:`s_natural` is set the stored
            ``-ln(s)`` column is decoded, so a true ``s`` column always sits on the
            same scale as ``s_pred`` and errors recomputed from the table match
            :attr:`metrics_`.
        """
        meta = [c for c in META_COLS if c in df.columns]
        out = df.select(meta)
        if self.s_natural and "s" in meta:
            out = out.with_columns(
                pl.col("s").map_batches(lambda v: pl.Series(np.exp(-v.to_numpy())))
            )
        return out

    def _feature_tensor(self, df: pl.DataFrame, stats: list) -> np.ndarray:
        """
        Build the CNN input tensor from per-stat regex blocks.

        Columns are gathered one statistic at a time so the channel order follows
        ``stats`` rather than the parquet's column order — the same selection
        :meth:`flexsweep.cnn.CNN.load_training_data` performs.

        Parameters
        ----------
        df : pl.DataFrame
            Feature table with ``{stat}_{window}_{center}`` columns.
        stats : list of str
            Statistics to select, in channel order.

        Returns
        -------
        np.ndarray
            ``(N, num_stats, W*C, 1)`` float32 tensor. Non-finite values are
            replaced by 0.0, which is the mean of the z-scored features.

        Raises
        ------
        ValueError
            If the number of selected columns does not match the grid implied by
            ``stats`` x :attr:`windows` x :attr:`center`.
        """
        blocks = [df.select(pl.col(f"^{i}_[0-9]+_[0-9]+$")) for i in stats]
        X_df = pl.concat(blocks, how="horizontal")
        n_expected = len(stats) * self.windows.size * self.center.size
        if X_df.width != n_expected:
            raise ValueError(
                f"selected {X_df.width} feature columns but the configured grid "
                f"implies {n_expected} ({len(stats)} stats x "
                f"{self.windows.size} windows x {self.center.size} centers). "
                "Pass matching `center`/`step`/`windows` to the constructor, or "
                "trim `_stats` to the statistics actually in the table."
            )
        X = np.nan_to_num(
            X_df.to_numpy().astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        return X.reshape(
            X.shape[0], len(stats), self.windows.size * self.center.size, 1
        )

    def load_training_data(self, _stats=None, w=None, n=None, one_dim=False):
        """
        Load sweep-only feature tensors and continuous targets, split 80/10/10.

        Mirrors :meth:`flexsweep.cnn.CNN.load_training_data` but drops neutral rows
        (:meth:`_drop_neutral`) and returns ``(s, t)`` instead of a 0/1 label.

        Parameters
        ----------
        _stats : list[str] | None, default=None
            Statistic base names to include. Defaults to :data:`DEFAULT_STATS`.
        w : int | list[int] | None, default=None
            Override :attr:`center` with these center coordinates before selecting
            columns.
        n : int | None, default=None
            Optional number of rows to sample from the table.
        one_dim : bool, default=False
            Accepted for signature compatibility with the classifier; **ignored**.
            The regression head consumes the 2D ``(num_stats, W*C, 1)`` layout only.

        Returns
        -------
        tuple
            ``(X_train, X_test, Y_train, Y_test, X_valid, Y_valid)``. The ``Y``
            arrays are STANDARDIZED when :attr:`scale_targets` is set; natural-unit
            test targets and meta columns are stashed on
            :attr:`~flexsweep.cnn.CNN.test_train_data`.

        Raises
        ------
        AssertionError
            If ``train_data`` is unset.
        ValueError
            If the table is missing a requested target column, or if no sweep rows
            survive :meth:`_drop_neutral`.
        """
        assert self.train_data is not None, "Please input training data"

        if isinstance(self.train_data, pl.DataFrame):
            tmp = self.train_data
        elif self.train_data.endswith(".parquet"):
            tmp = pl.read_parquet(self.train_data)
        else:
            tmp = pl.read_csv(self.train_data, separator=",")
        if n is not None:
            tmp = tmp.sample(n)

        missing = [c for c in self.targets if c not in tmp.columns]
        if missing:
            raise ValueError(f"targets={self.targets} but table is missing {missing}")

        tmp = self._drop_neutral(tmp, "train")

        if w is not None:
            self.center = (
                np.array([int(w)])
                if np.ndim(w) == 0
                else np.sort(np.asarray(w).astype(int))
            )

        stats = list(_stats) if _stats is not None else list(DEFAULT_STATS)
        self.num_stats = len(stats)
        self.feature_names = stats

        X = self._feature_tensor(tmp, stats)
        Y = self._targets_from(tmp)
        params = self._meta_frame(tmp)
        print(
            f"{tmp.height} rows | input {X.shape[1:]} "
            f"({self.num_stats} stats x {self.windows.size} windows x "
            f"{self.center.size} centers)"
        )

        # Subclass labels (hard/soft x young/old x complete/incomplete) are unevenly
        # sized, so stratify to keep every subclass represented in all three splits.
        strat = (
            tmp["model"].to_numpy()
            if self.stratify and "model" in tmp.columns
            else None
        )
        idx = np.arange(X.shape[0])
        i_tr, i_rest = train_test_split(
            idx, test_size=round(1 - self.train_split, 2), shuffle=True, stratify=strat
        )
        i_val, i_te = train_test_split(
            i_rest, test_size=0.5, stratify=strat[i_rest] if strat is not None else None
        )
        print(f"train={len(i_tr)} valid={len(i_val)} test={len(i_te)}")

        # Scaling statistics come from the TRAIN split only — using all rows would
        # leak the test distribution into the decode step.
        if self.scale_targets:
            self.y_mu = Y[i_tr].mean(0)
            self.y_sd = Y[i_tr].std(0)
            self.y_sd[self.y_sd == 0] = 1.0
            print(
                "target scaling:",
                {
                    nm: (round(float(a), 6), round(float(b), 6))
                    for nm, a, b in zip(self.targets, self.y_mu, self.y_sd)
                },
            )
        else:
            self.y_mu = np.zeros(len(self.targets), np.float32)
            self.y_sd = np.ones(len(self.targets), np.float32)
            print(
                "target scaling: DISABLED — targets enter the loss in native "
                "units; expect the small-scale target to collapse to its mean"
            )
        Yz = (Y - self.y_mu) / self.y_sd

        self.test_params = params[i_te.tolist()]
        self.test_train_data = [X[i_te], self.test_params, Y[i_te]]
        return X[i_tr], X[i_te], Yz[i_tr], Yz[i_te], X[i_val], Yz[i_val]

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    def cnn_flexsweep_regression(self, model_input):
        """
        Build the inherited tower with a linear ``Dense(n_targets)`` head.

        Cuts :meth:`flexsweep.cnn.CNN.cnn_flexsweep`'s graph at
        :data:`TRUNK_LAYER` — equivalent to swapping its
        ``Dense(1, activation="sigmoid")`` output for ``Dense(n_targets)``,
        without editing ``cnn.py``. The orphaned sigmoid is dropped by Keras when
        the functional model is assembled, since it no longer reaches an output.

        Parameters
        ----------
        model_input : tf.keras.layers.Input
            Input tensor of shape ``(num_stats, W*C, 1)``.

        Returns
        -------
        tf.Tensor
            Linear ``(n_targets,)`` output tensor.
        """
        base = tf.keras.Model(
            inputs=model_input, outputs=super().cnn_flexsweep(model_input)
        )
        trunk = base.get_layer(TRUNK_LAYER).output
        return tf.keras.layers.Dense(
            len(self.targets), activation=None, name="reg_head"
        )(trunk)

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------

    def train(
        self,
        _stats=None,
        w=None,
        loss="mse",
        batch_size=64,
        epochs=300,
        patience=10,
        lr=1e-4,
        verbose=2,
    ):
        """
        Fit the regressor and report test-split errors in natural units.

        Parameters
        ----------
        _stats : list[str] | None, default=None
            Statistic base names. Defaults to :data:`DEFAULT_STATS`.
        w : int | list[int] | None, default=None
            Center coordinate(s) to select (see :meth:`load_training_data`).
        loss : {"mse", "mae", "huber"}, default="mse"
            Applied to the (optionally standardized) targets. ``"huber"``
            down-weights the sparse old-sweep tail, which otherwise dominates the
            ``t`` gradient.
        batch_size : int, default=64
            Minibatch size for fitting and for the test-split prediction.
        epochs : int, default=300
            Maximum epochs; early stopping usually fires first.
        patience : int, default=10
            Early-stopping patience on ``val_mae``. Much larger than the
            classifier's 5 because ``CosineDecayRestarts`` makes the loss
            oscillate by design.
        lr : float, default=1e-4
            Initial learning rate of the cosine-restart schedule.
        verbose : int, default=2
            Keras verbosity, forwarded to ``fit`` and to the callbacks.

        Returns
        -------
        pl.DataFrame
            Test-split predictions: meta columns + ``{target}_pred``.

        Raises
        ------
        ValueError
            If ``loss`` is not one of :data:`SUPPORTED_LOSSES`.

        Notes
        -----
        - Optimizer: Adam with cosine-restarts schedule, as in ``CNN.train``.
        - Early stopping and checkpointing monitor ``val_mae``; ``val_accuracy``
          does not exist for a regression head.
        - Saves ``model_regressor.keras`` plus its scaling sidecar to
          ``output_folder`` when one is set.
        """
        # Standardized test targets are unpacked but unused: the test split is
        # scored against `test_train_data`, which holds them in natural units.
        X_tr, X_te, Y_tr, _Y_te, X_va, Y_va = self.load_training_data(
            _stats=_stats, w=w
        )

        inp = tf.keras.Input(X_tr.shape[1:])
        model = tf.keras.Model(
            inputs=inp,
            outputs=self.cnn_flexsweep_regression(inp),
            name="cnn_flexsweep_regressor",
        )

        losses = {"mse": "mse", "mae": "mae", "huber": tf.keras.losses.Huber()}
        if loss not in losses:
            raise ValueError(f"loss={loss!r} not in {SUPPORTED_LOSSES}")

        lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=lr, first_decay_steps=300
        )
        opt_adam = tf.keras.optimizers.Adam(
            learning_rate=lr_decayed_fn, epsilon=0.0000001, amsgrad=True
        )

        model.compile(
            optimizer=opt_adam,
            loss=losses[loss],
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )

        model_path = os.path.join(self.output_folder or ".", "model_regressor.keras")

        earlystop = tf.keras.callbacks.EarlyStopping(
            monitor="val_mae",
            min_delta=1e-5,
            patience=patience,
            verbose=verbose,
            mode="min",
            restore_best_weights=True,
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_mae",
            verbose=verbose,
            save_best_only=True,
            mode="min",
        )

        callbacks_list = [checkpoint, earlystop]

        start = time.time()

        history = model.fit(
            X_tr,
            Y_tr,
            validation_data=(X_va, Y_va),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose,
        )
        print(f"Training took {round(time.time() - start, 3)} seconds")

        self.model = model
        self.history = pl.DataFrame(history.history)

        if self.output_folder is not None:
            model.save(model_path)
            self._save_scaling(model_path)

        _, params, Y_true = self.test_train_data
        preds = model.predict(X_te, batch_size=batch_size, verbose=0)
        preds = preds * self.y_sd + self.y_mu

        self.metrics_ = self._score(preds, Y_true)
        print(self.metrics_)

        df_prediction = pl.concat(
            [
                params,
                pl.DataFrame(
                    {f"{nm}_pred": preds[:, i] for i, nm in enumerate(self.targets)}
                ),
            ],
            how="horizontal",
        )
        self.prediction = df_prediction

        if self.output_folder is not None:
            df_prediction.write_csv(
                os.path.join(self.output_folder, "regressor_predictions.txt")
            )
            self.metrics_.write_csv(
                os.path.join(self.output_folder, "regressor_metrics.csv")
            )

        return df_prediction

    def _score(self, preds: np.ndarray, truth: np.ndarray) -> pl.DataFrame:
        """
        Build the per-target error table, in natural units.

        Parameters
        ----------
        preds : np.ndarray
            ``(N, n_targets)`` decoded predictions.
        truth : np.ndarray
            ``(N, n_targets)`` true values, same units and column order.

        Returns
        -------
        pl.DataFrame
            One row per target with ``mae``, ``rmse``, ``pearson_r``, ``bias`` and
            ``baseline_mae``.

        Notes
        -----
        ``baseline_mae`` is the MAE of a constant mean-predictor: a head at or
        above it, with ``pearson_r ~ 0`` and ``bias ~ 0``, has collapsed to the
        prior mean rather than learned anything.
        """
        rows = []
        for i, nm in enumerate(self.targets):
            y, p = truth[:, i], preds[:, i]
            e = p - y
            rows.append(
                dict(
                    target=nm,
                    mae=float(np.mean(np.abs(e))),
                    rmse=float(np.sqrt(np.mean(e**2))),
                    # Pearson corr(pred, true) — NOT the recombination rate `r`
                    # that appears as a meta column in the prediction tables.
                    pearson_r=(
                        float(np.corrcoef(p, y)[0, 1]) if len(y) > 2 else float("nan")
                    ),
                    bias=float(np.mean(e)),
                    baseline_mae=float(np.mean(np.abs(y - y.mean()))),
                )
            )
        return pl.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Persistence of the target scaling (needed to decode predictions)
    # -----------------------------------------------------------------------

    def _scaling_path(self, model_path: str) -> str:
        """Path of the JSON sidecar sitting next to a saved model."""
        return f"{model_path}.scaling.json"

    def _save_scaling(self, model_path: str) -> None:
        """
        Write the target scaling, stat list and grid next to the saved model.

        Parameters
        ----------
        model_path : str
            Path the Keras model was saved to; the sidecar is written alongside it.
        """
        with open(self._scaling_path(model_path), "w") as fh:
            json.dump(
                dict(
                    targets=self.targets,
                    y_mu=np.asarray(self.y_mu).tolist(),
                    y_sd=np.asarray(self.y_sd).tolist(),
                    s_natural=self.s_natural,
                    scale_targets=self.scale_targets,
                    feature_names=self.feature_names,
                    center=np.asarray(self.center).tolist(),
                    windows=np.asarray(self.windows).tolist(),
                ),
                fh,
            )

    def _load_scaling(self, model_path: str) -> bool:
        """
        Restore the target scaling, stat list and grid from a model's sidecar.

        Parameters
        ----------
        model_path : str
            Path of the saved Keras model whose sidecar should be read.

        Returns
        -------
        bool
            True when a sidecar was found and applied, False when none exists.
        """
        path = self._scaling_path(model_path)
        if not os.path.exists(path):
            return False
        with open(path) as fh:
            d = json.load(fh)
        self.targets = d["targets"]
        self.y_mu = np.asarray(d["y_mu"], dtype=np.float32)
        self.y_sd = np.asarray(d["y_sd"], dtype=np.float32)
        self.s_natural = d.get("s_natural", self.s_natural)
        self.feature_names = d.get("feature_names")
        self.center = np.asarray(d["center"])
        self.windows = np.asarray(d["windows"])
        return True

    # -----------------------------------------------------------------------
    # Predict
    # -----------------------------------------------------------------------

    def predict(self, _stats=None, w=None, fname=None, batch_size=256):
        """
        Score ``predict_data`` and return meta columns + ``{target}_pred``.

        Parameters
        ----------
        _stats : list[str] | None, default=None
            Statistic base names. Defaults to :attr:`feature_names` when a model
            or sidecar supplied them, otherwise :data:`DEFAULT_STATS`.
        w : int | list[int] | None, default=None
            Center coordinate(s) to select (see :meth:`load_training_data`).
        fname : str | None, default=None
            File name written under ``output_folder``; nothing is written when
            either is None.
        batch_size : int, default=256
            Minibatch size for inference.

        Returns
        -------
        pl.DataFrame
            Meta columns + one ``{target}_pred`` column per target, decoded to the
            units the model was fit in.

        Raises
        ------
        AssertionError
            If no model or no ``predict_data`` is set, or if the model is a path
            with neither a sidecar nor an in-memory scaling to decode with.
        ValueError
            If every row of ``predict_data`` is neutral (see :meth:`_drop_neutral`).

        Notes
        -----
        - Neutral rows are dropped here too, so an empirical table must already be
          reduced to the windows the classifier flagged as sweeps.
        - When :attr:`model` is a path, the sidecar written next to it restores the
          target scaling, stat list and grid, so a model can be reloaded in a fresh
          session without re-specifying them.
        """
        assert self.model is not None, "Train a model or set `model` to a path"
        assert self.predict_data is not None, "Please input predict_data"

        if isinstance(self.model, str):
            if not self._load_scaling(self.model):
                assert self.y_mu is not None, (
                    "no scaling sidecar next to the model and none in memory — "
                    "predictions cannot be decoded to natural units"
                )
            model = tf.keras.models.load_model(self.model, compile=False)
        else:
            model = self.model

        df_test = (
            pl.read_parquet(self.predict_data)
            if isinstance(self.predict_data, str)
            else self.predict_data
        )
        df_test = self._drop_neutral(df_test, "predict")

        if w is not None:
            self.center = (
                np.array([int(w)])
                if np.ndim(w) == 0
                else np.sort(np.asarray(w).astype(int))
            )

        stats = (
            list(_stats)
            if _stats is not None
            else (self.feature_names or list(DEFAULT_STATS))
        )
        test_X = self._feature_tensor(df_test, stats)
        preds = model.predict(test_X, batch_size=batch_size, verbose=0)
        preds = preds * self.y_sd + self.y_mu

        df_prediction = pl.concat(
            [
                self._meta_frame(df_test),
                pl.DataFrame(
                    {f"{nm}_pred": preds[:, i] for i, nm in enumerate(self.targets)}
                ),
            ],
            how="horizontal",
        )

        # Labeled tables (simulations) get scored too; empirical ones cannot be.
        if all(nm in df_test.columns for nm in self.targets):
            self.metrics_ = self._score(preds, self._targets_from(df_test))
            print(self.metrics_)

        self.prediction = df_prediction

        if self.output_folder is not None and fname is not None:
            df_prediction.write_csv(os.path.join(self.output_folder, fname))

        return df_prediction
