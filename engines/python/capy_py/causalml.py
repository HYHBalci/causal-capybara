"""Heterogeneous effects, causal machine learning, and TMLE (plan 7.10 + 7.12).

Six adapters, one shared cross-fitting engine:

``ml.dml_plr``       double/debiased ML, partially linear model
``ml.dml_irm``       interactive regression model / cross-fitted AIPW (ATE and ATT)
``ml.causal_forest`` honest causal forest with local centring and OOB CATEs
``ml.metalearner``   T-, S-, X- and DR-learners
``ml.policy_tree``   shallow policy tree with an honest held-out policy value
``obs.tmle``         targeted maximum likelihood for the ATE and the ATT

The house rules that shape this module:

* A learner is a nuisance model, never an identification argument. Nothing in
  here can make treatment assignment ignorable; every result says so out loud.
* Whenever a learner is used the run must carry ``nuisance_rmse``,
  ``propensity_clipping``, ``cate_distribution``, ``cate_calibration``,
  ``rate`` and ``fold_stability`` -- and ``seed_stability`` when the user asks
  for repeats. Those are the diagnostics that tell you whether the
  cross-fitting actually worked.
* Default learners are boring: a regularised linear model, or an honest forest.
  A 12-layer net would not make the confounding go away.
* Where we implement a simplified version of a published procedure, the
  simplification is named in the classic printout and in the method card's
  ``what_can_go_wrong``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from . import roles, stats, vega
from .contracts import (
    CapyError,
    DataError,
    EngineError,
    ResultBuilder,
    RunContext,
    SpecError,
    adapter,
)

try:  # scikit-learn is the only extra dependency of this module
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.linear_model import LassoCV, LogisticRegression, RidgeCV
    from sklearn.model_selection import KFold, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    _SKLEARN_ERROR: str | None = None
except Exception as _exc:  # pragma: no cover - the engine must still load
    _SKLEARN_ERROR = f"{type(_exc).__name__}: {_exc}"


try:
    from sklearn.metrics import roc_auc_score as _roc_auc_score
except Exception:  # pragma: no cover
    _roc_auc_score = None


def _logreg_uses_l1_ratio() -> bool:
    """scikit-learn >= 1.8 spells the penalty as ``l1_ratio``; older versions do not."""
    if _SKLEARN_ERROR:
        return False
    try:
        import inspect

        params = inspect.signature(LogisticRegression.__init__).parameters
        return "l1_ratio" in params and params["penalty"].default == "deprecated"
    except Exception:  # pragma: no cover
        return False


_LOGREG_USES_L1_RATIO = _logreg_uses_l1_ratio()


PACKAGE = "capy.py"
DESIGN = "observational"

LEARNER_CHOICES = ("linear", "lasso", "forest", "gradient_boosting")
LEARNER_LABEL = {
    "linear": "ridge-regularised linear / logistic",
    "lasso": "lasso (L1) linear / logistic",
    "forest": "random forest",
    "gradient_boosting": "histogram gradient boosting",
}

_TREE_LEAF = -1


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(fn(np.asarray(y, dtype=float), np.asarray(x, dtype=float)))


def package_version() -> str:
    try:
        import sklearn

        return f"0.1.0 (scikit-learn {sklearn.__version__})"
    except Exception:  # pragma: no cover
        return "0.1.0"


def _require_sklearn() -> None:
    if _SKLEARN_ERROR:
        raise EngineError(
            "This method needs scikit-learn, which the Python engine could not import.",
            detail=_SKLEARN_ERROR,
        )


# ---------------------------------------------------------------------------
# Option plumbing -- user errors get a sentence, never a traceback
# ---------------------------------------------------------------------------


def _choice(ctx: RunContext, key: str, choices: Sequence[str], default: str) -> str:
    raw = ctx.opt(key, default)
    value = str(default if raw is None else raw).strip().lower()
    if value not in choices:
        raise SpecError(
            f"Option '{key}' must be one of: {', '.join(choices)}. You asked for '{value}'.",
            detail="Pick one of the listed values in the method card's Advanced panel.",
        )
    return value


def _int_opt(ctx: RunContext, key: str, default: int, lo: int, hi: int) -> int:
    raw = ctx.opt(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise SpecError(f"Option '{key}' must be a whole number; you gave '{raw}'.") from None
    if value < lo or value > hi:
        raise SpecError(f"Option '{key}' must be between {lo} and {hi}; you gave {value}.")
    return value


def _float_opt(ctx: RunContext, key: str, default: float, lo: float, hi: float) -> float:
    raw = ctx.opt(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise SpecError(f"Option '{key}' must be a number; you gave '{raw}'.") from None
    if not np.isfinite(value) or value < lo or value > hi:
        raise SpecError(f"Option '{key}' must be between {lo} and {hi}; you gave {value}.")
    return value


def _bool_opt(ctx: RunContext, key: str, default: bool) -> bool:
    raw = ctx.opt(key, default)
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(raw)


def _columns_opt(ctx: RunContext, key: str) -> list[str]:
    raw = ctx.opt(key, None)
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    return [str(c) for c in raw if str(c).strip()]


def _fmt(value: Any, nd: int = 6) -> str:
    if value is None:
        return "."
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "."
    return f"{v:.{nd}g}"


def _text_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], widths: Sequence[int]) -> list[str]:
    def cell(value: Any, width: int, first: bool) -> str:
        text = str(value)[: width - 1]
        return text.ljust(width) if first else text.rjust(width)

    out = ["".join(cell(h, w, i == 0) for i, (h, w) in enumerate(zip(headers, widths)))]
    out.append("-" * int(sum(widths)))
    for row in rows:
        out.append("".join(cell(c, w, i == 0) for i, (c, w) in enumerate(zip(row, widths))))
    return out


# ---------------------------------------------------------------------------
# Learners. Boring on purpose.
# ---------------------------------------------------------------------------


def _cv_folds(n: int) -> int:
    return int(min(5, max(2, n // 25))) if n >= 50 else 2


def make_regressor(name: str, seed: int, n_train: int):
    if name == "linear":
        return Pipeline(
            [("scale", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-4.0, 4.0, 25)))]
        )
    if name == "lasso":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LassoCV(n_alphas=30, cv=_cv_folds(n_train), random_state=seed,
                                  max_iter=20000, selection="cyclic")),
            ]
        )
    if name == "forest":
        return RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=max(3, min(10, n_train // 60 or 3)),
            max_features=None,
            random_state=seed,
            bootstrap=True,
        )
    if name == "gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
            min_samples_leaf=max(5, n_train // 60 or 5), early_stopping=False, random_state=seed,
        )
    raise SpecError(f"Unknown learner '{name}'.")


def _logistic(*, l1: bool, seed: int):
    """L1/L2 logistic regression, spelled the way the installed scikit-learn wants.

    scikit-learn 1.8 deprecated ``penalty=`` in favour of ``l1_ratio=``; asking for
    the old spelling on a new install prints a FutureWarning on every single fit.
    """
    kwargs: dict[str, Any] = {"C": 1.0, "max_iter": 3000,
                              "solver": "liblinear" if l1 else "lbfgs"}
    if l1:
        kwargs["random_state"] = int(seed)
    if _LOGREG_USES_L1_RATIO:
        kwargs["l1_ratio"] = 1.0 if l1 else 0.0
    else:  # pragma: no cover - older scikit-learn
        kwargs["penalty"] = "l1" if l1 else "l2"
    return LogisticRegression(**kwargs)


def make_classifier(name: str, seed: int, n_train: int):
    if name == "linear":
        return Pipeline([("scale", StandardScaler()), ("model", _logistic(l1=False, seed=seed))])
    if name == "lasso":
        return Pipeline([("scale", StandardScaler()), ("model", _logistic(l1=True, seed=seed))])
    if name == "forest":
        return RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=max(3, min(10, n_train // 60 or 3)),
            max_features=None,
            random_state=seed,
        )
    if name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
            min_samples_leaf=max(5, n_train // 60 or 5), early_stopping=False, random_state=seed,
        )
    raise SpecError(f"Unknown learner '{name}'.")


def _predict_reg(
    name: str, seed: int, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
    notes: list[str], tag: str,
) -> np.ndarray:
    """Fit and predict, falling back to the training mean and *saying so*."""
    n_te = X_te.shape[0]
    if y_tr.size == 0:
        notes.append(f"{tag}: no training rows; predicted 0.")
        return np.zeros(n_te)
    mean = float(np.mean(y_tr))
    if y_tr.size < 8 or float(np.std(y_tr)) < 1e-12:
        notes.append(f"{tag}: {y_tr.size} training rows or no variation; predicted the training mean.")
        return np.full(n_te, mean)
    try:
        model = make_regressor(name, seed, int(y_tr.size))
        model.fit(X_tr, y_tr)
        pred = np.asarray(model.predict(X_te), dtype=float).ravel()
    except Exception as exc:  # a learner failing must not kill the run
        notes.append(f"{tag}: {type(exc).__name__} while fitting; predicted the training mean.")
        return np.full(n_te, mean)
    if not np.all(np.isfinite(pred)):
        notes.append(f"{tag}: non-finite predictions replaced by the training mean.")
        pred = np.where(np.isfinite(pred), pred, mean)
    return pred


def _predict_prob(
    name: str, seed: int, X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
    notes: list[str], tag: str,
) -> np.ndarray:
    n_te = X_te.shape[0]
    if y_tr.size == 0:
        notes.append(f"{tag}: no training rows; predicted 0.5.")
        return np.full(n_te, 0.5)
    rate = float(np.mean(y_tr))
    if y_tr.size < 8 or len(np.unique(y_tr)) < 2:
        notes.append(f"{tag}: one class only (or too few rows) in training; predicted the base rate.")
        return np.full(n_te, rate)
    try:
        model = make_classifier(name, seed, int(y_tr.size))
        model.fit(X_tr, y_tr)
        classes = np.asarray(getattr(model, "classes_", np.array([0.0, 1.0])), dtype=float)
        proba = np.asarray(model.predict_proba(X_te), dtype=float)
        where = np.flatnonzero(np.isclose(classes, 1.0))
        if where.size == 0:
            notes.append(f"{tag}: the classifier never saw a positive case; predicted the base rate.")
            return np.full(n_te, rate)
        pred = proba[:, int(where[0])]
    except Exception as exc:
        notes.append(f"{tag}: {type(exc).__name__} while fitting; predicted the base rate.")
        return np.full(n_te, rate)
    if not np.all(np.isfinite(pred)):
        notes.append(f"{tag}: non-finite probabilities replaced by the base rate.")
        pred = np.where(np.isfinite(pred), pred, rate)
    return np.clip(pred, 0.0, 1.0)


def _rmse(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.size == 0:
        return None
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _auc(y: np.ndarray, p: np.ndarray) -> float | None:
    y = np.asarray(y, dtype=float)
    if y.size < 4 or len(np.unique(y)) < 2 or _roc_auc_score is None:
        return None
    try:
        return float(_roc_auc_score(y, p))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Cross-fitting: one engine, six methods
# ---------------------------------------------------------------------------


@dataclass
class Nuisance:
    """Out-of-fold nuisance predictions plus the honest report on how they did."""

    fold_id: np.ndarray
    m_hat: np.ndarray          # E[Y | X]
    g_hat: np.ndarray          # E[D | X], clipped when D is binary
    g_raw: np.ndarray          # E[D | X], before clipping
    mu0: np.ndarray            # E[Y | X, D = 0]
    mu1: np.ndarray            # E[Y | X, D = 1]
    n_clipped: int
    clip_lo: float
    clip_hi: float
    fold_rows: list[dict[str, Any]]
    n_folds: int
    learner: str
    notes: list[str]
    treatment_binary: bool
    outcome_binary: bool
    seed: int

    def subset(self, mask: np.ndarray) -> "Nuisance":
        mask = np.asarray(mask, dtype=bool)
        g_raw = self.g_raw[mask]
        return Nuisance(
            fold_id=self.fold_id[mask],
            m_hat=self.m_hat[mask],
            g_hat=self.g_hat[mask],
            g_raw=g_raw,
            mu0=self.mu0[mask],
            mu1=self.mu1[mask],
            n_clipped=int(np.sum((g_raw < self.clip_lo) | (g_raw > self.clip_hi))),
            clip_lo=self.clip_lo,
            clip_hi=self.clip_hi,
            fold_rows=self.fold_rows,
            n_folds=self.n_folds,
            learner=self.learner,
            notes=self.notes,
            treatment_binary=self.treatment_binary,
            outcome_binary=self.outcome_binary,
            seed=self.seed,
        )


def crossfit(
    X: np.ndarray,
    y: np.ndarray,
    d: np.ndarray,
    *,
    folds: int,
    learner: str,
    seed: int,
    clip: float = 0.01,
    treatment_binary: bool = True,
    outcome_binary: bool = False,
    ctx: RunContext | None = None,
    progress_from: float = 0.10,
    progress_to: float = 0.55,
    label: str = "fitting nuisance models",
) -> Nuisance:
    """K-fold cross-fitted nuisances: E[Y|X], E[D|X], E[Y|X,D=0], E[Y|X,D=1]."""
    n = int(np.asarray(y).size)
    folds = int(folds)
    if treatment_binary:
        n1, n0 = int(np.sum(d > 0.5)), int(np.sum(d <= 0.5))
        if min(n1, n0) < 2 * folds:
            raise DataError(
                f"Cross-fitting with {folds} folds needs at least {2 * folds} treated and "
                f"{2 * folds} control rows; this sample has {n1} treated and {n0} control.",
                detail="Lower the number of folds, widen the sample, or check the treatment coding.",
            )
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
        splits = list(splitter.split(X, (d > 0.5).astype(int)))
    else:
        if n < 4 * folds:
            raise DataError(
                f"Cross-fitting with {folds} folds needs at least {4 * folds} rows; this sample has {n}.",
                detail="Lower the number of folds or widen the sample.",
            )
        splitter = KFold(n_splits=folds, shuffle=True, random_state=int(seed))
        splits = list(splitter.split(X))

    fold_id = np.zeros(n, dtype=int)
    m_hat = np.zeros(n)
    g_raw = np.zeros(n)
    mu0 = np.zeros(n)
    mu1 = np.zeros(n)
    rows: list[dict[str, Any]] = []
    notes: list[str] = []

    for k, (tr, te) in enumerate(splits):
        if ctx is not None:
            frac = progress_from + (progress_to - progress_from) * (k / max(len(splits), 1))
            ctx.tick(frac, f"{label} (fold {k + 1} of {len(splits)})")
        fold_id[te] = k
        fold_seed = int(seed) + 1013 * (k + 1)
        X_tr, X_te = X[tr], X[te]
        y_tr, d_tr = y[tr], d[tr]

        if outcome_binary:
            m_hat[te] = _predict_prob(learner, fold_seed, X_tr, y_tr, X_te, notes, f"fold {k + 1} E[Y|X]")
        else:
            m_hat[te] = _predict_reg(learner, fold_seed, X_tr, y_tr, X_te, notes, f"fold {k + 1} E[Y|X]")

        if treatment_binary:
            g_raw[te] = _predict_prob(learner, fold_seed + 7, X_tr, (d_tr > 0.5).astype(float),
                                      X_te, notes, f"fold {k + 1} E[D|X]")
            sel0, sel1 = d_tr <= 0.5, d_tr > 0.5
            fn = _predict_prob if outcome_binary else _predict_reg
            mu0[te] = fn(learner, fold_seed + 13, X_tr[sel0], y_tr[sel0], X_te, notes,
                         f"fold {k + 1} E[Y|X,D=0]")
            mu1[te] = fn(learner, fold_seed + 19, X_tr[sel1], y_tr[sel1], X_te, notes,
                         f"fold {k + 1} E[Y|X,D=1]")
        else:
            g_raw[te] = _predict_reg(learner, fold_seed + 7, X_tr, d_tr, X_te, notes,
                                     f"fold {k + 1} E[D|X]")
            mu0[te] = m_hat[te]
            mu1[te] = m_hat[te]

        y_te, d_te = y[te], d[te]
        row: dict[str, Any] = {
            "fold": k + 1,
            "n": int(np.asarray(te).size),
            "n_treated": int(np.sum(d_te > 0.5)) if treatment_binary else None,
            "outcome_rmse": _rmse(y_te, m_hat[te]),
            "outcome_rmse_mean_model": _rmse(y_te, np.full(np.asarray(te).size, float(np.mean(y_tr)))),
            "treatment_rmse": _rmse(d_te, g_raw[te]),
            "treatment_auc": _auc((d_te > 0.5).astype(float), g_raw[te]) if treatment_binary else None,
        }
        if treatment_binary:
            c0, c1 = d_te <= 0.5, d_te > 0.5
            row["mu0_rmse"] = _rmse(y_te[c0], mu0[te][c0])
            row["mu1_rmse"] = _rmse(y_te[c1], mu1[te][c1])
        rows.append(row)

    if treatment_binary:
        lo, hi = float(clip), 1.0 - float(clip)
        g_hat, n_clipped = stats.clip_propensity(g_raw, lo, hi)
    else:
        lo, hi = float("-inf"), float("inf")
        g_hat, n_clipped = g_raw.copy(), 0

    return Nuisance(
        fold_id=fold_id,
        m_hat=m_hat,
        g_hat=g_hat,
        g_raw=g_raw,
        mu0=mu0,
        mu1=mu1,
        n_clipped=int(n_clipped),
        clip_lo=lo,
        clip_hi=hi,
        fold_rows=rows,
        n_folds=len(splits),
        learner=learner,
        notes=notes,
        treatment_binary=treatment_binary,
        outcome_binary=outcome_binary,
        seed=int(seed),
    )


def aipw_scores(y: np.ndarray, d: np.ndarray, nu: Nuisance) -> np.ndarray:
    """The doubly robust (AIPW) score whose mean is the ATE."""
    g = np.clip(nu.g_hat, 1e-8, 1 - 1e-8)
    return nu.mu1 - nu.mu0 + d * (y - nu.mu1) / g - (1.0 - d) * (y - nu.mu0) / (1.0 - g)


def att_scores(y: np.ndarray, d: np.ndarray, nu: Nuisance) -> tuple[np.ndarray, float]:
    """(psi_b, p) for the ATT. theta = mean(psi_b); influence curve = psi_b - (D/p) theta."""
    g = np.clip(nu.g_hat, 1e-8, 1 - 1e-8)
    p = float(np.mean(d))
    if p <= 0 or p >= 1:
        raise DataError("The ATT needs both treated and untreated rows in the analysis sample.")
    resid0 = y - nu.mu0
    psi_b = (d * resid0 - (1.0 - d) * (g / (1.0 - g)) * resid0) / p
    return psi_b, p


def dr_learner_cate(
    Xc: np.ndarray, gamma: np.ndarray, *, folds: int, learner: str, seed: int
) -> np.ndarray:
    """Cross-fitted regression of the AIPW score on the effect modifiers (Kennedy 2020)."""
    n = gamma.size
    k = int(min(folds, max(2, n // 10)))
    splitter = KFold(n_splits=k, shuffle=True, random_state=int(seed))
    tau = np.zeros(n)
    notes: list[str] = []
    for j, (tr, te) in enumerate(splitter.split(Xc)):
        tau[te] = _predict_reg(learner, int(seed) + 71 * (j + 1), Xc[tr], gamma[tr], Xc[te],
                               notes, f"DR-learner stage 2 fold {j + 1}")
    return tau


# ---------------------------------------------------------------------------
# Analysis sample and feature sets
# ---------------------------------------------------------------------------


@dataclass
class Setup:
    df: pd.DataFrame
    treat_col: str
    out_col: str
    conf: list[str]
    mods: list[str]
    X: np.ndarray
    x_names: list[str]
    Xc: np.ndarray
    c_names: list[str]
    y: np.ndarray
    d: np.ndarray
    treatment_binary: bool
    outcome_binary: bool
    modifiers_supplied: bool

    @property
    def n(self) -> int:
        return int(self.y.size)

    @property
    def n_treated(self) -> int:
        return int(np.sum(self.d > 0.5))

    @property
    def n_control(self) -> int:
        return int(np.sum(self.d <= 0.5))


def _matrix(df: pd.DataFrame, cols: Sequence[str], what: str) -> tuple[np.ndarray, list[str]]:
    try:
        dm = stats.design_matrix(df, list(cols), intercept=False)
    except (KeyError, ValueError) as exc:
        raise SpecError(f"The {what} could not be turned into a model matrix: {exc}") from None
    if dm.X.shape[1] == 0:
        raise SpecError(
            f"The {what} has no usable columns after dropping constants.",
            detail="Every candidate column takes a single value in this sample.",
        )
    if not np.all(np.isfinite(dm.X)):
        bad = [dm.names[j] for j in range(dm.X.shape[1]) if not np.all(np.isfinite(dm.X[:, j]))]
        raise DataError(
            f"These columns are not numeric after cleaning: {', '.join(bad[:6])}.",
            detail="Fix the column type on the data sheet, or drop the column from the role.",
        )
    return dm.X, list(dm.names)


def setup(
    ctx: RunContext,
    rb: ResultBuilder,
    *,
    allow_continuous_treatment: bool = False,
    extra_cols: Sequence[str] = (),
) -> Setup:
    _require_sklearn()
    spec = ctx.spec
    roles.require_design_roles(spec, DESIGN)
    treat_col = str(roles.require_role(spec, "treatment"))
    out_col = str(roles.require_role(spec, "outcome"))
    conf = list(roles.confounders(spec))
    if not conf:
        raise SpecError(
            "Causal machine learning needs measured confounders.",
            detail="Drop the variables you are willing to adjust for onto the "
                   "'measured confounders' slot. A learner cannot adjust for a variable you did not give it.",
        )
    forbidden = set(roles.get_role(spec, "forbidden"))
    mods = [m for m in roles.get_role(spec, "effect_modifiers") if m not in forbidden]

    # Decide binary-versus-continuous BEFORE building the sample: the CONSORT counter
    # splits every step by arm, which it can only do for a two-level treatment.
    raw_levels = None
    if treat_col in getattr(ctx.data, "columns", []):
        raw_levels = int(ctx.data[treat_col].nunique(dropna=True))
    if raw_levels is not None and raw_levels > 2 and not allow_continuous_treatment:
        raise SpecError(
            f"Treatment '{treat_col}' has {raw_levels} distinct values; this method needs a "
            "binary treatment.",
            detail="Define the contrast (which level counts as treated) in the inspector, or use "
                   "the partially linear DML method, which accepts a continuous dose.",
        )
    counts_by_arm = raw_levels is None or raw_levels == 2
    sample = roles.build_sample(
        ctx,
        needed=["treatment", "outcome", "confounders", "effect_modifiers"],
        extra=[c for c in extra_cols if isinstance(c, str)],
        treat_col=(treat_col if counts_by_arm else "__capy_continuous_treatment__"),
    )
    rb.extend_flow(sample.flow)
    df = sample.df

    for warn in roles.bad_control_warnings(spec):
        rb.add_warning(
            f"{warn['variable']}: {warn['reason']}",
            level="caution",
            code="bad_control",
            explain_key="guardrail.bad_control",
        )

    series = df[treat_col]
    n_levels = int(series.nunique(dropna=True))
    if n_levels < 2:
        raise DataError(
            f"Treatment '{treat_col}' takes only one value in this sample.",
            detail="With no contrast there is nothing to compare.",
        )
    if n_levels == 2:
        d = stats.to01(series)
        treatment_binary = True
    elif allow_continuous_treatment:
        d = roles.numeric(df, treat_col, "treatment")
        treatment_binary = False
    else:
        raise SpecError(
            f"Treatment '{treat_col}' has {n_levels} distinct values; this method needs a binary treatment.",
            detail="Define the contrast (which level counts as treated) in the inspector, "
                   "or use the partially linear DML method, which accepts a continuous dose.",
        )

    y = roles.numeric(df, out_col, "outcome")
    outcome_binary = bool(stats.is_binary(df[out_col]) and set(np.unique(y)).issubset({0.0, 1.0}))

    x_cols = list(dict.fromkeys(conf + mods))
    X, x_names = _matrix(df, x_cols, "confounder set")
    c_cols = mods if mods else conf
    Xc, c_names = _matrix(df, c_cols, "effect-modifier set")

    st = Setup(
        df=df, treat_col=treat_col, out_col=out_col, conf=conf, mods=mods,
        X=X, x_names=x_names, Xc=Xc, c_names=c_names, y=y, d=d,
        treatment_binary=treatment_binary, outcome_binary=outcome_binary,
        modifiers_supplied=bool(mods),
    )
    rb.set_counts(n=st.n, n_treated=st.n_treated if treatment_binary else None,
                  n_control=st.n_control if treatment_binary else None)
    rb.set_roles_used(
        {
            "treatment": treat_col,
            "outcome": out_col,
            "confounders": conf,
            "effect_modifiers": mods,
        }
    )
    if not mods:
        rb.add_warning(
            "No effect modifiers were named, so heterogeneity is explored over the full confounder set. "
            "That is exploratory: pre-register the modifiers you care about.",
            level="info",
            code="no_effect_modifiers",
        )
    return st


# ---------------------------------------------------------------------------
# Diagnostics -- mandatory whenever a learner is used (plan 8.1)
# ---------------------------------------------------------------------------


def diag_nuisance_rmse(rb: ResultBuilder, nu: Nuisance) -> str:
    rows = [dict(r) for r in nu.fold_rows]
    table_id = rb.artifact(
        "table",
        title="Out-of-fold nuisance fit, by fold",
        data=rows,
        columns=vega.table_artifact_columns(rows),
        explain_key="diagnostic.nuisance_rmse",
        caption="Each fold is scored by models that never saw it.",
    )
    bar_rows = [{"label": f"Fold {r['fold']}", "value": r.get("outcome_rmse") or 0.0} for r in rows]
    chart_id = rb.artifact(
        "vega",
        title="Outcome-model RMSE by fold",
        spec=vega.bar_chart(bar_rows, x="label", y="value", horizontal=False, sort_desc=False,
                            y_title="Out-of-fold RMSE", x_title="Fold",
                            title="Outcome-model RMSE by fold"),
        data=bar_rows,
        explain_key="diagnostic.nuisance_rmse",
    )
    rmses = [r["outcome_rmse"] for r in rows if r.get("outcome_rmse") is not None]
    bases = [r["outcome_rmse_mean_model"] for r in rows if r.get("outcome_rmse_mean_model") is not None]
    aucs = [r["treatment_auc"] for r in rows if r.get("treatment_auc") is not None]
    briers = [r["treatment_rmse"] for r in rows if r.get("treatment_rmse") is not None]
    mean_rmse = float(np.mean(rmses)) if rmses else None
    mean_base = float(np.mean(bases)) if bases else None
    mean_auc = float(np.mean(aucs)) if aucs else None
    gain = None
    if mean_rmse is not None and mean_base and mean_base > 0:
        gain = float(1.0 - (mean_rmse / mean_base) ** 2)
    spread = float(max(rmses) - min(rmses)) if len(rmses) > 1 else 0.0

    status = "info"
    if mean_rmse is not None and mean_base is not None and mean_rmse > mean_base * 1.02:
        status = "weakens"
    summary_bits = []
    if mean_rmse is not None:
        summary_bits.append(
            f"Out-of-fold outcome RMSE averages {mean_rmse:.4g} across {nu.n_folds} folds"
            + (f", against {mean_base:.4g} for a model that just predicts the mean" if mean_base else "")
            + (f" (out-of-fold R2 {gain:.2f})" if gain is not None else "")
        )
    if mean_auc is not None:
        summary_bits.append(f"the treatment model reaches AUC {mean_auc:.3f}")
    summary = "; ".join(summary_bits) + "." if summary_bits else "Nuisance fit recorded by fold."
    if nu.notes:
        summary += f" {len(nu.notes)} fold-level fallback(s) were recorded."

    rb.add_diagnostic(
        "nuisance_rmse",
        "Nuisance fit by fold",
        status=status,
        summary=summary,
        worry_when="One fold is far worse than the others, or the outcome model does no better than "
                   "predicting the mean -- then the residuals the estimator relies on are mostly noise. "
                   "A treatment AUC near 1 is the other failure: assignment is nearly deterministic, "
                   "which is a positivity problem, not a modelling success.",
        artifact_ids=[chart_id, table_id],
        explain_key="diagnostic.nuisance_rmse",
        values={
            "learner": nu.learner,
            "learner_label": LEARNER_LABEL.get(nu.learner, nu.learner),
            "folds": nu.n_folds,
            "outcome_rmse_mean": mean_rmse,
            "outcome_rmse_mean_model": mean_base,
            "outcome_r2_out_of_fold": gain,
            "outcome_rmse_spread": spread,
            "propensity_brier_rmse": float(np.mean(briers)) if briers else None,
            "propensity_auc_mean": mean_auc,
            "fallbacks": nu.notes[:20],
        },
    )
    if nu.notes:
        rb.add_warning(
            f"{len(nu.notes)} nuisance fit(s) fell back to a mean/base-rate prediction. "
            "Read the nuisance diagnostic before trusting the estimate.",
            level="caution",
            code="nuisance_fallback",
        )
    return "nuisance_rmse"


def diag_propensity(
    rb: ResultBuilder, nu: Nuisance, d: np.ndarray, *, rule: str, n_dropped: int = 0
) -> str:
    if not nu.treatment_binary:
        var_d = float(np.var(d, ddof=1)) if d.size > 1 else float("nan")
        resid_var = float(np.var(d - nu.g_hat, ddof=1)) if d.size > 1 else float("nan")
        share = float(resid_var / var_d) if var_d > 0 else float("nan")
        rb.add_diagnostic(
            "propensity_clipping",
            "Propensity clipping",
            status="not_applicable",
            summary="The treatment is continuous, so there is no propensity score to clip. "
                    f"The share of treatment variance left after removing E[D|X] is {share:.3f}.",
            worry_when="That share heads to zero: the confounders explain nearly all of the treatment, "
                       "so the residual variation the estimator divides by is almost nothing and the "
                       "standard error is not to be believed.",
            explain_key="diagnostic.propensity_clipping",
            values={"residual_variance_share": None if not np.isfinite(share) else share,
                    "treatment_binary": False},
        )
        rb.set_assumption_status(
            "positivity",
            "supported" if np.isfinite(share) and share > 0.05 else "weakened",
            note=f"Residual treatment variation after adjustment: {share:.3f} of the total.",
        )
        return "propensity_clipping"

    ps = nu.g_raw
    treated = d > 0.5
    lo_edge = float(min(np.min(ps), 0.0))
    hi_edge = float(max(np.max(ps), 1.0))
    rows: list[dict[str, Any]] = []
    for arm, sel in (("Treated", treated), ("Control", ~treated)):
        for row in stats.histogram_rows(ps[sel], bins=30, lo=lo_edge, hi=hi_edge):
            rows.append({"x": row["x"], "count": row["count"], "arm": arm})
    hist_id = rb.artifact(
        "vega",
        title="Estimated propensity score by arm",
        spec=vega.overlap_histogram(rows, x_title="Cross-fitted propensity score",
                                    title="Overlap by arm (cross-fitted propensity)"),
        data=rows,
        explain_key="diagnostic.overlap",
        caption="Cross-fitted scores: each unit is scored by a model that never saw it.",
    )
    weights = np.where(treated, 1.0 / np.clip(nu.g_hat, 1e-8, 1), 1.0 / np.clip(1 - nu.g_hat, 1e-8, 1))
    wrows = stats.histogram_rows(weights, bins=30)
    weight_id = rb.artifact(
        "vega",
        title="Inverse-probability weights",
        spec=vega.weight_histogram(wrows, title="Inverse-probability weight distribution"),
        data=wrows,
        explain_key="diagnostic.weights",
    )
    ess = stats.effective_sample_size(weights)
    rb.set_counts(n_effective=ess)

    n = int(ps.size)
    n_below = int(np.sum(ps < nu.clip_lo))
    n_above = int(np.sum(ps > nu.clip_hi))
    n_clipped = n_below + n_above
    share = n_clipped / max(n, 1)
    status = "supports"
    if share > 0.05 or float(np.min(ps[treated])) < 0.01 or float(np.max(ps[~treated])) > 0.99:
        status = "weakens"
    verb = {"clip": "clipped", "drop": "dropped"}.get(rule, "flagged (the score itself uses the unclipped values)")
    summary = (
        f"{n_clipped} of {n} units ({share:.1%}) had a cross-fitted propensity outside "
        f"[{nu.clip_lo:.3g}, {nu.clip_hi:.3g}] and were {verb}. "
        f"Scores run from {float(np.min(ps)):.3g} to {float(np.max(ps)):.3g}; "
        f"the Kish effective sample size under IPW is {ess:.0f} of {n}."
    )
    if rule == "drop" and n_dropped:
        summary += f" {n_dropped} row(s) left the analysis sample; see the sample flow."
    rb.add_diagnostic(
        "propensity_clipping",
        "Propensity clipping and overlap",
        status=status,
        summary=summary,
        worry_when="More than a few per cent of units sit at the clip, or one arm has scores the other "
                   "never reaches. Then the estimate is being carried by extrapolation rather than by "
                   "comparable units, and clipping has quietly changed the estimand.",
        artifact_ids=[hist_id, weight_id],
        explain_key="diagnostic.propensity_clipping",
        values={
            "threshold_low": nu.clip_lo,
            "threshold_high": nu.clip_hi,
            "n_clipped": n_clipped,
            "n_below": n_below,
            "n_above": n_above,
            "share_clipped": share,
            "rule": rule,
            "n_dropped": int(n_dropped),
            "ps_min": float(np.min(ps)),
            "ps_max": float(np.max(ps)),
            "ps_min_treated": float(np.min(ps[treated])) if treated.any() else None,
            "ps_max_control": float(np.max(ps[~treated])) if (~treated).any() else None,
            "effective_sample_size": ess,
            "max_weight": float(np.max(weights)),
        },
    )
    rb.set_assumption_status(
        "positivity",
        "supported" if status == "supports" else "weakened",
        note=f"{share:.1%} of units sat outside the [{nu.clip_lo:.3g}, {nu.clip_hi:.3g}] propensity band.",
    )
    if status == "weakens":
        rb.add_warning(
            f"{share:.1%} of units have a propensity score outside the clipping band. "
            "Positivity is doing real work here; read the overlap plot before quoting the number.",
            level="warning",
            code="positivity",
            explain_key="assumption.positivity",
        )
    return "propensity_clipping"


def diag_cate_distribution(rb: ResultBuilder, tau: np.ndarray, *, label: str) -> str:
    rows = stats.histogram_rows(tau, bins=30)
    art = rb.artifact(
        "vega",
        title="Distribution of estimated effects (CATE)",
        spec=vega.histogram(rows, x_title="Estimated effect for a unit", rule_at=0.0,
                            title="Estimated effect by unit"),
        data=rows,
        explain_key="diagnostic.cate_distribution",
        caption=label,
    )
    qs = np.quantile(tau, [0.05, 0.25, 0.5, 0.75, 0.95])
    sd = float(np.std(tau, ddof=1)) if tau.size > 1 else 0.0
    rb.add_diagnostic(
        "cate_distribution",
        "Distribution of estimated effects",
        status="info",
        summary=(
            f"Estimated effects have mean {float(np.mean(tau)):.4g} and spread (SD) {sd:.4g}; "
            f"the 5th-95th percentile range is {qs[0]:.4g} to {qs[4]:.4g}, and "
            f"{float(np.mean(tau > 0)):.0%} of units are estimated to benefit."
        ),
        worry_when="The spread is wide but the calibration check below says the ranking is noise -- "
                   "then you are looking at estimation error, not heterogeneity. A spread of zero "
                   "means the learner found nothing, which is a finding, not a failure.",
        artifact_ids=[art],
        explain_key="diagnostic.cate_distribution",
        values={
            "source": label,
            "mean": float(np.mean(tau)),
            "sd": sd,
            "q05": float(qs[0]), "q25": float(qs[1]), "median": float(qs[2]),
            "q75": float(qs[3]), "q95": float(qs[4]),
            "share_positive": float(np.mean(tau > 0)),
        },
    )
    return "cate_distribution"


def diag_cate_calibration(
    rb: ResultBuilder,
    tau: np.ndarray,
    gamma: np.ndarray,
    *,
    level: float = 0.95,
    groups: int = 5,
    score_label: str = "cross-fitted AIPW scores",
) -> str:
    n = tau.size
    n_groups = int(max(2, min(groups, n // 30 if n >= 60 else 2)))
    if float(np.std(tau)) < 1e-12:
        rb.add_diagnostic(
            "cate_calibration",
            "Calibration of predicted against realised effects",
            status="not_applicable",
            summary="The model predicts the same effect for every unit, so there is no ranking to calibrate.",
            worry_when="Nothing here -- but do not report heterogeneity you did not find.",
            explain_key="diagnostic.cate_calibration",
            values={"n_groups": 0, "reason": "constant CATE"},
        )
        return "cate_calibration"

    edges = np.quantile(tau, np.linspace(0, 1, n_groups + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    bucket = np.clip(np.digitize(tau, edges[1:-1], right=False), 0, n_groups - 1)
    z = stats.z_for(level)
    rows: list[dict[str, Any]] = []
    forest_rows: list[dict[str, Any]] = []
    for q in range(n_groups):
        sel = bucket == q
        if not sel.any():
            continue
        pred = float(np.mean(tau[sel]))
        realised = float(np.mean(gamma[sel]))
        se = float(np.std(gamma[sel], ddof=1) / math.sqrt(sel.sum())) if sel.sum() > 1 else float("nan")
        lo, hi = (realised - z * se, realised + z * se) if np.isfinite(se) else (None, None)
        rows.append({
            "group": f"Q{q + 1}",
            "n": int(sel.sum()),
            "predicted_effect": pred,
            "realised_effect": realised,
            "se": None if not np.isfinite(se) else se,
            "ci_low": lo,
            "ci_high": hi,
        })
        forest_rows.append({
            "label": f"Q{q + 1} (predicted {pred:+.3g})",
            "estimate": realised, "se": None if not np.isfinite(se) else se,
            "ci_low": lo, "ci_high": hi, "n": int(sel.sum()), "engine": "python",
        })
        rb.add_estimate(
            f"Effect in CATE quintile Q{q + 1}",
            realised,
            se=None if not np.isfinite(se) else se,
            ci=(lo, hi),
            p_value=stats.norm_sf2(realised / se) if np.isfinite(se) and se > 0 else None,
            group="cate_quantile",
            term=q + 1,
            n=int(sel.sum()),
        )
    forest_id = rb.artifact(
        "vega",
        title="Realised effect by predicted-effect group",
        spec=vega.forest(forest_rows, title="Realised effect by predicted-effect group",
                         x_title="Realised effect (doubly robust)"),
        data=forest_rows,
        explain_key="diagnostic.cate_calibration",
        caption=f"Realised effects come from {score_label}, not from the model being checked.",
    )
    table_id = rb.artifact(
        "table",
        title="CATE calibration by group",
        data=rows,
        columns=vega.table_artifact_columns(rows),
        explain_key="diagnostic.cate_calibration",
    )
    scatter_rows = [{"predicted": r["predicted_effect"], "realised": r["realised_effect"]} for r in rows]
    scatter_id = rb.artifact(
        "vega",
        title="Predicted against realised effect",
        spec=vega.scatter(scatter_rows, x="predicted", y="realised", fit_line=True,
                          x_title="Predicted effect", y_title="Realised effect",
                          title="Predicted against realised effect"),
        data=scatter_rows,
        explain_key="diagnostic.cate_calibration",
    )

    mean_pred = float(np.mean(tau))
    Z = np.column_stack([np.full(n, mean_pred), tau - mean_pred])
    fit = stats.ols(gamma, Z, ["mean_prediction", "differential_prediction"], vcov="HC1")
    alpha, beta = float(fit.params[0]), float(fit.params[1])
    se_alpha, se_beta = float(fit.se[0]), float(fit.se[1])
    p_beta = stats.norm_sf2(beta / se_beta) if se_beta > 0 else None
    beta_ci = stats.wald_ci(beta, se_beta, level)

    status = "info"
    if se_beta > 0:
        if beta > 0 and (beta_ci[0] is not None and beta_ci[0] <= 1.0 <= (beta_ci[1] or 1.0)):
            status = "supports"
        elif beta <= 0 or (beta_ci[1] is not None and beta_ci[1] < 0.5):
            status = "weakens"
    rb.add_diagnostic(
        "cate_calibration",
        "Calibration of predicted against realised effects",
        status=status,
        summary=(
            f"Splitting units into {len(rows)} groups by predicted effect, the realised (doubly robust) "
            f"effect runs from {rows[0]['realised_effect']:.4g} in the lowest group to "
            f"{rows[-1]['realised_effect']:.4g} in the highest. The best linear predictor test gives a "
            f"calibration slope of {beta:.3f} (SE {se_beta:.3f}); 1.0 means the predicted spread is "
            f"exactly right, 0 means the ranking carries no information."
        ),
        worry_when="The calibration slope is near zero or negative: the model is ranking units at random, "
                   "and any targeting built on it will not beat treating everyone. A slope far above 1 "
                   "means the model understates how different units really are.",
        artifact_ids=[forest_id, scatter_id, table_id],
        explain_key="diagnostic.cate_calibration",
        values={
            "n_groups": len(rows),
            "groups": rows,
            "blp_mean_prediction": alpha,
            "blp_mean_prediction_se": se_alpha,
            "blp_differential_prediction": beta,
            "blp_differential_prediction_se": se_beta,
            "blp_differential_ci_low": beta_ci[0],
            "blp_differential_ci_high": beta_ci[1],
            "blp_differential_p": p_beta,
            "score_label": score_label,
        },
    )
    return "cate_calibration"


def _toc_on_grid(tau: np.ndarray, gamma: np.ndarray, grid: np.ndarray) -> np.ndarray:
    n = tau.size
    order = np.argsort(-tau, kind="stable")
    g = gamma[order]
    cum = np.cumsum(g)
    base = float(np.mean(gamma))
    ks = np.clip(np.ceil(np.asarray(grid, dtype=float) * n).astype(int), 1, n)
    return cum[ks - 1] / ks - base


def diag_rate(
    rb: ResultBuilder,
    tau: np.ndarray,
    gamma: np.ndarray,
    *,
    seed: int,
    reps: int = 200,
    level: float = 0.95,
) -> str:
    n = tau.size
    q_full = np.arange(1, n + 1) / n
    toc_full = _toc_on_grid(tau, gamma, q_full)
    autoc = _trapz(toc_full, q_full)
    qini = _trapz(q_full * toc_full, q_full)
    grid = np.linspace(0.05, 1.0, 20)
    toc_grid = _toc_on_grid(tau, gamma, grid)

    rng = np.random.default_rng(int(seed))
    autoc_draws: list[float] = []
    qini_draws: list[float] = []
    curves: list[np.ndarray] = []
    for _ in range(int(max(reps, 0))):
        idx = rng.integers(0, n, size=n)
        t2, g2 = tau[idx], gamma[idx]
        tf = _toc_on_grid(t2, g2, q_full)
        autoc_draws.append(_trapz(tf, q_full))
        qini_draws.append(_trapz(q_full * tf, q_full))
        curves.append(_toc_on_grid(t2, g2, grid))

    se_autoc = float(np.std(autoc_draws, ddof=1)) if len(autoc_draws) > 4 else None
    se_qini = float(np.std(qini_draws, ddof=1)) if len(qini_draws) > 4 else None
    if curves:
        band = np.std(np.vstack(curves), axis=0, ddof=1)
    else:
        band = np.full(grid.size, np.nan)
    z = stats.z_for(level)
    rows = [
        {
            "x": float(q),
            "estimate": float(t),
            "ci_low": float(t - z * b) if np.isfinite(b) else None,
            "ci_high": float(t + z * b) if np.isfinite(b) else None,
        }
        for q, t, b in zip(grid, toc_grid, band)
    ]
    art = rb.artifact(
        "vega",
        title="TOC curve (targeting operator characteristic)",
        spec=vega.path_plot(rows, title="Targeting operator characteristic",
                            x_title="Share of the population treated, best-ranked first",
                            y_title="Gain over treating everyone"),
        data=rows,
        explain_key="diagnostic.rate",
        caption="How much better than treating everyone you do by treating only the top-ranked share.",
    )
    p_autoc = stats.norm_sf2(autoc / se_autoc) if se_autoc and se_autoc > 0 else None
    ci = stats.wald_ci(autoc, se_autoc, level) if se_autoc else (None, None)
    status = "info"
    if p_autoc is not None and p_autoc < 0.05 and autoc > 0:
        status = "supports"
    rb.add_diagnostic(
        "rate",
        "RATE: is the ranking worth targeting on?",
        status=status,
        summary=(
            f"AUTOC = {autoc:.4g}"
            + (f" (bootstrap SE {se_autoc:.4g}, p = {p_autoc:.3f})" if se_autoc else " (SE unavailable)")
            + f"; Qini = {qini:.4g}. AUTOC above zero means treating the highest-ranked units first "
              "beats treating a random subset of the same size."
        ),
        worry_when="AUTOC is indistinguishable from zero: the CATE ranking is not usable for targeting, "
                   "whatever the histogram looks like. Note the bootstrap holds the fitted ranking fixed, "
                   "so it understates the uncertainty coming from having estimated that ranking at all.",
        artifact_ids=[art],
        explain_key="diagnostic.rate",
        values={
            "autoc": autoc,
            "autoc_se": se_autoc,
            "autoc_ci_low": ci[0],
            "autoc_ci_high": ci[1],
            "autoc_p": p_autoc,
            "qini": qini,
            "qini_se": se_qini,
            "bootstrap_reps": int(len(autoc_draws)),
            "inference": "bootstrap over rows, fitted ranking held fixed",
        },
    )
    return "rate"


def diag_fold_stability(
    rb: ResultBuilder, fold_values: Sequence[dict[str, Any]], overall: float, overall_se: float | None
) -> str:
    rows = [dict(r) for r in fold_values]
    forest_rows = [
        {"label": f"Fold {r['fold']}", "estimate": r["estimate"], "se": r.get("se"),
         "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high"), "n": r.get("n"), "engine": "python"}
        for r in rows
    ]
    forest_rows.append({"label": "All folds", "estimate": overall, "se": overall_se,
                        "ci_low": stats.wald_ci(overall, overall_se)[0] if overall_se else None,
                        "ci_high": stats.wald_ci(overall, overall_se)[1] if overall_se else None,
                        "engine": "python"})
    art = rb.artifact(
        "vega",
        title="Estimate by cross-fitting fold",
        spec=vega.forest(forest_rows, title="Estimate by cross-fitting fold", x_title="Estimate"),
        data=forest_rows,
        explain_key="diagnostic.fold_stability",
    )
    table_id = rb.artifact("table", title="Fold-level estimates", data=rows,
                           columns=vega.table_artifact_columns(rows),
                           explain_key="diagnostic.fold_stability")
    vals = np.array([r["estimate"] for r in rows if r.get("estimate") is not None], dtype=float)
    spread = float(np.max(vals) - np.min(vals)) if vals.size > 1 else 0.0
    ratio = float(spread / (4.0 * overall_se)) if overall_se and overall_se > 0 else None
    sign_flip = bool(vals.size > 1 and np.min(vals) < 0 < np.max(vals))
    # A fold estimate uses a fraction of the data, so it is inherently noisier than the
    # pooled one: comparing the raw spread with the pooled interval would cry wolf on a
    # perfectly healthy run. Ask instead whether the folds disagree by more than their own
    # standard errors allow (a Cochran Q heterogeneity test).
    fold_ses = [r.get("se") for r in rows if r.get("estimate") is not None]
    q_stat = q_p = None
    if vals.size > 1 and all(se is not None and np.isfinite(se) and se > 0 for se in fold_ses):
        w = 1.0 / np.asarray(fold_ses, dtype=float) ** 2
        pooled = float(np.sum(w * vals) / np.sum(w))
        q_stat = float(np.sum(w * (vals - pooled) ** 2))
        q_p = float(stats.chi2_sf(q_stat, int(vals.size - 1)))
    status = "supports"
    if sign_flip or (q_p is not None and q_p < 0.01) or (q_p is None and ratio is not None and ratio > 1.5):
        status = "weakens"
    rb.add_diagnostic(
        "fold_stability",
        "Stability across cross-fitting folds",
        status=status,
        summary=(
            f"Fold-level estimates run from {float(np.min(vals)):.4g} to {float(np.max(vals)):.4g} "
            f"(range {spread:.4g}) around a pooled estimate of {overall:.4g}. "
            + (f"They disagree no more than their own standard errors allow "
               f"(Q = {q_stat:.2f} on {vals.size - 1} d.f., p = {q_p:.3f})."
               if (q_p is not None and q_p >= 0.01) else
               f"They disagree by more than their own standard errors allow "
               f"(Q = {q_stat:.2f} on {vals.size - 1} d.f., p = {q_p:.3f})."
               if q_p is not None else
               (f"That range is {ratio:.2f} times the width of the pooled confidence interval."
                if ratio is not None else ""))
        ),
        worry_when="Fold estimates that straddle zero, or that disagree by more than their own "
                   "standard errors allow. Each fold sees a fraction of the data, so some spread is "
                   "expected; what is not expected is folds that contradict each other. That would "
                   "mean the answer depends on which rows happened to land where, and you need more "
                   "data or a simpler learner.",
        artifact_ids=[art, table_id],
        explain_key="diagnostic.fold_stability",
        values={
            "folds": rows,
            "min": float(np.min(vals)) if vals.size else None,
            "max": float(np.max(vals)) if vals.size else None,
            "range": spread,
            "range_over_ci_width": ratio,
            "sign_flip": sign_flip,
            "q_statistic": q_stat,
            "q_p_value": q_p,
        },
    )
    if status == "weakens":
        rb.add_warning(
            "The estimate moves a lot across cross-fitting folds. Treat the confidence interval as "
            "optimistic and rerun with more folds or a simpler learner.",
            level="caution",
            code="fold_instability",
        )
    return "fold_stability"


def diag_seed_stability(rb: ResultBuilder, values: Sequence[dict[str, Any]], overall: float) -> str:
    rows = [dict(r) for r in values]
    forest_rows = [
        {"label": f"Seed {r['seed']}", "estimate": r["estimate"], "se": r.get("se"),
         "ci_low": r.get("ci_low"), "ci_high": r.get("ci_high"), "engine": "python"}
        for r in rows
    ]
    art = rb.artifact(
        "vega",
        title="Estimate across cross-fitting seeds",
        spec=vega.forest(forest_rows, title="Estimate across cross-fitting seeds", x_title="Estimate"),
        data=forest_rows,
        explain_key="diagnostic.seed_stability",
    )
    vals = np.array([r["estimate"] for r in rows], dtype=float)
    sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    spread = float(np.max(vals) - np.min(vals)) if vals.size else 0.0
    sign_flip = bool(vals.size > 1 and np.min(vals) < 0 < np.max(vals))
    status = "weakens" if sign_flip else "supports"
    rb.add_diagnostic(
        "seed_stability",
        "Stability across cross-fitting seeds",
        status=status,
        summary=(
            f"Repeating the whole cross-fitting under {vals.size} seeds gives estimates from "
            f"{float(np.min(vals)):.4g} to {float(np.max(vals)):.4g} (SD across seeds {sd:.4g}) "
            f"around {overall:.4g}."
        ),
        worry_when="The spread across seeds is comparable to the standard error, or the sign changes. "
                   "The split is an arbitrary choice; the answer should not depend on it.",
        artifact_ids=[art],
        explain_key="diagnostic.seed_stability",
        values={"reps": rows, "sd": sd, "range": spread, "sign_flip": sign_flip},
    )
    if sign_flip:
        rb.add_warning(
            "The sign of the estimate changes with the cross-fitting seed. This number is not stable.",
            level="warning",
            code="seed_instability",
        )
    return "seed_stability"


def subgroup_estimates(
    rb: ResultBuilder,
    df: pd.DataFrame,
    gamma: np.ndarray,
    cols: Sequence[str],
    *,
    level: float = 0.95,
    max_levels: int = 12,
) -> int:
    """Pre-registered subgroup GATEs, with the multiplicity caution attached."""
    if not cols:
        return 0
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SpecError(
            f"Subgroup variable(s) not in the analysis sample: {', '.join(missing)}.",
            detail="Add them to a role, or remove them from the subgroups option.",
        )
    z = stats.z_for(level)
    rows: list[dict[str, Any]] = []
    k = 0
    for col in cols:
        series = df[col]
        levels = list(pd.unique(series.dropna()))
        if len(levels) > max_levels:
            raise SpecError(
                f"Subgroup variable '{col}' has {len(levels)} levels; that is a fishing expedition, "
                f"not a subgroup analysis (the limit is {max_levels}).",
            )
        for lev in sorted(levels, key=lambda v: str(v)):
            sel = (series == lev).to_numpy()
            if sel.sum() < 10:
                continue
            est = float(np.mean(gamma[sel]))
            se = float(np.std(gamma[sel], ddof=1) / math.sqrt(sel.sum()))
            lo, hi = (est - z * se, est + z * se)
            p = stats.norm_sf2(est / se) if se > 0 else None
            rb.add_estimate(f"{col} = {lev}", est, se=se, ci=(lo, hi), p_value=p,
                            group="subgroup", term=str(lev), n=int(sel.sum()))
            rows.append({"variable": col, "level": str(lev), "n": int(sel.sum()),
                         "estimate": est, "se": se, "ci_low": lo, "ci_high": hi, "p_value": p})
            k += 1
    if not rows:
        return 0
    forest_rows = [
        {"label": f"{r['variable']} = {r['level']}", "estimate": r["estimate"], "se": r["se"],
         "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": r["n"], "engine": "python"}
        for r in rows
    ]
    art = rb.artifact(
        "vega",
        title="Subgroup effects",
        spec=vega.forest(forest_rows, title="Subgroup effects (doubly robust)", x_title="Effect"),
        data=forest_rows,
        explain_key="diagnostic.subgroups",
    )
    rb.artifact("table", title="Subgroup effects", data=rows,
                columns=vega.table_artifact_columns(rows), explain_key="diagnostic.subgroups")
    rb.add_diagnostic(
        "subgroups",
        "Pre-registered subgroups",
        status="info",
        summary=f"{k} subgroup contrasts were estimated with the same doubly robust score as the headline.",
        worry_when="Any of these was chosen after seeing the data. With "
                   f"{k} contrasts, the chance that at least one crosses p < 0.05 by luck alone is "
                   f"about {1 - 0.95 ** k:.0%}.",
        artifact_ids=[art],
        explain_key="diagnostic.subgroups",
        values={"n_contrasts": k, "bonferroni_alpha": 0.05 / k, "rows": rows},
    )
    rb.add_warning(
        f"{k} subgroup contrasts were computed. Treating each at the 5% level gives roughly a "
        f"{1 - 0.95 ** k:.0%} chance of at least one false positive; the Bonferroni threshold for "
        f"{k} tests is p < {0.05 / k:.4f}. Report subgroups you pre-registered, and label the rest "
        "as exploratory.",
        level="caution",
        code="multiplicity",
        explain_key="stats.multiplicity",
    )
    return k


def note_exchangeability(rb: ResultBuilder) -> None:
    """The sentence this whole product exists to say."""
    rb.set_assumption_status(
        "exchangeability",
        "untested",
        note="Nothing in this run tests it. Cross-fitting and flexible learners reduce bias from the "
             "functional form of the nuisance models; they do nothing about a confounder you did not "
             "measure. Use the Probe tab (Rosenbaum bounds, Cinelli-Hazlett, negative controls) to ask "
             "how strong an unmeasured confounder would have to be.",
    )
    rb.set_assumption_status(
        "consistency",
        "assumed",
        note="The treatment variable is taken to name a single, well-defined intervention.",
    )
    rb.set_assumption_status(
        "sutva",
        "assumed",
        note="One unit's treatment is taken not to affect another unit's outcome.",
    )


# ---------------------------------------------------------------------------
# Shared run scaffolding
# ---------------------------------------------------------------------------


@dataclass
class Run:
    """Everything the six adapters share: the sample, the options, the ledger."""

    st: Setup
    level: float
    learner: str
    folds: int
    clip: float
    stability_reps: int
    rate_reps: int
    cluster: np.ndarray | None
    cluster_name: str | None
    subgroups: list[str]

    @property
    def n(self) -> int:
        return self.st.n


def _begin(
    ctx: RunContext,
    rb: ResultBuilder,
    *,
    allow_continuous_treatment: bool = False,
    default_learner: str = "linear",
    default_folds: int = 5,
) -> Run:
    """Seed the ledger, build the sample, read the options every method shares."""
    roles.seed_ledger(rb, DESIGN)
    cluster_name = roles.get_role(ctx.spec, "cluster")
    subgroups = _columns_opt(ctx, "subgroups")
    extra = [c for c in ([cluster_name] if cluster_name else []) + subgroups if c]
    st = setup(ctx, rb, allow_continuous_treatment=allow_continuous_treatment, extra_cols=extra)
    note_exchangeability(rb)

    learner = _choice(ctx, "learner", LEARNER_CHOICES, default_learner)
    folds = _int_opt(ctx, "folds", default_folds, 2, 20)
    clip = _float_opt(ctx, "clip", 0.01, 0.0, 0.45)
    level = _float_opt(ctx, "ci_level", 0.95, 0.50, 0.999)
    reps = _int_opt(ctx, "stability_reps", 1, 1, 10)
    rate_reps = _int_opt(ctx, "rate_reps", 200, 0, 2000)

    cluster = None
    if cluster_name and cluster_name in st.df.columns:
        cluster = st.df[cluster_name].to_numpy()
    used = dict(rb.result.get("roles_used") or {})
    if cluster_name:
        used["cluster"] = cluster_name
    if subgroups:
        used["subgroups"] = subgroups
    rb.set_roles_used(used)
    return Run(
        st=st, level=level, learner=learner, folds=folds, clip=clip,
        stability_reps=reps, rate_reps=rate_reps, cluster=cluster,
        cluster_name=cluster_name, subgroups=subgroups,
    )


def _ic_se(psi: np.ndarray, cluster: np.ndarray | None) -> tuple[float, str, int | None]:
    """Influence-function standard error, clustered when a cluster role is set."""
    psi = np.asarray(psi, dtype=float)
    n = int(psi.size)
    if n < 2:
        return float("nan"), "influence function (too few rows)", None
    if cluster is not None:
        codes, uniq = pd.factorize(pd.Series(cluster).astype(str))
        g = int(len(uniq))
        if g < 2:
            return (float(np.std(psi, ddof=1) / math.sqrt(n)),
                    "influence function (robust; the cluster column has one level)", g)
        sums = np.zeros(g)
        np.add.at(sums, codes, psi)
        se = float(np.sqrt(np.sum(sums ** 2)) / n) * math.sqrt(g / max(g - 1, 1))
        return se, f"influence function, clustered by {g} groups", g
    return float(np.std(psi, ddof=1) / math.sqrt(n)), "influence function (robust)", None


def _finish(
    rb: ResultBuilder, est: float, se: float | None, *, inference: str, level: float = 0.95
) -> tuple[float | None, float | None]:
    if se is None or not np.isfinite(se) or se <= 0:
        rb.set_estimate(est, se=None, inference=inference, ci_level=level)
        rb.add_warning(
            "No usable standard error came out of this fit, so the estimate is reported "
            "without an interval.",
            level="warning", code="no_se",
        )
        rb.mark_provisional("No standard error could be computed for the headline estimate.")
        return (None, None)
    lo, hi = stats.wald_ci(est, se, level)
    z = est / se
    rb.set_estimate(est, se=se, ci=(lo, hi), p_value=stats.norm_sf2(z), statistic=z,
                    inference=inference, ci_level=level)
    return (lo, hi)


def _estimand_opt(
    ctx: RunContext, rb: ResultBuilder, choices: Sequence[str], default: str
) -> str:
    explicit = ctx.opt("estimand", None)
    raw = explicit if explicit is not None else (ctx.estimand or default)
    value = str(raw).strip().upper()
    if value in choices:
        return value
    if explicit is not None:
        raise SpecError(
            f"This method can target the {' or the '.join(choices)}; you asked for '{raw}'.",
            detail="Pick a supported population in the inspector, or choose another method.",
        )
    rb.add_warning(
        f"The project asks for the {value}. This method reports the {default}; the two answer "
        "different questions, so read the estimand line before quoting the number.",
        level="caution", code="estimand_substituted",
    )
    return default


def _set_estimand(rb: ResultBuilder, name: str, st: Setup, sentence: str | None = None) -> None:
    rb.result["estimand"] = name
    rb.result["estimand_label"] = sentence or roles.describe_estimand(name, st.treat_col, st.out_col)


def _fold_rows(
    fold_id: np.ndarray, scores: np.ndarray, *, level: float, ic: np.ndarray | None = None
) -> list[dict[str, Any]]:
    """Per-fold estimates from a score whose mean is the parameter."""
    rows: list[dict[str, Any]] = []
    z = stats.z_for(level)
    for k in sorted(int(f) for f in np.unique(fold_id)):
        sel = fold_id == k
        m = int(sel.sum())
        if m < 2:
            continue
        vals = scores[sel]
        est = float(np.mean(vals))
        infl = (vals - est) if ic is None else ic[sel]
        se = float(np.std(infl, ddof=1) / math.sqrt(m))
        rows.append({
            "fold": k + 1, "n": m, "estimate": est, "se": se,
            "ci_low": est - z * se, "ci_high": est + z * se,
        })
    return rows


def _cate_not_applicable(rb: ResultBuilder, why: str) -> None:
    """Every learner-driven run carries these three; sometimes the honest value is 'not here'."""
    rb.add_diagnostic(
        "cate_distribution", "Distribution of estimated effects",
        status="not_applicable", summary=why,
        worry_when="Nothing here -- but do not describe heterogeneity this run did not estimate.",
        explain_key="diagnostic.cate_distribution", values={"reason": why},
    )
    rb.add_diagnostic(
        "cate_calibration", "Calibration of predicted against realised effects",
        status="not_applicable", summary=why,
        worry_when="Nothing here. With no per-unit effects there is no ranking to calibrate.",
        explain_key="diagnostic.cate_calibration", values={"reason": why},
    )
    rb.add_diagnostic(
        "rate", "RATE: is the ranking worth targeting on?",
        status="not_applicable", summary=why,
        worry_when="Nothing here. Targeting needs a per-unit ranking to target on.",
        explain_key="diagnostic.rate", values={"reason": why},
    )


def _heterogeneity_block(
    rb: ResultBuilder,
    run: Run,
    tau: np.ndarray | None,
    gamma: np.ndarray | None,
    *,
    source_label: str,
    seed: int,
    score_label: str = "cross-fitted doubly robust (AIPW) scores",
) -> None:
    if tau is None or gamma is None:
        _cate_not_applicable(
            rb,
            "This run estimates one average effect, not an effect per unit, so there is no "
            "distribution of effects to show. A causal forest or a meta-learner does that.",
        )
        return
    diag_cate_distribution(rb, tau, label=source_label)
    diag_cate_calibration(rb, tau, gamma, level=run.level, score_label=score_label)
    diag_rate(rb, tau, gamma, seed=seed, reps=run.rate_reps, level=run.level)


def _seed_stability_block(
    ctx: RunContext,
    rb: ResultBuilder,
    run: Run,
    point: Callable[[int], tuple[float, float | None]],
    overall: float,
) -> None:
    """Rerun the whole cross-fitting under fresh splits when the user asks for repeats."""
    if run.stability_reps <= 1:
        return
    z = stats.z_for(run.level)
    rows: list[dict[str, Any]] = [{
        "seed": int(ctx.seed), "estimate": float(overall), "se": None,
        "ci_low": None, "ci_high": None,
    }]
    for r in range(1, int(run.stability_reps)):
        seed_r = int(ctx.seed) + 9973 * r
        ctx.tick(0.88, f"stability repeat {r} of {run.stability_reps - 1}")
        try:
            est_r, se_r = point(seed_r)
        except CapyError:
            continue
        if est_r is None or not np.isfinite(est_r):
            continue
        rows.append({
            "seed": seed_r, "estimate": float(est_r), "se": se_r,
            "ci_low": (est_r - z * se_r) if se_r else None,
            "ci_high": (est_r + z * se_r) if se_r else None,
        })
    if len(rows) > 1:
        diag_seed_stability(rb, rows, float(overall))


def _clipping_line(nu: Nuisance) -> str:
    if not nu.treatment_binary:
        return "Propensity clipping : not applicable (continuous treatment)"
    share = nu.n_clipped / max(int(nu.g_raw.size), 1)
    return (f"Propensity clipping : {nu.n_clipped} of {nu.g_raw.size} units ({share:.1%}) "
            f"outside [{nu.clip_lo:.3g}, {nu.clip_hi:.3g}]")


def _classic(st: Setup, title: str, run: Run, body: Sequence[str]) -> str:
    head = [
        title,
        "=" * len(title),
        f"Outcome          : {st.out_col}" + ("  (binary)" if st.outcome_binary else ""),
        f"Treatment        : {st.treat_col}" + ("" if st.treatment_binary else "  (continuous dose)"),
        f"N                : {st.n}" + (f"  ({st.n_treated} treated, {st.n_control} control)"
                                        if st.treatment_binary else ""),
        f"Confounders      : {', '.join(st.conf) if st.conf else '(none)'}",
        f"Effect modifiers : {', '.join(st.mods) if st.mods else '(none named; confounder set used)'}",
        f"Learner          : {LEARNER_LABEL.get(run.learner, run.learner)}",
        f"Cross-fitting    : {run.folds} folds, driven by the run seed and reproducible",
    ]
    if run.cluster_name:
        head.append(f"Clustered by     : {run.cluster_name}")
    return "\n".join(head + [""] + list(body))


# ---------------------------------------------------------------------------
# ml.dml_plr -- partially linear DML (Chernozhukov et al. 2018)
# ---------------------------------------------------------------------------


def _plr_theta(y: np.ndarray, d: np.ndarray, nu: Nuisance) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Residual-on-residual least squares with the Neyman-orthogonal score.

    Returns (theta, y_residual, d_residual, denominator).
    """
    y_res = y - nu.m_hat
    d_res = d - nu.g_raw          # the PLR score wants the unclipped E[D|X]
    den = float(np.mean(d_res ** 2))
    if den <= 1e-12:
        raise DataError(
            "After adjusting for the confounders there is essentially no variation left in the "
            "treatment, so nothing can be learned about its effect.",
            detail="The learner predicts treatment almost perfectly from the confounders. "
                   "That is a positivity failure, not a modelling success.",
        )
    theta = float(np.mean(d_res * y_res) / den)
    return theta, y_res, d_res, den


@adapter("ml.dml_plr", label="Double ML (partially linear)", package=PACKAGE,
         needs=["numpy", "scipy", "pandas", "scikit-learn"])
def dml_plr(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Double ML (partially linear)", package=PACKAGE,
                       package_version=package_version())
    run = _begin(ctx, rb, allow_continuous_treatment=True)
    st = run.st
    binary = st.treatment_binary
    estimand = "ATE" if binary else "dose_response"
    _set_estimand(
        rb, estimand, st,
        None if binary else
        f"If the dose of {st.treat_col} rose by one unit, how would average {st.out_col} change?",
    )

    def fit(seed: int) -> dict[str, Any]:
        nu = crossfit(
            st.X, st.y, st.d, folds=run.folds, learner=run.learner, seed=seed, clip=run.clip,
            treatment_binary=binary, outcome_binary=False, ctx=ctx,
            label="fitting the outcome and treatment models",
        )
        theta, y_res, d_res, den = _plr_theta(st.y, st.d, nu)
        psi = d_res * (y_res - theta * d_res)      # mean-zero orthogonal score
        infl = psi / den                            # influence function for theta
        se, inference, n_clusters = _ic_se(infl, run.cluster)
        return {"nu": nu, "theta": theta, "y_res": y_res, "d_res": d_res, "den": den,
                "psi": psi, "infl": infl, "se": se, "inference": inference,
                "n_clusters": n_clusters}

    ctx.tick(0.05, "preparing the analysis sample")
    main = fit(int(ctx.seed))
    nu, theta, se = main["nu"], main["theta"], main["se"]
    ci = _finish(rb, theta, se, level=run.level,
                 inference=f"{main['inference']}, cross-fitted partially linear score")

    ctx.tick(0.60, "scoring the nuisance models")
    diag_nuisance_rmse(rb, nu)
    diag_propensity(rb, nu, st.d, rule="report")

    # Per-fold estimates: refit theta inside each fold's held-out rows.
    z = stats.z_for(run.level)
    fold_est: list[dict[str, Any]] = []
    for k in range(nu.n_folds):
        sel = nu.fold_id == k
        if int(sel.sum()) < 5:
            continue
        dr, yr = main["d_res"][sel], main["y_res"][sel]
        den_k = float(np.mean(dr ** 2))
        if den_k <= 1e-12:
            continue
        th_k = float(np.mean(dr * yr) / den_k)
        infl_k = dr * (yr - th_k * dr) / den_k
        se_k = float(np.std(infl_k, ddof=1) / math.sqrt(int(sel.sum())))
        fold_est.append({"fold": k + 1, "n": int(sel.sum()), "estimate": th_k, "se": se_k,
                         "ci_low": th_k - z * se_k, "ci_high": th_k + z * se_k})
    if fold_est:
        diag_fold_stability(rb, fold_est, theta, se if np.isfinite(se) else None)

    ctx.tick(0.72, "looking for heterogeneity")
    gamma = tau = None
    if binary:
        gamma = aipw_scores(st.y, st.d, nu)
        tau = dr_learner_cate(st.Xc, gamma, folds=run.folds, learner=run.learner,
                              seed=int(ctx.seed) + 5)
    _heterogeneity_block(
        rb, run, tau, gamma, seed=int(ctx.seed),
        source_label="a cross-fitted DR-learner over the effect modifiers -- the partially linear "
                     "model itself assumes one effect for everyone, so this is the check on that",
    )
    if binary and gamma is not None:
        subgroup_estimates(rb, st.df, gamma, run.subgroups, level=run.level)

    _seed_stability_block(ctx, rb, run, lambda s: (lambda f: (f["theta"], f["se"]))(fit(s)), theta)

    rb.add_diagnostic(
        "constant_effect", "The partially linear restriction",
        status="info",
        summary="The partially linear model writes the outcome as tau*D plus a flexible function of "
                "the confounders. That forces one effect on everyone; with a binary treatment the "
                "number it recovers is a variance-weighted average effect, which equals the ATE "
                "only when the effect really is constant.",
        worry_when="The calibration diagnostic finds real heterogeneity. Then read the interactive "
                   "regression model (ml.dml_irm) instead, which targets the ATE directly.",
        explain_key="diagnostic.constant_effect",
        values={"treatment_binary": binary},
    )
    if binary:
        rb.add_warning(
            "The partially linear model reports a variance-weighted effect, not the ATE, when the "
            "effect varies across units. ml.dml_irm targets the ATE without that restriction.",
            level="info", code="plr_weighting",
        )

    forest_rows = [{"label": "Double ML (partially linear)", "estimate": theta, "se": se,
                    "ci_low": ci[0], "ci_high": ci[1], "n": st.n, "engine": "python"}]
    rb.artifact("vega", title="Estimate",
                spec=vega.forest(forest_rows, x_title=f"Effect on {st.out_col}"),
                data=forest_rows,
                caption="One method is not a comparison. Run a second before believing this one.")
    resid_rows = [{"d_res": float(a), "y_res": float(b)}
                  for a, b in zip(main["d_res"][:4000], main["y_res"][:4000])]
    rb.artifact("vega", title="Outcome residual against treatment residual",
                spec=vega.scatter(resid_rows, x="d_res", y="y_res", fit_line=True,
                                  x_title="Treatment residual (D - E[D|X])",
                                  y_title="Outcome residual (Y - E[Y|X])",
                                  title="The regression the estimator actually runs"),
                data=resid_rows,
                caption="The slope of this line is the estimate.")

    rb.set_classic(_classic(st, "Double machine learning -- partially linear model", run, [
        f"Estimand         : {estimand}",
        _clipping_line(nu),
        "",
        f"theta            : {_fmt(theta)}",
        f"  SE             : {_fmt(se)}   ({main['inference']})",
        f"  {run.level:.0%} CI        : [{_fmt(ci[0])}, {_fmt(ci[1])}]",
        f"  denominator    : E[(D - E[D|X])^2] = {_fmt(main['den'])}",
        "",
        "How it works: fit E[Y|X] and E[D|X] out of fold, then regress the outcome residual on the",
        "treatment residual. The score is Neyman-orthogonal, so small errors in either nuisance",
        "model do not move the estimate to first order. The standard error is the analytic one",
        "implied by that score, not a bootstrap.",
        "",
        "What it does not do: it cannot make treatment assignment ignorable. Every number here is",
        "conditional on the confounders you supplied being the ones that mattered.",
    ]))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# ml.dml_irm -- interactive regression model / cross-fitted AIPW
# ---------------------------------------------------------------------------


@adapter("ml.dml_irm", label="Double ML (interactive, AIPW)", package=PACKAGE,
         needs=["numpy", "scipy", "pandas", "scikit-learn"])
def dml_irm(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Double ML (interactive, AIPW)", package=PACKAGE,
                       package_version=package_version())
    run = _begin(ctx, rb)
    st = run.st
    estimand = _estimand_opt(ctx, rb, ("ATE", "ATT"), "ATE")
    _set_estimand(rb, estimand, st)

    def fit(seed: int) -> dict[str, Any]:
        nu = crossfit(
            st.X, st.y, st.d, folds=run.folds, learner=run.learner, seed=seed, clip=run.clip,
            treatment_binary=True, outcome_binary=st.outcome_binary, ctx=ctx,
            label="fitting the outcome and treatment models",
        )
        if estimand == "ATE":
            scores = aipw_scores(st.y, st.d, nu)
            est = float(np.mean(scores))
            infl = scores - est
        else:
            scores, p = att_scores(st.y, st.d, nu)
            est = float(np.mean(scores))
            infl = scores - (st.d / p) * est
        se, inference, n_clusters = _ic_se(infl, run.cluster)
        return {"nu": nu, "est": est, "scores": scores, "infl": infl, "se": se,
                "inference": inference, "n_clusters": n_clusters}

    ctx.tick(0.05, "preparing the analysis sample")
    main = fit(int(ctx.seed))
    nu, est, se = main["nu"], main["est"], main["se"]
    ci = _finish(rb, est, se, level=run.level,
                 inference=f"{main['inference']}, cross-fitted doubly robust score")

    ctx.tick(0.60, "scoring the nuisance models")
    diag_nuisance_rmse(rb, nu)
    diag_propensity(rb, nu, st.d, rule="clip")

    share_clipped = nu.n_clipped / max(st.n, 1)
    if nu.n_clipped:
        rb.add_warning(
            f"{nu.n_clipped} of {st.n} units ({share_clipped:.1%}) had a cross-fitted propensity "
            f"outside [{nu.clip_lo:.3g}, {nu.clip_hi:.3g}] and were clipped to the boundary. "
            "Clipping keeps the weights finite and quietly changes the population the estimate is "
            "about; it is reported here rather than buried.",
            level="caution" if share_clipped > 0.01 else "info",
            code="propensity_clipped", explain_key="diagnostic.propensity_clipping",
        )
    if share_clipped > 0.10:
        rb.mark_provisional(
            f"{share_clipped:.0%} of units sit at the propensity clip; the estimate is carried by "
            "extrapolation into regions where one arm is nearly absent."
        )

    fold_est = _fold_rows(nu.fold_id, main["scores"], level=run.level, ic=main["infl"])
    if fold_est:
        diag_fold_stability(rb, fold_est, est, se if np.isfinite(se) else None)

    ctx.tick(0.72, "looking for heterogeneity")
    gamma = aipw_scores(st.y, st.d, nu)
    tau = dr_learner_cate(st.Xc, gamma, folds=run.folds, learner=run.learner,
                          seed=int(ctx.seed) + 5)
    _heterogeneity_block(rb, run, tau, gamma, seed=int(ctx.seed),
                         source_label="a cross-fitted DR-learner over the effect modifiers")
    subgroup_estimates(rb, st.df, gamma, run.subgroups, level=run.level)
    _seed_stability_block(ctx, rb, run, lambda s: (lambda f: (f["est"], f["se"]))(fit(s)), est)

    if estimand == "ATE":
        rb.add_estimate("Mean outcome if everyone were treated", float(np.mean(nu.mu1)))
        rb.add_estimate("Mean outcome if nobody were treated", float(np.mean(nu.mu0)))
    forest_rows = [{"label": f"Double ML ({estimand})", "estimate": est, "se": se,
                    "ci_low": ci[0], "ci_high": ci[1], "n": st.n, "engine": "python"}]
    rb.artifact("vega", title="Estimate",
                spec=vega.forest(forest_rows, x_title=f"Effect on {st.out_col}"),
                data=forest_rows,
                caption="One method is not a comparison. Run a second before believing this one.")

    rb.set_classic(_classic(st, "Double machine learning -- interactive regression model", run, [
        f"Estimand         : {estimand}",
        _clipping_line(nu),
        f"Clip threshold   : propensities held inside [{nu.clip_lo:.4g}, {nu.clip_hi:.4g}]",
        "",
        f"{estimand} estimate     : {_fmt(est)}",
        f"  SE             : {_fmt(se)}   ({main['inference']})",
        f"  {run.level:.0%} CI        : [{_fmt(ci[0])}, {_fmt(ci[1])}]",
        "",
        "How it works: E[Y|X,D=1], E[Y|X,D=0] and E[D|X] are fitted out of fold, then combined in",
        "the doubly robust score whose mean is the effect. Right if either the outcome models or",
        "the treatment model is right; the standard error comes from that score's own influence",
        "function.",
        "",
        "Clipping is the honest weak point: every clipped unit is a unit for which the data have",
        "almost no comparison, and the estimator replaces it with the value at the boundary.",
    ]))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# ml.causal_forest -- an honest forest built out of scikit-learn trees
# ---------------------------------------------------------------------------

MTRY_CHOICES = ("all", "half", "sqrt")
_MTRY = {"all": None, "half": 0.5, "sqrt": "sqrt"}


def honest_causal_forest(
    X: np.ndarray,
    y_res: np.ndarray,
    d_res: np.ndarray,
    *,
    n_trees: int,
    min_leaf: int,
    subsample_frac: float,
    honest_frac: float,
    max_depth: int | None,
    max_features: Any,
    seed: int,
    ctx: RunContext | None = None,
    progress_from: float = 0.55,
    progress_to: float = 0.75,
) -> dict[str, Any]:
    """Honest causal forest on locally centred residuals.

    The scheme, spelled out because honesty is the whole point of the estimator:

    1. Nuisances E[Y|X] and E[D|X] are cross-fitted elsewhere; this routine sees
       only the residuals Y - E[Y|X] and D - E[D|X] (R-learner local centring).
    2. Each tree draws a subsample without replacement and splits it in two. The
       first half chooses the splits; the second half -- which never influenced
       the tree's shape -- supplies the leaf estimates. That is honesty.
    3. A leaf estimate is the local R-learner solution
       sum(Y~ D~) / sum(D~^2) over the estimating half of that leaf.
    4. A unit's effect is averaged over the trees that never saw it at all
       (out of bag), so no unit's own outcome inflates its own prediction.

    The simplification against grf: the splitting labels are the gradient of the
    local moment condition taken once at the root, not recomputed at every node.
    """
    n = int(X.shape[0])
    den_all = float(np.mean(d_res ** 2))
    if den_all <= 1e-12:
        raise DataError(
            "After adjusting for the confounders there is no variation left in the treatment, so "
            "no forest can learn how the effect varies.",
            detail="This is a positivity failure: the confounders predict treatment almost exactly.",
        )
    tau_global = float(np.mean(d_res * y_res) / den_all)
    rho = d_res * (y_res - tau_global * d_res) / den_all

    rng = np.random.default_rng(int(seed))
    oob_sum = np.zeros(n)
    oob_cnt = np.zeros(n)
    all_sum = np.zeros(n)
    all_cnt = 0
    importance = np.zeros(int(X.shape[1]))
    n_leaves = 0
    n_fallback = 0
    n_grown = 0

    s = int(round(float(subsample_frac) * n))
    s = max(min(s, n - 1), 4)
    for b in range(int(n_trees)):
        if ctx is not None and (b % 25 == 0):
            frac = progress_from + (progress_to - progress_from) * (b / max(int(n_trees), 1))
            ctx.tick(frac, f"growing tree {b + 1} of {n_trees}")
        perm = rng.permutation(n)
        sub, oob = perm[:s], perm[s:]
        cut = int(round(float(honest_frac) * s))
        cut = max(min(cut, s - 1), 1)
        j1, j2 = sub[:cut], sub[cut:]
        if j1.size < max(2 * min_leaf, 4) or j2.size < max(min_leaf, 2):
            continue
        tree = DecisionTreeRegressor(
            min_samples_leaf=int(min_leaf),
            max_depth=(int(max_depth) if max_depth else None),
            max_features=max_features,
            random_state=int(seed) + 7919 * (b + 1),
        )
        tree.fit(X[j1], rho[j1])
        nodes = tree.apply(X)
        nn = int(tree.tree_.node_count)
        num = np.bincount(nodes[j2], weights=(y_res[j2] * d_res[j2]), minlength=nn)
        den = np.bincount(nodes[j2], weights=(d_res[j2] ** 2), minlength=nn)
        cnt = np.bincount(nodes[j2], minlength=nn)
        value = np.full(nn, tau_global)
        ok = (cnt >= max(int(min_leaf), 2)) & (den > 1e-10)
        value[ok] = num[ok] / den[ok]
        used = np.unique(nodes[j2])
        n_leaves += int(used.size)
        n_fallback += int(np.sum(~ok[used]))
        preds = value[nodes]
        if oob.size:
            oob_sum[oob] += preds[oob]
            oob_cnt[oob] += 1
        all_sum += preds
        all_cnt += 1
        importance += np.asarray(tree.feature_importances_, dtype=float)
        n_grown += 1

    if n_grown == 0:
        raise DataError(
            "The sample is too small to grow an honest forest: every tree would have fewer rows "
            "than one leaf needs.",
            detail="Lower the minimum leaf size, or use a method that does not split the sample twice.",
        )
    tau_in = all_sum / max(all_cnt, 1)
    tau = np.where(oob_cnt > 0, oob_sum / np.maximum(oob_cnt, 1.0), tau_in)
    return {
        "tau": tau,
        "tau_in_sample": tau_in,
        "tau_global": tau_global,
        "n_trees_grown": n_grown,
        "n_without_oob": int(np.sum(oob_cnt == 0)),
        "mean_oob_trees": float(np.mean(oob_cnt)),
        "importance": (importance / max(n_grown, 1)),
        "leaf_fallback_share": float(n_fallback / max(n_leaves, 1)),
        "subsample_size": int(s),
        "split_half": int(round(float(honest_frac) * s)),
    }


@adapter("ml.causal_forest", label="Honest causal forest", package=PACKAGE,
         needs=["numpy", "scipy", "pandas", "scikit-learn"])
def causal_forest(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Honest causal forest", package=PACKAGE,
                       package_version=package_version())
    run = _begin(ctx, rb)
    st = run.st
    estimand = _estimand_opt(ctx, rb, ("ATE", "ATT"), "ATE")
    _set_estimand(rb, estimand, st)

    n_trees = _int_opt(ctx, "n_trees", 400, 50, 2000)
    min_leaf = _int_opt(ctx, "min_leaf", 5, 1, 200)
    max_depth = _int_opt(ctx, "max_depth", 0, 0, 30)
    subsample = _float_opt(ctx, "subsample_fraction", 0.5, 0.1, 0.9)
    honest = _float_opt(ctx, "honest_fraction", 0.5, 0.2, 0.8)
    mtry = _choice(ctx, "mtry", MTRY_CHOICES, "all")

    def fit(seed: int) -> dict[str, Any]:
        nu = crossfit(
            st.X, st.y, st.d, folds=run.folds, learner=run.learner, seed=seed, clip=run.clip,
            treatment_binary=True, outcome_binary=st.outcome_binary, ctx=ctx,
            progress_from=0.08, progress_to=0.50,
            label="centring the outcome and treatment",
        )
        if estimand == "ATE":
            scores = aipw_scores(st.y, st.d, nu)
            est = float(np.mean(scores))
            infl = scores - est
        else:
            scores, p = att_scores(st.y, st.d, nu)
            est = float(np.mean(scores))
            infl = scores - (st.d / p) * est
        se, inference, _ = _ic_se(infl, run.cluster)
        return {"nu": nu, "est": est, "scores": scores, "infl": infl, "se": se,
                "inference": inference}

    ctx.tick(0.05, "preparing the analysis sample")
    main = fit(int(ctx.seed))
    nu, est, se = main["nu"], main["est"], main["se"]
    ci = _finish(rb, est, se, level=run.level,
                 inference=f"{main['inference']}, doubly robust average over the forest's nuisances")

    ctx.tick(0.55, "growing the honest forest")
    forest = honest_causal_forest(
        st.Xc, st.y - nu.m_hat, st.d - nu.g_raw,
        n_trees=n_trees, min_leaf=min_leaf, subsample_frac=subsample, honest_frac=honest,
        max_depth=(max_depth or None), max_features=_MTRY[mtry], seed=int(ctx.seed) + 31, ctx=ctx,
    )
    tau = forest["tau"]

    ctx.tick(0.78, "scoring the nuisance models")
    diag_nuisance_rmse(rb, nu)
    diag_propensity(rb, nu, st.d, rule="clip")
    fold_est = _fold_rows(nu.fold_id, main["scores"], level=run.level, ic=main["infl"])
    if fold_est:
        diag_fold_stability(rb, fold_est, est, se if np.isfinite(se) else None)

    gamma = aipw_scores(st.y, st.d, nu)
    _heterogeneity_block(rb, run, tau, gamma, seed=int(ctx.seed),
                         source_label="out-of-bag predictions from the honest forest")
    subgroup_estimates(rb, st.df, gamma, run.subgroups, level=run.level)
    _seed_stability_block(ctx, rb, run, lambda s: (lambda f: (f["est"], f["se"]))(fit(s)), est)

    imp_rows = [{"variable": nm, "importance": float(v)}
                for nm, v in zip(st.c_names, forest["importance"])]
    imp_rows.sort(key=lambda r: -r["importance"])
    imp_id = rb.artifact(
        "vega", title="What the forest split on",
        spec=vega.bar_chart(imp_rows[:20], x="variable", y="importance", horizontal=True,
                            x_title="Variable", y_title="Share of splitting signal",
                            title="What the forest split on"),
        data=imp_rows, explain_key="diagnostic.forest_honesty",
        caption="Split frequency, weighted by how much each split moved the effect. It ranks "
                "variables; it does not say the effect really varies with them -- the calibration "
                "diagnostic is what says that.",
    )
    honesty_id = rb.artifact(
        "table", title="Honesty scheme",
        data=[
            {"setting": "Trees grown", "value": forest["n_trees_grown"]},
            {"setting": "Rows per tree (subsample, no replacement)", "value": forest["subsample_size"]},
            {"setting": "Rows choosing the splits", "value": forest["split_half"]},
            {"setting": "Rows estimating the leaves",
             "value": forest["subsample_size"] - forest["split_half"]},
            {"setting": "Average out-of-bag trees per unit", "value": round(forest["mean_oob_trees"], 1)},
            {"setting": "Units with no out-of-bag tree", "value": forest["n_without_oob"]},
            {"setting": "Leaves falling back to the overall effect",
             "value": f"{forest['leaf_fallback_share']:.1%}"},
        ],
        columns=["setting", "value"], explain_key="diagnostic.forest_honesty",
    )
    rb.add_diagnostic(
        "forest_honesty", "Honest splitting and out-of-bag prediction",
        status="info",
        summary=(
            f"{forest['n_trees_grown']} trees, each grown on {forest['subsample_size']} rows drawn "
            f"without replacement and split in two: {forest['split_half']} rows chose the splits and "
            f"{forest['subsample_size'] - forest['split_half']} rows -- which never influenced the "
            "tree's shape -- supplied the leaf estimates. Each unit's effect averages the "
            f"{forest['mean_oob_trees']:.0f} trees on average that never saw it. "
            f"{forest['leaf_fallback_share']:.1%} of leaves had too little residual treatment "
            "variation to estimate on their own and fell back to the overall effect."
        ),
        worry_when="A large share of leaves falling back, or units with no out-of-bag tree: then the "
                   "per-unit effects are mostly the overall effect wearing a disguise. Honesty costs "
                   "sample size, which is the price of not reading noise as heterogeneity.",
        artifact_ids=[honesty_id, imp_id],
        explain_key="diagnostic.forest_honesty",
        values={k: forest[k] for k in ("n_trees_grown", "subsample_size", "split_half",
                                       "mean_oob_trees", "n_without_oob", "leaf_fallback_share",
                                       "tau_global")},
    )
    if forest["leaf_fallback_share"] > 0.5:
        rb.add_warning(
            "Most leaves had too little residual treatment variation to estimate an effect of their "
            "own, so the forest is close to reporting one effect for everyone. Read the calibration "
            "diagnostic before describing heterogeneity.",
            level="caution", code="forest_degenerate",
        )
    if forest["n_without_oob"]:
        rb.add_warning(
            f"{forest['n_without_oob']} unit(s) appeared in every tree's subsample, so their effect "
            "had to be predicted in sample. Grow more trees if you plan to use those predictions.",
            level="info", code="forest_no_oob",
        )

    forest_rows = [{"label": f"Honest causal forest ({estimand})", "estimate": est, "se": se,
                    "ci_low": ci[0], "ci_high": ci[1], "n": st.n, "engine": "python"}]
    rb.artifact("vega", title="Estimate",
                spec=vega.forest(forest_rows, x_title=f"Effect on {st.out_col}"),
                data=forest_rows,
                caption="The average effect comes from the doubly robust score, not from averaging "
                        "the forest's own predictions.")

    rb.set_classic(_classic(st, "Honest causal forest", run, [
        f"Estimand         : {estimand}",
        f"Trees            : {forest['n_trees_grown']}   (min leaf {min_leaf}, mtry {mtry}"
        + (f", max depth {max_depth}" if max_depth else ", depth unlimited") + ")",
        f"Honesty          : {forest['subsample_size']} rows per tree, "
        f"{forest['split_half']} choose the splits, "
        f"{forest['subsample_size'] - forest['split_half']} estimate the leaves",
        f"Out-of-bag       : {forest['mean_oob_trees']:.0f} trees per unit on average; "
        f"{forest['n_without_oob']} unit(s) had none",
        _clipping_line(nu),
        "",
        f"{estimand} estimate     : {_fmt(est)}",
        f"  SE             : {_fmt(se)}   ({main['inference']})",
        f"  {run.level:.0%} CI        : [{_fmt(ci[0])}, {_fmt(ci[1])}]",
        f"Effect spread    : SD of the out-of-bag effects = {_fmt(float(np.std(tau, ddof=1)))}",
        "",
        "The honesty scheme, in full:",
        "  1. E[Y|X] and E[D|X] are cross-fitted, and the forest works on the residuals only",
        "     (R-learner local centring), so the trees cannot rediscover the confounding.",
        "  2. Every tree draws a subsample WITHOUT replacement and cuts it in half. One half",
        "     chooses the splits; the other half, which never touched the tree's shape, supplies",
        "     the leaf estimates. A leaf's effect is sum(Y~ D~) / sum(D~^2) over that half.",
        "  3. A unit's effect averages only the trees whose subsample excluded it entirely.",
        "",
        "Simplification against grf: the splitting labels are the gradient of the local moment",
        "condition taken once at the root rather than recomputed at every node, and there is no",
        "bootstrap-of-little-bags, so the per-unit effects carry no pointwise interval. Judge the",
        "ranking by the calibration and RATE diagnostics, not by eye.",
    ]))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# ml.metalearner -- T, S, X and DR learners over the same cross-fitting
# ---------------------------------------------------------------------------

METALEARNERS = ("t", "s", "x", "dr")
METALEARNER_LABEL = {
    "t": "T-learner (one outcome model per arm)",
    "s": "S-learner (one model with treatment as a feature)",
    "x": "X-learner (imputed effects, propensity-weighted)",
    "dr": "DR-learner (regression on the doubly robust score)",
}


def _kfold_predict(
    X: np.ndarray,
    target: np.ndarray,
    *,
    train_mask: np.ndarray | None,
    folds: int,
    learner: str,
    seed: int,
    notes: list[str],
    tag: str,
) -> np.ndarray:
    """Cross-fitted prediction for every row, trained on a subset of the training folds."""
    n = int(X.shape[0])
    k = int(min(max(folds, 2), max(2, n // 10)))
    out = np.zeros(n)
    splitter = KFold(n_splits=k, shuffle=True, random_state=int(seed))
    for j, (tr, te) in enumerate(splitter.split(X)):
        rows = tr if train_mask is None else tr[train_mask[tr]]
        out[te] = _predict_reg(learner, int(seed) + 101 * (j + 1), X[rows], target[rows], X[te],
                               notes, f"{tag} fold {j + 1}")
    return out


def _s_learner_cate(
    X: np.ndarray, y: np.ndarray, d: np.ndarray, *, folds: int, learner: str, seed: int
) -> tuple[np.ndarray, list[str]]:
    n = int(X.shape[0])
    Xd = np.hstack([X, d.reshape(-1, 1)])
    X1 = np.hstack([X, np.ones((n, 1))])
    X0 = np.hstack([X, np.zeros((n, 1))])
    notes: list[str] = []
    tau = np.zeros(n)
    k = int(min(max(folds, 2), max(2, n // 10)))
    splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(seed))
    for j, (tr, te) in enumerate(splitter.split(Xd, (d > 0.5).astype(int))):
        p1 = _predict_reg(learner, int(seed) + 211 * (j + 1), Xd[tr], y[tr], X1[te], notes,
                          f"S-learner fold {j + 1} (treated arm)")
        p0 = _predict_reg(learner, int(seed) + 211 * (j + 1), Xd[tr], y[tr], X0[te], notes,
                          f"S-learner fold {j + 1} (control arm)")
        tau[te] = p1 - p0
    return tau, notes


def _x_learner_cate(
    Xc: np.ndarray, y: np.ndarray, d: np.ndarray, nu: Nuisance, *,
    folds: int, learner: str, seed: int,
) -> tuple[np.ndarray, list[str]]:
    treated = d > 0.5
    imputed = np.where(treated, y - nu.mu0, nu.mu1 - y)
    notes: list[str] = []
    tau1 = _kfold_predict(Xc, imputed, train_mask=treated, folds=folds, learner=learner,
                          seed=int(seed) + 3, notes=notes, tag="X-learner (treated arm)")
    tau0 = _kfold_predict(Xc, imputed, train_mask=~treated, folds=folds, learner=learner,
                          seed=int(seed) + 5, notes=notes, tag="X-learner (control arm)")
    g = np.clip(nu.g_hat, 1e-6, 1 - 1e-6)
    return g * tau0 + (1.0 - g) * tau1, notes


@adapter("ml.metalearner", label="Meta-learner (T, S, X, DR)", package=PACKAGE,
         needs=["numpy", "scipy", "pandas", "scikit-learn"])
def metalearner(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Meta-learner (T, S, X, DR)", package=PACKAGE,
                       package_version=package_version())
    run = _begin(ctx, rb)
    st = run.st
    kind = _choice(ctx, "metalearner", METALEARNERS, "dr")
    estimand = _estimand_opt(ctx, rb, ("ATE", "ATT"), "ATE")
    _set_estimand(rb, estimand, st)
    if kind == "s" and run.learner in ("linear", "lasso"):
        rb.add_warning(
            "An S-learner built on a linear model has no treatment-by-covariate term, so it can "
            "only ever report the same effect for everyone. If you are looking for heterogeneity, "
            "use the T-, X- or DR-learner, or give the S-learner a flexible learner.",
            level="caution", code="s_learner_linear",
        )

    def fit(seed: int) -> dict[str, Any]:
        nu = crossfit(
            st.X, st.y, st.d, folds=run.folds, learner=run.learner, seed=seed, clip=run.clip,
            treatment_binary=True, outcome_binary=st.outcome_binary, ctx=ctx,
            label="fitting the outcome and treatment models",
        )
        if estimand == "ATE":
            scores = aipw_scores(st.y, st.d, nu)
            est = float(np.mean(scores))
            infl = scores - est
        else:
            scores, p = att_scores(st.y, st.d, nu)
            est = float(np.mean(scores))
            infl = scores - (st.d / p) * est
        se, inference, _ = _ic_se(infl, run.cluster)
        return {"nu": nu, "est": est, "scores": scores, "infl": infl, "se": se,
                "inference": inference}

    ctx.tick(0.05, "preparing the analysis sample")
    main = fit(int(ctx.seed))
    nu, est, se = main["nu"], main["est"], main["se"]
    ci = _finish(rb, est, se, level=run.level,
                 inference=f"{main['inference']}, cross-fitted doubly robust score")

    ctx.tick(0.62, f"fitting the {kind.upper()}-learner")
    gamma = aipw_scores(st.y, st.d, nu)
    notes: list[str] = []
    if kind == "t":
        tau = nu.mu1 - nu.mu0
    elif kind == "s":
        tau, notes = _s_learner_cate(st.X, st.y, st.d, folds=run.folds, learner=run.learner,
                                     seed=int(ctx.seed) + 11)
    elif kind == "x":
        tau, notes = _x_learner_cate(st.Xc, st.y, st.d, nu, folds=run.folds, learner=run.learner,
                                     seed=int(ctx.seed) + 13)
    else:
        tau = dr_learner_cate(st.Xc, gamma, folds=run.folds, learner=run.learner,
                              seed=int(ctx.seed) + 17)
    plug_in = float(np.mean(tau))

    ctx.tick(0.74, "scoring the nuisance models")
    diag_nuisance_rmse(rb, nu)
    diag_propensity(rb, nu, st.d, rule="clip")
    fold_est = _fold_rows(nu.fold_id, main["scores"], level=run.level, ic=main["infl"])
    if fold_est:
        diag_fold_stability(rb, fold_est, est, se if np.isfinite(se) else None)
    _heterogeneity_block(rb, run, tau, gamma, seed=int(ctx.seed),
                         source_label=f"{METALEARNER_LABEL[kind]}, cross-fitted")
    subgroup_estimates(rb, st.df, gamma, run.subgroups, level=run.level)
    _seed_stability_block(ctx, rb, run, lambda s: (lambda f: (f["est"], f["se"]))(fit(s)), est)

    rb.add_estimate(
        f"{METALEARNER_LABEL[kind]}: plain average of its own predicted effects",
        plug_in, se=None, group="diagnostic_only",
    )
    rb.add_diagnostic(
        "metalearner_plug_in", "The meta-learner's own average, and why it is not the headline",
        status="info",
        summary=(
            f"Averaging the {kind.upper()}-learner's per-unit predictions gives {plug_in:.4g}, "
            f"against the doubly robust headline of {est:.4g}. The headline is the one with a valid "
            "standard error: a regularised learner's predictions are shrunk towards the middle, and "
            "that bias does not average away, so the plug-in average is reported without an interval."
        ),
        worry_when="The two are far apart. That is regularisation bias in the learner, and it means "
                   "the per-unit effects are shrunk -- fine for ranking units, wrong for quoting "
                   "a magnitude.",
        explain_key="diagnostic.metalearner_plug_in",
        values={"metalearner": kind, "plug_in_mean": plug_in, "doubly_robust": est,
                "difference": plug_in - est, "fallbacks": notes[:20]},
    )
    if notes:
        rb.add_warning(
            f"{len(notes)} stage(s) of the meta-learner fell back to a mean prediction. "
            "The per-unit effects are less informative than they look.",
            level="caution", code="metalearner_fallback",
        )

    forest_rows = [
        {"label": f"Doubly robust {estimand} (headline)", "estimate": est, "se": se,
         "ci_low": ci[0], "ci_high": ci[1], "n": st.n, "engine": "python"},
        {"label": f"{kind.upper()}-learner plug-in average", "estimate": plug_in,
         "ci_low": None, "ci_high": None, "n": st.n, "engine": "python", "provisional": True},
    ]
    rb.artifact("vega", title="Estimate",
                spec=vega.forest(forest_rows, x_title=f"Effect on {st.out_col}"),
                data=forest_rows,
                caption="The triangle has no interval on purpose: a plug-in average of shrunk "
                        "predictions has no honest one.")

    rb.set_classic(_classic(st, f"Meta-learner -- {METALEARNER_LABEL[kind]}", run, [
        f"Estimand         : {estimand}",
        f"Meta-learner     : {METALEARNER_LABEL[kind]}",
        _clipping_line(nu),
        "",
        f"{estimand} estimate     : {_fmt(est)}   (cross-fitted doubly robust score)",
        f"  SE             : {_fmt(se)}   ({main['inference']})",
        f"  {run.level:.0%} CI        : [{_fmt(ci[0])}, {_fmt(ci[1])}]",
        "",
        f"{kind.upper()}-learner average of its own predictions : {_fmt(plug_in)}   (no interval)",
        f"Spread of predicted effects (SD)              : {_fmt(float(np.std(tau, ddof=1)))}",
        "",
        "Division of labour: the meta-learner's job is the effect for each unit, and the doubly",
        "robust score's job is the average with a standard error you can defend. Reporting the",
        "learner's own average as if it had an interval is the usual mistake, so it is shown here",
        "without one.",
        "",
        "T fits one outcome model per arm and subtracts. S fits a single model with treatment as a",
        "feature, which can shrink the effect to nothing if the learner regularises it away. X",
        "imputes each unit's effect from the other arm's model and weights the two by the",
        "propensity, which helps when the arms are very different sizes. DR regresses the doubly",
        "robust score on the effect modifiers and is the one whose errors matter least.",
    ]))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# ml.policy_tree -- a shallow tree you can read, valued on held-out data
# ---------------------------------------------------------------------------


def _split_candidates(x: np.ndarray, max_candidates: int) -> np.ndarray:
    u = np.unique(x)
    if u.size < 2:
        return np.empty(0)
    if u.size <= max_candidates + 1:
        return (u[:-1] + u[1:]) / 2.0
    qs = np.linspace(0.0, 1.0, max_candidates + 2)[1:-1]
    cand = np.unique(np.quantile(x, qs))
    return cand[(cand > u[0]) & (cand < u[-1])]


def grow_policy_tree(
    X: np.ndarray,
    gamma: np.ndarray,
    names: Sequence[str],
    *,
    depth: int,
    min_leaf: int,
    max_candidates: int = 32,
) -> dict[str, Any]:
    """Greedy shallow tree that maximises the sum of doubly robust scores it treats.

    Greedy, with a one-step lookahead at each node: not the exact optimum a
    depth-2 exhaustive search would find, which is said out loud wherever the
    tree is reported.
    """

    def leaf(idx: np.ndarray) -> dict[str, Any]:
        s = float(np.sum(gamma[idx]))
        return {"kind": "leaf", "action": 1 if s > 0 else 0, "value": max(s, 0.0),
                "n": int(idx.size), "score_sum": s}

    def build(idx: np.ndarray, d: int) -> dict[str, Any]:
        node = leaf(idx)
        if d <= 0 or idx.size < 2 * min_leaf:
            return node
        best: tuple[float, int, float, np.ndarray] | None = None
        for j in range(int(X.shape[1])):
            xs = X[idx, j]
            for t in _split_candidates(xs, max_candidates):
                left = xs <= t
                nl = int(left.sum())
                if nl < min_leaf or (idx.size - nl) < min_leaf:
                    continue
                sl = float(np.sum(gamma[idx[left]]))
                sr = node["score_sum"] - sl
                v = max(sl, 0.0) + max(sr, 0.0)
                if best is None or v > best[0] + 1e-12:
                    best = (v, j, float(t), left)
        if best is None or best[0] <= node["value"] + 1e-12:
            return node
        _, j, t, left = best
        return {
            "kind": "split", "col": int(j), "variable": str(names[j]), "threshold": float(t),
            "n": int(idx.size), "left": build(idx[left], d - 1), "right": build(idx[~left], d - 1),
        }

    return build(np.arange(int(X.shape[0])), int(depth))


def policy_apply(node: dict[str, Any], X: np.ndarray) -> np.ndarray:
    out = np.zeros(int(X.shape[0]))

    def walk(nd: dict[str, Any], idx: np.ndarray) -> None:
        if idx.size == 0:
            return
        if nd["kind"] == "leaf":
            out[idx] = float(nd["action"])
            return
        left = X[idx, nd["col"]] <= nd["threshold"]
        walk(nd["left"], idx[left])
        walk(nd["right"], idx[~left])

    walk(node, np.arange(int(X.shape[0])))
    return out


def policy_rules(node: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def walk(nd: dict[str, Any], path: list[str]) -> None:
        if nd["kind"] == "leaf":
            rows.append({
                "rule": " and ".join(path) if path else "everyone",
                "action": "treat" if nd["action"] else "do not treat",
                "n_training_rows": nd["n"],
            })
            return
        walk(nd["left"], path + [f"{nd['variable']} <= {nd['threshold']:.4g}"])
        walk(nd["right"], path + [f"{nd['variable']} > {nd['threshold']:.4g}"])

    walk(node, [])
    return rows


def policy_text(node: dict[str, Any], indent: int = 0) -> list[str]:
    pad = "  " * indent
    if node["kind"] == "leaf":
        return [f"{pad}-> {'TREAT' if node['action'] else 'do not treat'}  ({node['n']} rows)"]
    out = [f"{pad}if {node['variable']} <= {node['threshold']:.4g}:"]
    out += policy_text(node["left"], indent + 1)
    out.append(f"{pad}else:")
    out += policy_text(node["right"], indent + 1)
    return out


@adapter("ml.policy_tree", label="Policy tree (who should be treated)", package=PACKAGE,
         needs=["numpy", "scipy", "pandas", "scikit-learn"])
def policy_tree(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Policy tree (who should be treated)", package=PACKAGE,
                       package_version=package_version())
    run = _begin(ctx, rb)
    st = run.st
    depth = _int_opt(ctx, "depth", 2, 1, 3)
    min_leaf = _int_opt(ctx, "min_leaf", 25, 5, 5000)
    eval_share = _float_opt(ctx, "eval_share", 0.5, 0.2, 0.8)
    _set_estimand(
        rb, "policy_value", st,
        f"If only the units this tree picks out were given {st.treat_col}, how much better off "
        f"would average {st.out_col} be than treating nobody?",
    )

    ctx.tick(0.05, "preparing the analysis sample")
    nu = crossfit(
        st.X, st.y, st.d, folds=run.folds, learner=run.learner, seed=int(ctx.seed), clip=run.clip,
        treatment_binary=True, outcome_binary=st.outcome_binary, ctx=ctx,
        label="fitting the outcome and treatment models",
    )
    gamma = aipw_scores(st.y, st.d, nu)

    n = st.n
    rng = np.random.default_rng(int(ctx.seed) + 41)
    perm = rng.permutation(n)
    n_eval = int(round(eval_share * n))
    n_eval = max(min(n_eval, n - 2 * min_leaf), 10)
    if n - n_eval < 4 * min_leaf:
        raise DataError(
            f"An honest policy evaluation splits the sample in two, and {n} rows do not leave "
            f"enough on the learning side for leaves of at least {min_leaf} rows.",
            detail="Lower the minimum leaf size, lower the evaluation share, or use more data.",
        )
    eval_idx = np.sort(perm[:n_eval])
    train_idx = np.sort(perm[n_eval:])
    rb.add_flow("Held out to value the policy", n=int(train_idx.size), dropped=int(eval_idx.size),
                reason=f"{eval_idx.size} row(s) were kept back so the learned policy could be "
                       "valued on data that did not choose it")

    ctx.tick(0.62, "learning the policy")
    tree_train = grow_policy_tree(st.Xc[train_idx], gamma[train_idx], st.c_names,
                                  depth=depth, min_leaf=min_leaf)
    pi_eval = policy_apply(tree_train, st.Xc[eval_idx])
    pi_train = policy_apply(tree_train, st.Xc[train_idx])
    g_eval = gamma[eval_idx]
    cluster_eval = run.cluster[eval_idx] if run.cluster is not None else None

    value = float(np.mean(g_eval * pi_eval))
    se, inference, _ = _ic_se(g_eval * pi_eval - value, cluster_eval)
    if float(np.sum(pi_eval)) == 0.0:
        # A policy that treats nobody is worth exactly nothing, with no sampling error in it.
        value, se = 0.0, 0.0
        inference = "exact: the learned policy treats nobody"
        rb.set_estimate(0.0, se=0.0, ci=(0.0, 0.0), inference=inference, ci_level=run.level)
        ci = (0.0, 0.0)
        rb.add_warning(
            "The tree recommends treating nobody. On these data no rule it could find beat doing "
            "nothing, so the value of the policy is zero by construction rather than by estimation.",
            level="caution", code="policy_empty",
        )
    else:
        ci = _finish(rb, value, se, level=run.level,
                     inference=f"{inference}, on the {eval_idx.size} held-out rows")

    treat_all = float(np.mean(g_eval))
    se_all, _, _ = _ic_se(g_eval - treat_all, cluster_eval)
    gain = float(np.mean(g_eval * (pi_eval - 1.0)))
    se_gain, _, _ = _ic_se(g_eval * (pi_eval - 1.0) - gain, cluster_eval)
    in_sample = float(np.mean(gamma[train_idx] * pi_train))
    z = stats.z_for(run.level)

    rb.add_estimate("Value of the learned policy (held out, against treating nobody)", value,
                    se=se, ci=(ci[0], ci[1]), group="policy", n=int(eval_idx.size))
    rb.add_estimate("Value of treating everyone (held out)", treat_all, se=se_all,
                    ci=stats.wald_ci(treat_all, se_all, run.level), group="policy",
                    n=int(eval_idx.size))
    rb.add_estimate("Gain of the learned policy over treating everyone (held out)", gain,
                    se=se_gain, ci=stats.wald_ci(gain, se_gain, run.level),
                    p_value=stats.norm_sf2(gain / se_gain) if se_gain > 0 else None,
                    group="policy", n=int(eval_idx.size))
    rb.add_estimate("Value of treating nobody", 0.0, group="policy")
    rb.add_estimate("Value on the rows the tree was learned from (optimistic)", in_sample,
                    se=None, group="policy", n=int(train_idx.size))

    ctx.tick(0.74, "scoring the nuisance models")
    diag_nuisance_rmse(rb, nu)
    diag_propensity(rb, nu, st.d, rule="clip")
    fold_est = _fold_rows(nu.fold_id, gamma, level=run.level)
    if fold_est:
        ate = float(np.mean(gamma))
        se_ate, _, _ = _ic_se(gamma - ate, run.cluster)
        diag_fold_stability(rb, fold_est, ate, se_ate)
    tau = dr_learner_cate(st.Xc, gamma, folds=run.folds, learner=run.learner,
                          seed=int(ctx.seed) + 19)
    _heterogeneity_block(rb, run, tau, gamma, seed=int(ctx.seed),
                         source_label="a cross-fitted DR-learner over the effect modifiers; the "
                                      "tree itself is grown on the doubly robust scores directly")
    subgroup_estimates(rb, st.df, gamma, run.subgroups, level=run.level)

    # The tree the user reads is grown on everything; its value is the held-out one above.
    tree_full = grow_policy_tree(st.Xc, gamma, st.c_names, depth=depth, min_leaf=min_leaf)
    pi_full = policy_apply(tree_full, st.Xc)
    rules = policy_rules(tree_full)
    rules_id = rb.artifact("table", title="The policy, as rules", data=rules,
                           columns=["rule", "action", "n_training_rows"],
                           explain_key="diagnostic.policy_value",
                           caption="Grown on the whole sample so you can read it; valued on the "
                                   "held-out half so the value is not self-graded.")
    text_id = rb.artifact("text", title="The policy, as a tree",
                          data="\n".join(policy_text(tree_full)),
                          explain_key="diagnostic.policy_value")
    bar_rows = [
        {"label": "Learned policy (held out)", "value": value},
        {"label": "Treat everyone (held out)", "value": treat_all},
        {"label": "Treat nobody", "value": 0.0},
        {"label": "Learned policy, in sample (optimistic)", "value": in_sample},
    ]
    bar_id = rb.artifact(
        "vega", title="What each policy is worth",
        spec=vega.bar_chart(bar_rows, x="label", y="value", horizontal=True, sort_desc=False,
                            x_title="Policy", y_title=f"Average gain in {st.out_col} per person",
                            title="What each policy is worth"),
        data=bar_rows, explain_key="diagnostic.policy_value",
    )
    rb.add_diagnostic(
        "policy_value", "What the policy is worth, valued honestly",
        status="supports" if (se_gain > 0 and gain - z * se_gain > 0) else "info",
        summary=(
            f"The tree was learned on {train_idx.size} rows and valued on {eval_idx.size} rows it "
            f"never saw. On those held-out rows it treats {float(np.mean(pi_eval)):.0%} of units and "
            f"is worth {value:.4g} per person against treating nobody"
            + (f" (SE {se:.4g})" if se and np.isfinite(se) else "")
            + f", against {treat_all:.4g} for treating everyone -- a difference of {gain:+.4g}"
            + (f" (SE {se_gain:.4g})" if se_gain and np.isfinite(se_gain) else "")
            + f". On the rows it was learned from the same policy scores {in_sample:.4g}; that "
              "number is optimistic by construction and must not be quoted."
        ),
        worry_when="The held-out value is much lower than the in-sample one -- the tree has fitted "
                   "noise. And a gain over treating everyone whose interval includes zero means "
                   "you have not shown that targeting beats the simple policy, which is usually "
                   "cheaper to run and easier to defend. Note the interval holds the fitted "
                   "nuisance models fixed, so it is a little narrower than the truth.",
        artifact_ids=[bar_id, rules_id, text_id],
        explain_key="diagnostic.policy_value",
        values={
            "depth": depth, "min_leaf": min_leaf, "n_train": int(train_idx.size),
            "n_eval": int(eval_idx.size), "eval_share": eval_share,
            "value_held_out": value, "value_se": se if np.isfinite(se) else None,
            "value_ci_low": ci[0], "value_ci_high": ci[1],
            "value_in_sample": in_sample,
            "value_treat_all": treat_all, "value_treat_all_se": se_all,
            "gain_over_treat_all": gain, "gain_se": se_gain,
            "share_treated_held_out": float(np.mean(pi_eval)),
            "share_treated_full_sample": float(np.mean(pi_full)),
            "search": "greedy, one-step lookahead per node",
            "rules": rules,
        },
    )
    rb.add_warning(
        "The value of a policy measured on the same rows that chose it is optimistic, always. The "
        f"headline here is the held-out value on {eval_idx.size} rows the tree never saw; the "
        f"in-sample figure ({in_sample:.4g}) is shown only so you can see the size of the gap.",
        level="caution", code="policy_optimism", explain_key="diagnostic.policy_value",
    )
    if not (se_gain > 0 and gain - z * se_gain > 0):
        rb.add_warning(
            "This tree has not been shown to beat treating everyone. Before deploying targeting, "
            "weigh that against the cost of running a rule at all.",
            level="caution", code="policy_no_gain",
        )
    if depth >= 3:
        rb.add_warning(
            "A depth-3 tree has many more ways to fit noise than a depth-1 rule, and the search is "
            "greedy rather than exhaustive. Compare the held-out values across depths before "
            "choosing one.",
            level="info", code="policy_depth",
        )

    rb.set_classic(_classic(st, "Policy tree -- who should be treated", run, [
        f"Estimand         : value of the learned policy against treating nobody",
        f"Depth            : {depth}   (minimum leaf {min_leaf} rows, greedy search)",
        f"Honest split     : {train_idx.size} rows learned the tree, "
        f"{eval_idx.size} rows valued it",
        _clipping_line(nu),
        "",
        "The policy:",
        *policy_text(tree_full, indent=1),
        "",
        f"Held-out value vs treating nobody  : {_fmt(value)}   SE {_fmt(se)}",
        f"  {run.level:.0%} CI                          : [{_fmt(ci[0])}, {_fmt(ci[1])}]",
        f"Held-out value of treating everyone: {_fmt(treat_all)}   SE {_fmt(se_all)}",
        f"Gain over treating everyone        : {_fmt(gain)}   SE {_fmt(se_gain)}",
        f"Share treated (held out)           : {float(np.mean(pi_eval)):.1%}",
        "",
        f"In-sample value (DO NOT QUOTE)     : {_fmt(in_sample)}",
        "  This number is optimistic by construction: a policy chosen to maximise a sum will",
        "  always score well on the rows it chose from. The held-out number above is the one",
        "  that means anything.",
        "",
        "The tree printed here was grown on the whole sample so that it is the one you would",
        "deploy; the value attached to it comes from the held-out half. Growing and valuing on",
        "the same rows is the mistake this method exists to avoid.",
    ]))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# obs.tmle -- targeted maximum likelihood
# ---------------------------------------------------------------------------


def _expit_np(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _logit_np(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    q = np.clip(p, eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def fluctuate(
    ystar: np.ndarray, H: np.ndarray, q_init: np.ndarray, *, max_iter: int = 100
) -> tuple[float, bool, int]:
    """One-parameter logistic fluctuation: MLE of eps in logit(Q) + eps*H.

    Newton with step halving on the quasi-binomial log-likelihood. Deterministic,
    and it reports whether it actually converged rather than pretending.
    """
    offset = _logit_np(q_init)

    def loglik(eps: float) -> float:
        p = np.clip(_expit_np(offset + eps * H), 1e-12, 1 - 1e-12)
        return float(np.sum(ystar * np.log(p) + (1.0 - ystar) * np.log(1.0 - p)))

    eps = 0.0
    ll = loglik(eps)
    for it in range(int(max_iter)):
        p = _expit_np(offset + eps * H)
        score = float(np.sum(H * (ystar - p)))
        info = float(np.sum(H * H * p * (1.0 - p)))
        if info <= 1e-12 or not np.isfinite(score):
            return eps, abs(score) < 1e-6, it
        step = score / info
        if not np.isfinite(step):
            return eps, False, it
        alpha = 1.0
        for _ in range(30):
            cand = eps + alpha * step
            ll_new = loglik(cand)
            if np.isfinite(ll_new) and ll_new >= ll - 1e-12:
                eps, ll = cand, ll_new
                break
            alpha /= 2.0
        else:
            return eps, False, it
        if abs(alpha * step) < 1e-10:
            return eps, True, it + 1
    return eps, False, int(max_iter)


@adapter("obs.tmle", label="Targeted maximum likelihood (TMLE)", package=PACKAGE,
         needs=["numpy", "scipy", "pandas", "scikit-learn"])
def tmle(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Targeted maximum likelihood (TMLE)", package=PACKAGE,
                       package_version=package_version())
    run = _begin(ctx, rb)
    st = run.st
    estimand = _estimand_opt(ctx, rb, ("ATE", "ATT"), "ATE")
    _set_estimand(rb, estimand, st)

    y = st.y
    lo_y, hi_y = float(np.min(y)), float(np.max(y))
    if hi_y - lo_y <= 0:
        raise DataError(f"The outcome '{st.out_col}' takes a single value in this sample.")
    if st.outcome_binary:
        a_bound, b_bound = 0.0, 1.0
        bound_note = "binary outcome; no transform needed"
    else:
        pad = _float_opt(ctx, "bound_padding", 0.0, 0.0, 0.25) * (hi_y - lo_y)
        a_bound, b_bound = lo_y - pad, hi_y + pad
        bound_note = (f"continuous outcome mapped to [0, 1] by (y - {a_bound:.6g}) / "
                      f"{(b_bound - a_bound):.6g}, then mapped back")
    scale = b_bound - a_bound
    ystar = (y - a_bound) / scale
    ystar = np.clip(ystar, 0.0, 1.0)

    def fit(seed: int) -> dict[str, Any]:
        nu = crossfit(
            st.X, st.y, st.d, folds=run.folds, learner=run.learner, seed=seed, clip=run.clip,
            treatment_binary=True, outcome_binary=st.outcome_binary, ctx=ctx,
            label="fitting the initial outcome and treatment models",
        )
        q1 = np.clip((nu.mu1 - a_bound) / scale, 1e-4, 1 - 1e-4)
        q0 = np.clip((nu.mu0 - a_bound) / scale, 1e-4, 1 - 1e-4)
        qa = np.where(st.d > 0.5, q1, q0)
        g = np.clip(nu.g_hat, 1e-6, 1 - 1e-6)
        p_treated = float(np.mean(st.d))
        if estimand == "ATE":
            H = st.d / g - (1.0 - st.d) / (1.0 - g)
            H1, H0 = 1.0 / g, -1.0 / (1.0 - g)
        else:
            if p_treated <= 0 or p_treated >= 1:
                raise DataError("The ATT needs both treated and untreated rows.")
            H = (st.d - (1.0 - st.d) * g / (1.0 - g)) / p_treated
            H1 = np.ones_like(g) / p_treated
            H0 = -(g / (1.0 - g)) / p_treated
        eps, converged, iters = fluctuate(ystar, H, qa)
        q1s = _expit_np(_logit_np(q1) + eps * H1)
        q0s = _expit_np(_logit_np(q0) + eps * H0)
        qas = np.where(st.d > 0.5, q1s, q0s)
        contrast = q1s - q0s
        resid = H * (ystar - qas)
        if estimand == "ATE":
            psi_star = float(np.mean(contrast))
            score = (contrast + resid) * scale
            est = psi_star * scale
            infl = score - est
        else:
            psi_star = float(np.mean(st.d * contrast) / p_treated)
            score = (resid + (st.d / p_treated) * contrast) * scale
            est = psi_star * scale
            infl = score - (st.d / p_treated) * est
        se, inference, _ = _ic_se(infl, run.cluster)
        initial = (float(np.mean(q1 - q0)) if estimand == "ATE"
                   else float(np.mean(st.d * (q1 - q0)) / p_treated)) * scale
        return {"nu": nu, "est": est, "se": se, "inference": inference, "score": score,
                "infl": infl, "eps": eps, "converged": converged, "iters": iters,
                "initial": initial, "H": H, "resid": resid, "q1s": q1s, "q0s": q0s,
                "qas": qas, "ystar": ystar}

    ctx.tick(0.05, "preparing the analysis sample")
    main = fit(int(ctx.seed))
    nu, est, se = main["nu"], main["est"], main["se"]
    ci = _finish(rb, est, se, level=run.level,
                 inference=f"{main['inference']}, TMLE influence curve")

    ctx.tick(0.62, "scoring the nuisance models")
    diag_nuisance_rmse(rb, nu)
    diag_propensity(rb, nu, st.d, rule="clip")
    fold_est = _fold_rows(nu.fold_id, main["score"], level=run.level, ic=main["infl"])
    if fold_est:
        diag_fold_stability(rb, fold_est, est, se if np.isfinite(se) else None)

    gamma = aipw_scores(st.y, st.d, nu)
    tau = dr_learner_cate(st.Xc, gamma, folds=run.folds, learner=run.learner,
                          seed=int(ctx.seed) + 23)
    _heterogeneity_block(rb, run, tau, gamma, seed=int(ctx.seed),
                         source_label="a cross-fitted DR-learner over the effect modifiers")
    subgroup_estimates(rb, st.df, gamma, run.subgroups, level=run.level)
    _seed_stability_block(ctx, rb, run, lambda s: (lambda f: (f["est"], f["se"]))(fit(s)), est)

    mean_score = float(np.mean(main["resid"]))
    score_scaled = abs(mean_score) * scale
    rel = score_scaled / se if se and np.isfinite(se) and se > 0 else None
    move = est - main["initial"]
    conv_rows = [
        {"quantity": "Fluctuation parameter (epsilon)", "value": _fmt(main["eps"])},
        {"quantity": "Newton iterations", "value": main["iters"]},
        {"quantity": "Converged", "value": bool(main["converged"])},
        {"quantity": "Initial (untargeted) estimate", "value": _fmt(main["initial"])},
        {"quantity": "Targeted estimate", "value": _fmt(est)},
        {"quantity": "Moved by", "value": _fmt(move)},
        {"quantity": "Mean of the score equation", "value": _fmt(score_scaled)},
        {"quantity": "Score as a share of the SE", "value": _fmt(rel) if rel is not None else "."},
    ]
    conv_id = rb.artifact("table", title="Targeting step", data=conv_rows,
                          columns=["quantity", "value"], explain_key="diagnostic.tmle_targeting")
    ic_rows = stats.histogram_rows(main["infl"] , bins=30)
    ic_id = rb.artifact("vega", title="Influence curve",
                        spec=vega.histogram(ic_rows, x_title="Influence curve value", rule_at=0.0,
                                            title="Influence curve, unit by unit"),
                        data=ic_rows, explain_key="diagnostic.tmle_targeting",
                        caption="A few units carrying most of the mass means the standard error "
                                "rests on those units.")
    solved = rel is not None and rel < 0.05
    rb.add_diagnostic(
        "tmle_targeting", "The targeting step",
        status="supports" if (main["converged"] and solved) else "weakens",
        summary=(
            f"The fluctuation moved the estimate from {main['initial']:.4g} (the plain outcome-model "
            f"answer) to {est:.4g}, a shift of {move:+.4g}, with epsilon = {main['eps']:.4g} after "
            f"{main['iters']} Newton step(s). The efficient score equation is solved to "
            f"{score_scaled:.3g}"
            + (f", which is {rel:.1%} of the standard error." if rel is not None else ".")
        ),
        worry_when="Epsilon is large, or the score equation is not solved: then the targeting step "
                   "did most of the work and the initial outcome model was a long way off. A big "
                   "move is also a warning that the propensity weights are extreme, because it is "
                   "the weights that pull the estimate.",
        artifact_ids=[conv_id, ic_id],
        explain_key="diagnostic.tmle_targeting",
        values={
            "epsilon": main["eps"], "converged": bool(main["converged"]),
            "iterations": main["iters"], "initial_estimate": main["initial"],
            "targeted_estimate": est, "shift": move,
            "score_equation": score_scaled, "score_over_se": rel,
            "outcome_bounds": [a_bound, b_bound], "transform": bound_note,
        },
    )
    if not main["converged"]:
        rb.mark_provisional(
            "The targeting step did not converge, so the estimate is not the targeted one it claims "
            "to be. Treat it as a doubly robust estimate with an approximate interval."
        )
        rb.add_warning(
            "The fluctuation step did not converge. The number is reported so you can see it, but "
            "the efficiency argument behind TMLE does not hold here.",
            level="warning", code="tmle_no_convergence",
        )
    if rel is not None and rel > 0.05:
        rb.mark_provisional(
            "The efficient score equation is not solved to within a small fraction of the standard "
            "error, so the influence-curve interval understates the uncertainty."
        )

    rb.add_estimate("Initial (untargeted) outcome-model estimate", main["initial"], se=None,
                    group="diagnostic_only")
    if estimand == "ATE":
        rb.add_estimate("Mean outcome if everyone were treated",
                        float(np.mean(main["q1s"])) * scale + a_bound)
        rb.add_estimate("Mean outcome if nobody were treated",
                        float(np.mean(main["q0s"])) * scale + a_bound)

    forest_rows = [{"label": f"TMLE ({estimand})", "estimate": est, "se": se,
                    "ci_low": ci[0], "ci_high": ci[1], "n": st.n, "engine": "python"}]
    rb.artifact("vega", title="Estimate",
                spec=vega.forest(forest_rows, x_title=f"Effect on {st.out_col}"),
                data=forest_rows,
                caption="One method is not a comparison. Run a second before believing this one.")

    rb.set_classic(_classic(st, "Targeted maximum likelihood estimation", run, [
        f"Estimand         : {estimand}",
        f"Outcome scale    : {bound_note}",
        _clipping_line(nu),
        "",
        f"Initial estimate : {_fmt(main['initial'])}   (outcome models only, before targeting)",
        f"{estimand} estimate     : {_fmt(est)}",
        f"  SE             : {_fmt(se)}   ({main['inference']})",
        f"  {run.level:.0%} CI        : [{_fmt(ci[0])}, {_fmt(ci[1])}]",
        f"  epsilon        : {_fmt(main['eps'])} after {main['iters']} Newton step(s), "
        f"converged: {main['converged']}",
        f"  score equation : {_fmt(score_scaled)}"
        + (f"  ({rel:.1%} of the SE)" if rel is not None else ""),
        "",
        "The four steps, in order:",
        "  1. Initial fit. E[Y|X,D] and E[D|X] are fitted out of fold by the chosen learner.",
        "  2. Clever covariate. H = D/g - (1-D)/(1-g) for the ATE; for the ATT,",
        "     H = (D - (1-D) g/(1-g)) / P(D=1).",
        "  3. Fluctuation. One logistic step, logit(Q) + eps*H, fitted by maximum likelihood on the",
        "     bounded outcome. That is what makes the estimate solve the efficient score equation.",
        "  4. Inference. The standard error is the standard deviation of the influence curve, which",
        "     is where the efficiency claim comes from.",
        "",
        "Continuous outcomes are squeezed into [0, 1] before the fluctuation and mapped back after",
        "(Gruber and van der Laan). That keeps the update inside the bounds of the outcome, which a",
        "linear fluctuation would not.",
        "",
        "What it shares with every method in this bench: nothing here tests whether the confounders",
        "you supplied were the ones that mattered.",
    ]))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# Method cards
# ---------------------------------------------------------------------------

_ML_ROLES = {
    "roles_required": ["treatment", "outcome", "confounders"],
    "roles_optional": ["effect_modifiers", "cluster"],
    "roles_forbidden": ["forbidden"],
}

_LEARNER_OPTION = {
    "name": "learner", "type": "select", "default": "linear", "choices": list(LEARNER_CHOICES),
    "label": "Nuisance learner", "profile": "standard",
    "help": "The model used for E[Y|X] and E[D|X]. Boring is usually right: a regularised linear "
            "model is hard to beat when the sample is small, and a flexible learner cannot fix a "
            "confounder you did not measure.",
}


def _shared_options(*, standard_learner: bool = True) -> list[dict[str, Any]]:
    learner = dict(_LEARNER_OPTION)
    if not standard_learner:
        learner["profile"] = "advanced"
    return [
        learner,
        {"name": "folds", "type": "int", "default": 5, "min": 2, "max": 20,
         "label": "Cross-fitting folds", "profile": "standard",
         "help": "Each fold is scored by models that never saw it. More folds use the data better "
                 "and cost more time; below 5 the estimate starts to depend on the split."},
        {"name": "clip", "type": "number", "default": 0.01, "min": 0.0, "max": 0.45,
         "label": "Clip propensity scores at", "profile": "advanced",
         "help": "Keeps the weights finite when a unit is nearly certain to be treated. Every "
                 "clipped unit is reported, because clipping quietly changes the population."},
        {"name": "ci_level", "type": "number", "default": 0.95, "min": 0.5, "max": 0.999,
         "label": "Confidence level", "profile": "advanced",
         "help": "The interval width. Changing it after seeing the result is not a neutral act."},
        {"name": "stability_reps", "type": "int", "default": 1, "min": 1, "max": 10,
         "label": "Repeat under new fold splits", "profile": "advanced",
         "help": "Reruns the whole cross-fitting under fresh random splits. The split is an "
                 "arbitrary choice; the answer should not depend on it."},
        {"name": "rate_reps", "type": "int", "default": 200, "min": 0, "max": 2000,
         "label": "Bootstrap draws for the RATE curve", "profile": "advanced",
         "help": "Draws used for the standard error of the targeting curve. 0 skips it."},
        {"name": "subgroups", "type": "columns", "default": [],
         "label": "Pre-registered subgroups", "profile": "standard",
         "help": "Groups you named BEFORE looking. Each one gets the same doubly robust score as "
                 "the headline, and the multiplicity caution comes with them."},
    ]


_ML_DIAGNOSTICS = ["nuisance_rmse", "propensity_clipping", "cate_distribution",
                   "cate_calibration", "rate", "fold_stability", "seed_stability"]

_ML_PROBES = ["probe.cinelli_hazlett", "probe.rosenbaum", "probe.placebo_outcome",
              "probe.subset", "probe.random_common_cause", "probe.trim_curve"]

_ML_NEEDS = ["numpy", "scipy", "pandas", "scikit-learn"]

_ML_CAVEAT = ("Machine learning does not identify anything. Cross-fitting removes the bias that "
              "comes from fitting flexible models on the same rows you estimate with; it does "
              "nothing about a confounder you never measured.")


METHOD_CARDS: list[dict[str, Any]] = [
    {
        "id": "ml.dml_plr",
        "title": "Double ML (partially linear)",
        "one_liner": "Predict the outcome and the treatment from the confounders, then regress "
                     "what is left of one on what is left of the other.",
        "designs": [DESIGN],
        "estimands": ["ATE", "dose_response"],
        **_ML_ROLES,
        "options": _shared_options(),
        "diagnostics": _ML_DIAGNOSTICS + ["constant_effect"],
        "probes": _ML_PROBES,
        "needs": _ML_NEEDS,
        "explain_key": "method.ml.dml_plr",
        "status": "recommended",
        "why_recommended": "The one causal ML method that also handles a continuous dose, with an "
                           "analytic standard error that comes from the estimator's own orthogonal "
                           "score rather than a bootstrap.",
        "what_can_go_wrong": "It assumes one effect for everyone. With a binary treatment and a "
                             "varying effect it returns a variance-weighted average that is not the "
                             "ATE. If the confounders predict treatment almost perfectly there is "
                             "no residual variation left to divide by, and the interval collapses "
                             "into nonsense. " + _ML_CAVEAT,
        "needs_overlap": True,
        "engines": {"python": True, "r": "DoubleML"},
        "references": [
            "Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey & Robins (2018), "
            "Double/debiased machine learning for treatment and structural parameters",
            "Robinson (1988), Root-N-consistent semiparametric regression",
        ],
        "disrecommend_when": "The effect plainly varies across units -- use ml.dml_irm instead.",
    },
    {
        "id": "ml.dml_irm",
        "title": "Double ML (interactive)",
        "one_liner": "A separate outcome model for each arm plus a treatment model, combined in a "
                     "doubly robust score: right if either side is right.",
        "designs": [DESIGN],
        "estimands": ["ATE", "ATT"],
        **_ML_ROLES,
        "options": _shared_options() + [
            {"name": "estimand", "type": "select", "default": "ATE", "choices": ["ATE", "ATT"],
             "label": "Population", "profile": "standard",
             "help": "ATE is everyone; ATT is the units that actually got treated. They differ "
                     "whenever the effect varies, which is the usual case."},
        ],
        "diagnostics": _ML_DIAGNOSTICS,
        "probes": _ML_PROBES,
        "needs": _ML_NEEDS,
        "explain_key": "method.ml.dml_irm",
        "status": "recommended",
        "why_recommended": "No restriction on how the effect varies, two chances to get the "
                           "adjustment right, and a standard error from the score's own influence "
                           "function. This is the default causal ML estimator for a binary treatment.",
        "what_can_go_wrong": "Doubly robust is not doubly safe. When propensities approach 0 or 1 "
                             "the weights explode, clipping kicks in, and the estimand quietly "
                             "becomes something about a population you did not choose. The run "
                             "reports how many units were clipped and at what threshold. " + _ML_CAVEAT,
        "needs_overlap": True,
        "engines": {"python": True, "r": "DoubleML"},
        "references": [
            "Chernozhukov et al. (2018), Double/debiased machine learning",
            "Robins, Rotnitzky & Zhao (1994), Estimation of regression coefficients when some "
            "regressors are not always observed",
        ],
        "disrecommend_when": "Fewer than about 200 rows, where cross-fitting has too little to "
                             "work with.",
    },
    {
        "id": "ml.causal_forest",
        "title": "Honest causal forest",
        "one_liner": "Grow trees that look for units whose effect differs, keeping the rows that "
                     "chose each split away from the rows that estimate it.",
        "designs": [DESIGN],
        "estimands": ["CATE", "ATE", "ATT"],
        **_ML_ROLES,
        "options": _shared_options(standard_learner=False) + [
            {"name": "estimand", "type": "select", "default": "ATE", "choices": ["ATE", "ATT"],
             "label": "Population for the headline average", "profile": "standard",
             "help": "The per-unit effects are the point of the method; this is the population the "
                     "single headline number averages over."},
            {"name": "n_trees", "type": "int", "default": 400, "min": 50, "max": 2000,
             "label": "Trees", "profile": "standard",
             "help": "More trees means a smoother, more stable set of per-unit effects. It does not "
                     "buy you significance."},
            {"name": "min_leaf", "type": "int", "default": 5, "min": 1, "max": 200,
             "label": "Minimum rows per leaf", "profile": "advanced",
             "help": "Larger leaves mean steadier but blunter effects. Because of honesty, a leaf "
                     "needs enough rows on BOTH sides of the split to be usable."},
            {"name": "max_depth", "type": "int", "default": 0, "min": 0, "max": 30,
             "label": "Maximum tree depth (0 = unlimited)", "profile": "advanced",
             "help": "A cap on how finely the forest may cut the population."},
            {"name": "subsample_fraction", "type": "number", "default": 0.5, "min": 0.1, "max": 0.9,
             "label": "Rows per tree", "profile": "advanced",
             "help": "The share of the sample each tree draws, without replacement. The rest are "
                     "that tree's out-of-bag rows."},
            {"name": "honest_fraction", "type": "number", "default": 0.5, "min": 0.2, "max": 0.8,
             "label": "Share of a tree's rows that choose the splits", "profile": "advanced",
             "help": "The remainder estimate the leaves. Splitting these two jobs is what stops the "
                     "forest from reading its own noise as heterogeneity."},
            {"name": "mtry", "type": "select", "default": "all",
             "choices": list(MTRY_CHOICES), "label": "Variables tried per split",
             "profile": "advanced",
             "help": "Trying fewer variables at each split makes the trees more different from one "
                     "another, which can help when there are many modifiers."},
        ],
        "diagnostics": _ML_DIAGNOSTICS + ["forest_honesty"],
        "probes": _ML_PROBES,
        "needs": _ML_NEEDS,
        "explain_key": "method.ml.causal_forest",
        "status": "recommended",
        "why_recommended": "The standard way to ask who benefits without fishing: honest sample "
                           "splitting inside every tree, out-of-bag predictions, and a calibration "
                           "test that tells you whether the ranking is real.",
        "what_can_go_wrong": "A forest will always return a spread of effects, including when there "
                             "is none. Read the calibration slope and the RATE before describing "
                             "heterogeneity. This implementation has no pointwise intervals on the "
                             "per-unit effects, and its splitting labels are a one-step "
                             "approximation of grf's. " + _ML_CAVEAT,
        "needs_overlap": True,
        "engines": {"python": True, "r": "grf"},
        "references": [
            "Wager & Athey (2018), Estimation and inference of heterogeneous treatment effects "
            "using random forests",
            "Athey, Tibshirani & Wager (2019), Generalized random forests",
            "Nie & Wager (2021), Quasi-oracle estimation of heterogeneous treatment effects",
        ],
        "disrecommend_when": "Small samples: honesty spends half of every tree's rows on estimation, "
                             "so under a few hundred rows the leaves are mostly the overall effect.",
    },
    {
        "id": "ml.metalearner",
        "title": "Meta-learner (T, S, X, DR)",
        "one_liner": "Build the per-unit effect out of ordinary prediction models, four different "
                     "ways, on the same cross-fitting.",
        "designs": [DESIGN],
        "estimands": ["CATE", "ATE", "ATT"],
        **_ML_ROLES,
        "options": _shared_options() + [
            {"name": "metalearner", "type": "select", "default": "dr",
             "choices": list(METALEARNERS), "label": "Which meta-learner", "profile": "standard",
             "help": "T fits one outcome model per arm. S fits one model with treatment as a "
                     "feature. X imputes each unit's effect from the other arm and weights by the "
                     "propensity. DR regresses the doubly robust score on the modifiers and is the "
                     "safest default."},
            {"name": "estimand", "type": "select", "default": "ATE", "choices": ["ATE", "ATT"],
             "label": "Population for the headline average", "profile": "standard",
             "help": "The headline average always comes from the doubly robust score, because that "
                     "is the number with a defensible interval."},
        ],
        "diagnostics": _ML_DIAGNOSTICS + ["metalearner_plug_in"],
        "probes": _ML_PROBES,
        "needs": _ML_NEEDS,
        "explain_key": "method.ml.metalearner",
        "status": "reasonable",
        "why_recommended": "Comparing meta-learners on the same folds is the cheapest way to see "
                           "how much of your 'heterogeneity' is the learner rather than the world.",
        "what_can_go_wrong": "The S-learner can regularise the treatment effect away to nothing. "
                             "The T-learner inherits both outcome models' biases. None of the "
                             "per-unit effects carries an interval, and the plain average of a "
                             "learner's predictions is shrunk, which is why it is never the "
                             "headline here. " + _ML_CAVEAT,
        "needs_overlap": True,
        "engines": {"python": True, "r": "grf"},
        "references": [
            "Kunzel, Sekhon, Bickel & Yu (2019), Metalearners for estimating heterogeneous "
            "treatment effects using machine learning",
            "Kennedy (2023), Towards optimal doubly robust estimation of heterogeneous causal effects",
        ],
        "disrecommend_when": "You want one average effect and nothing else -- ml.dml_irm is the "
                             "same score with less machinery.",
    },
    {
        "id": "ml.policy_tree",
        "title": "Policy tree (who should be treated)",
        "one_liner": "Learn a short, readable rule for who to treat, then value that rule on data "
                     "it never saw.",
        "designs": [DESIGN],
        "estimands": ["policy_value", "CATE"],
        **_ML_ROLES,
        "options": _shared_options(standard_learner=False) + [
            {"name": "depth", "type": "int", "default": 2, "min": 1, "max": 3,
             "label": "Tree depth", "profile": "standard",
             "help": "Depth 1 is a single rule. Depth 3 can fit noise in a way you will not see "
                     "until the policy is deployed."},
            {"name": "min_leaf", "type": "int", "default": 25, "min": 5, "max": 5000,
             "label": "Minimum rows per leaf", "profile": "standard",
             "help": "Stops the tree from carving out a handful of lucky rows."},
            {"name": "eval_share", "type": "number", "default": 0.5, "min": 0.2, "max": 0.8,
             "label": "Share held back to value the policy", "profile": "advanced",
             "help": "These rows do not help choose the rule, so the value measured on them is not "
                     "self-graded. Holding back less makes the value less reliable, not more."},
        ],
        "diagnostics": _ML_DIAGNOSTICS + ["policy_value"],
        "probes": _ML_PROBES,
        "needs": _ML_NEEDS,
        "explain_key": "method.ml.policy_tree",
        "status": "reasonable",
        "why_recommended": "A rule you can print on one page and hand to the people who run the "
                           "programme, with a value measured on rows that did not choose it.",
        "what_can_go_wrong": "The in-sample value of a learned policy is optimistic, always, and "
                             "the gap can be large. The search here is greedy rather than "
                             "exhaustive, so the tree is a good rule, not the best one. A policy "
                             "that does not beat treating everyone is usually not worth the "
                             "administrative cost of targeting. " + _ML_CAVEAT,
        "needs_overlap": True,
        "engines": {"python": True, "r": "policytree"},
        "references": [
            "Athey & Wager (2021), Policy learning with observational data",
            "Zhou, Athey & Wager (2023), Offline multi-action policy learning",
        ],
        "disrecommend_when": "The calibration diagnostic says the per-unit effects carry no signal: "
                             "then there is nothing to target on and the tree is decoration.",
    },
    {
        "id": "obs.tmle",
        "title": "Targeted maximum likelihood (TMLE)",
        "one_liner": "Fit the outcome model, then nudge it once in the direction the treatment "
                     "model says it is wrong.",
        "designs": [DESIGN],
        "estimands": ["ATE", "ATT"],
        **_ML_ROLES,
        "options": _shared_options() + [
            {"name": "estimand", "type": "select", "default": "ATE", "choices": ["ATE", "ATT"],
             "label": "Population", "profile": "standard",
             "help": "ATE is everyone; ATT is the units that actually got treated."},
            {"name": "bound_padding", "type": "number", "default": 0.0, "min": 0.0, "max": 0.25,
             "label": "Padding on the outcome bounds", "profile": "advanced",
             "help": "Continuous outcomes are squeezed into [0, 1] using the observed range before "
                     "the targeting step. Padding widens that range if you expect values outside "
                     "the sample's own minimum and maximum."},
        ],
        "diagnostics": _ML_DIAGNOSTICS + ["tmle_targeting"],
        "probes": _ML_PROBES,
        "needs": _ML_NEEDS,
        "explain_key": "method.obs.tmle",
        "status": "recommended",
        "why_recommended": "Doubly robust like AIPW, but the estimate is a prediction from a model, "
                           "so it stays inside the bounds of the outcome -- a weighted estimator "
                           "can return a risk difference no data could produce.",
        "what_can_go_wrong": "The targeting step is driven by inverse propensity weights, so "
                             "extreme propensities move the estimate a long way and the interval "
                             "with it. The run reports how far the targeting moved the number and "
                             "whether the score equation was actually solved. " + _ML_CAVEAT,
        "needs_overlap": True,
        "engines": {"python": True, "r": "tmle"},
        "references": [
            "van der Laan & Rubin (2006), Targeted maximum likelihood learning",
            "Gruber & van der Laan (2010), A targeted maximum likelihood estimator of a causal "
            "effect on a bounded continuous outcome",
            "Schuler & Rose (2017), Targeted maximum likelihood estimation for causal inference in "
            "observational studies",
        ],
        "disrecommend_when": "Overlap is so poor that most units sit at the propensity clip; then "
                             "the fluctuation is fitting the boundary, not the data.",
    },
]
