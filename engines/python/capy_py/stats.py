"""Shared numerics for Causal Capybara's Python adapters.

Small, boring, and tested. Adapters translate a spec into calls here or into a
third-party package; they do not re-derive linear algebra per design.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .contracts import DataError

try:  # scipy is a hard dependency of the engine, but keep the failure legible
    from scipy import stats as _sps
except Exception:  # pragma: no cover
    _sps = None


#: Below this many clusters the cluster-robust sandwich is known to be
#: anti-conservative -- the interval comes out too narrow and the p-value too
#: small (Bertrand, Duflo and Mullainathan 2004; Cameron and Miller 2015).
FEW_CLUSTERS = 30

#: Past this ratio between the largest and smallest direction in the design,
#: double-precision arithmetic has nothing left to work with: the answer would
#: be made mostly of rounding error even though the rank test still passes.
#: Chosen well clear of the ill-conditioned-but-usable range (a pair of
#: covariates correlated at 0.999999 comes in around 1e6).
SINGULAR_CONDITION = 1e12

#: A standard error this many times the spread of the outcome is not a wide
#: interval, it is a fit that never resolved: the normal equations were solved
#: through a pivot that is numerically zero, and every number downstream of it
#: is rounding noise dressed up as a finding.
ABSURD_SE_RATIO = 1e6


# ---------------------------------------------------------------------------
# Distributions (thin wrappers so adapters do not import scipy directly)
# ---------------------------------------------------------------------------


def norm_ppf(q: float) -> float:
    if _sps is not None:
        return float(_sps.norm.ppf(q))
    # Acklam's rational approximation, plenty for a CI multiplier
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        x = math.sqrt(-2 * math.log(q))
        return (((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x + c[5]) / (
            (((d[0] * x + d[1]) * x + d[2]) * x + d[3]) * x + 1)
    if q > phigh:
        x = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x + c[5]) / (
            (((d[0] * x + d[1]) * x + d[2]) * x + d[3]) * x + 1)
    x = q - 0.5
    r = x * x
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * x / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def norm_sf2(z: float) -> float:
    """Two-sided normal p-value."""
    z = abs(float(z))
    if _sps is not None:
        return float(2.0 * _sps.norm.sf(z))
    return math.erfc(z / math.sqrt(2.0))


def t_sf2(t: float, df: float) -> float:
    """Two-sided t p-value; falls back to the normal when df is large or scipy is absent."""
    if _sps is not None and df and df > 0:
        return float(2.0 * _sps.t.sf(abs(float(t)), df))
    return norm_sf2(t)


def t_ppf(q: float, df: float) -> float:
    if _sps is not None and df and df > 0:
        return float(_sps.t.ppf(q, df))
    return norm_ppf(q)


def chi2_sf(x: float, df: int) -> float:
    if _sps is not None:
        return float(_sps.chi2.sf(float(x), int(df)))
    return float("nan")


def f_sf(x: float, dfn: int, dfd: int) -> float:
    if _sps is not None:
        return float(_sps.f.sf(float(x), int(dfn), int(dfd)))
    return float("nan")


def z_for(level: float = 0.95) -> float:
    return norm_ppf(0.5 + level / 2.0)


def wald_ci(estimate: float, se: float | None, level: float = 0.95) -> tuple[float | None, float | None]:
    if se is None or not np.isfinite(se) or se <= 0:
        return (None, None)
    z = z_for(level)
    return (float(estimate - z * se), float(estimate + z * se))


# ---------------------------------------------------------------------------
# Design matrices
# ---------------------------------------------------------------------------


def is_binary(series: pd.Series) -> bool:
    vals = pd.unique(series.dropna())
    if len(vals) != 2:
        return False
    try:
        as_num = {float(v) for v in vals}
    except (TypeError, ValueError):
        return True
    return as_num in ({0.0, 1.0}, {0.0, 1.0})


def to01(series: pd.Series) -> np.ndarray:
    """Coerce a two-valued column to 0/1 with the larger/"true" level as 1."""
    s = series
    if s.dtype == bool:
        return s.to_numpy().astype(float)
    vals = sorted(pd.unique(s.dropna()), key=lambda v: (str(type(v)), str(v)))
    if len(vals) > 2:
        raise ValueError(f"Column '{series.name}' has {len(vals)} levels; expected 2.")
    try:
        num = pd.to_numeric(s, errors="raise").to_numpy(dtype=float)
        uniq = sorted({v for v in np.unique(num[~np.isnan(num)])})
        if uniq == [0.0, 1.0] or uniq == [0.0] or uniq == [1.0]:
            return num
        hi = max(uniq)
        return (num == hi).astype(float)
    except (TypeError, ValueError):
        pass
    truthy = {"1", "true", "yes", "treated", "treatment", "t", "y"}
    lowered = s.astype(str).str.strip().str.lower()
    if lowered.dropna().isin(truthy | {"0", "false", "no", "control", "untreated", "f", "n"}).all():
        return lowered.isin(truthy).astype(float).to_numpy()
    hi = str(vals[-1])
    return (s.astype(str) == hi).astype(float).to_numpy()


@dataclass
class DesignMatrix:
    X: np.ndarray
    names: list[str]
    index: pd.Index
    dropped: list[str] = field(default_factory=list)

    @property
    def k(self) -> int:
        return self.X.shape[1]


def design_matrix(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    intercept: bool = True,
    drop_constant: bool = True,
    max_levels: int = 40,
) -> DesignMatrix:
    """Numeric design matrix with one-hot expansion for categorical columns."""
    n = len(df)
    parts: list[np.ndarray] = []
    names: list[str] = []
    if intercept:
        parts.append(np.ones((n, 1)))
        names.append("(Intercept)")
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' is not in the analysis sample.")
        s = df[col]
        if s.dtype == bool:
            parts.append(s.to_numpy(dtype=float).reshape(-1, 1))
            names.append(col)
            continue
        if pd.api.types.is_numeric_dtype(s):
            parts.append(pd.to_numeric(s, errors="coerce").to_numpy(dtype=float).reshape(-1, 1))
            names.append(col)
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            parts.append(s.astype("int64").to_numpy(dtype=float).reshape(-1, 1))
            names.append(col)
            continue
        levels = pd.unique(s.dropna().astype(str))
        levels = sorted(levels)
        if len(levels) > max_levels:
            raise ValueError(
                f"Column '{col}' has {len(levels)} levels; that is a fixed effect, not a covariate."
            )
        for lev in levels[1:]:
            parts.append((s.astype(str) == lev).to_numpy(dtype=float).reshape(-1, 1))
            names.append(f"{col}[{lev}]")
    if not parts:
        return DesignMatrix(np.zeros((n, 0)), [], df.index)
    X = np.hstack(parts)
    dropped: list[str] = []
    if drop_constant and X.shape[1] > (1 if intercept else 0):
        keep = []
        for j in range(X.shape[1]):
            if intercept and j == 0:
                keep.append(j)
                continue
            col = X[:, j]
            if np.nanstd(col) <= 1e-12:
                dropped.append(names[j])
            else:
                keep.append(j)
        X = X[:, keep]
        names = [names[j] for j in keep]
    return DesignMatrix(X, names, df.index, dropped)


# ---------------------------------------------------------------------------
# Least squares with robust / clustered inference
# ---------------------------------------------------------------------------


@dataclass
class OLSFit:
    params: np.ndarray
    names: list[str]
    se: np.ndarray
    vcov: np.ndarray
    resid: np.ndarray
    fitted: np.ndarray
    n: int
    k: int
    df_resid: float
    vcov_type: str
    n_clusters: int | None = None
    weights: np.ndarray | None = None
    rank: int | None = None
    rank_deficient: bool = False
    condition: float = 1.0
    cluster_warning: str | None = None

    def index(self, name: str) -> int:
        return self.names.index(name)

    def coef(self, name: str) -> float:
        return float(self.params[self.index(name)])

    def stderr(self, name: str) -> float:
        return float(self.se[self.index(name)])

    def tstat(self, name: str) -> float:
        se = self.stderr(name)
        return float(self.coef(name) / se) if se > 0 else float("nan")

    def pvalue(self, name: str) -> float:
        return t_sf2(self.tstat(name), self.df_resid)

    def conf_int(self, name: str, level: float = 0.95) -> tuple[float, float]:
        crit = t_ppf(0.5 + level / 2.0, self.df_resid)
        est, se = self.coef(name), self.stderr(name)
        return (est - crit * se, est + crit * se)

    def summary_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "term": nm,
                "estimate": float(self.params[j]),
                "se": float(self.se[j]),
                "statistic": float(self.params[j] / self.se[j]) if self.se[j] > 0 else None,
                "p_value": t_sf2(self.params[j] / self.se[j], self.df_resid) if self.se[j] > 0 else None,
            }
            for j, nm in enumerate(self.names)
        ]

    def classic_text(self, title: str = "Linear model") -> str:
        head = f"{title}\n  n = {self.n}   vcov = {self.vcov_type}"
        if self.n_clusters:
            head += f"   clusters = {self.n_clusters}"
        lines = [head, "", f"{'term':<28}{'estimate':>14}{'se':>14}{'t':>10}{'p':>10}"]
        for row in self.summary_rows():
            stat = row["statistic"]
            pval = row["p_value"]
            lines.append(
                f"{row['term'][:28]:<28}{row['estimate']:>14.6g}{row['se']:>14.6g}"
                f"{(stat if stat is not None else float('nan')):>10.3f}"
                f"{(pval if pval is not None else float('nan')):>10.4f}"
            )
        return "\n".join(lines)


def _conditioning(Xw: np.ndarray) -> tuple[int, bool, float]:
    """Rank, a rank-deficiency flag and the condition number of a design matrix.

    The rank is taken on the columns scaled to unit length rather than on the
    raw cross-product, because the raw version answers a question nobody asked:
    with columns whose sizes differ by ten orders of magnitude the small ones
    look like rounding error next to the big ones and get counted out. Scaling
    first means the singular values are compared like with like, so what comes
    back is the number of genuinely distinct directions the data contain, and
    the condition number says how nearly two of them coincide.
    """
    n, k = Xw.shape
    if k == 0 or n == 0:
        return 0, False, 1.0
    scale = np.linalg.norm(Xw, axis=0)
    live = scale > 0.0
    if not np.any(live):
        return 0, True, float("inf")
    Z = Xw[:, live] / scale[live]
    # A thin QR first: R has the same singular values as Z but is only k by k,
    # which keeps this affordable on the wide dummy designs the panel methods
    # build. Going through the cross-product instead would square the condition
    # number and start calling merely awkward designs singular.
    if n >= Z.shape[1]:
        Z = np.linalg.qr(Z, mode="r")
    sv = np.linalg.svd(Z, compute_uv=False)
    if sv.size == 0 or not np.isfinite(sv[0]) or sv[0] <= 0:
        return 0, True, float("inf")
    tol = float(max(n, int(live.sum()))) * float(np.finfo(float).eps) * float(sv[0])
    rank = int(np.sum(sv > tol))
    smallest = float(sv[-1])
    cond = float(sv[0] / smallest) if smallest > 0 else float("inf")
    return rank, bool(rank < k), cond


def fit_failure_reason(fit: OLSFit, *, outcome_scale: float | None = None) -> str | None:
    """Say why this fit is arithmetic noise rather than a result, or ``None`` if it is sound.

    Every adapter should ask this before it prints a number. A singular design
    does not raise: ``numpy.linalg.pinv`` resolves it and returns coefficients
    of the order of 1e14 with standard errors of the order of 1e29, which then
    sail through every sanity check written in terms of "is it finite" and come
    out the other end as a confident-looking estimate with p = 1.
    """
    if not np.all(np.isfinite(fit.params)):
        return ("Some of the coefficients came out as infinity or as 'not a number', which means "
                "the fit did not resolve rather than that the effect is very large.")
    if not np.all(np.isfinite(fit.se)):
        return ("Some of the standard errors came out as infinity or as 'not a number', so no "
                "confidence interval from this fit would mean anything.")
    if fit.rank_deficient:
        return ("Two or more of the things being estimated move together exactly in these data, "
                "so no amount of arithmetic can tell their effects apart.")
    if not np.isfinite(fit.condition) or fit.condition > SINGULAR_CONDITION:
        return ("Two or more of the things being estimated move together so closely in these data "
                "that separating them is past what the arithmetic can do: what came back is mostly "
                "rounding error.")
    if outcome_scale is not None and outcome_scale > 0:
        biggest_se = float(np.max(fit.se)) if fit.se.size else 0.0
        if biggest_se > ABSURD_SE_RATIO * float(outcome_scale):
            return (f"The standard error came out as {biggest_se:.3g} against an outcome that "
                    f"varies by about {float(outcome_scale):.3g}. An uncertainty that much larger "
                    "than the outcome itself is a failed fit, not a wide interval.")
    return None


def outcome_scale(y: np.ndarray | pd.Series) -> float:
    """A typical size for an outcome, for judging whether a standard error is absurd."""
    v = np.asarray(y, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    return max(float(np.std(v)), abs(float(np.mean(v))), 1e-12)


def ols(
    y: np.ndarray | pd.Series,
    X: np.ndarray,
    names: Sequence[str],
    *,
    weights: np.ndarray | None = None,
    cluster: np.ndarray | pd.Series | None = None,
    vcov: str = "HC1",
    absorb_df: int = 0,
) -> OLSFit:
    """Weighted least squares with HC0/HC1/HC2/HC3, cluster, or classical vcov."""
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    if n != y.shape[0]:
        raise ValueError(f"y has {y.shape[0]} rows but X has {n}.")
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()
    # Weights reach here from two places -- a survey weight the user picked, and
    # a weight some method worked out for itself -- and a bad one used to come
    # out of numpy as a raw error message. Say what is wrong in words instead.
    if np.any(~np.isfinite(w)) or np.any(w < 0):
        raise DataError(
            "Some of the weights are blank, negative, or not a number, so the rows cannot be "
            "added up.",
            detail="A weight says how much each row counts towards the answer, so every weight "
                   "has to be zero or more. Check the column in the weight box for blanks, minus "
                   "signs and text, or leave the box empty to count every row once.",
        )
    if n > 0 and float(np.sum(w)) <= 0.0:
        raise DataError(
            "Every weight is zero, so no row counts towards the answer.",
            detail="A weight says how much each row counts towards the answer. A column of zeros "
                   "-- or of blanks read as zeros -- gives every row no say at all, and there is "
                   "nothing left to work the answer out from. Check the column in the weight box, "
                   "or leave the box empty to count every row once.",
        )
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw

    XtX = Xw.T @ Xw
    xtx_inv = np.linalg.pinv(XtX)
    beta = xtx_inv @ (Xw.T @ yw)
    fitted = X @ beta
    resid = y - fitted
    rank, rank_deficient, condition = _conditioning(Xw)

    names = list(names)
    vcov_type = vcov
    cluster_warning: str | None = None
    if cluster is not None:
        g = pd.Series(np.asarray(cluster).ravel()).astype(str).to_numpy()
        codes, uniq = pd.factorize(g)
        n_g = len(uniq)
        # With one cluster the within-cluster scores sum to zero by the normal
        # equations, so the sandwich variance is exactly zero and the p-value
        # comes out at zero as well. That is not a very precise answer, it is no
        # answer at all, so refuse rather than print it.
        if n_g < 2:
            raise DataError(
                "Every row falls in the same cluster, so a clustered standard error cannot be "
                "worked out here.",
                detail="Clustering measures uncertainty by comparing one cluster with another, and "
                       "with only one cluster there is nothing to compare it with -- the interval "
                       "would come out as a single point. Choose the column that says which group "
                       "each row was assigned as a block, or leave the cluster box empty to treat "
                       "the rows as independent.",
            )
        if n_g < FEW_CLUSTERS:
            cluster_warning = (
                f"There are only {n_g} clusters. Clustered standard errors are worked out from the "
                "spread between clusters, and with this few there is not enough spread to measure: "
                "the standard error comes out too small, so the confidence interval below is "
                f"narrower than it should be and the p-value smaller. Roughly {FEW_CLUSTERS} "
                "clusters are needed before this settles down. Treat a borderline result as not "
                "settled, and prefer the wild cluster bootstrap p-value -- a re-sampling check "
                "that holds up with few clusters -- where one is offered."
            )
        meat = np.zeros((k, k))
        u = (resid * w)[:, None] * X
        for gi in range(n_g):
            sel = codes == gi
            s = u[sel].sum(axis=0)
            meat += np.outer(s, s)
        dof_c = (n_g / max(n_g - 1, 1)) * ((n - 1) / max(n - rank - absorb_df, 1))
        V = xtx_inv @ meat @ xtx_inv * dof_c
        df_resid = float(max(n_g - 1, 1))
        vcov_type = f"cluster({n_g})"
        n_clusters: int | None = n_g
    else:
        n_clusters = None
        df_resid = float(max(n - rank - absorb_df, 1))
        if vcov in ("classical", "const", "iid"):
            sigma2 = float((w * resid**2).sum() / df_resid)
            V = xtx_inv * sigma2
            vcov_type = "classical"
        else:
            h = np.einsum("ij,jk,ik->i", Xw, xtx_inv, Xw)
            h = np.clip(h, 0.0, 1 - 1e-10)
            u2 = (resid * w) ** 2
            if vcov == "HC0":
                omega = u2
            elif vcov == "HC2":
                omega = u2 / (1 - h)
            elif vcov == "HC3":
                omega = u2 / (1 - h) ** 2
            else:  # HC1 default
                omega = u2 * (n / df_resid)
                vcov_type = "HC1"
            meat = (X * omega[:, None]).T @ X
            V = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    return OLSFit(
        params=beta,
        names=names,
        se=se,
        vcov=V,
        resid=resid,
        fitted=fitted,
        n=n,
        k=k,
        df_resid=df_resid,
        vcov_type=vcov_type,
        n_clusters=n_clusters,
        weights=w,
        rank=rank,
        rank_deficient=rank_deficient,
        condition=condition,
        cluster_warning=cluster_warning,
    )


def ols_df(
    df: pd.DataFrame,
    outcome: str,
    regressors: Sequence[str],
    *,
    weights: np.ndarray | None = None,
    cluster: str | np.ndarray | None = None,
    vcov: str = "HC1",
    intercept: bool = True,
) -> OLSFit:
    dm = design_matrix(df, regressors, intercept=intercept)
    cl = df[cluster].to_numpy() if isinstance(cluster, str) else cluster
    return ols(df[outcome].to_numpy(dtype=float), dm.X, dm.names, weights=weights, cluster=cl, vcov=vcov)


def hac_se(fit: OLSFit, X: np.ndarray, lags: int | None = None) -> np.ndarray:
    """Newey-West standard errors for a time-ordered fit (ITS, single series)."""
    n, k = X.shape
    u = fit.resid
    if lags is None:
        lags = int(math.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lags = max(int(lags), 0)
    xtx_inv = np.linalg.pinv(X.T @ X)
    S = (X * (u**2)[:, None]).T @ X
    for l in range(1, lags + 1):
        w_l = 1.0 - l / (lags + 1.0)
        Xl, Xr = X[l:], X[:-l]
        ul, ur = u[l:], u[:-l]
        G = (Xl * (ul * ur)[:, None]).T @ Xr
        S += w_l * (G + G.T)
    V = xtx_inv @ S @ xtx_inv * (n / max(n - k, 1))
    return np.sqrt(np.clip(np.diag(V), 0.0, None))


# ---------------------------------------------------------------------------
# Logistic regression (IRLS, with ridge fallback for separation)
# ---------------------------------------------------------------------------


@dataclass
class LogitFit:
    params: np.ndarray
    names: list[str]
    fitted: np.ndarray
    converged: bool
    iterations: int
    separation: bool = False
    ridge: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _expit(np.asarray(X, dtype=float) @ self.params)


def _expit(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def logit(
    y: np.ndarray,
    X: np.ndarray,
    names: Sequence[str] | None = None,
    *,
    weights: np.ndarray | None = None,
    max_iter: int = 60,
    tol: float = 1e-8,
    ridge: float = 1e-6,
) -> LogitFit:
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()
    beta = np.zeros(k)
    converged = False
    it = 0
    lam = float(ridge)
    for attempt in range(4):
        beta = np.zeros(k)
        converged = False
        for it in range(1, max_iter + 1):
            eta = X @ beta
            mu = _expit(eta)
            W = np.clip(mu * (1 - mu), 1e-10, None) * w
            z = eta + (y - mu) / np.clip(mu * (1 - mu), 1e-10, None)
            XtWX = (X * W[:, None]).T @ X + lam * np.eye(k)
            XtWz = (X * W[:, None]).T @ z
            try:
                new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                new = np.linalg.pinv(XtWX) @ XtWz
            if not np.all(np.isfinite(new)):
                break
            step = np.max(np.abs(new - beta))
            beta = new
            if step < tol:
                converged = True
                break
        if converged and np.all(np.isfinite(beta)) and np.max(np.abs(beta)) < 50:
            break
        lam = max(lam * 100, 1e-3)  # separation: shrink rather than diverge
    fitted = _expit(X @ beta)
    separation = bool(np.max(np.abs(beta)) >= 25 or lam > 1e-5)
    return LogitFit(beta, list(names or [f"x{j}" for j in range(k)]), fitted, converged, it, separation, lam)


# ---------------------------------------------------------------------------
# Balance, weights, effective sample size
# ---------------------------------------------------------------------------


def weighted_mean(x: np.ndarray, w: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=float)
    if w is None:
        return float(np.nanmean(x)) if x.size else float("nan")
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w)
    tot = w[m].sum()
    return float((x[m] * w[m]).sum() / tot) if tot > 0 else float("nan")


def weighted_var(x: np.ndarray, w: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=float)
    if w is None:
        return float(np.nanvar(x, ddof=1)) if x.size > 1 else float("nan")
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w)
    x, w = x[m], w[m]
    sw = w.sum()
    if sw <= 0:
        return float("nan")
    mu = (x * w).sum() / sw
    denom = sw - (w**2).sum() / sw
    if denom <= 0:
        return float("nan")
    return float((w * (x - mu) ** 2).sum() / denom)


def smd(x: np.ndarray, treat: np.ndarray, w: np.ndarray | None = None, *,
        pooled_from: np.ndarray | None = None, pooled_treat: np.ndarray | None = None) -> float:
    """Standardised mean difference.

    The denominator uses the *unadjusted* pooled SD when ``pooled_from`` is given,
    which is the cobalt convention for a Love plot: it keeps the before and after
    points on one scale instead of letting the adjustment move the yardstick.
    ``pooled_treat`` is the treatment vector that goes with ``pooled_from``, which
    is not the same vector as ``treat`` once matching has subset the sample.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(treat, dtype=float) > 0.5
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)
    m1 = weighted_mean(x[t], w[t])
    m0 = weighted_mean(x[~t], w[~t])
    if pooled_from is not None:
        base = np.asarray(pooled_from, dtype=float)
        bt = np.asarray(pooled_treat if pooled_treat is not None else treat, dtype=float) > 0.5
        if bt.shape[0] != base.shape[0]:
            raise ValueError("pooled_treat must be the treatment vector for pooled_from")
        v1, v0 = np.nanvar(base[bt], ddof=1), np.nanvar(base[~bt], ddof=1)
    else:
        v1, v0 = weighted_var(x[t], w[t]), weighted_var(x[~t], w[~t])
    denom = math.sqrt(max((v1 + v0) / 2.0, 0.0))
    if not np.isfinite(denom) or denom <= 1e-12:
        return 0.0 if abs(m1 - m0) < 1e-12 else float("nan")
    return float((m1 - m0) / denom)


def variance_ratio(x: np.ndarray, treat: np.ndarray, w: np.ndarray | None = None) -> float:
    x = np.asarray(x, dtype=float)
    t = np.asarray(treat, dtype=float) > 0.5
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)
    v1, v0 = weighted_var(x[t], w[t]), weighted_var(x[~t], w[~t])
    if not np.isfinite(v0) or v0 <= 1e-12:
        return float("nan")
    return float(v1 / v0)


def effective_sample_size(w: np.ndarray) -> float:
    """Kish ESS. The number the weight histogram is really about."""
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if w.size == 0:
        return 0.0
    return float(w.sum() ** 2 / (w**2).sum())


def balance_table(
    df: pd.DataFrame,
    covariates: Sequence[str],
    treat: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    matched_index: pd.Index | None = None,
) -> list[dict[str, Any]]:
    """Rows for a Love plot: SMD before and after adjustment, plus variance ratios."""
    rows: list[dict[str, Any]] = []
    t = np.asarray(treat, dtype=float)
    for col in covariates:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s) and s.dtype != bool:
            levels = sorted(pd.unique(s.dropna().astype(str)))[:20]
            for lev in levels:
                x = (s.astype(str) == lev).to_numpy(dtype=float)
                rows.append(_balance_row(f"{col}[{lev}]", x, t, weights, matched_index, df.index))
            continue
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        rows.append(_balance_row(col, x, t, weights, matched_index, df.index))
    return rows


def _balance_row(
    name: str,
    x: np.ndarray,
    t: np.ndarray,
    weights: np.ndarray | None,
    matched_index: pd.Index | None,
    index: pd.Index,
) -> dict[str, Any]:
    before = smd(x, t)
    if matched_index is not None:
        sel = np.asarray(index.isin(matched_index))
        wsel = None if weights is None else np.asarray(weights, dtype=float)[sel]
        after = smd(x[sel], t[sel], wsel, pooled_from=x, pooled_treat=t)
        vr = variance_ratio(x[sel], t[sel], wsel)
    elif weights is not None:
        after = smd(x, t, weights, pooled_from=x, pooled_treat=t)
        vr = variance_ratio(x, t, weights)
    else:
        after = before
        vr = variance_ratio(x, t)
    return {
        "variable": name,
        "smd_before": _f(before),
        "smd_after": _f(after),
        "abs_smd_before": _f(abs(before)) if np.isfinite(before) else None,
        "abs_smd_after": _f(abs(after)) if np.isfinite(after) else None,
        "variance_ratio": _f(vr),
        "mean_treated": _f(weighted_mean(x[t > 0.5], None if weights is None else weights[t > 0.5])),
        "mean_control": _f(weighted_mean(x[t <= 0.5], None if weights is None else weights[t <= 0.5])),
    }


def _f(v: Any) -> float | None:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def bootstrap_se(
    fn,
    n: int,
    *,
    reps: int = 200,
    seed: int = 0,
    cluster_codes: np.ndarray | None = None,
    on_error: str = "skip",
) -> tuple[float | None, list[float]]:
    """Nonparametric (optionally block/cluster) bootstrap of a scalar statistic."""
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    if cluster_codes is not None:
        codes = np.asarray(cluster_codes)
        uniq = np.unique(codes)
        groups = {g: np.flatnonzero(codes == g) for g in uniq}
    for _ in range(int(reps)):
        if cluster_codes is not None:
            picked = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([groups[g] for g in picked]) if len(picked) else np.array([], dtype=int)
        else:
            idx = rng.integers(0, n, size=n)
        try:
            val = fn(idx)
        except Exception:
            if on_error == "raise":
                raise
            continue
        if val is not None and np.isfinite(val):
            draws.append(float(val))
    if len(draws) < 5:
        return None, draws
    return float(np.std(draws, ddof=1)), draws


def percentile_ci(draws: Sequence[float], level: float = 0.95) -> tuple[float | None, float | None]:
    if len(draws) < 5:
        return (None, None)
    a = (1 - level) / 2
    lo, hi = np.quantile(np.asarray(draws, dtype=float), [a, 1 - a])
    return (float(lo), float(hi))


# ---------------------------------------------------------------------------
# Binning / smoothing helpers used by the boards and the diagnostics
# ---------------------------------------------------------------------------


def binned_means(
    x: np.ndarray,
    y: np.ndarray,
    *,
    bins: int = 20,
    lo: float | None = None,
    hi: float | None = None,
) -> list[dict[str, Any]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        return []
    lo = float(np.min(x)) if lo is None else lo
    hi = float(np.max(x)) if hi is None else hi
    if hi <= lo:
        return []
    edges = np.linspace(lo, hi, int(bins) + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, int(bins) - 1)
    rows = []
    for b in range(int(bins)):
        sel = idx == b
        if not sel.any():
            continue
        yy = y[sel]
        rows.append(
            {
                "bin": b,
                "x": float((edges[b] + edges[b + 1]) / 2),
                "x_lo": float(edges[b]),
                "x_hi": float(edges[b + 1]),
                "y": float(np.mean(yy)),
                "se": float(np.std(yy, ddof=1) / math.sqrt(yy.size)) if yy.size > 1 else None,
                "n": int(yy.size),
            }
        )
    return rows


def quantile_bins(x: np.ndarray, bins: int = 20) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    qs = np.linspace(0, 1, int(bins) + 1)
    edges = np.unique(np.quantile(x[np.isfinite(x)], qs))
    return edges


def histogram_rows(x: np.ndarray, *, bins: int = 30, lo: float | None = None, hi: float | None = None,
                   density: bool = False) -> list[dict[str, Any]]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return []
    rng = (lo if lo is not None else float(np.min(x)), hi if hi is not None else float(np.max(x)))
    if rng[1] <= rng[0]:
        rng = (rng[0] - 0.5, rng[0] + 0.5)
    counts, edges = np.histogram(x, bins=int(bins), range=rng, density=density)
    return [
        {"x_lo": float(edges[i]), "x_hi": float(edges[i + 1]),
         "x": float((edges[i] + edges[i + 1]) / 2), "count": float(counts[i])}
        for i in range(len(counts))
    ]


def triangular_kernel(u: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.abs(u), 0.0, None)


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    return np.clip(0.75 * (1.0 - u**2), 0.0, None)


def uniform_kernel(u: np.ndarray) -> np.ndarray:
    return (np.abs(u) <= 1).astype(float)


KERNELS = {
    "triangular": triangular_kernel,
    "epanechnikov": epanechnikov_kernel,
    "uniform": uniform_kernel,
}


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) and np.isfinite(b) else float("nan")


def clip_propensity(ps: np.ndarray, lo: float = 0.01, hi: float = 0.99) -> tuple[np.ndarray, int]:
    ps = np.asarray(ps, dtype=float)
    n_clipped = int(np.sum((ps < lo) | (ps > hi)))
    return np.clip(ps, lo, hi), n_clipped


def factorize(values: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    codes, uniq = pd.factorize(pd.Series(list(values)))
    return np.asarray(codes), np.asarray(uniq)
