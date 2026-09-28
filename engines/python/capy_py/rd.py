"""Regression discontinuity for capy.py (plan 7.5).

What this module is, and what it is not
---------------------------------------
``rdrobust`` (R; Calonico, Cattaneo, Farrell, Titiunik) is the authoritative
implementation of modern RD inference. There is no R on this machine, so this is
an independent Python implementation of the same *ideas*, written to be read and
audited rather than to impersonate rdrobust:

* local polynomial of order ``p`` either side of the cutoff, kernel weighted,
  fitted by weighted least squares (:func:`capy_py.stats.ols`);
* an MSE-optimal bandwidth from a **finite-sample plug-in rule implemented here**
  (bias and variance of the actual estimator on this sample, with preliminary
  bias constants from a global polynomial), or the Imbens-Kalyanaraman (2012)
  formula, or the user's own number -- *not* rdrobust's CCT selector;
* CCT-style bias correction: the leading smoothing bias is estimated with an
  order ``p+1`` polynomial at a pilot bandwidth ``b`` and subtracted, and the
  robust variance accounts for the fact that the correction is estimated on the
  same data (both estimators are written as linear maps of the outcome, so the
  covariance is exact);
* heteroskedasticity-robust (HC1-type) or cluster-robust variance, rather than
  rdrobust's default nearest-neighbour variance estimator.

Every simplification is named in the classic printout and on the method card.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from . import roles, stats, vega
from .contracts import DataError, ResultBuilder, RunContext, SpecError, adapter

PACKAGE = "capy.py"
PACKAGE_VERSION = "0.1.0"
DEFAULT_KERNEL = "triangular"

SELECTOR_LABELS = {
    "mse": "MSE-optimal, capy finite-sample plug-in",
    "ik": "MSE-optimal, Imbens-Kalyanaraman (2012)",
    "manual": "user-supplied",
}

NOT_RDROBUST = (
    "Estimated by capy.py's own local-polynomial code, not by rdrobust. "
    "The bandwidth selector and the variance estimator are named in the Classic tab; "
    "numbers will be close to rdrobust but need not match to the last digit."
)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def _opt_float(ctx: RunContext, name: str, default: float | None = None, *,
               minimum: float | None = None, maximum: float | None = None) -> float | None:
    raw = ctx.opt(name, default)
    if raw is None or raw == "":
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        raise SpecError(f"Option '{name}' must be a number (got {raw!r}).") from None
    if not np.isfinite(val):
        raise SpecError(f"Option '{name}' must be a finite number.")
    if minimum is not None and val < minimum:
        raise SpecError(f"Option '{name}' must be at least {minimum:g} (got {val:g}).")
    if maximum is not None and val > maximum:
        raise SpecError(f"Option '{name}' must be at most {maximum:g} (got {val:g}).")
    return val


def _opt_int(ctx: RunContext, name: str, default: int, *, minimum: int | None = None,
             maximum: int | None = None) -> int:
    raw = ctx.opt(name, default)
    if raw is None or raw == "":
        return default
    try:
        val = int(round(float(raw)))
    except (TypeError, ValueError):
        raise SpecError(f"Option '{name}' must be a whole number (got {raw!r}).") from None
    if minimum is not None and val < minimum:
        raise SpecError(f"Option '{name}' must be at least {minimum} (got {val}).")
    if maximum is not None and val > maximum:
        raise SpecError(f"Option '{name}' must be at most {maximum} (got {val}).")
    return val


def _opt_bool(ctx: RunContext, name: str, default: bool) -> bool:
    raw = ctx.opt(name, default)
    if isinstance(raw, bool):
        return raw
    if raw is None or raw == "":
        return default
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(raw)


def _opt_choice(ctx: RunContext, name: str, default: str, choices: Sequence[str]) -> str:
    raw = ctx.opt(name, default)
    if raw is None or raw == "":
        return default
    val = str(raw).strip().lower()
    if val not in choices:
        raise SpecError(
            f"Option '{name}' must be one of: {', '.join(choices)} (got {raw!r}).")
    return val


def _kernel(name: Any) -> tuple[str, Callable[[np.ndarray], np.ndarray]]:
    key = str(name or DEFAULT_KERNEL).strip().lower()
    if key not in stats.KERNELS:
        raise SpecError(
            f"Unknown kernel '{name}'.",
            detail=f"Choose one of: {', '.join(sorted(stats.KERNELS))}.",
        )
    return key, stats.KERNELS[key]


# ---------------------------------------------------------------------------
# Local polynomial window: design, linear map, and the pieces bias correction
# needs. Everything inside a window is fitted in u = (x - c) / h so the design
# is well conditioned; coefficients are converted back to x units on the way out.
# ---------------------------------------------------------------------------


@dataclass
class _Window:
    h: float
    order: int
    idx: np.ndarray          # positions into the full-sample arrays
    u: np.ndarray            # scaled running variable inside the window
    x: np.ndarray            # raw centred running variable inside the window
    w: np.ndarray            # kernel weights
    B: np.ndarray            # design matrix (m x k)
    A: np.ndarray            # (B'WB)^-1 B'W  -- the linear map, k x m
    names: list[str]
    n_left: int
    n_right: int
    ok: bool

    def row(self, deriv: int, side: str) -> np.ndarray:
        """Linear weights (in x units) for mu^(deriv)/deriv! on one side."""
        base = 0 if side == "left" else self.order + 1
        return self.A[base + deriv] / (self.h ** deriv)

    def contrast(self, deriv: int) -> np.ndarray:
        return self.row(deriv, "right") - self.row(deriv, "left")

    def contrast_vector(self, deriv: int) -> np.ndarray:
        """Coefficient contrast in *fitted* (u) units, for use with an OLSFit vcov."""
        c = np.zeros(self.B.shape[1])
        c[self.order + 1 + deriv] = 1.0
        c[deriv] = -1.0
        return c


def _window(
    xc: np.ndarray,
    h: float,
    order: int,
    kernel_fn: Callable[[np.ndarray], np.ndarray],
    covars: np.ndarray | None = None,
    covar_names: Sequence[str] = (),
) -> _Window:
    h = float(h)
    if not np.isfinite(h) or h <= 0:
        raise SpecError("The bandwidth must be a positive number.")
    w_all = np.asarray(kernel_fn(xc / h), dtype=float)
    idx = np.flatnonzero(np.isfinite(w_all) & (w_all > 0))
    x = xc[idx]
    u = x / h
    w = w_all[idx]
    d = (x >= 0).astype(float)
    cols: list[np.ndarray] = []
    names: list[str] = []
    for j in range(order + 1):
        cols.append((1.0 - d) * u**j)
        names.append(f"left:u^{j}")
    for j in range(order + 1):
        cols.append(d * u**j)
        names.append(f"right:u^{j}")
    if covars is not None and covars.size and covars.shape[1]:
        sub = covars[idx]
        for j in range(sub.shape[1]):
            cols.append(sub[:, j])
            names.append(str(covar_names[j]) if j < len(covar_names) else f"cov{j}")
    B = np.column_stack(cols) if cols else np.zeros((len(idx), 0))
    Bw = B * w[:, None]
    XtWX = Bw.T @ B
    k = B.shape[1]
    ok = bool(len(idx) > k and k > 0 and np.linalg.matrix_rank(XtWX) == k)
    A = np.linalg.pinv(XtWX) @ Bw.T
    return _Window(
        h=h, order=order, idx=idx, u=u, x=x, w=w, B=B, A=A, names=names,
        n_left=int((x < 0).sum()), n_right=int((x >= 0).sum()), ok=ok,
    )


def _embed(n: int, idx: np.ndarray, vals: np.ndarray) -> np.ndarray:
    out = np.zeros(n, dtype=float)
    out[idx] = vals
    return out


def _cov_of(ell: np.ndarray, u1: np.ndarray, u2: np.ndarray,
            cluster_codes: np.ndarray | None = None) -> float:
    """Sandwich (co)variance of two linear-in-y statistics sharing weights ``ell``."""
    if cluster_codes is None:
        return float(np.sum(ell * ell * u1 * u2))
    codes = np.asarray(cluster_codes, dtype=int)
    g1 = np.bincount(codes, weights=ell * u1)
    g2 = np.bincount(codes, weights=ell * u2)
    used = np.bincount(codes, weights=(ell != 0).astype(float))
    n_g = int(np.sum(used > 0))
    fac = n_g / max(n_g - 1, 1)
    return float(np.sum(g1 * g2) * fac)


def _hc_resid(y: np.ndarray, win: _Window, cluster_codes: np.ndarray | None,
              vce: str = "HC1") -> np.ndarray:
    """Residuals inside a window, HC1-scaled, embedded at full length."""
    cl = None if cluster_codes is None else cluster_codes[win.idx]
    fit = stats.ols(y[win.idx], win.B, win.names, weights=win.w, cluster=cl, vcov=vce)
    m, k = win.B.shape
    scale = math.sqrt(m / max(m - k, 1))
    return _embed(len(y), win.idx, fit.resid * scale)


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------


@dataclass
class _RDFit:
    tau: float
    se: float
    z: float
    p_value: float
    ci: tuple[float, float]
    tau_bc: float | None
    se_rb: float | None
    ci_rb: tuple[float, float] | None
    p_rb: float | None
    z_rb: float | None
    ell: np.ndarray
    ell_rb: np.ndarray | None
    resid: np.ndarray
    resid_rb: np.ndarray
    beta: np.ndarray
    win: _Window
    h: float
    b: float | None
    n_left: int
    n_right: int
    order: int
    deriv: int
    vcov_type: str
    covar_mean: np.ndarray | None = None
    bias_note: str | None = None

    @property
    def n_effective(self) -> int:
        return int(self.n_left + self.n_right)


def _rd_fit(
    xc: np.ndarray,
    y: np.ndarray,
    *,
    h: float,
    b: float | None,
    order: int,
    deriv: int,
    kernel_fn: Callable[[np.ndarray], np.ndarray],
    covars: np.ndarray | None = None,
    covar_names: Sequence[str] = (),
    cluster_codes: np.ndarray | None = None,
    level: float = 0.95,
    vce: str = "HC1",
) -> _RDFit:
    n = len(xc)
    win = _window(xc, h, order, kernel_fn, covars, covar_names)
    need = order + 2
    if win.n_left < need or win.n_right < need:
        raise DataError(
            f"Only {win.n_left} observation(s) below and {win.n_right} above the cutoff fall inside "
            f"the bandwidth h = {h:.6g}; an order-{order} polynomial needs at least {need} a side.",
            detail="Widen the bandwidth, lower the polynomial order, or accept that this cutoff "
                   "does not have enough data around it.",
        )
    cl = None if cluster_codes is None else cluster_codes[win.idx]
    fit = stats.ols(y[win.idx], win.B, win.names, weights=win.w, cluster=cl, vcov=vce)
    cvec = win.contrast_vector(deriv)
    scale = h ** deriv
    tau = float(cvec @ fit.params) / scale
    var = float(cvec @ fit.vcov @ cvec) / (scale**2)
    se = math.sqrt(max(var, 0.0))
    z = tau / se if se > 0 else float("nan")
    p_value = stats.norm_sf2(z) if se > 0 else float("nan")
    ci = stats.wald_ci(tau, se, level)

    ell = _embed(n, win.idx, win.contrast(deriv))
    resid = _hc_resid(y, win, cluster_codes, vce="HC1")

    tau_bc = se_rb = ci_rb = p_rb = z_rb = None
    ell_rb = None
    resid_rb = resid
    bias_note = None
    if b:
        q = order + 1
        winb = _window(xc, float(b), q, kernel_fn, covars, covar_names)
        if winb.ok and winb.n_left >= q + 2 and winb.n_right >= q + 2:
            a = win.contrast(deriv)
            v = win.x ** (order + 1)
            right = win.x >= 0
            L_left = float(a @ np.where(right, 0.0, v))
            L_right = float(a @ np.where(right, v, 0.0))
            c_left = winb.A[q] / (float(b) ** q)
            c_right = winb.A[2 * q + 1] / (float(b) ** q)
            ell_rb = (
                ell
                - L_left * _embed(n, winb.idx, c_left)
                - L_right * _embed(n, winb.idx, c_right)
            )
            tau_bc = float(ell_rb @ y)
            win_r = _window(xc, max(float(h), float(b)), q, kernel_fn, covars, covar_names)
            resid_rb = _hc_resid(y, win_r, cluster_codes, vce="HC1")
            var_rb = _cov_of(ell_rb, resid_rb, resid_rb, cluster_codes)
            se_rb = math.sqrt(max(var_rb, 0.0))
            z_rb = tau_bc / se_rb if se_rb > 0 else float("nan")
            p_rb = stats.norm_sf2(z_rb) if se_rb > 0 else float("nan")
            ci_rb = stats.wald_ci(tau_bc, se_rb, level)
        else:
            bias_note = (
                f"The pilot fit for bias correction (order {q} at b = {float(b):.6g}) did not have "
                "enough observations either side, so only the conventional estimator was computed."
            )

    covar_mean = None
    if covars is not None and covars.size and covars.shape[1]:
        covar_mean = covars[win.idx].mean(axis=0)

    return _RDFit(
        tau=tau, se=se, z=z, p_value=p_value, ci=(ci[0], ci[1]),
        tau_bc=tau_bc, se_rb=se_rb, ci_rb=ci_rb, p_rb=p_rb, z_rb=z_rb,
        ell=ell, ell_rb=ell_rb, resid=resid, resid_rb=resid_rb,
        beta=fit.params, win=win, h=float(h), b=(float(b) if b else None),
        n_left=win.n_left, n_right=win.n_right, order=order, deriv=deriv,
        vcov_type=fit.vcov_type, covar_mean=covar_mean, bias_note=bias_note,
    )


# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------


@dataclass
class _Prelim:
    order: int
    scale: float
    params: np.ndarray
    vcov: np.ndarray
    sigma2: np.ndarray

    def theta(self, j: int) -> tuple[float, float, float, float, float]:
        """(theta_left, theta_right, var_l, var_r, cov) for the x^j coefficient."""
        if j > self.order:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        li, ri = j, self.order + 1 + j
        s = self.scale ** j
        return (
            float(self.params[li]) / s,
            float(self.params[ri]) / s,
            float(self.vcov[li, li]) / (s * s),
            float(self.vcov[ri, ri]) / (s * s),
            float(self.vcov[li, ri]) / (s * s),
        )


def _prelim_fit(xc: np.ndarray, y: np.ndarray, order: int,
                cluster_codes: np.ndarray | None = None) -> _Prelim:
    """Global side-separated polynomial: preliminary bias constants and sigma^2."""
    n = len(xc)
    scale = float(np.max(np.abs(xc)))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    z = xc / scale
    d = (xc >= 0).astype(float)
    cols, names = [], []
    for j in range(order + 1):
        cols.append((1.0 - d) * z**j)
        names.append(f"L{j}")
    for j in range(order + 1):
        cols.append(d * z**j)
        names.append(f"R{j}")
    B = np.column_stack(cols)
    fit = stats.ols(y, B, names, cluster=cluster_codes, vcov="HC1")
    u = fit.resid
    sigma2 = np.full(n, float(np.mean(u**2)))
    for mask in ((xc < 0), (xc >= 0)):
        if mask.sum() < 10:
            continue
        ax = np.abs(xc[mask])
        thr = float(np.quantile(ax, 0.5))
        near = mask & (np.abs(xc) <= thr)
        if near.sum() < 10:
            near = mask
        sigma2[mask] = float(np.mean(u[near] ** 2))
    sigma2 = np.clip(sigma2, 1e-300, None)
    return _Prelim(order=order, scale=scale, params=fit.params, vcov=fit.vcov, sigma2=sigma2)


def _prelim_order(xc: np.ndarray, wanted: int) -> int:
    """Back off the global polynomial order when a side is thin."""
    n_left = int((xc < 0).sum())
    n_right = int((xc >= 0).sum())
    u_left = int(np.unique(xc[xc < 0]).size)
    u_right = int(np.unique(xc[xc >= 0]).size)
    for order in range(wanted, 0, -1):
        need = 5 * (order + 1)
        if min(n_left, n_right) >= need and min(u_left, u_right) >= order + 1:
            return order
    return 1


def _bw_grid(xc: np.ndarray, order: int, points: int = 24, min_side: int = 10) -> np.ndarray:
    left = np.sort(-xc[xc < 0])
    right = np.sort(xc[xc >= 0])
    n_min = max(min_side, 3 * (order + 2))
    if len(left) < n_min or len(right) < n_min:
        n_min = max(order + 2, min(len(left), len(right)))
    if len(left) < n_min or len(right) < n_min or n_min < order + 2:
        raise DataError(
            "There are too few observations on one side of the cutoff to choose a bandwidth.",
            detail=f"{len(left)} below and {len(right)} at or above the cutoff; an order-{order} "
                   f"polynomial needs at least {order + 2} a side, and a bandwidth search needs more.",
        )
    h_lo = max(float(left[n_min - 1]), float(right[n_min - 1]))
    h_hi = max(float(left[-1]), float(right[-1]))
    if h_lo <= 0:
        h_lo = h_hi / 100.0 if h_hi > 0 else 1.0
    if h_hi <= h_lo:
        h_hi = h_lo * 1.5
    return np.geomspace(h_lo * 1.0001, h_hi, int(points))


def _bw_select(
    xc: np.ndarray,
    y: np.ndarray,
    *,
    order: int,
    deriv: int,
    mode: str,
    kernel_fn: Callable[[np.ndarray], np.ndarray],
    prelim: _Prelim,
    covars: np.ndarray | None = None,
    covar_names: Sequence[str] = (),
    cluster_codes: np.ndarray | None = None,
    points: int = 24,
    tick: Callable[[float, str], None] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Finite-sample plug-in MSE bandwidth.

    For each candidate ``h`` the estimator is written as a linear map of y, so its
    variance is exact given a preliminary sigma^2, and its leading bias is
    ``L_left * theta_left + L_right * theta_right`` with ``theta`` the order-(p+1)
    coefficients from a global polynomial. The criterion adds the *variance of the
    estimated bias constants*, which is what stops the search running away to the
    whole sample when the true curvature is zero (the role IK's regularisation
    term plays).

    ``mode='contrast'`` targets the jump (or kink); ``mode='pair'`` targets the two
    side-specific coefficients, which is how the pilot bandwidth ``b`` is chosen.
    """
    grid = _bw_grid(xc, order, points=points)
    th_l, th_r, v_l, v_r, c_lr = prelim.theta(order + 1)
    sigma2 = prelim.sigma2
    rows: list[dict[str, Any]] = []
    best_h, best_mse = None, float("inf")
    for i, h in enumerate(grid):
        if tick is not None:
            tick(0.05 + 0.15 * (i / max(len(grid) - 1, 1)), "choosing a bandwidth")
        try:
            win = _window(xc, float(h), order, kernel_fn, covars, covar_names)
        except SpecError:
            continue
        if not win.ok or win.n_left < order + 2 or win.n_right < order + 2:
            continue
        v = win.x ** (order + 1)
        right = win.x >= 0
        vl = np.where(right, 0.0, v)
        vr = np.where(right, v, 0.0)
        targets = ([win.contrast(deriv)] if mode == "contrast"
                   else [win.row(deriv, "left"), win.row(deriv, "right")])
        mse = 0.0
        for a in targets:
            L_l = float(a @ vl)
            L_r = float(a @ vr)
            bias = L_l * th_l + L_r * th_r
            var_bias = L_l * L_l * v_l + L_r * L_r * v_r + 2.0 * L_l * L_r * c_lr
            ell = _embed(len(xc), win.idx, a)
            var = _cov_of(ell, np.sqrt(sigma2), np.sqrt(sigma2), cluster_codes)
            mse += bias * bias + max(var_bias, 0.0) + max(var, 0.0)
        if not np.isfinite(mse):
            continue
        rows.append({"h": float(h), "mse": float(mse),
                     "n_left": win.n_left, "n_right": win.n_right})
        if mse < best_mse:
            best_mse, best_h = mse, float(h)
    if best_h is None:
        raise DataError(
            "No bandwidth around the cutoff could be fitted.",
            detail="Every candidate bandwidth left one side of the cutoff without enough "
                   "observations to fit the local polynomial.",
        )
    return best_h, rows


def _bw_ik(xc: np.ndarray, y: np.ndarray) -> float:
    """Imbens-Kalyanaraman (2012) MSE-optimal bandwidth, edge (triangular) kernel, p = 1."""
    n = len(xc)
    sx = float(np.std(xc, ddof=1))
    if not np.isfinite(sx) or sx <= 0:
        raise DataError("The running variable does not vary; there is no cutoff to look at.")
    h1 = 1.84 * sx * n ** (-0.2)
    near = np.abs(xc) <= h1
    left = near & (xc < 0)
    right = near & (xc >= 0)
    n_l, n_r = int(left.sum()), int(right.sum())
    if n_l < 5 or n_r < 5:
        raise DataError(
            "Too few observations near the cutoff for the Imbens-Kalyanaraman bandwidth.",
            detail=f"{n_l} below and {n_r} above within the pilot window.",
        )
    f_hat = (n_l + n_r) / (2.0 * n * h1)
    resid = np.concatenate([y[left] - y[left].mean(), y[right] - y[right].mean()])
    sigma2 = float(np.mean(resid**2))
    # global cubic for the third derivative
    d = (xc >= 0).astype(float)
    B = np.column_stack([np.ones(n), d, xc, xc**2, xc**3])
    cub = stats.ols(y, B, ["1", "d", "x", "x2", "x3"], vcov="HC1")
    m3 = 6.0 * float(cub.params[4])
    denom = max(m3 * m3, 0.01)
    h2_r = 3.56 * (sigma2 / (f_hat * denom)) ** (1.0 / 7.0) * max(int((xc >= 0).sum()), 1) ** (-1.0 / 7.0)
    h2_l = 3.56 * (sigma2 / (f_hat * denom)) ** (1.0 / 7.0) * max(int((xc < 0).sum()), 1) ** (-1.0 / 7.0)
    m2 = {}
    n2 = {}
    for side, hh, mask in (("right", h2_r, xc >= 0), ("left", h2_l, xc < 0)):
        sel = mask & (np.abs(xc) <= hh)
        n2[side] = int(sel.sum())
        if n2[side] < 4:
            m2[side] = 0.0
            continue
        xs = xc[sel]
        Bq = np.column_stack([np.ones(len(xs)), xs, xs**2])
        q = stats.ols(y[sel], Bq, ["1", "x", "x2"], vcov="HC1")
        m2[side] = 2.0 * float(q.params[2])
    r_r = 2160.0 * sigma2 / max(n2["right"] * h2_r**4, 1e-12)
    r_l = 2160.0 * sigma2 / max(n2["left"] * h2_l**4, 1e-12)
    bottom = f_hat * ((m2["right"] - m2["left"]) ** 2 + r_r + r_l)
    if bottom <= 0:
        raise DataError("The Imbens-Kalyanaraman formula did not produce a usable bandwidth.")
    return float(3.4375 * (2.0 * sigma2 / bottom) ** 0.2 * n ** -0.2)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _bin_rows(x: np.ndarray, y: np.ndarray, side: str, n_bins: int, method: str,
              cutoff: float) -> list[dict[str, Any]]:
    if x.size == 0:
        return []
    n_bins = max(int(n_bins), 1)
    rows: list[dict[str, Any]] = []
    if method == "evenly":
        for r in stats.binned_means(x, y, bins=n_bins):
            rows.append({"x": float(r["x"] + cutoff), "y": float(r["y"]),
                         "side": side, "n": int(r["n"])})
        return rows
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        return [{"x": float(np.mean(x) + cutoff), "y": float(np.mean(y)),
                 "side": side, "n": int(x.size)}]
    pos = np.clip(np.digitize(x, edges[1:-1], right=False), 0, edges.size - 2)
    for b in range(edges.size - 1):
        sel = pos == b
        if not sel.any():
            continue
        rows.append({"x": float(np.mean(x[sel]) + cutoff), "y": float(np.mean(y[sel])),
                     "side": side, "n": int(sel.sum())})
    return rows


def _auto_bins(n_side: int) -> int:
    """~2 n^(1/3) bins: the rate of the IMSE-optimal rdplot selector, not its constant."""
    return int(np.clip(round(2.0 * max(n_side, 1) ** (1.0 / 3.0)), 4, 40))


def _fit_curve(res: _RDFit, cutoff: float, points: int = 40) -> list[dict[str, Any]]:
    """The fitted polynomial each side, inside the bandwidth, at mean covariates."""
    order = res.order
    beta = res.beta
    base = 0.0
    if res.covar_mean is not None and res.covar_mean.size:
        k_poly = 2 * (order + 1)
        base = float(beta[k_poly: k_poly + res.covar_mean.size] @ res.covar_mean)
    rows: list[dict[str, Any]] = []
    for side, lo, hi, offset in (("left", -res.h, 0.0, 0), ("right", 0.0, res.h, order + 1)):
        xs = np.linspace(lo, hi, points)
        us = xs / res.h
        vals = np.zeros_like(xs) + base
        for j in range(order + 1):
            vals = vals + float(beta[offset + j]) * us**j
        for xv, yv in zip(xs, vals):
            rows.append({"x": float(xv + cutoff), "y": float(yv), "side": side})
    return rows


def _binned_scatter_artifact(
    rb: ResultBuilder, xc: np.ndarray, y: np.ndarray, res: _RDFit, cutoff: float,
    *, running: str, outcome: str, bins: int, method: str, title: str,
) -> str:
    left = xc < 0
    right = xc >= 0
    n_left_bins = bins or _auto_bins(int(left.sum()))
    n_right_bins = bins or _auto_bins(int(right.sum()))
    points = (_bin_rows(xc[left], y[left], "left", n_left_bins, method, cutoff)
              + _bin_rows(xc[right], y[right], "right", n_right_bins, method, cutoff))
    spec = vega.binned_scatter(
        points,
        cutoff=cutoff,
        fits=_fit_curve(res, cutoff),
        title=title,
        x_title=running,
        y_title=outcome,
    )
    return rb.artifact(
        "vega",
        title=title,
        spec=spec,
        data=points,
        explain_key="plot.rd.binned_scatter",
        caption=(
            f"{'Quantile-spaced' if method == 'quantile' else 'Evenly spaced'} bin means of "
            f"{outcome} against {running} over the whole support, with the order-{res.order} "
            f"polynomial fitted inside the bandwidth (h = {res.h:.4g}) drawn either side. "
            "The vertical gap at the cutoff is the estimate."
        ),
    )


# ---------------------------------------------------------------------------
# Discreteness of the running variable
# ---------------------------------------------------------------------------


def _discreteness(xc: np.ndarray, h: float) -> dict[str, Any]:
    inside = np.abs(xc) < h
    left_pts = np.unique(xc[inside & (xc < 0)])
    right_pts = np.unique(xc[inside & (xc >= 0)])
    all_pts = np.unique(xc)
    at_cut = int(np.sum(xc == 0))
    counts = pd.Series(xc).value_counts()
    modal_share = float(counts.iloc[0] / len(xc)) if len(counts) else 0.0
    return {
        "mass_points_total": int(all_pts.size),
        "mass_points_left_in_h": int(left_pts.size),
        "mass_points_right_in_h": int(right_pts.size),
        "n_at_cutoff": at_cut,
        "modal_value_share": modal_share,
        "n_inside_bandwidth": int(inside.sum()),
    }


# ---------------------------------------------------------------------------
# McCrary (2008) density discontinuity test
# ---------------------------------------------------------------------------


def _mccrary(x: np.ndarray, cutoff: float, bandwidth: float | None = None) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 30:
        raise DataError("A density test needs at least 30 observations of the running variable.")
    sd = float(np.std(x, ddof=1))
    if not np.isfinite(sd) or sd <= 0:
        raise DataError("The running variable does not vary; there is no density to test.")
    binsize = 2.0 * sd * n ** -0.5
    z = x - cutoff
    k = np.floor(z / binsize).astype(int)
    k_lo, k_hi = int(k.min()), int(k.max())
    edges_k = np.arange(k_lo, k_hi + 1)
    counts = np.bincount(k - k_lo, minlength=edges_k.size).astype(float)
    mid = (edges_k + 0.5) * binsize
    dens = counts / (n * binsize)

    def _side_bw(mask: np.ndarray) -> float | None:
        zz, dd = mid[mask], dens[mask]
        if zz.size < 6:
            return None
        s = float(np.max(np.abs(zz))) or 1.0
        t = zz / s
        B = np.column_stack([t**j for j in range(5)])
        try:
            fit = stats.ols(dd, B, [f"p{j}" for j in range(5)], vcov="classical")
        except Exception:
            return None
        a = fit.params
        f2 = (2 * a[2] + 6 * a[3] * t + 12 * a[4] * t**2) / (s * s)
        ss = float(np.sum(f2**2))
        sigma2 = float(np.mean(fit.resid**2))
        span = float(np.max(zz) - np.min(zz))
        if ss <= 0 or sigma2 <= 0 or span <= 0:
            return None
        return float(3.348 * (sigma2 * span / ss) ** 0.2)

    if bandwidth is None:
        hl = _side_bw(mid < 0)
        hr = _side_bw(mid >= 0)
        cands = [v for v in (hl, hr) if v and np.isfinite(v) and v > 0]
        span_l = float(np.max(np.abs(mid[mid < 0]))) if np.any(mid < 0) else 0.0
        span_r = float(np.max(mid[mid >= 0])) if np.any(mid >= 0) else 0.0
        fallback = max(span_l, span_r) / 2.0 if max(span_l, span_r) > 0 else 5 * binsize
        h = float(np.mean(cands)) if cands else fallback
        h = float(np.clip(h, 3.0 * binsize, max(span_l, span_r) if max(span_l, span_r) > 0 else 3.0 * binsize))
    else:
        h = float(bandwidth)
    if not np.isfinite(h) or h <= 0:
        raise DataError("A usable bandwidth for the density test could not be found.")

    out: dict[str, Any] = {"binsize": float(binsize), "bandwidth": float(h), "n": n}
    fhat: dict[str, float] = {}
    for side, mask in (("left", mid < 0), ("right", mid >= 0)):
        zz, dd = mid[mask], dens[mask]
        w = np.clip(1.0 - np.abs(zz) / h, 0.0, None)
        keep = w > 0
        if int(keep.sum()) < 3:
            raise DataError(
                f"Only {int(keep.sum())} histogram bin(s) fall inside the density-test bandwidth "
                f"on the {side}.",
                detail="The running variable is too coarse near the cutoff for a McCrary test; "
                       "consider a local-randomisation analysis instead.",
            )
        B = np.column_stack([np.ones(int(keep.sum())), zz[keep]])
        fit = stats.ols(dd[keep], B, ["1", "z"], weights=w[keep], vcov="classical")
        fhat[side] = float(fit.params[0])
    f_l, f_r = fhat["left"], fhat["right"]
    if f_l <= 0 or f_r <= 0:
        out.update({"f_left": f_l, "f_right": f_r, "theta": None, "se": None,
                    "statistic": None, "p_value": None,
                    "note": "The local linear density estimate was not positive on one side; "
                            "the log difference is undefined."})
    else:
        theta = math.log(f_r) - math.log(f_l)
        var = (1.0 / (n * h)) * (24.0 / 5.0) * (1.0 / f_r + 1.0 / f_l)
        se = math.sqrt(max(var, 0.0))
        zstat = theta / se if se > 0 else float("nan")
        out.update({
            "f_left": f_l, "f_right": f_r, "theta": float(theta), "se": float(se),
            "statistic": float(zstat), "p_value": float(stats.norm_sf2(zstat)) if se > 0 else None,
        })
    out["bins"] = [
        {"x_lo": float(edges_k[i] * binsize + cutoff),
         "x_hi": float((edges_k[i] + 1) * binsize + cutoff),
         "x": float(mid[i] + cutoff),
         "count": float(counts[i]),
         "side": "left" if mid[i] < 0 else "right"}
        for i in range(edges_k.size)
    ]
    out["n_bins"] = int(edges_k.size)
    return out


# ---------------------------------------------------------------------------
# Shared preparation
# ---------------------------------------------------------------------------


@dataclass
class _Prep:
    ctx: RunContext
    sample: roles.Sample
    df: pd.DataFrame
    running: str
    outcome: str
    treatment: str | None
    cutoff: float
    xc: np.ndarray
    y: np.ndarray
    covars: np.ndarray | None
    covar_names: list[str]
    covar_cols: list[str]
    cluster_col: str | None
    cluster_codes: np.ndarray | None
    kernel_name: str
    kernel_fn: Callable[[np.ndarray], np.ndarray]
    order: int
    deriv: int
    level: float
    vce: str
    warnings: list[dict[str, str]] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.df)


def _prepare(
    ctx: RunContext,
    *,
    default_order: int = 1,
    deriv: int = 0,
    need_treatment: bool = False,
    use_covariates: bool = True,
) -> _Prep:
    roles.require_design_roles(ctx.spec, "rd")
    running = roles.require_role(ctx.spec, "running")
    outcome = roles.require_role(ctx.spec, "outcome")
    raw_cutoff = roles.get_role(ctx.spec, "cutoff")
    try:
        cutoff = float(raw_cutoff)
    except (TypeError, ValueError):
        raise SpecError(
            f"The cutoff must be a number; got {raw_cutoff!r}.",
            detail="Type the threshold value of the running variable in the inspector, "
                   "or drag the orange line on the RD board.",
        ) from None
    treatment = roles.get_role(ctx.spec, "treatment")
    if need_treatment and not treatment:
        raise SpecError(
            "Fuzzy RD needs the treatment actually received, as well as the running variable.",
            detail="Drop the variable that records who was treated on the treatment slot. "
                   "If crossing the cutoff determines treatment exactly, use sharp RD instead.",
        )
    cluster_col = roles.get_role(ctx.spec, "cluster")
    confs = roles.confounders(ctx.spec) if use_covariates else []

    order = _opt_int(ctx, "polynomial_order", default_order, minimum=deriv + 1, maximum=4)
    kernel_name, kernel_fn = _kernel(ctx.opt("kernel", DEFAULT_KERNEL))
    level = _opt_float(ctx, "ci_level", 0.95, minimum=0.5, maximum=0.999) or 0.95
    vce = _opt_choice(ctx, "vce", "HC1", ("hc0", "hc1", "hc2", "hc3")).upper()

    needed = ["outcome", "running", "cluster"]
    if treatment:
        needed.append("treatment")
    if confs:
        needed.append("confounders")
    sample = roles.build_sample(ctx, needed=needed)
    df = sample.df

    x = roles.numeric(df, running, "running variable")
    y = roles.numeric(df, outcome, "outcome")
    xc = x - cutoff
    if not np.all(np.isfinite(xc)):
        raise DataError(f"The running variable '{running}' has non-numeric values in the sample.")

    donut = _opt_float(ctx, "donut", 0.0, minimum=0.0) or 0.0
    if donut > 0:
        keep = np.abs(xc) > donut
        if int(keep.sum()) == 0:
            raise SpecError(f"A donut radius of {donut:g} removes every observation.")
        sample.apply(keep, "Donut hole", f"|{running} - {cutoff:g}| <= {donut:g} removed "
                                         "(heaping / manipulation right at the threshold)")
        df = sample.df
        x, y = x[keep], y[keep]
        xc = xc[keep]

    n_left = int((xc < 0).sum())
    n_right = int((xc >= 0).sum())
    if n_left == 0 or n_right == 0:
        raise DataError(
            f"Every observation falls on one side of the cutoff "
            f"({n_left} below, {n_right} at or above {cutoff:g}).",
            detail=f"Check that the cutoff is on the scale of '{running}' "
                   f"(its range here is {np.nanmin(x):.6g} to {np.nanmax(x):.6g}).",
        )

    covars = None
    covar_names: list[str] = []
    covar_cols: list[str] = []
    if confs:
        present = [c for c in confs if c in df.columns]
        if present:
            dm = stats.design_matrix(df, present, intercept=False)
            if dm.X.shape[1]:
                mu = np.nanmean(dm.X, axis=0)
                sd = np.nanstd(dm.X, axis=0)
                sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
                covars = (dm.X - mu) / sd
                covar_names = list(dm.names)
                covar_cols = present

    cluster_codes = None
    if cluster_col and cluster_col in df.columns:
        codes, uniq = pd.factorize(df[cluster_col].astype(str))
        cluster_codes = np.asarray(codes, dtype=int)
        if len(uniq) < 2:
            raise DataError(
                f"The clustering variable '{cluster_col}' has a single cluster in this sample.",
                detail="Cluster-robust inference needs more than one cluster.",
            )

    return _Prep(
        ctx=ctx, sample=sample, df=df, running=running, outcome=outcome, treatment=treatment,
        cutoff=cutoff, xc=xc, y=y, covars=covars, covar_names=covar_names, covar_cols=covar_cols,
        cluster_col=cluster_col, cluster_codes=cluster_codes, kernel_name=kernel_name,
        kernel_fn=kernel_fn, order=order, deriv=deriv, level=level, vce=vce,
    )


def _choose_bandwidths(
    prep: _Prep, *, order: int, deriv: int, y: np.ndarray,
) -> tuple[float, float | None, str, dict[str, Any]]:
    ctx = prep.ctx
    selector = _opt_choice(ctx, "bandwidth_selector", "mse", ("mse", "ik"))
    h_user = _opt_float(ctx, "bandwidth", None, minimum=0.0)
    b_user = _opt_float(ctx, "pilot_bandwidth", None, minimum=0.0)
    info: dict[str, Any] = {"selector": selector}
    if h_user:
        h = float(h_user)
        selector = "manual"
    elif selector == "ik":
        if order != 1 or deriv != 0:
            raise SpecError(
                "The Imbens-Kalyanaraman bandwidth as implemented here is for a sharp "
                "local *linear* RD (polynomial order 1, level jump).",
                detail="Use the plug-in MSE selector, or set the bandwidth by hand.",
            )
        h = _bw_ik(prep.xc, y)
        info["ik_bandwidth"] = h
    else:
        prelim = _prelim_fit(prep.xc, y, _prelim_order(prep.xc, order + 4), prep.cluster_codes)
        h, path = _bw_select(
            prep.xc, y, order=order, deriv=deriv, mode="contrast",
            kernel_fn=prep.kernel_fn, prelim=prelim, covars=prep.covars,
            covar_names=prep.covar_names, cluster_codes=prep.cluster_codes, tick=ctx.tick,
        )
        info["mse_path"] = path
        info["prelim_order"] = prelim.order
    if b_user:
        b: float | None = float(b_user)
    else:
        try:
            prelim_b = _prelim_fit(prep.xc, y, _prelim_order(prep.xc, order + 5),
                                   prep.cluster_codes)
            b, _ = _bw_select(
                prep.xc, y, order=order + 1, deriv=order + 1, mode="pair",
                kernel_fn=prep.kernel_fn, prelim=prelim_b, covars=prep.covars,
                covar_names=prep.covar_names, cluster_codes=prep.cluster_codes,
                points=18, tick=ctx.tick,
            )
        except DataError:
            b = None
    if b is not None and h is not None and b < h:
        b = float(h)  # the pilot fit is one order higher; it should not be narrower
    info["h"] = h
    info["b"] = b
    return float(h), (float(b) if b else None), selector, info


# ===========================================================================
# From here down: the adapters. Everything above is machinery they share.
# ===========================================================================

DESIGN = "rd"

# How few distinct values of the running variable near the cutoff before the
# local polynomial is really interpolating between a handful of points.
MASS_POINTS_WARN = 20
MASS_POINTS_BAD = 10
# A first stage this small makes the Wald ratio a division by almost nothing.
FIRST_STAGE_WARN = 0.10
FIRST_STAGE_T_WARN = 3.0

CONTINUITY_NOTE = (
    "The estimate is the size of the step at the cutoff. It is the causal effect only if nothing "
    "else about these units changes discontinuously there -- no other programme with the same "
    "threshold, no different data collection above and below."
)


def _fmt(value: Any, digits: int = 4) -> str:
    """Numbers for the Classic tab, with a dash where a number does not exist."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(val):
        return "--"
    return f"{val:.{digits}g}"


def _ci_text(ci: tuple[float | None, float | None] | None, digits: int = 4) -> str:
    if not ci or ci[0] is None or ci[1] is None:
        return "[--, --]"
    return f"[{_fmt(ci[0], digits)}, {_fmt(ci[1], digits)}]"


# ---------------------------------------------------------------------------
# Wiring a _Prep into a ResultBuilder
# ---------------------------------------------------------------------------


def _wire(rb: ResultBuilder, prep: _Prep, *, estimand: str | None, estimand_label: str) -> None:
    """Sample flow, counts, roles and the assumption ledger -- once, for every RD adapter."""
    rb.result["estimand"] = estimand
    rb.result["estimand_label"] = estimand_label
    rb.result["treatment"] = prep.treatment
    rb.result["outcome"] = prep.outcome
    rb.extend_flow(prep.sample.flow)
    rb.set_counts(n=prep.n)
    rb.set_roles_used({
        "running": prep.running,
        "cutoff": prep.cutoff,
        "outcome": prep.outcome,
        "treatment": prep.treatment,
        "confounders": prep.covar_cols,
        "cluster": prep.cluster_col,
    })
    roles.seed_ledger(rb, DESIGN)
    rb.set_assumption_status("continuity", "assumed", CONTINUITY_NOTE)
    rb.set_assumption_status(
        "consistency", "assumed",
        f"Crossing {prep.cutoff:g} on '{prep.running}' is treated as one well-defined switch.",
    )
    rb.set_assumption_status(
        "sutva", "assumed",
        "Units just below the cutoff are assumed not to be affected by units just above being "
        "treated -- worth a thought when the cutoff rations a fixed number of places.",
    )


def _flow_bandwidth(rb: ResultBuilder, prep: _Prep, res: _RDFit,
                    *, label: str = "Inside the bandwidth") -> None:
    """Bandwidth truncation is a sample edit, so it is a CONSORT row like any other."""
    n_eff = res.n_effective
    rb.add_flow(
        label, n_eff,
        n_treated=res.n_right, n_control=res.n_left,
        dropped=int(prep.n - n_eff),
        reason=(f"|{prep.running} - {prep.cutoff:g}| < h = {res.h:.6g}; rows outside the bandwidth get "
                f"zero kernel weight and do not enter the estimate"),
    )
    rb.set_counts(n_treated=res.n_right, n_control=res.n_left, n_effective=n_eff)


def _safe_fit(prep: _Prep, y: np.ndarray, *, h: float, order: int, deriv: int,
              b: float | None = None, covars: np.ndarray | None = None) -> _RDFit | None:
    """A fit that returns None instead of raising -- for paths and placebos."""
    try:
        return _rd_fit(
            prep.xc, y, h=h, b=b, order=order, deriv=deriv, kernel_fn=prep.kernel_fn,
            covars=covars, covar_names=(prep.covar_names if covars is not None else ()),
            cluster_codes=prep.cluster_codes, level=prep.level, vce=prep.vce,
        )
    except (DataError, SpecError, np.linalg.LinAlgError, ValueError):
        return None


# ---------------------------------------------------------------------------
# The hero plot
# ---------------------------------------------------------------------------


def _plot_options(ctx: RunContext) -> tuple[int, str]:
    bins = _opt_int(ctx, "plot_bins", 0, minimum=0, maximum=100)
    method = _opt_choice(ctx, "plot_bin_spacing", "quantile", ("quantile", "evenly"))
    return bins, method


def _hero_plot(rb: ResultBuilder, ctx: RunContext, prep: _Prep, res: _RDFit,
               *, y: np.ndarray | None = None, y_label: str | None = None,
               title: str | None = None) -> str:
    bins, method = _plot_options(ctx)
    yy = prep.y if y is None else y
    label = y_label or prep.outcome
    return _binned_scatter_artifact(
        rb, prep.xc, yy, res, prep.cutoff,
        running=prep.running, outcome=label, bins=bins, method=method,
        title=title or f"{label} against {prep.running}",
    )


def _raw_scatter_artifact(rb: ResultBuilder, ctx: RunContext, prep: _Prep) -> str:
    """Bin means with no fitted curve -- the fallback when no local fit is possible."""
    bins, method = _plot_options(ctx)
    left, right = prep.xc < 0, prep.xc >= 0
    points = (_bin_rows(prep.xc[left], prep.y[left], "left",
                        bins or _auto_bins(int(left.sum())), method, prep.cutoff)
              + _bin_rows(prep.xc[right], prep.y[right], "right",
                          bins or _auto_bins(int(right.sum())), method, prep.cutoff))
    title = f"{prep.outcome} against {prep.running}"
    return rb.artifact(
        "vega", title=title,
        spec=vega.binned_scatter(points, cutoff=prep.cutoff, fits=None, title=title,
                                 x_title=prep.running, y_title=prep.outcome),
        data=points, explain_key="plot.rd.binned_scatter",
        caption=("Bin means of the outcome either side of the cutoff. No local polynomial is drawn "
                 "because none could be fitted on this data."),
    )


def _diag_discontinuity_plot(rb: ResultBuilder, prep: _Prep, res: _RDFit, art: str,
                             *, what: str = "outcome") -> None:
    rb.add_diagnostic(
        "rd_plot", "The picture at the cutoff",
        status="info",
        summary=(f"Bin means either side of {prep.cutoff:g} with the fitted local polynomial drawn "
                 f"inside the bandwidth. The step you can see in the {what} is {res.tau:.4g}; that step "
                 "is the estimate."),
        worry_when=("The bins wander in a way the fitted line does not follow, or the step at the cutoff "
                    "is smaller than the wiggle in the bins away from it. Then the number is a feature "
                    "of the polynomial, not of the cutoff."),
        artifact_ids=[art], explain_key="plot.rd.binned_scatter",
        values={"estimate": res.tau, "bandwidth": res.h,
                "n_inside_bandwidth": res.n_effective,
                "n_below": res.n_left, "n_at_or_above": res.n_right},
    )


# ---------------------------------------------------------------------------
# Bandwidth sensitivity: a plot, not a hidden option
# ---------------------------------------------------------------------------


def _diag_bandwidth_path(
    rb: ResultBuilder,
    prep: _Prep,
    *,
    estimate_at: Callable[[float], tuple[float, float | None, float | None] | None],
    h_star: float,
    ci_star: tuple[float | None, float | None],
    y_title: str,
    points: int = 13,
    diag_id: str = "bandwidth_sensitivity",
) -> dict[str, Any]:
    span = float(np.max(np.abs(prep.xc)))
    lo = max(h_star * 0.4, 1e-12)
    hi = min(max(h_star * 2.5, lo * 1.5), span if span > 0 else h_star * 2.5)
    grid = sorted(set(np.geomspace(lo, hi, int(points)).tolist() + [float(h_star)]))
    rows: list[dict[str, Any]] = []
    for h in grid:
        out = estimate_at(float(h))
        if out is None:
            continue
        est, ci_lo, ci_hi = out
        if est is None or not np.isfinite(est):
            continue
        rows.append({
            "x": float(h), "estimate": float(est),
            "ci_low": (float(ci_lo) if ci_lo is not None and np.isfinite(ci_lo) else None),
            "ci_high": (float(ci_hi) if ci_hi is not None and np.isfinite(ci_hi) else None),
            "chosen": bool(abs(h - h_star) <= 1e-12),
        })
    if len(rows) < 3:
        rb.add_diagnostic(
            diag_id, "Does the answer depend on the bandwidth?",
            status="untested",
            summary="Too few bandwidths around the chosen one could be fitted to draw a sensitivity path.",
            worry_when="A result that only exists at one bandwidth is a bandwidth, not a finding.",
            explain_key="diagnostic.rd.bandwidth_path",
        )
        return {}
    art = rb.artifact(
        "vega", title="Estimate against bandwidth",
        spec=vega.path_plot(
            rows, title="The estimate at every bandwidth",
            x_title=f"Bandwidth h (units of {prep.running})",
            y_title=y_title, marker_x=float(h_star),
        ),
        data=rows, columns=["x", "estimate", "ci_low", "ci_high"],
        explain_key="diagnostic.rd.bandwidth_path",
        caption=("Each point refits the same local polynomial at a different bandwidth. The orange line "
                 "is the bandwidth the selector chose. Wide bandwidths use more data and lean harder on "
                 "the functional form; narrow ones are local but noisy."),
    )
    ests = [r["estimate"] for r in rows]
    lo_e, hi_e = float(min(ests)), float(max(ests))
    signs = {1 if e > 0 else (-1 if e < 0 else 0) for e in ests}
    flips = len(signs - {0}) > 1
    inside = 0
    if ci_star[0] is not None and ci_star[1] is not None:
        inside = sum(1 for e in ests if ci_star[0] <= e <= ci_star[1])
    outside = len(ests) - inside
    # A sign change only matters when the headline itself is distinguishable from
    # zero; an estimate that is nothing at every bandwidth is allowed to be
    # nothing with either sign.
    called = bool(ci_star[0] is not None and ci_star[1] is not None
                  and not (ci_star[0] <= 0.0 <= ci_star[1]))
    unstable = bool((flips and called) or outside > len(ests) / 3.0)
    rb.add_diagnostic(
        diag_id, "Does the answer depend on the bandwidth?",
        status="weakens" if unstable else "supports",
        summary=(f"Refitting at {len(rows)} bandwidths between {rows[0]['x']:.4g} and {rows[-1]['x']:.4g} "
                 f"moves the estimate between {lo_e:.4g} and {hi_e:.4g}; {outside} of {len(ests)} fall "
                 f"outside the interval reported at the chosen bandwidth ({h_star:.4g})."
                 + (" The sign changes along the path." if flips else "")),
        worry_when=("The estimate sliding steadily as the window widens, or changing sign. That is the "
                    "shape of a functional-form artefact rather than a jump at the cutoff."),
        artifact_ids=[art],
        values={"h_chosen": float(h_star), "h_min": rows[0]["x"], "h_max": rows[-1]["x"],
                "estimate_min": lo_e, "estimate_max": hi_e,
                "n_outside_chosen_ci": int(outside), "n_bandwidths": len(rows),
                "sign_changes": flips},
        explain_key="diagnostic.rd.bandwidth_path",
    )
    if unstable:
        rb.add_warning(
            f"The estimate moves from {lo_e:.4g} to {hi_e:.4g} across bandwidths around the chosen one. "
            "Report the path, not just the number.",
            level="caution", code="bandwidth_sensitive",
            explain_key="diagnostic.rd.bandwidth_path",
        )
    return {"rows": rows, "unstable": unstable, "artifact": art}


# ---------------------------------------------------------------------------
# Is the running variable continuous enough for any of this?
# ---------------------------------------------------------------------------


def _diag_discreteness(rb: ResultBuilder, prep: _Prep, h: float) -> dict[str, Any]:
    info = _discreteness(prep.xc, h)
    rows = stats.histogram_rows(prep.xc + prep.cutoff, bins=40)
    art = rb.artifact(
        "vega", title=f"Distribution of {prep.running}",
        spec=vega.histogram(rows, x_title=prep.running,
                            title=f"Where {prep.running} actually sits", rule_at=prep.cutoff),
        data=rows, columns=["x_lo", "x_hi", "count"],
        explain_key="diagnostic.rd.discreteness",
        caption=("Local polynomial RD assumes the running variable is continuous near the cutoff. When "
                 "it only takes a handful of values there, the fit interpolates between a few points and "
                 "the usual standard errors are optimistic."),
    )
    few = min(info["mass_points_left_in_h"], info["mass_points_right_in_h"])
    status = "weakens" if few < MASS_POINTS_BAD else ("info" if few < MASS_POINTS_WARN else "supports")
    rb.add_diagnostic(
        "running_variable_discreteness", "Is the running variable continuous near the cutoff?",
        status=status,
        summary=(f"Inside the bandwidth '{prep.running}' takes {info['mass_points_left_in_h']} distinct "
                 f"value(s) below the cutoff and {info['mass_points_right_in_h']} at or above it, out of "
                 f"{info['mass_points_total']} distinct values in the whole sample; "
                 f"{info['n_at_cutoff']} row(s) sit exactly on {prep.cutoff:g}."),
        worry_when=("Only a few distinct values either side. Then this is a comparison of a few groups, "
                    "standard errors ought to be clustered on the running variable, and a "
                    "local-randomisation analysis fits better than a local polynomial."),
        artifact_ids=[art], values=info, explain_key="diagnostic.rd.discreteness",
    )
    if few < MASS_POINTS_WARN:
        rb.add_warning(
            f"'{prep.running}' is coarse near the cutoff: {info['mass_points_left_in_h']} distinct "
            f"value(s) below and {info['mass_points_right_in_h']} at or above, inside the bandwidth. "
            "Local polynomial standard errors assume a continuous running variable.",
            level="warning" if few < MASS_POINTS_BAD else "caution",
            code="discrete_running_variable", explain_key="diagnostic.rd.discreteness",
        )
    if few < MASS_POINTS_BAD:
        rb.mark_provisional(
            f"The running variable takes fewer than {MASS_POINTS_BAD} distinct values on one side of the "
            "cutoff inside the bandwidth. Read this as a comparison of a few groups, not as a smooth "
            "discontinuity."
        )
    return info


# ---------------------------------------------------------------------------
# Manipulation: the McCrary density test, wired as a diagnostic
# ---------------------------------------------------------------------------


def _diag_density(rb: ResultBuilder, prep: _Prep,
                  *, bandwidth: float | None = None) -> dict[str, Any] | None:
    try:
        out = _mccrary(prep.xc + prep.cutoff, prep.cutoff, bandwidth)
    except (DataError, SpecError) as exc:
        rb.add_diagnostic(
            "manipulation", "Did units sort across the cutoff?",
            status="untested",
            summary=f"The density test could not run here: {exc.message}",
            worry_when="A pile-up of units just on the winning side of the cutoff.",
            explain_key="assumption.no_manipulation",
        )
        rb.set_assumption_status("no_manipulation", "untested", exc.message)
        return None
    art = rb.artifact(
        "vega", title=f"Density of {prep.running} either side of the cutoff",
        spec=vega.density_by_side(out["bins"], cutoff=prep.cutoff,
                                  title=f"How many units sit at each value of {prep.running}",
                                  x_title=prep.running),
        data=out["bins"], columns=["x_lo", "x_hi", "count", "side"],
        explain_key="assumption.no_manipulation",
        caption=("A step in the height of this histogram at the cutoff is the signature of units moving "
                 "themselves across it."),
    )
    p = out.get("p_value")
    theta = out.get("theta")
    if p is None or theta is None:
        rb.add_diagnostic(
            "manipulation", "Did units sort across the cutoff?",
            status="untested",
            summary=out.get("note") or "The density either side of the cutoff could not be compared.",
            worry_when="A pile-up of units just on the winning side of the cutoff.",
            artifact_ids=[art], values={k: v for k, v in out.items() if k != "bins"},
            explain_key="assumption.no_manipulation",
        )
        rb.set_assumption_status("no_manipulation", "untested",
                                 "The density test did not produce a comparison.")
        return out
    ratio = math.exp(theta)
    suspicious = bool(p < 0.05)
    values = {k: v for k, v in out.items() if k != "bins"}
    values["density_ratio_right_over_left"] = ratio
    rb.add_diagnostic(
        "manipulation", "Did units sort across the cutoff?",
        status="weakens" if suspicious else "supports",
        summary=(f"McCrary density test: the estimated density just above the cutoff is {ratio:.2f} "
                 f"times the density just below (log difference {theta:.3f}, z = {out['statistic']:.2f}, "
                 f"p = {p:.3f})."
                 + (" That is a bigger gap than sampling noise comfortably explains."
                    if suspicious else
                    " That is the kind of gap sampling noise produces, so the test does not contradict "
                    "the no-manipulation assumption.")),
        worry_when=("More units just above the cutoff than just below. It means people could see the "
                    "threshold and move themselves over it, and the units either side are then no "
                    "longer comparable."),
        artifact_ids=[art], values=values, explain_key="assumption.no_manipulation",
    )
    rb.set_assumption_status(
        "no_manipulation", "weakened" if suspicious else "supported",
        ("The density of the running variable jumps at the cutoff, which is what sorting looks like."
         if suspicious else
         "The density test did not find a jump. 'Supported' means the test did not contradict the "
         "assumption, not that no manipulation happened."),
    )
    if suspicious:
        rb.add_warning(
            f"Units appear to pile up on one side of {prep.cutoff:g}: the density is {ratio:.2f} times "
            f"higher just above than just below (p = {p:.3f}). Consider a donut hole, and read the "
            "estimate as describing whoever ended up either side rather than a clean comparison.",
            level="warning", code="density_jump", explain_key="assumption.no_manipulation",
        )
    return out


# ---------------------------------------------------------------------------
# Covariate jumps at the cutoff
# ---------------------------------------------------------------------------


def _covariate_jumps(prep: _Prep, *, h: float, order: int) -> dict[str, Any]:
    """Local polynomial jump in every covariate, sharing one window.

    Because the window, the kernel and the design depend only on the running
    variable, every covariate is the same linear map ``ell`` applied to a
    different column. That makes the whole vector of jumps -- and the joint test
    -- exact rather than a loop of unrelated regressions.
    """
    if not prep.covar_cols:
        raise SpecError(
            "This check needs covariates to look at, and none are set.",
            detail="Drop the variables that should be unaffected by the programme -- age, sex, "
                   "pre-programme outcomes -- onto the covariates zone of the RD board.",
        )
    dm = stats.design_matrix(prep.df, prep.covar_cols, intercept=False)
    if not dm.names:
        raise DataError(
            "None of the covariates vary in this sample, so there is nothing to test for a jump.")
    win = _window(prep.xc, h, order, prep.kernel_fn)
    need = order + 2
    if win.n_left < need or win.n_right < need:
        raise DataError(
            f"Only {win.n_left} observation(s) below and {win.n_right} above the cutoff fall inside "
            f"h = {h:.6g}; an order-{order} polynomial needs at least {need} a side.",
            detail="Widen the bandwidth or lower the polynomial order.",
        )
    n = len(prep.xc)
    ell = _embed(n, win.idx, win.contrast(0))
    inside = np.zeros(n, dtype=bool)
    inside[win.idx] = True
    right = inside & (prep.xc >= 0)
    left = inside & (prep.xc < 0)

    jumps: list[float] = []
    resids: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for j, name in enumerate(dm.names):
        z = np.asarray(dm.X[:, j], dtype=float)
        jumps.append(float(ell @ z))
        resids.append(_hc_resid(z, win, prep.cluster_codes))
        sd = float(np.std(z, ddof=1))
        rows.append({"variable": name, "sd": sd if np.isfinite(sd) and sd > 0 else None,
                     "raw_gap": (float(np.mean(z[right])) - float(np.mean(z[left])))
                     if right.any() and left.any() else None})
    k = len(jumps)
    V = np.zeros((k, k))
    for a in range(k):
        for b in range(a, k):
            v = _cov_of(ell, resids[a], resids[b], prep.cluster_codes)
            V[a, b] = V[b, a] = v
    jvec = np.asarray(jumps, dtype=float)
    ses = np.sqrt(np.clip(np.diag(V), 0.0, None))
    out_rows: list[dict[str, Any]] = []
    for j, name in enumerate(dm.names):
        se = float(ses[j])
        est = float(jvec[j])
        z = est / se if se > 0 else float("nan")
        ci = stats.wald_ci(est, se, prep.level)
        sd = rows[j]["sd"]
        out_rows.append({
            "variable": name,
            "jump": est,
            "se": se if se > 0 else None,
            "ci_low": ci[0], "ci_high": ci[1],
            "statistic": float(z) if np.isfinite(z) else None,
            "p_value": float(stats.norm_sf2(z)) if se > 0 else None,
            "standardised_jump": (est / sd) if sd else None,
            "raw_gap_in_window": rows[j]["raw_gap"],
            "standardised_raw_gap": (rows[j]["raw_gap"] / sd) if sd and rows[j]["raw_gap"] is not None else None,
        })
    rank = int(np.linalg.matrix_rank(V)) if k else 0
    wald = float(jvec @ np.linalg.pinv(V) @ jvec) if rank else float("nan")
    p_joint = float(stats.chi2_sf(wald, rank)) if rank and np.isfinite(wald) else None
    return {
        "rows": out_rows, "wald": wald if np.isfinite(wald) else None, "df": rank,
        "p_value": p_joint, "h": float(h), "order": int(order),
        "n_left": int(win.n_left), "n_right": int(win.n_right),
        "dropped_constant": list(dm.dropped),
    }


def _diag_covariate_balance(rb: ResultBuilder, prep: _Prep, jumps: dict[str, Any],
                            *, diag_id: str = "covariate_balance") -> dict[str, Any]:
    rows = jumps["rows"]
    love_rows = [
        {"variable": r["variable"],
         "abs_smd_before": abs(r["standardised_raw_gap"]) if r["standardised_raw_gap"] is not None else None,
         "abs_smd_after": abs(r["standardised_jump"]) if r["standardised_jump"] is not None else None}
        for r in rows
    ]
    art = rb.artifact(
        "vega", title="Covariate jumps at the cutoff",
        spec=vega.love_plot(
            love_rows, threshold=0.1,
            title="Covariate gaps at the cutoff (standardised)"),
        data=love_rows, columns=["variable", "abs_smd_before", "abs_smd_after"],
        explain_key="diagnostic.rd.covariate_balance",
        caption=("'Before' is the raw gap between units above and below inside the bandwidth; 'After' is "
                 "the local polynomial jump, which is the quantity that should be zero if the cutoff is "
                 "the only thing changing there. Both are in standard deviations of the covariate."),
    )
    tab = rb.artifact(
        "table", title="Covariate jumps at the cutoff", data=rows,
        columns=["variable", "jump", "se", "ci_low", "ci_high", "statistic", "p_value",
                 "standardised_jump", "raw_gap_in_window"],
        caption="Each row is the same local polynomial estimator with that covariate as the outcome.",
    )
    p = jumps.get("p_value")
    sig = [r for r in rows if r["p_value"] is not None and r["p_value"] < 0.05]
    worst = max((abs(r["standardised_jump"]) for r in rows
                 if r["standardised_jump"] is not None), default=0.0)
    weakened = bool((p is not None and p < 0.05) or sig)
    rb.add_diagnostic(
        diag_id, "Do the covariates jump at the cutoff too?",
        status="weakens" if weakened else "supports",
        summary=(f"{len(rows)} covariate(s) tested at h = {jumps['h']:.4g}. "
                 + (f"Joint test: chi-square {jumps['wald']:.2f} on {jumps['df']} d.f., p = {p:.3f}. "
                    if p is not None else "The joint test could not be formed. ")
                 + f"The largest single jump is {worst:.3f} standard deviations"
                 + (f"; {len(sig)} covariate(s) jump significantly on their own."
                    if sig else "; none jumps significantly on its own.")),
        worry_when=("A covariate that the programme cannot possibly have changed jumping at the cutoff. "
                    "If age or sex steps at the threshold, something other than eligibility changes "
                    "there and the outcome jump is not the programme."),
        artifact_ids=[art, tab],
        values={"p_value": p, "wald": jumps.get("wald"), "df": jumps.get("df"),
                "max_abs_standardised_jump": worst,
                "n_significant": len(sig),
                "significant": [r["variable"] for r in sig],
                "bandwidth": jumps["h"]},
        explain_key="diagnostic.rd.covariate_balance",
    )
    rb.set_assumption_status(
        "continuity", "weakened" if weakened else "supported",
        (CONTINUITY_NOTE + " Here at least one covariate jumps at the cutoff, which is evidence against it.")
        if weakened else
        (CONTINUITY_NOTE + " The measured covariates do not jump at the cutoff. 'Supported' means this "
         "check did not contradict continuity -- it says nothing about what you did not measure."),
    )
    if weakened:
        rb.add_warning(
            "Covariates that the programme cannot have changed jump at the cutoff ("
            + ", ".join(r["variable"] for r in sig[:4])
            + "). Something other than eligibility differs across the threshold.",
            level="warning", code="covariate_jump",
            explain_key="diagnostic.rd.covariate_balance",
        )
    return {"artifact": art, "table": tab, "weakened": weakened, "max": worst, "p_value": p}


def _maybe_covariate_balance(rb: ResultBuilder, prep: _Prep, *, h: float, order: int,
                             columns: Sequence[str]) -> None:
    """Always leave a covariate-balance row, even when there is nothing to check.

    A missing diagnostic reads as 'fine'. An untested one reads as untested,
    which is what a design with no covariates on it actually is.
    """
    if not columns:
        rb.add_diagnostic(
            "covariate_balance", "Do the covariates jump at the cutoff too?",
            status="untested",
            summary="No covariates were supplied, so the design was not checked against variables the "
                    "programme cannot have changed.",
            worry_when="Continuity is assumed here and tested against nothing. Drop a pre-programme "
                       "variable on the covariates zone and this becomes a real check.",
            explain_key="diagnostic.rd.covariate_balance",
        )
        return
    try:
        jumps = _covariate_jumps(prep, h=h, order=order)
    except (DataError, SpecError) as exc:
        rb.add_diagnostic(
            "covariate_balance", "Do the covariates jump at the cutoff too?",
            status="untested", summary=f"The balance check could not run: {exc.message}",
            worry_when="A covariate the programme cannot have changed jumping at the cutoff.",
            explain_key="diagnostic.rd.covariate_balance",
        )
        return
    _diag_covariate_balance(rb, prep, jumps)


# ---------------------------------------------------------------------------
# Placebo cutoffs: the same estimator where nothing happened
# ---------------------------------------------------------------------------


def _sens_placebo_cutoffs(rb: ResultBuilder, prep: _Prep, *, h: float, order: int,
                          y: np.ndarray, actual: float, y_title: str) -> None:
    """Refit at fake cutoffs, using only data from one side of the real one."""
    rows: list[dict[str, Any]] = []
    for side, mask in (("below", prep.xc < 0), ("at or above", prep.xc >= 0)):
        xs = prep.xc[mask]
        ys = y[mask]
        if xs.size < 8 * (order + 2):
            continue
        for q in (0.33, 0.67):
            fake = float(np.quantile(xs, q))
            shifted = xs - fake
            if np.min(shifted) >= 0 or np.max(shifted) <= 0:
                continue
            try:
                fit = _rd_fit(shifted, ys, h=h, b=None, order=order, deriv=0,
                              kernel_fn=prep.kernel_fn, level=prep.level, vce=prep.vce)
            except (DataError, SpecError, np.linalg.LinAlgError, ValueError):
                continue
            rows.append({
                "label": f"Fake cutoff at {fake + prep.cutoff:.4g} ({side})",
                "estimate": fit.tau, "se": fit.se,
                "ci_low": fit.ci[0], "ci_high": fit.ci[1],
                "p_value": fit.p_value, "engine": "python",
            })
    if not rows:
        return
    rows.append({"label": f"Real cutoff at {prep.cutoff:.4g}", "estimate": actual,
                 "ci_low": None, "ci_high": None, "engine": "python"})
    art = rb.artifact(
        "vega", title="The same estimator at fake cutoffs",
        spec=vega.forest(rows, title="Placebo cutoffs", x_title=y_title),
        data=rows, columns=["label", "estimate", "se", "ci_low", "ci_high", "p_value"],
        explain_key="probe.rd.placebo_cutoff",
        caption=("Each fake cutoff uses only data from one side of the real one, so there is no real "
                 "discontinuity to find. Jumps as big as the real one here mean the estimator finds "
                 "steps in this data wherever you point it."),
    )
    placebo = [r for r in rows if r.get("ci_low") is not None]
    bigger = sum(1 for r in placebo if abs(r["estimate"]) >= abs(actual))
    rb.add_sensitivity(
        "placebo_cutoffs",
        title="The same estimator at fake cutoffs",
        summary=(f"{bigger} of {len(placebo)} placebo cutoffs produced a jump at least as large as the "
                 f"{abs(actual):.4g} found at the real cutoff."),
        values={"n_placebos": len(placebo), "n_at_least_as_large": bigger,
                "actual": float(actual),
                "placebos": [{"label": r["label"], "estimate": r["estimate"], "se": r["se"],
                              "p_value": r["p_value"]} for r in placebo]},
        artifact_ids=[art],
    )
    if placebo and bigger >= max(1, len(placebo) // 2):
        rb.add_warning(
            f"{bigger} of {len(placebo)} fake cutoffs produce a jump as large as the real one. The "
            "estimator is finding steps where the policy has none.",
            level="caution", code="placebo_cutoff", explain_key="probe.rd.placebo_cutoff",
        )


# ---------------------------------------------------------------------------
# The Classic tab
# ---------------------------------------------------------------------------


def _classic_head(prep: _Prep, title: str, *, selector: str, h: float, b: float | None,
                  order: int, deriv: int, res: _RDFit | None) -> list[str]:
    lines = [
        title,
        "=" * len(title),
        f"Running variable : {prep.running}   cutoff = {prep.cutoff:g}",
        f"Outcome          : {prep.outcome}",
    ]
    if prep.treatment:
        lines.append(f"Treatment taken  : {prep.treatment}")
    lines += [
        f"Kernel           : {prep.kernel_name}    polynomial order = {order}"
        + ("    derivative = 1 (kink)" if deriv else ""),
        f"Bandwidth        : h = {h:.6g}   ({SELECTOR_LABELS.get(selector, selector)})",
        f"Bias pilot       : b = {b:.6g}" if b else "Bias pilot       : none (conventional estimator only)",
    ]
    if res is not None:
        lines.append(
            f"Variance         : {res.vcov_type}"
            + (f"   clustered on {prep.cluster_col}" if prep.cluster_col else "")
        )
        lines.append(
            f"Observations     : {prep.n} in the sample, {res.n_effective} inside the bandwidth "
            f"({res.n_left} below, {res.n_right} at or above the cutoff)"
        )
    lines.append("")
    return lines


def _classic_tail(rb: ResultBuilder, prep: _Prep) -> list[str]:
    lines = ["", "Diagnostics", "-----------"]
    for d in rb.result["diagnostics"]:
        lines.append(f"  [{d['status']:<9}] {d['title']}")
        if d.get("summary"):
            lines.append(f"              {d['summary']}")
    if rb.result["warnings"]:
        lines += ["", "Warnings", "--------"]
        for w in rb.result["warnings"]:
            lines.append(f"  ({w['level']}) {w['message']}")
    if rb.result["provisional_reasons"]:
        lines += ["", "Provisional", "-----------"]
        for r in rb.result["provisional_reasons"]:
            lines.append(f"  {r}")
    lines += ["", "About this implementation", "-------------------------", "  " + NOT_RDROBUST]
    return lines


# ---------------------------------------------------------------------------
# Headline plumbing shared by the estimating adapters
# ---------------------------------------------------------------------------


def _selector_text(selector: str, h: float, b: float | None) -> str:
    text = f"bandwidth h = {h:.4g} ({SELECTOR_LABELS.get(selector, selector)})"
    if b:
        text += f", bias pilot b = {b:.4g}"
    return text


def _set_headline(
    rb: ResultBuilder,
    prep: _Prep,
    *,
    est: float,
    se: float | None,
    ci: tuple[float | None, float | None] | None,
    p_value: float | None,
    statistic: float | None,
    est_bc: float | None,
    se_rb: float | None,
    ci_rb: tuple[float | None, float | None] | None,
    p_rb: float | None,
    z_rb: float | None,
    selector: str,
    h: float,
    b: float | None,
    vcov_type: str,
    what: str,
    bias_note: str | None = None,
) -> None:
    """The conventional point estimate, reported with robust bias-corrected inference.

    That pairing is deliberate. The point estimate is the step you can see in the
    binned scatter, so the headline number and the hero plot agree. The interval
    is the bias-corrected one, because at an MSE-optimal bandwidth the smoothing
    bias of the conventional estimator is the same order as its standard error,
    and a conventional interval is then narrower than the evidence.
    """
    level_pct = f"{prep.level * 100:g}%"
    have_robust = (est_bc is not None and se_rb is not None
                   and np.isfinite(se_rb) and se_rb > 0 and ci_rb is not None)
    if have_robust:
        rb.set_estimate(
            est, se=se_rb, ci=ci_rb, p_value=p_rb, statistic=z_rb,
            inference=(f"conventional local-polynomial point estimate with robust bias-corrected "
                       f"{level_pct} interval (Calonico-Cattaneo-Titiunik style, implemented here); "
                       f"{_selector_text(selector, h, b)}; {vcov_type} variance. The interval is "
                       f"centred on the bias-corrected estimate, so it is not the estimate plus or "
                       f"minus z times the standard error."),
            ci_level=prep.level,
        )
    else:
        rb.set_estimate(
            est, se=se, ci=ci, p_value=p_value, statistic=statistic,
            inference=(f"conventional local polynomial, {level_pct} Wald interval; "
                       f"{_selector_text(selector, h, b)}; {vcov_type} variance"),
            ci_level=prep.level,
        )
        rb.add_warning(
            "Only the conventional estimator was available here: "
            + (bias_note or "the pilot fit needed for bias correction could not be formed.")
            + " Its confidence interval does not allow for the smoothing bias of the local polynomial, "
              "so it is narrower than it should be at an MSE-optimal bandwidth.",
            level="caution", code="no_bias_correction",
            explain_key="method.rd.bias_correction",
        )
    rb.add_estimate(f"Conventional (h = {h:.4g})", est, se=se, ci=ci, p_value=p_value,
                    group=what, term="conventional")
    if have_robust:
        rb.add_estimate("Bias-corrected, robust interval", est_bc, se=se_rb, ci=ci_rb,
                        p_value=p_rb, group=what, term="robust")


def _diag_bias_correction(rb: ResultBuilder, prep: _Prep, *, est: float, ci: tuple,
                          est_bc: float | None, se_rb: float | None, ci_rb: tuple | None,
                          se: float | None, h: float, b: float | None, what: str) -> None:
    rows = [
        {"label": "Conventional", "estimate": est, "ci_low": ci[0], "ci_high": ci[1],
         "se": se, "engine": "python"},
    ]
    if est_bc is not None and ci_rb is not None:
        rows.append({"label": "Bias-corrected, robust CI", "estimate": est_bc,
                     "ci_low": ci_rb[0], "ci_high": ci_rb[1], "se": se_rb, "engine": "python"})
    art = rb.artifact(
        "vega", title="Conventional and bias-corrected",
        spec=vega.forest(rows, title="The same jump, two ways of accounting for smoothing bias",
                         x_title=what),
        data=rows, columns=["label", "estimate", "se", "ci_low", "ci_high"],
        explain_key="method.rd.bias_correction",
        caption=("The local polynomial is fitted over a window, so it is slightly biased towards the "
                 "shape of the function inside that window. The bias-corrected row estimates that bias "
                 "with a higher-order fit and widens the interval to account for having estimated it."),
    )
    if est_bc is None:
        rb.add_diagnostic(
            "bias_correction", "Smoothing bias of the local polynomial",
            status="untested",
            summary="The higher-order pilot fit needed to estimate the smoothing bias could not be "
                    "formed here, so only the conventional estimator is reported.",
            worry_when="A conventional interval at an MSE-optimal bandwidth is narrower than the "
                       "evidence, because the bias is the same order as the standard error.",
            artifact_ids=[art], explain_key="method.rd.bias_correction",
        )
        return
    shift = est_bc - est
    ratio = abs(shift) / se if se and se > 0 else float("nan")
    big = bool(np.isfinite(ratio) and ratio > 1.0)
    rb.add_diagnostic(
        "bias_correction", "Smoothing bias of the local polynomial",
        status="weakens" if big else "info",
        summary=(f"Correcting the smoothing bias moves the estimate by {shift:+.4g} "
                 + (f"({ratio:.2f} conventional standard errors) " if np.isfinite(ratio) else "")
                 + f"and widens the interval to {_ci_text(ci_rb)}."),
        worry_when=("A correction bigger than a standard error. It means the fitted curve is doing real "
                    "work inside the window, and the answer depends on the polynomial you chose."),
        artifact_ids=[art],
        values={"conventional": est, "bias_corrected": est_bc, "shift": shift,
                "shift_in_conventional_se": (float(ratio) if np.isfinite(ratio) else None),
                "h": h, "b": b},
        explain_key="method.rd.bias_correction",
    )
    if big:
        rb.add_warning(
            f"The bias correction moves the estimate by {shift:+.4g}, more than one conventional "
            "standard error. Look hard at the binned scatter and at the bandwidth path.",
            level="caution", code="large_bias_correction",
            explain_key="method.rd.bias_correction",
        )


# ---------------------------------------------------------------------------
# rd.local_linear -- sharp RD
# ---------------------------------------------------------------------------


@adapter("rd.local_linear", label="Sharp RD (local polynomial at the cutoff)", package=PACKAGE,
         needs=("numpy", "pandas"))
def local_linear(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Sharp RD (local polynomial at the cutoff)",
                       package=PACKAGE, package_version=PACKAGE_VERSION)
    ctx.tick(0.02, "reading the design")
    prep = _prepare(ctx, default_order=1, deriv=0)
    adjust = _opt_bool(ctx, "covariate_adjust", False)
    balance_cols = list(prep.covar_cols)
    if not adjust:
        # Covariates are here to be checked for jumps, not to be controlled for.
        prep.covars = None
    _wire(
        rb, prep, estimand="LATE",
        estimand_label=(f"For units right at the {prep.running} cutoff of {prep.cutoff:g}, what did "
                        f"crossing it do to {prep.outcome}?"),
    )
    if adjust and balance_cols:
        rb.add_warning(
            "The covariates are entering the local regression as controls. In RD they are normally "
            "used to check the design rather than to fix it; adjusting changes the estimand slightly "
            "and can hide a jump the covariates would otherwise have revealed.",
            level="info", code="rd_covariate_adjustment",
        )

    ctx.tick(0.05, "choosing a bandwidth")
    h, b, selector, bwinfo = _choose_bandwidths(prep, order=prep.order, deriv=0, y=prep.y)
    ctx.tick(0.45, "fitting the local polynomial")
    res = _rd_fit(
        prep.xc, prep.y, h=h, b=b, order=prep.order, deriv=0, kernel_fn=prep.kernel_fn,
        covars=prep.covars, covar_names=prep.covar_names, cluster_codes=prep.cluster_codes,
        level=prep.level, vce=prep.vce,
    )
    _flow_bandwidth(rb, prep, res)
    _set_headline(
        rb, prep, est=res.tau, se=res.se, ci=res.ci, p_value=res.p_value, statistic=res.z,
        est_bc=res.tau_bc, se_rb=res.se_rb, ci_rb=res.ci_rb, p_rb=res.p_rb, z_rb=res.z_rb,
        selector=selector, h=h, b=b, vcov_type=res.vcov_type,
        what=f"Effect on {prep.outcome}", bias_note=res.bias_note,
    )

    ctx.tick(0.55, "drawing the discontinuity")
    art_plot = _hero_plot(rb, ctx, prep, res)
    _diag_discontinuity_plot(rb, prep, res, art_plot)
    _diag_bias_correction(
        rb, prep, est=res.tau, ci=res.ci, est_bc=res.tau_bc, se_rb=res.se_rb, ci_rb=res.ci_rb,
        se=res.se, h=h, b=b, what=f"Effect on {prep.outcome}",
    )

    ctx.tick(0.65, "walking the bandwidth")

    def _at(hh: float) -> tuple[float, float | None, float | None] | None:
        fit = _safe_fit(prep, prep.y, h=hh, order=prep.order, deriv=0, covars=prep.covars)
        return None if fit is None else (fit.tau, fit.ci[0], fit.ci[1])

    _diag_bandwidth_path(
        rb, prep, estimate_at=_at, h_star=h,
        ci_star=(rb.result["ci_low"], rb.result["ci_high"]),
        y_title=f"Jump in {prep.outcome}",
    )

    ctx.tick(0.8, "checking the running variable")
    _diag_discreteness(rb, prep, h)
    _diag_density(rb, prep)

    ctx.tick(0.88, "checking covariate balance")
    _maybe_covariate_balance(rb, prep, h=h, order=prep.order, columns=balance_cols)

    ctx.tick(0.94, "placebo cutoffs")
    _sens_placebo_cutoffs(rb, prep, h=h, order=prep.order, y=prep.y, actual=res.tau,
                          y_title=f"Jump in {prep.outcome}")

    body = [
        f"{'':<30}{'estimate':>12}{'SE':>12}{'  ' + str(int(prep.level * 100)) + '% interval':>26}{'p':>10}",
        f"{'Conventional':<30}{res.tau:>12.6g}{res.se:>12.6g}{_ci_text(res.ci):>26}{res.p_value:>10.4f}",
    ]
    if res.tau_bc is not None and res.se_rb:
        body.append(
            f"{'Bias-corrected (robust)':<30}{res.tau_bc:>12.6g}{res.se_rb:>12.6g}"
            f"{_ci_text(res.ci_rb):>26}{(res.p_rb if res.p_rb is not None else float('nan')):>10.4f}"
        )
    body += [
        "",
        "Headline: the conventional point estimate -- the step you can see in the binned scatter --",
        "reported with the robust bias-corrected interval.",
    ]
    rb.set_classic("\n".join(
        _classic_head(prep, "Sharp regression discontinuity", selector=selector, h=h, b=b,
                      order=prep.order, deriv=0, res=res)
        + body + _classic_tail(rb, prep)
    ))
    rb.set_scripts(python=_script_sharp(prep, h=h, b=b, order=prep.order))
    ctx.tick(1.0, "done")
    return rb.finish()


def _script_sharp(prep: _Prep, *, h: float, b: float | None, order: int, deriv: int = 0) -> str:
    return (
        "from capy_py.contracts import run_method\n"
        f"spec = {{'design': 'rd', 'roles': {{'running': {prep.running!r}, 'cutoff': {prep.cutoff!r},\n"
        f"         'outcome': {prep.outcome!r}"
        + (f", 'treatment': {prep.treatment!r}" if prep.treatment else "")
        + (f", 'confounders': {prep.covar_cols!r}" if prep.covar_cols else "")
        + (f", 'cluster': {prep.cluster_col!r}" if prep.cluster_col else "")
        + "}}\n"
        f"options = {{'bandwidth': {h!r}, 'pilot_bandwidth': {b!r}, 'polynomial_order': {order},\n"
        f"           'kernel': {prep.kernel_name!r}, 'ci_level': {prep.level!r}}}\n"
        "result = run_method('rd.local_linear', spec, df, options=options)\n"
    )


# ---------------------------------------------------------------------------
# rd.fuzzy -- the local Wald ratio
# ---------------------------------------------------------------------------


def _local_wald(prep: _Prep, fy: _RDFit, fd: _RDFit, *, robust: bool) -> dict[str, Any] | None:
    """Ratio of two linear-in-outcome statistics that share the same weights."""
    if robust:
        ell = fy.ell_rb
        num, den = fy.tau_bc, fd.tau_bc
        ry, rd_ = fy.resid_rb, fd.resid_rb
    else:
        ell = fy.ell
        num, den = fy.tau, fd.tau
        ry, rd_ = fy.resid, fd.resid
    if ell is None or num is None or den is None or abs(den) < 1e-12:
        return None
    ratio = float(num) / float(den)
    v_y = _cov_of(ell, ry, ry, prep.cluster_codes)
    v_d = _cov_of(ell, rd_, rd_, prep.cluster_codes)
    c_yd = _cov_of(ell, ry, rd_, prep.cluster_codes)
    var = (v_y - 2.0 * ratio * c_yd + ratio * ratio * v_d) / (float(den) ** 2)
    se = math.sqrt(max(var, 0.0))
    if not np.isfinite(se) or se <= 0:
        return {"estimate": ratio, "se": None, "ci": (None, None), "z": None, "p": None}
    z = ratio / se
    return {"estimate": ratio, "se": se, "ci": stats.wald_ci(ratio, se, prep.level),
            "z": z, "p": float(stats.norm_sf2(z))}


def _diag_first_stage(rb: ResultBuilder, prep: _Prep, fd: _RDFit, art: str) -> dict[str, Any]:
    order = fd.order
    take_left = float(fd.beta[0])
    take_right = float(fd.beta[order + 1])
    t = fd.z
    weak = bool(abs(fd.tau) < FIRST_STAGE_WARN
                or (np.isfinite(t) and abs(t) < FIRST_STAGE_T_WARN))
    values = {
        "first_stage_jump": fd.tau, "se": fd.se, "statistic": (float(t) if np.isfinite(t) else None),
        "p_value": (float(fd.p_value) if np.isfinite(fd.p_value) else None),
        "ci_low": fd.ci[0], "ci_high": fd.ci[1],
        "take_up_just_below": take_left, "take_up_just_above": take_right,
        "bandwidth": fd.h,
    }
    rb.add_diagnostic(
        "first_stage", "Does crossing the cutoff actually change who is treated?",
        status="weakens" if weak else "supports",
        summary=(f"Take-up rises from about {take_left:.1%} just below the cutoff to {take_right:.1%} "
                 f"just above: a jump of {fd.tau:.3f} (SE {fd.se:.3f}, "
                 f"{_ci_text(fd.ci, 3)}). Everything below divides by this number."),
        worry_when=("A jump much smaller than about 0.1, or one that is not clearly different from "
                    "zero. The Wald ratio then divides by almost nothing, the interval explodes, and "
                    "the number can be anything."),
        artifact_ids=[art], values=values, explain_key="diagnostic.rd.first_stage",
    )
    rb.add_assumption(
        "relevance", label="Crossing the cutoff moves treatment",
        status="weakened" if weak else "supported",
        note=(f"The treatment probability jumps by {fd.tau:.3f} at the cutoff."
              + (" That is small enough that the ratio is unstable." if weak else "")),
        diagnostic_ids=["first_stage"], explain_key="assumption.relevance",
    )
    rb.add_assumption(
        "monotonicity", label="No defiers at the cutoff", status="assumed",
        note="Nobody is assumed to take the programme only because they fell just below the cutoff. "
             "Nothing in the data can check this.",
        explain_key="assumption.monotonicity",
    )
    rb.add_assumption(
        "exclusion", label="The cutoff only matters through treatment", status="assumed",
        note=f"Crossing {prep.cutoff:g} on '{prep.running}' is assumed to affect '{prep.outcome}' only "
             f"by changing '{prep.treatment}'. If the threshold also switches on something else, this "
             "number is not a treatment effect.",
        explain_key="assumption.exclusion",
    )
    if weak:
        rb.add_warning(
            f"The treatment jump at the cutoff is only {fd.tau:.3f} (t = "
            + (f"{t:.2f}" if np.isfinite(t) else "n/a")
            + "). A fuzzy RD divides the outcome jump by this, so the estimate and its interval are "
              "both unreliable. Report the reduced form -- the jump in the outcome -- as well.",
            level="warning", code="weak_first_stage", explain_key="diagnostic.rd.first_stage",
        )
        rb.mark_provisional(
            "The first stage at the cutoff is small, so the local Wald ratio is dividing by a number "
            "that is close to zero. Read the reduced form alongside it."
        )
    return values


@adapter("rd.fuzzy", label="Fuzzy RD (local Wald ratio)", package=PACKAGE,
         needs=("numpy", "pandas"))
def fuzzy(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Fuzzy RD (local Wald ratio)",
                       package=PACKAGE, package_version=PACKAGE_VERSION)
    ctx.tick(0.02, "reading the design")
    prep = _prepare(ctx, default_order=1, deriv=0, need_treatment=True)
    adjust = _opt_bool(ctx, "covariate_adjust", False)
    balance_cols = list(prep.covar_cols)
    if not adjust:
        prep.covars = None
    _wire(
        rb, prep, estimand="LATE",
        estimand_label=(f"For units at the {prep.running} cutoff of {prep.cutoff:g} whose "
                        f"{prep.treatment} was switched by crossing it, what did it do to "
                        f"{prep.outcome}?"),
    )
    d = roles.treatment_vector(prep.df, prep.treatment)
    if float(np.std(d)) <= 0:
        raise DataError(
            f"'{prep.treatment}' does not vary in this sample, so there is no first stage to divide by.")

    ctx.tick(0.05, "choosing a bandwidth")
    h, b, selector, bwinfo = _choose_bandwidths(prep, order=prep.order, deriv=0, y=prep.y)
    ctx.tick(0.4, "fitting both sides")
    kw = dict(h=h, b=b, order=prep.order, deriv=0, kernel_fn=prep.kernel_fn,
              covars=prep.covars, covar_names=prep.covar_names,
              cluster_codes=prep.cluster_codes, level=prep.level, vce=prep.vce)
    fy = _rd_fit(prep.xc, prep.y, **kw)
    fd = _rd_fit(prep.xc, d, **kw)
    _flow_bandwidth(rb, prep, fy)

    if abs(fd.tau) < 1e-8:
        raise DataError(
            f"The share treated does not change at the cutoff (jump {fd.tau:.3g}), so a fuzzy RD has "
            "nothing to divide by.",
            detail="Check that the cutoff and the treatment variable belong to the same rule. If "
                   "crossing the cutoff determines treatment exactly, use sharp RD instead.",
        )

    conv = _local_wald(prep, fy, fd, robust=False)
    rob = _local_wald(prep, fy, fd, robust=True)
    if conv is None:
        raise DataError("The local Wald ratio could not be formed at the chosen bandwidth.")
    _set_headline(
        rb, prep, est=conv["estimate"], se=conv["se"], ci=conv["ci"], p_value=conv["p"],
        statistic=conv["z"],
        est_bc=(rob or {}).get("estimate"), se_rb=(rob or {}).get("se"),
        ci_rb=(rob or {}).get("ci"), p_rb=(rob or {}).get("p"), z_rb=(rob or {}).get("z"),
        selector=selector, h=h, b=b, vcov_type=fy.vcov_type,
        what=f"Effect on {prep.outcome} per unit of {prep.treatment}",
        bias_note=fy.bias_note,
    )
    rb.add_estimate("Reduced form: jump in the outcome", fy.tau, se=fy.se, ci=fy.ci,
                    p_value=fy.p_value, group="pieces", term="reduced_form")
    rb.add_estimate("First stage: jump in treatment", fd.tau, se=fd.se, ci=fd.ci,
                    p_value=fd.p_value, group="pieces", term="first_stage")

    ctx.tick(0.55, "drawing the discontinuity")
    art_y = _hero_plot(rb, ctx, prep, fy)
    _diag_discontinuity_plot(rb, prep, fy, art_y, what="outcome (the reduced form)")
    art_d = _hero_plot(rb, ctx, prep, fd, y=d, y_label=f"share with {prep.treatment}",
                       title=f"Take-up of {prep.treatment} against {prep.running}")
    _diag_first_stage(rb, prep, fd, art_d)
    _diag_bias_correction(
        rb, prep, est=conv["estimate"], ci=conv["ci"],
        est_bc=(rob or {}).get("estimate"), se_rb=(rob or {}).get("se"),
        ci_rb=(rob or {}).get("ci"), se=conv["se"], h=h, b=b,
        what=f"Effect on {prep.outcome}",
    )

    ctx.tick(0.7, "walking the bandwidth")

    def _at(hh: float) -> tuple[float, float | None, float | None] | None:
        a = _safe_fit(prep, prep.y, h=hh, order=prep.order, deriv=0, covars=prep.covars)
        c = _safe_fit(prep, d, h=hh, order=prep.order, deriv=0, covars=prep.covars)
        if a is None or c is None:
            return None
        out = _local_wald(prep, a, c, robust=False)
        return None if out is None else (out["estimate"], out["ci"][0], out["ci"][1])

    _diag_bandwidth_path(
        rb, prep, estimate_at=_at, h_star=h,
        ci_star=(rb.result["ci_low"], rb.result["ci_high"]),
        y_title=f"LATE on {prep.outcome}",
    )

    ctx.tick(0.85, "checking the running variable")
    _diag_discreteness(rb, prep, h)
    _diag_density(rb, prep)
    _maybe_covariate_balance(rb, prep, h=h, order=prep.order, columns=balance_cols)
    ctx.tick(0.94, "placebo cutoffs")
    # The placebo is run on the reduced form: at a fake cutoff there is no first
    # stage to divide by, so a ratio there would be noise over noise.
    _sens_placebo_cutoffs(rb, prep, h=h, order=prep.order, y=prep.y, actual=fy.tau,
                          y_title=f"Jump in {prep.outcome} (reduced form)")

    body = [
        f"{'':<30}{'estimate':>12}{'SE':>12}{'  ' + str(int(prep.level * 100)) + '% interval':>26}{'p':>10}",
        f"{'Reduced form (outcome)':<30}{fy.tau:>12.6g}{fy.se:>12.6g}{_ci_text(fy.ci):>26}"
        f"{fy.p_value:>10.4f}",
        f"{'First stage (treatment)':<30}{fd.tau:>12.6g}{fd.se:>12.6g}{_ci_text(fd.ci):>26}"
        f"{fd.p_value:>10.4f}",
        f"{'LATE, conventional':<30}{conv['estimate']:>12.6g}"
        f"{(conv['se'] if conv['se'] else float('nan')):>12.6g}{_ci_text(conv['ci']):>26}"
        f"{(conv['p'] if conv['p'] is not None else float('nan')):>10.4f}",
    ]
    if rob:
        body.append(
            f"{'LATE, bias-corrected':<30}{rob['estimate']:>12.6g}"
            f"{(rob['se'] if rob['se'] else float('nan')):>12.6g}{_ci_text(rob['ci']):>26}"
            f"{(rob['p'] if rob['p'] is not None else float('nan')):>10.4f}"
        )
    body += [
        "",
        "The LATE is the reduced form divided by the first stage, with a delta-method standard error",
        "that uses the covariance between the two (both are the same weighted contrast applied to a",
        "different column, so that covariance is exact rather than assumed away).",
    ]
    rb.set_classic("\n".join(
        _classic_head(prep, "Fuzzy regression discontinuity (local Wald ratio)", selector=selector,
                      h=h, b=b, order=prep.order, deriv=0, res=fy)
        + body + _classic_tail(rb, prep)
    ))
    rb.set_scripts(python=_script_sharp(prep, h=h, b=b, order=prep.order).replace(
        "rd.local_linear", "rd.fuzzy"))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# rd.density_test -- manipulation at the cutoff
# ---------------------------------------------------------------------------


@adapter("rd.density_test", label="Manipulation test at the cutoff (McCrary)", package=PACKAGE,
         needs=("numpy", "pandas"))
def density_test(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Manipulation test at the cutoff (McCrary)",
                       package=PACKAGE, package_version=PACKAGE_VERSION)
    ctx.tick(0.05, "reading the design")
    prep = _prepare(ctx, default_order=1, deriv=0, use_covariates=False)
    prep.covars = None
    _wire(
        rb, prep, estimand=None,
        estimand_label=(f"Do more units sit just above the {prep.running} cutoff of {prep.cutoff:g} "
                        f"than just below? A step in the density is what sorting across the threshold "
                        f"looks like."),
    )
    user_bw = _opt_float(ctx, "density_bandwidth", None, minimum=0.0)
    ctx.tick(0.3, "estimating the density either side")
    out = _mccrary(prep.xc + prep.cutoff, prep.cutoff, user_bw)
    _diag_density(rb, prep, bandwidth=user_bw)

    hd = float(out["bandwidth"])
    inside = int(np.sum(np.abs(prep.xc) <= hd))
    rb.add_flow(
        "Inside the density-test bandwidth", inside,
        n_treated=int(np.sum((prep.xc >= 0) & (np.abs(prep.xc) <= hd))),
        n_control=int(np.sum((prep.xc < 0) & (np.abs(prep.xc) <= hd))),
        dropped=int(prep.n - inside),
        reason=(f"|{prep.running} - {prep.cutoff:g}| <= {hd:.6g}; the local linear density fit uses "
                f"histogram bins of width {out['binsize']:.6g} inside this window"),
    )
    rb.set_counts(n_effective=inside)

    theta = out.get("theta")
    se = out.get("se")
    if theta is None or se is None:
        raise DataError(
            "The density either side of the cutoff could not be compared.",
            detail=out.get("note") or "The local linear density estimate was not positive on one side.",
        )
    ci = stats.wald_ci(float(theta), float(se), prep.level)
    rb.set_estimate(
        float(theta), se=float(se), ci=ci, p_value=out.get("p_value"),
        statistic=out.get("statistic"),
        inference=(f"McCrary (2008) local linear density discontinuity, implemented here; "
                   f"bin width {out['binsize']:.4g}, density bandwidth {hd:.4g}"),
        ci_level=prep.level,
    )
    rb.add_estimate("Density just below the cutoff", out["f_left"])
    rb.add_estimate("Density just above the cutoff", out["f_right"])
    rb.add_estimate("Ratio above / below", math.exp(float(theta)))

    ctx.tick(0.6, "the picture at the cutoff")
    res = _safe_fit(prep, prep.y, h=_opt_float(ctx, "bandwidth", None, minimum=0.0) or hd,
                    order=prep.order, deriv=0)
    if res is not None:
        art = _hero_plot(rb, ctx, prep, res)
        _diag_discontinuity_plot(rb, prep, res, art)
    else:
        # No fit was possible, but the plot is the point of an RD screen, so the
        # bin means still go out -- without a fitted curve drawn through them.
        art = _raw_scatter_artifact(rb, ctx, prep)
        rb.add_diagnostic(
            "rd_plot", "The picture at the cutoff",
            status="untested",
            summary=("Bin means either side of the cutoff. No local polynomial could be fitted here, so "
                     "no fitted curve is drawn and no step is estimated."),
            worry_when="Bins that jump around as much away from the cutoff as at it.",
            artifact_ids=[art], explain_key="plot.rd.binned_scatter",
        )
    _diag_discreteness(rb, prep, hd)

    ctx.tick(0.85, "walking the density bandwidth")

    def _density_at(hh: float) -> tuple[float, float | None, float | None] | None:
        try:
            alt = _mccrary(prep.xc + prep.cutoff, prep.cutoff, hh)
        except (DataError, SpecError):
            return None
        if alt.get("theta") is None or alt.get("se") is None:
            return None
        aci = stats.wald_ci(float(alt["theta"]), float(alt["se"]), prep.level)
        return (float(alt["theta"]), aci[0], aci[1])

    _diag_bandwidth_path(
        rb, prep, estimate_at=_density_at, h_star=hd, ci_star=ci,
        y_title="Log difference in density (above - below)", points=9,
    )

    body = [
        f"Bin width        : {out['binsize']:.6g}   ({out['n_bins']} bins over the support)",
        f"Density bandwidth: {hd:.6g}" + ("  (user supplied)" if user_bw else "  (McCrary rule of thumb)"),
        f"Observations     : {out['n']} with a usable {prep.running}, {inside} inside the bandwidth",
        "",
        f"Density just below the cutoff : {out['f_left']:.6g}",
        f"Density just above the cutoff : {out['f_right']:.6g}",
        f"Log difference (above - below): {theta:.6g}   SE {se:.6g}",
        f"z = {out['statistic']:.4f}    p = {out['p_value']:.4f}",
        f"{int(prep.level * 100)}% interval for the log difference: {_ci_text(ci)}",
        "",
        "A ratio near one, and a p-value that is not small, is the pattern you want here. It does not",
        "prove nobody moved across the cutoff; it says this test did not find them.",
    ]
    rb.set_classic("\n".join(
        [
            "Manipulation test at the cutoff (McCrary 2008)",
            "=============================================",
            f"Running variable : {prep.running}   cutoff = {prep.cutoff:g}",
            "",
        ] + body + _classic_tail(rb, prep)
    ))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# rd.covariate_balance -- the same estimator, one covariate at a time
# ---------------------------------------------------------------------------


@adapter("rd.covariate_balance", label="Covariate balance at the cutoff", package=PACKAGE,
         needs=("numpy", "pandas"))
def covariate_balance(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Covariate balance at the cutoff",
                       package=PACKAGE, package_version=PACKAGE_VERSION)
    ctx.tick(0.03, "reading the design")
    prep = _prepare(ctx, default_order=1, deriv=0)
    if not prep.covar_cols:
        raise SpecError(
            "This check needs covariates to look at, and none are set.",
            detail="Drop the variables the programme cannot possibly have changed -- age, sex, "
                   "pre-programme measurements -- onto the covariates zone of the RD board.",
        )
    prep.covars = None  # here the covariates are the outcomes, never the controls
    _wire(
        rb, prep, estimand=None,
        estimand_label=(f"Do the covariates jump at the {prep.running} cutoff of {prep.cutoff:g}? "
                        f"If they do, something other than eligibility changes there and the outcome "
                        f"jump is not the programme."),
    )
    ctx.tick(0.1, "choosing a bandwidth")
    h, b, selector, bwinfo = _choose_bandwidths(prep, order=prep.order, deriv=0, y=prep.y)
    ctx.tick(0.4, "one covariate at a time")
    jumps = _covariate_jumps(prep, h=h, order=prep.order)

    res = _rd_fit(prep.xc, prep.y, h=h, b=None, order=prep.order, deriv=0,
                  kernel_fn=prep.kernel_fn, cluster_codes=prep.cluster_codes,
                  level=prep.level, vce=prep.vce)
    _flow_bandwidth(rb, prep, res)

    rb.set_estimate(
        None, se=None, ci=None, p_value=jumps.get("p_value"), statistic=jumps.get("wald"),
        inference=(f"joint chi-square test that all {jumps['df']} covariate jumps are zero, from the "
                   f"exact covariance of the jumps (they share one weighted contrast); "
                   f"{_selector_text(selector, h, None)}; {res.vcov_type} variance"),
        ci_level=prep.level,
    )
    for row in jumps["rows"]:
        rb.add_estimate(f"Jump in {row['variable']}", row["jump"], se=row["se"],
                        ci=(row["ci_low"], row["ci_high"]), p_value=row["p_value"],
                        group="covariate", term=row["variable"])

    ctx.tick(0.6, "the picture at the cutoff")
    bal = _diag_covariate_balance(rb, prep, jumps)
    art = _hero_plot(rb, ctx, prep, res)
    _diag_discontinuity_plot(rb, prep, res, art)

    worst = max(jumps["rows"],
                key=lambda r: abs(r["standardised_jump"] or 0.0)) if jumps["rows"] else None
    if worst is not None:
        dm = stats.design_matrix(prep.df, prep.covar_cols, intercept=False)
        col = dm.names.index(worst["variable"])
        z = np.asarray(dm.X[:, col], dtype=float)
        sd = float(np.std(z, ddof=1)) or 1.0

        def _at(hh: float) -> tuple[float, float | None, float | None] | None:
            fit = _safe_fit(prep, z, h=hh, order=prep.order, deriv=0)
            if fit is None:
                return None
            return (fit.tau / sd, fit.ci[0] / sd, fit.ci[1] / sd)

        ci_star = ((worst["ci_low"] / sd) if worst["ci_low"] is not None else None,
                   (worst["ci_high"] / sd) if worst["ci_high"] is not None else None)
        ctx.tick(0.75, "walking the bandwidth")
        _diag_bandwidth_path(
            rb, prep, estimate_at=_at, h_star=h, ci_star=ci_star,
            y_title=f"Standardised jump in {worst['variable']}",
        )
    ctx.tick(0.9, "checking the running variable")
    _diag_discreteness(rb, prep, h)

    header = f"{'covariate':<26}{'jump':>12}{'SE':>12}{'std. jump':>12}{'p':>10}"
    body = [header, "-" * len(header)]
    for row in jumps["rows"]:
        body.append(
            f"{row['variable'][:26]:<26}{row['jump']:>12.6g}"
            f"{(row['se'] if row['se'] else float('nan')):>12.6g}"
            f"{(row['standardised_jump'] if row['standardised_jump'] is not None else float('nan')):>12.4f}"
            f"{(row['p_value'] if row['p_value'] is not None else float('nan')):>10.4f}"
        )
    body += [
        "",
        f"Joint test that every jump is zero: chi-square {_fmt(jumps.get('wald'), 6)} on "
        f"{jumps['df']} d.f., p = {_fmt(jumps.get('p_value'), 4)}",
        "",
        "Each row is the same local polynomial used for the outcome, with that covariate as the",
        "outcome instead. A significant jump weakens continuity: it says the units either side of the",
        "cutoff differ in something the programme did not cause.",
    ]
    rb.set_classic("\n".join(
        _classic_head(prep, "Covariate balance at the cutoff", selector=selector, h=h, b=None,
                      order=prep.order, deriv=0, res=res)
        + body + _classic_tail(rb, prep)
    ))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# rd.kink -- the estimand is the change in slope
# ---------------------------------------------------------------------------


@adapter("rd.kink", label="Regression kink (change in slope at the cutoff)", package=PACKAGE,
         needs=("numpy", "pandas"))
def kink(ctx: RunContext) -> dict[str, Any]:
    rb = ResultBuilder(ctx, method_label="Regression kink (change in slope at the cutoff)",
                       package=PACKAGE, package_version=PACKAGE_VERSION)
    ctx.tick(0.02, "reading the design")
    prep = _prepare(ctx, default_order=2, deriv=1)
    adjust = _opt_bool(ctx, "covariate_adjust", False)
    balance_cols = list(prep.covar_cols)
    if not adjust:
        prep.covars = None
    _wire(
        rb, prep, estimand="slope_change",
        estimand_label=(f"At {prep.running} = {prep.cutoff:g} the rule changes slope, not level. How "
                        f"much does the slope of {prep.outcome} with respect to {prep.running} change "
                        f"there?"),
    )
    rb.add_warning(
        "A kink design reads the change in slope at the threshold, so the number is per unit of "
        f"'{prep.running}', not a level effect. Kink estimates are far noisier than jump estimates and "
        "lean harder on the smoothness of everything else at the threshold.",
        level="info", code="kink_estimand", explain_key="method.rd.kink",
    )

    ctx.tick(0.05, "choosing a bandwidth")
    h, b, selector, bwinfo = _choose_bandwidths(prep, order=prep.order, deriv=1, y=prep.y)
    ctx.tick(0.45, "fitting the local polynomial")
    res = _rd_fit(
        prep.xc, prep.y, h=h, b=b, order=prep.order, deriv=1, kernel_fn=prep.kernel_fn,
        covars=prep.covars, covar_names=prep.covar_names, cluster_codes=prep.cluster_codes,
        level=prep.level, vce=prep.vce,
    )
    _flow_bandwidth(rb, prep, res)
    _set_headline(
        rb, prep, est=res.tau, se=res.se, ci=res.ci, p_value=res.p_value, statistic=res.z,
        est_bc=res.tau_bc, se_rb=res.se_rb, ci_rb=res.ci_rb, p_rb=res.p_rb, z_rb=res.z_rb,
        selector=selector, h=h, b=b, vcov_type=res.vcov_type,
        what=f"Change in the slope of {prep.outcome}", bias_note=res.bias_note,
    )
    order = prep.order
    slope_left = float(res.beta[1]) / h
    slope_right = float(res.beta[order + 2]) / h
    rb.add_estimate("Slope just below the cutoff", slope_left, group="pieces", term="slope_left")
    rb.add_estimate("Slope just above the cutoff", slope_right, group="pieces", term="slope_right")

    ctx.tick(0.55, "drawing the kink")
    art = _hero_plot(rb, ctx, prep, res)
    rb.add_diagnostic(
        "rd_plot", "The picture at the kink",
        status="info",
        summary=(f"Bin means either side of {prep.cutoff:g} with the fitted order-{order} polynomial. "
                 f"The slope goes from {slope_left:.4g} below the cutoff to {slope_right:.4g} above it, "
                 f"a change of {res.tau:.4g}. Look for a bend, not a step."),
        worry_when=("A visible step in the level at the cutoff -- that is a discontinuity design, not a "
                    "kink -- or bins whose slope wanders as much away from the cutoff as at it."),
        artifact_ids=[art], explain_key="plot.rd.binned_scatter",
        values={"slope_below": slope_left, "slope_above": slope_right,
                "slope_change": res.tau, "bandwidth": res.h,
                "n_inside_bandwidth": res.n_effective},
    )
    _diag_bias_correction(
        rb, prep, est=res.tau, ci=res.ci, est_bc=res.tau_bc, se_rb=res.se_rb, ci_rb=res.ci_rb,
        se=res.se, h=h, b=b, what=f"Change in the slope of {prep.outcome}",
    )

    # A kink design assumes the level is continuous. Check it.
    lvl = _safe_fit(prep, prep.y, h=h, order=prep.order, deriv=0, covars=prep.covars)
    if lvl is not None:
        rows = [{"label": "Level jump at the cutoff", "estimate": lvl.tau,
                 "ci_low": lvl.ci[0], "ci_high": lvl.ci[1], "se": lvl.se, "engine": "python"}]
        art_lvl = rb.artifact(
            "vega", title="Level jump at the kink",
            spec=vega.forest(rows, title="Is there a step as well as a bend?",
                             x_title=f"Jump in {prep.outcome}"),
            data=rows, columns=["label", "estimate", "se", "ci_low", "ci_high"],
            explain_key="diagnostic.rd.level_jump",
            caption="A kink design assumes the level is continuous at the threshold and only the slope "
                    "changes. A significant step here means it is not a pure kink.",
        )
        stepped = bool(np.isfinite(lvl.p_value) and lvl.p_value < 0.05)
        rb.add_diagnostic(
            "level_jump_at_kink", "Is the level continuous at the kink?",
            status="weakens" if stepped else "supports",
            summary=(f"Fitting the same window for a step rather than a bend gives {lvl.tau:.4g} "
                     f"(SE {lvl.se:.4g}, p = {lvl.p_value:.3f})."
                     + (" That is a step, so this threshold changes the level as well as the slope."
                        if stepped else
                        " No step worth speaking of, which is what a pure kink design needs.")),
            worry_when=("A significant step. Then the rule changes the level too, and the slope change "
                        "on its own is not the whole story -- run the sharp RD as well."),
            artifact_ids=[art_lvl],
            values={"level_jump": lvl.tau, "se": lvl.se, "p_value": lvl.p_value},
            explain_key="diagnostic.rd.level_jump",
        )
        if stepped:
            rb.add_warning(
                f"The outcome also steps by {lvl.tau:.4g} at the threshold (p = {lvl.p_value:.3f}). A "
                "kink design assumes only the slope changes; run the sharp RD alongside this.",
                level="warning", code="level_jump_at_kink",
                explain_key="diagnostic.rd.level_jump",
            )

    ctx.tick(0.7, "walking the bandwidth")

    def _at(hh: float) -> tuple[float, float | None, float | None] | None:
        fit = _safe_fit(prep, prep.y, h=hh, order=prep.order, deriv=1, covars=prep.covars)
        return None if fit is None else (fit.tau, fit.ci[0], fit.ci[1])

    _diag_bandwidth_path(
        rb, prep, estimate_at=_at, h_star=h,
        ci_star=(rb.result["ci_low"], rb.result["ci_high"]),
        y_title=f"Slope change in {prep.outcome}",
    )

    ctx.tick(0.85, "checking the running variable")
    _diag_discreteness(rb, prep, h)
    _diag_density(rb, prep)
    _maybe_covariate_balance(rb, prep, h=h, order=prep.order, columns=balance_cols)

    body = [
        f"{'':<30}{'estimate':>12}{'SE':>12}{'  ' + str(int(prep.level * 100)) + '% interval':>26}{'p':>10}",
        f"{'Slope change, conventional':<30}{res.tau:>12.6g}{res.se:>12.6g}{_ci_text(res.ci):>26}"
        f"{res.p_value:>10.4f}",
    ]
    if res.tau_bc is not None and res.se_rb:
        body.append(
            f"{'Slope change, bias-corrected':<30}{res.tau_bc:>12.6g}{res.se_rb:>12.6g}"
            f"{_ci_text(res.ci_rb):>26}{(res.p_rb if res.p_rb is not None else float('nan')):>10.4f}"
        )
    body += [
        "",
        f"Slope just below the cutoff : {slope_left:.6g}",
        f"Slope just above the cutoff : {slope_right:.6g}",
        "",
        f"The estimand is the change in slope: units of {prep.outcome} per unit of {prep.running}.",
        "It is not a level effect and must not be reported as one.",
    ]
    rb.set_classic("\n".join(
        _classic_head(prep, "Regression kink design", selector=selector, h=h, b=b,
                      order=prep.order, deriv=1, res=res)
        + body + _classic_tail(rb, prep)
    ))
    ctx.tick(1.0, "done")
    return rb.finish()


# ---------------------------------------------------------------------------
# Method cards -- what the UI renders before anything is run
# ---------------------------------------------------------------------------

_RD_ROLES = {
    "roles_required": ["running", "cutoff", "outcome"],
    "roles_optional": ["confounders", "cluster", "treatment"],
    "roles_forbidden": ["instruments", "forbidden"],
}

_OPT_KERNEL = {
    "name": "kernel", "type": "select", "default": DEFAULT_KERNEL,
    "choices": ["triangular", "epanechnikov", "uniform"],
    "label": "Weighting near the cutoff",
    "help": "Triangular gives the most weight to units closest to the cutoff, which is what the "
            "MSE-optimal theory assumes. Uniform is the plain 'everyone inside the window counts "
            "equally' choice.",
    "profile": "advanced",
}
_OPT_SELECTOR = {
    "name": "bandwidth_selector", "type": "select", "default": "mse", "choices": ["mse", "ik"],
    "label": "How to choose the window",
    "help": "'mse' is capy's own finite-sample plug-in rule: it measures the bias and variance of this "
            "estimator on this sample. 'ik' is the Imbens-Kalyanaraman (2012) formula, for a sharp "
            "local linear fit only. Neither is rdrobust's CCT selector.",
    "profile": "standard",
}
_OPT_BANDWIDTH = {
    "name": "bandwidth", "type": "number", "default": None, "min": 0.0,
    "label": "Bandwidth (leave empty to choose one)",
    "help": "The half-width of the window either side of the cutoff, in the units of the running "
            "variable. Setting it by hand overrides the selector; the sensitivity path shows you what "
            "every other choice would have given.",
    "profile": "standard",
}
_OPT_PILOT = {
    "name": "pilot_bandwidth", "type": "number", "default": None, "min": 0.0,
    "label": "Bias-correction pilot bandwidth",
    "help": "The wider window used by the higher-order fit that estimates the smoothing bias.",
    "profile": "advanced",
}
_OPT_DONUT = {
    "name": "donut", "type": "number", "default": 0.0, "min": 0.0,
    "label": "Donut hole radius",
    "help": "Drop units within this distance of the cutoff, where heaping and manipulation live. Every "
            "dropped row appears in the sample flow.",
    "profile": "standard",
}
_OPT_LEVEL = {
    "name": "ci_level", "type": "number", "default": 0.95, "min": 0.5, "max": 0.999,
    "label": "Confidence level", "help": "0.95 unless you have a reason.", "profile": "advanced",
}
_OPT_VCE = {
    "name": "vce", "type": "select", "default": "HC1", "choices": ["HC0", "HC1", "HC2", "HC3"],
    "label": "Standard errors",
    "help": "Heteroskedasticity-robust. HC3 is more conservative in small windows. Set the cluster role "
            "as well and the variance becomes cluster-robust.",
    "profile": "advanced",
}
_OPT_PLOT_BINS = {
    "name": "plot_bins", "type": "int", "default": 0, "min": 0, "max": 100,
    "label": "Bins per side in the plot", "help": "0 picks about 2 n^(1/3) bins a side.",
    "profile": "advanced",
}
_OPT_PLOT_SPACING = {
    "name": "plot_bin_spacing", "type": "select", "default": "quantile",
    "choices": ["quantile", "evenly"], "label": "Bin spacing",
    "help": "Quantile bins hold the same number of units each; even bins hold the same width each.",
    "profile": "advanced",
}
_OPT_COVADJ = {
    "name": "covariate_adjust", "type": "bool", "default": False,
    "label": "Put the covariates in the local regression",
    "help": "Off by default: in RD the covariates are how you check the design, not how you fix it. "
            "Turning this on tightens the interval a little and slightly changes the estimand.",
    "profile": "advanced",
}


def _order_option(default: int, minimum: int, help_text: str) -> dict[str, Any]:
    return {
        "name": "polynomial_order", "type": "int", "default": default, "min": minimum, "max": 4,
        "label": "Polynomial order either side", "help": help_text, "profile": "standard",
    }


_RD_REFERENCES = [
    "Imbens & Lemieux (2008), Regression discontinuity designs: a guide to practice",
    "Calonico, Cattaneo & Titiunik (2014), Robust nonparametric confidence intervals for "
    "regression-discontinuity designs",
    "Imbens & Kalyanaraman (2012), Optimal bandwidth choice for the regression discontinuity estimator",
    "Cattaneo, Idrobo & Titiunik (2020), A practical introduction to regression discontinuity designs",
]

_NOT_RDROBUST_CARD = (
    "This is capy.py's own local-polynomial code, not rdrobust. The bandwidth selector is named in the "
    "Classic tab and the variance is heteroskedasticity- or cluster-robust rather than rdrobust's "
    "nearest-neighbour estimator, so numbers will be close but need not match to the last digit."
)

METHOD_CARDS: list[dict[str, Any]] = [
    {
        "id": "rd.local_linear",
        "title": "Sharp RD at the cutoff",
        "one_liner": "Fit a line either side of the threshold and measure the step between them.",
        "designs": [DESIGN],
        "estimands": ["LATE"],
        **_RD_ROLES,
        "options": [
            _order_option(1, 1, "1 is the default and the one the literature defends. Higher orders "
                                "fit more wiggle inside the window and are easy to over-read."),
            _OPT_SELECTOR, _OPT_BANDWIDTH, _OPT_DONUT, _OPT_KERNEL, _OPT_PILOT,
            _OPT_COVADJ, _OPT_VCE, _OPT_LEVEL, _OPT_PLOT_BINS, _OPT_PLOT_SPACING,
        ],
        "diagnostics": ["rd_plot", "bias_correction", "bandwidth_sensitivity",
                        "running_variable_discreteness", "manipulation", "covariate_balance"],
        "probes": ["probe.placebo_outcome", "probe.negative_control", "probe.subset",
                   "probe.alternate_spec"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.rd.local_linear",
        "status": "recommended",
        "why_recommended": "When a cutoff really did decide eligibility, the units either side of it are "
                           "as good as randomly assigned, and you can see the whole argument in one "
                           "plot. Nothing else in observational work is this legible.",
        "what_can_go_wrong": "The answer is only about units at the cutoff, and it moves with the "
                            "bandwidth and the polynomial order. If people could see the threshold and "
                            "push themselves across it, the comparison is gone -- which is why the "
                            "density test and the covariate jumps travel with the estimate. "
                            + _NOT_RDROBUST_CARD,
        "needs_overlap": False,
        "engines": {"python": True, "r": "rdrobust"},
        "references": _RD_REFERENCES,
        "disrecommend_when": "The running variable has fewer than about 20 distinct values either side "
                             "of the cutoff inside the bandwidth",
    },
    {
        "id": "rd.fuzzy",
        "title": "Fuzzy RD (crossing the cutoff only nudged people in)",
        "one_liner": "Divide the jump in the outcome by the jump in take-up: the effect for the people "
                     "the cutoff actually moved.",
        "designs": [DESIGN],
        "estimands": ["LATE"],
        **{**_RD_ROLES, "roles_required": ["running", "cutoff", "outcome", "treatment"],
           "roles_optional": ["confounders", "cluster"]},
        "options": [
            _order_option(1, 1, "1 is the default. Higher orders fit more wiggle inside the window."),
            _OPT_SELECTOR, _OPT_BANDWIDTH, _OPT_DONUT, _OPT_KERNEL, _OPT_PILOT,
            _OPT_COVADJ, _OPT_VCE, _OPT_LEVEL, _OPT_PLOT_BINS, _OPT_PLOT_SPACING,
        ],
        "diagnostics": ["rd_plot", "first_stage", "bias_correction", "bandwidth_sensitivity",
                        "running_variable_discreteness", "manipulation", "covariate_balance"],
        "probes": ["probe.placebo_outcome", "probe.negative_control", "probe.subset",
                   "probe.alternate_spec"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.rd.fuzzy",
        "status": "recommended",
        "why_recommended": "The honest method when the cutoff changed the odds of getting the programme "
                           "rather than settling it. The first stage is shown as its own plot, not "
                           "buried in a table.",
        "what_can_go_wrong": "It divides by the jump in take-up. A small first stage makes the interval "
                            "explode and the point estimate meaningless, and the answer only describes "
                            "the people the cutoff moved -- not everyone at the cutoff, and certainly "
                            "not everyone. " + _NOT_RDROBUST_CARD,
        "needs_overlap": False,
        "engines": {"python": True, "r": "rdrobust"},
        "references": _RD_REFERENCES + [
            "Hahn, Todd & van der Klaauw (2001), Identification and estimation of treatment effects "
            "with a regression-discontinuity design",
        ],
        "disrecommend_when": "The treatment jump at the cutoff is below about 0.1",
    },
    {
        "id": "rd.density_test",
        "title": "Did people sort across the cutoff?",
        "one_liner": "Compare how many units sit just below the threshold with how many sit just above.",
        "designs": [DESIGN],
        "estimands": [],
        **{**_RD_ROLES, "roles_optional": ["cluster", "treatment", "confounders"]},
        "options": [
            {"name": "density_bandwidth", "type": "number", "default": None, "min": 0.0,
             "label": "Density bandwidth (leave empty to choose one)",
             "help": "The window the local linear density fit uses either side of the cutoff.",
             "profile": "standard"},
            _OPT_DONUT, _OPT_LEVEL, _OPT_PLOT_BINS, _OPT_PLOT_SPACING,
        ],
        "diagnostics": ["manipulation", "rd_plot", "running_variable_discreteness",
                        "bandwidth_sensitivity"],
        "probes": ["probe.subset"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.rd.density_test",
        "status": "recommended",
        "why_recommended": "It is the one check that speaks to the assumption RD lives on: that units "
                           "could not choose which side of the cutoff to land on. Run it before you "
                           "believe any RD number.",
        "what_can_go_wrong": "A clean density does not prove nobody manipulated anything -- sorting that "
                            "leaves the totals alone is invisible here. A dirty density can also come "
                            "from harmless heaping on round numbers, so look at the histogram before "
                            "concluding anything.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "rddensity"},
        "references": [
            "McCrary (2008), Manipulation of the running variable in the regression discontinuity "
            "design: a density test",
            "Cattaneo, Jansson & Ma (2020), Simple local polynomial density estimators",
        ],
        "disrecommend_when": "The running variable is coarse enough that fewer than about three "
                             "histogram bins fall inside the bandwidth",
    },
    {
        "id": "rd.covariate_balance",
        "title": "Do the covariates jump at the cutoff too?",
        "one_liner": "Run the same estimator on variables the programme cannot have changed; they "
                     "should not step at the threshold.",
        "designs": [DESIGN],
        "estimands": [],
        **{**_RD_ROLES, "roles_required": ["running", "cutoff", "outcome", "confounders"],
           "roles_optional": ["cluster", "treatment"]},
        "options": [
            _order_option(1, 1, "The same order you used for the outcome, so the check matches the "
                                "estimate."),
            _OPT_SELECTOR, _OPT_BANDWIDTH, _OPT_DONUT, _OPT_KERNEL, _OPT_VCE, _OPT_LEVEL,
            _OPT_PLOT_BINS, _OPT_PLOT_SPACING,
        ],
        "diagnostics": ["covariate_balance", "rd_plot", "bandwidth_sensitivity",
                        "running_variable_discreteness"],
        "probes": ["probe.subset", "probe.alternate_spec"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.rd.covariate_balance",
        "status": "recommended",
        "why_recommended": "It is the RD version of a balance table, and it uses exactly the estimator "
                           "that produced the headline number, at exactly the same bandwidth. A "
                           "covariate that jumps is the clearest possible evidence that something other "
                           "than eligibility changes at the threshold.",
        "what_can_go_wrong": "Passing this check does not make the design valid: it only speaks to the "
                            "variables you happened to measure. With many covariates one will cross the "
                            "5% line by luck, which is why the joint test is reported alongside the "
                            "individual ones.",
        "needs_overlap": False,
        "engines": {"python": True, "r": "rdrobust"},
        "references": _RD_REFERENCES + [
            "Calonico, Cattaneo, Farrell & Titiunik (2019), Regression discontinuity designs using "
            "covariates",
        ],
        "disrecommend_when": None,
    },
    {
        "id": "rd.kink",
        "title": "Regression kink (the rule bends, it does not switch)",
        "one_liner": "Where a formula changes slope rather than switching on, measure the change in "
                     "slope of the outcome.",
        "designs": [DESIGN],
        "estimands": ["slope_change"],
        **_RD_ROLES,
        "options": [
            _order_option(2, 2, "A slope needs at least a quadratic. 2 is the default, matching the "
                                "usual p = derivative + 1 convention."),
            _OPT_SELECTOR, _OPT_BANDWIDTH, _OPT_DONUT, _OPT_KERNEL, _OPT_PILOT,
            _OPT_COVADJ, _OPT_VCE, _OPT_LEVEL, _OPT_PLOT_BINS, _OPT_PLOT_SPACING,
        ],
        "diagnostics": ["rd_plot", "level_jump_at_kink", "bias_correction", "bandwidth_sensitivity",
                        "running_variable_discreteness", "manipulation", "covariate_balance"],
        "probes": ["probe.placebo_outcome", "probe.subset", "probe.alternate_spec"],
        "needs": ["numpy", "scipy", "pandas"],
        "explain_key": "method.rd.kink",
        "status": "reasonable",
        "why_recommended": "Benefit formulas, tax schedules and subsidy rules usually bend rather than "
                           "switch. When that is the rule you have, the kink is the discontinuity, and "
                           "reading it as a level jump would be wrong.",
        "what_can_go_wrong": "Estimating a derivative is far noisier than estimating a level, so kink "
                            "designs need much more data than they look like they need, and they are "
                            "more sensitive to curvature in the underlying relationship. The number is "
                            "a change in slope -- per unit of the running variable -- and must never be "
                            "reported as an effect size. " + _NOT_RDROBUST_CARD,
        "needs_overlap": False,
        "engines": {"python": True, "r": "rdrobust"},
        "references": _RD_REFERENCES + [
            "Card, Lee, Pei & Weber (2015), Inference on causal effects in a generalized regression "
            "kink design",
        ],
        "disrecommend_when": "Fewer than a few thousand observations inside a plausible bandwidth",
    },
]
