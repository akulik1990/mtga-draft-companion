"""Shared helpers for video_pick_tests_h1a/h1b/h5: skill groups, clustered conditional logit, clustered OLS, Bonferroni line."""
import numpy as np
from scipy.stats import norm
GR = ["all", "good", "avg", "weak"]
def gmask(skill, k):
    return {"all": ~np.isnan(skill), "good": skill >= .6 - 1e-9, "avg": (skill >= .52 - 1e-9) & (skill <= .58 + 1e-9), "weak": skill < .52 - 1e-9}[k]
NPRIM = 6; ALPHA = .05 / NPRIM; ZB = norm.isf(ALPHA / 2)   # six primary contrasts (H1a x2, H1b x2, H5 x2), two-sided
def clogit(X, gid, chosen, cluster):
    """Conditional logit, rows sorted by gid. Returns beta, cluster-robust covariance (clusters = drafts)."""
    starts = np.r_[0, np.flatnonzero(np.diff(gid)) + 1]; sz = np.diff(np.r_[starts, len(gid)]); ng = len(starts)
    mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1; Z = (X - mu) / sd; b = np.zeros(X.shape[1]); ch = chosen.astype(float)
    def probs(b):
        u = Z @ b; m = np.maximum.reduceat(u, starts); e = np.exp(u - np.repeat(m, sz)); s = np.add.reduceat(e, starts); return e / np.repeat(s, sz), (np.log(s) + m)
    for it in range(30):
        p, ls = probs(b); ll = (Z @ b)[chosen].sum() - ls.sum()
        gr = Z[chosen].sum(0) - (p[:, None] * Z).sum(0)
        pz = p[:, None] * Z; sx = np.add.reduceat(pz, starts); H = (pz.T @ Z) - sx.T @ sx
        step = np.linalg.solve(H + 1e-9 * np.eye(len(b)), gr); b = b + step
        if np.abs(step).max() < 1e-8: break
    p, _ = probs(b); sc = (ch - p)[:, None] * Z; sg = np.add.reduceat(sc, starts)       # per-choice-set score
    cl = cluster[starts]; uq, inv = np.unique(cl, return_inverse=True); sc_c = np.zeros((len(uq), Z.shape[1])); np.add.at(sc_c, inv, sg)
    pz = p[:, None] * Z; sx = np.add.reduceat(pz, starts); H = (pz.T @ Z) - sx.T @ sx; Hi = np.linalg.inv(H)
    V = Hi @ (sc_c.T @ sc_c) @ Hi * len(uq) / (len(uq) - 1)
    return b / sd, V / np.outer(sd, sd)
def ols(X, y, cluster):
    """OLS with cluster-robust (CR1) covariance; X includes the constant."""
    XtXi = np.linalg.inv(X.T @ X); b = XtXi @ (X.T @ y); e = y - X @ b
    uq, inv = np.unique(cluster, return_inverse=True); s = np.zeros((len(uq), X.shape[1])); np.add.at(s, inv, X * e[:, None])
    V = XtXi @ (s.T @ s) @ XtXi * len(uq) / (len(uq) - 1) * (len(y) - 1) / (len(y) - X.shape[1]); return b, V
def lin(b, V, w):
    w = np.asarray(w, float); e = w @ b; se = float(np.sqrt(w @ V @ w)); return e, se
def fm(e, se, d=3, scale=1.0):
    z = e / se if se > 0 else np.nan; return f"{e*scale:+.{d}f} [{(e-1.96*se)*scale:+.{d}f}, {(e+1.96*se)*scale:+.{d}f}] z={z:+.2f}" + ("**" if abs(z) > ZB else "*" if abs(z) > 1.96 else "")
def diff(a, b):
    """good minus weak, independent samples; a, b = (est, se)."""
    d = a[0] - b[0]; se = (a[1] ** 2 + b[1] ** 2) ** .5; return d, se
