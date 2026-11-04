"""Simulation: does the 7-win / 3-loss cap alone push the deck-effect estimator (tools/deck_context.py tau^2) negative?

    uv run python tools/tau_sim.py

No: it attenuates it (true 0.09 reads about 0.057) but keeps the sign. Constant-p model, logit(p) ~ N(0.15, tau).
"""
import numpy as np
rng = np.random.default_rng(7)
def sim(sd, n=200000, wins_cap=7, loss_cap=3):
    # true deck effect: logit(p) ~ N(0.15, sd); constant-p̄ model fitted on all games (what a context-only fit reduces to)
    p = 1 / (1 + np.exp(-(0.15 + sd * rng.standard_normal(n))))
    W = np.zeros(n); L = np.zeros(n)
    alive = np.ones(n, bool)
    while alive.any():
        w = rng.random(n) < p
        W += alive & w; L += alive & ~w
        alive = alive & (W < wins_cap) & (L < loss_cap)
    N = W + L
    pbar = W.sum() / N.sum()
    r = W - pbar * N; V = pbar * (1 - pbar) * N
    est = ((r ** 2 - V).sum()) / (V ** 2).sum()
    return est, N.mean(), pbar
for sd in (0.0, 0.1, 0.2, 0.3):
    e, nm, pb = sim(sd)
    print(f"true tau {sd:.1f} (tau^2 {sd*sd:.3f}): estimator {e:+.4f}   mean games {nm:.2f}  pbar {pb:.3f}")
print("no cap (fixed 9 games), true tau 0.3:")
rng = np.random.default_rng(7)
n=200000; p = 1/(1+np.exp(-(0.15+0.3*rng.standard_normal(n)))); W=(rng.random((n,9))<p[:,None]).sum(1); N=9
pbar=W.sum()/(N*n); r=W-pbar*N; V=pbar*(1-pbar)*N
print(f"  estimator {((r**2-V).sum())/(V**2*n):+.4f} (true 0.090)")
