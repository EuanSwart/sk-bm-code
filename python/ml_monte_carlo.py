"""
Monte Carlo check of the rest-law-agnostic MSD (Appendix D.5, Numerical Verification, label ssML:numerics).

Parameters: alpha = 3/2, delta = 1/2, tau_r = 1, beta = 0.6, lambda = 1e-4  (D = tau_m = 3, tau_eff = 23.8).

  1. Sampler check: Kozubowski-Rachev Mittag-Leffler rests reproduce E e^{-sR} = 1/(1+s^beta) and the
     numerically inverted cdf L^{-1}[1/(s(1+s^beta))].
  2. Walk: rest -> flight -> rest ..., flight L = U^{-1/alpha}, displacement +-L^delta (elapsed part counted
     if t falls inside a flight). Exponential, Mittag-Leffler and tempered (rejection, accept w.p. e^{-lambda R})
     rests. Ten batches; E[x^2] estimated both by x(t)^2 and by sum_k A_k^2 (same mean, far lower variance).
  3. Compared with the exact Talbot inversion of ml_exact.py at 31 log-spaced times in [1, 1e6].

Run:   python ml_monte_carlo.py            (full run, as in the report; ~10-20 min)
       python ml_monte_carlo.py --quick    (10x fewer paths, for a smoke test)
Writes results/ml_monte_carlo.json and prints the checkpoint table.
"""
import argparse, json, os, time
import numpy as np
import mpmath as mp
import ml_exact as mx

ALPHA, DELTA, TAU_R, BETA, LAM = 1.5, 0.5, 1.0, 0.6, 1e-4
T_CHECK = 10.0 ** (np.arange(31) / 5)          # 1 ... 1e6
N_BATCHES = 10
PATHS = {"exp": 1_000, "ml": 20_000, "tmp": 5_000}   # paths per batch
SEEDS = {"exp": 11, "ml": 22, "tmp": 33}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- resting times
def ml_rest(rng, n, tau=TAU_R, beta=BETA):
    """Kozubowski-Rachev inversion for Mittag-Leffler rests."""
    u, v = rng.random(n), rng.random(n)
    pb = np.pi * beta
    return -tau * np.log(u) * (np.sin(pb) / np.tan(pb * v) - np.cos(pb)) ** (1 / beta)


def tmp_rest(rng, n, lam=LAM):
    """Tempered Mittag-Leffler by rejection: accept a ML proposal R with probability e^{-lam R}."""
    out = np.empty(n); todo = np.arange(n)
    while todo.size:
        r = ml_rest(rng, todo.size)
        ok = rng.random(todo.size) < np.exp(-lam * r)
        out[todo[ok]] = r[ok]; todo = todo[~ok]
    return out


REST = {"exp": lambda rng, n: rng.exponential(TAU_R, n), "ml": ml_rest, "tmp": tmp_rest}


# ---------------------------------------------------------------- walk
def simulate(kind, n, rng, tc=T_CHECK, alpha=ALPHA, delta=DELTA):
    """Return (mean x^2, mean sum_k A_k^2) at each checkpoint, averaged over n paths."""
    nc = tc.size
    tcp = np.append(tc, np.inf)
    X2 = np.zeros(nc); S = np.zeros(nc)
    t = np.zeros(n); x = np.zeros(n); A = np.zeros(n); j = np.zeros(n, dtype=np.int64)
    while t.size:
        e = t + REST[kind](rng, t.size)                     # rest: position frozen
        while True:
            hit = tcp[j] <= e
            if not hit.any(): break
            np.add.at(X2, j[hit], x[hit] ** 2); np.add.at(S, j[hit], A[hit]); j[hit] += 1
        t = e
        L = rng.random(t.size) ** (-1 / alpha)               # flight
        sg = np.where(rng.random(t.size) < 0.5, -1.0, 1.0)
        e = t + L
        while True:
            hit = tcp[j] <= e
            if not hit.any(): break
            w = tcp[j[hit]] - t[hit]
            np.add.at(X2, j[hit], (x[hit] + sg[hit] * w ** delta) ** 2)
            np.add.at(S, j[hit], A[hit] + w ** (2 * delta)); j[hit] += 1
        x = x + sg * L ** delta; A = A + L ** (2 * delta); t = e
        keep = j < nc                                        # drop finished paths
        t, x, A, j = t[keep], x[keep], A[keep], j[keep]
    return X2 / n, S / n


# ---------------------------------------------------------------- checks
def sampler_check(n=2_000_000, seed=7):
    rng = np.random.default_rng(seed)
    R = np.sort(ml_rest(rng, n))
    for s in (0.1, 1.0, 5.0):
        print(f"  E exp(-sR), s={s:g}: {np.exp(-s * R).mean():.4f}  vs 1/(1+s^beta) = {1 / (1 + s ** BETA):.4f}")
    xs = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 1e3, 1e4])
    mp.mp.dps = 20
    F = np.array([float(mp.invertlaplace(lambda s: 1 / (s * (1 + s ** BETA)), x, method="talbot")) for x in xs])
    print(f"  max |empirical cdf - exact cdf| over 11 points: {np.abs(np.searchsorted(R, xs, 'right') / n - F).max():.1e}")


def main(quick):
    print("Sampler check (Kozubowski-Rachev):"); sampler_check(200_000 if quick else 2_000_000)
    R = {"exp": mx.R_exp(), "ml": mx.R_ml(), "tmp": mx.R_tmp(LAM)}
    out = {}
    for kind in ("exp", "ml", "tmp"):
        n = PATHS[kind] // (10 if quick else 1)
        t0 = time.time()
        res = [simulate(kind, n, np.random.default_rng(SEEDS[kind] * 1000 + b)) for b in range(N_BATCHES)]
        x2 = np.array([r[0] for r in res]); sa = np.array([r[1] for r in res])
        exact = np.array([float(mx.msd(t, R[kind])) for t in T_CHECK])
        se = lambda a: a.std(axis=0, ddof=1) / np.sqrt(N_BATCHES)
        pulls = (sa.mean(0) - exact) / se(sa)
        out[kind] = dict(t=T_CHECK.tolist(), exact=exact.tolist(), x2_mean=x2.mean(0).tolist(), x2_se=se(x2).tolist(),
                         sumA2_mean=sa.mean(0).tolist(), sumA2_se=se(sa).tolist())
        print(f"\n[{kind}] {N_BATCHES} x {n} paths, {time.time() - t0:.0f}s;  pulls of sum A^2 over 31 times: "
              f"rms {np.sqrt(np.mean(pulls ** 2)):.2f}, max |z| {np.abs(pulls).max():.1f}")
        print(f"   {'t':>6} {'exact':>12} {'MC x^2':>22} {'MC sum A_k^2':>22}")
        for i in (10, 20, 30):
            print(f"   {T_CHECK[i]:6.0e} {exact[i]:12.5g} {x2.mean(0)[i]:12.5g} +- {se(x2)[i]:<8.2g}"
                  f"{sa.mean(0)[i]:12.5g} +- {se(sa)[i]:.2g}")
    # asymptotes quoted in the appendix
    D, tm, te = 3.0, 3.0, float(mx.tau_eff(LAM))
    print(f"\nML leading term at 1e6: {D * 1e6 ** BETA / float(mp.gamma(1 + BETA)):.4g};"
          f"  tempered plateau D t/(tau_eff+tau_m) at 1e6: {D * 1e6 / (te + tm):.5g}")
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    with open(f"{ROOT}/results/ml_monte_carlo.json", "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    main(p.parse_args().quick)
