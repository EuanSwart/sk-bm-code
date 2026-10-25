"""
Data for the two TikZ figures of Section 6 (Extension I). Both figures are drawn in TikZ inside the report;
this script produces the numbers pasted into them.

  fig:ML-raster   One realisation of six traders, exponential vs Mittag-Leffler rest (beta = 0.6),
                  alpha = 3/2, tau_r = 1, t in [0, 120], seed 20261009.
                  -> results/raster_exp.txt, results/raster_ml.txt  (trader, start, end of each metaorder)
                     and the  \\foreach \\a/\\b in {...}  lists for each trader.
  fig:ML-regimes  Exact relative MSD E[x(t)^2]/E[x(1)^2] for the three rest laws (lambda = 5e-4),
                  37 log-spaced times in [1, 1e6], from the Talbot inversion in ml_exact.py.
                  -> results/ml_regimes_curves.txt and the  plot coordinates {...}  strings (log10 axes).

Run:  python ml_figure_data.py      (~1 min)
"""
import os
import numpy as np
import mpmath as mp
import ml_exact as mx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- Figure fig:ML-raster
def raster(seed=20261009, alpha=1.5, tau=1.0, beta=0.6, T=120.0, n_traders=6):
    rng = np.random.default_rng(seed)

    def ml_rest():
        u, v = rng.random(), rng.random()
        return -tau * np.log(u) * (np.sin(beta * np.pi) / np.tan(beta * np.pi * v) - np.cos(beta * np.pi)) ** (1 / beta)

    def exp_rest():
        return rng.exponential(tau)

    def path(rest):
        t, bars = 0.0, []
        while t < T:
            t += rest()                       # every path starts with a rest
            if t >= T:
                break
            L = rng.random() ** (-1 / alpha)  # Pareto flight
            bars.append((t, min(t + L, T)))
            t += L
        return bars

    for name, rest in (("exp", exp_rest), ("ml", ml_rest)):   # order matters: one shared RNG stream
        rows = {i: path(rest) for i in range(1, n_traders + 1)}
        with open(f"{OUT}/raster_{name}.txt", "w") as fh:
            for i, bars in rows.items():
                for a, b in bars:
                    fh.write(f"{i} {a:.2f} {b:.2f}\n")
        print(f"% raster ({name}); active time per trader:",
              {i: round(float(sum(b - a for a, b in v)), 1) for i, v in rows.items()})
        for i, bars in rows.items():
            print(f"%  trader {i}: \\foreach \\a/\\b in {{" + ",".join(f"{float(f'{a:.2f}'):g}/{float(f'{b:.2f}'):g}"
                                                             for a, b in bars) + "}")


# ---------------------------------------------------------------- Figure fig:ML-regimes
def regimes(lam="5e-4"):
    mp.mp.dps = 25
    laws = {"exp": mx.R_exp(), "ml": mx.R_ml(), "tmp": mx.R_tmp(lam)}
    ts = np.logspace(0, 6, 37)
    d = np.array([[t] + [float(mx.msd(t, R)) for R in laws.values()] for t in ts])
    np.savetxt(f"{OUT}/ml_regimes_curves.txt", d, header="t  exp  ml  tempered", fmt="%.10g")
    rel = d[:, 1:] / d[0, 1:]
    for k, name in enumerate(laws):
        print(f"% {name}: \\draw plot coordinates {{" +
              " ".join(f"({np.log10(t):.3f},{np.log10(v):.3f})" for t, v in zip(ts, rel[:, k])) + "};")
    print(f"% crossover t* = 1/lambda at log10 = {np.log10(1 / float(lam)):.3f}")


if __name__ == "__main__":
    raster()
    regimes()
