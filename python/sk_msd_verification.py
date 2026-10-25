"""
Monte Carlo check of the three MSD regimes, including the boundary alpha = 2 delta.

Report: Section 5.5 (Numerical Verification), Figures fig:msd-regimes and fig:msd-rho-eff.

    E[x(t)^2] ~ alpha / ((alpha - 2 delta) mu) * t,                                             alpha > 2 delta
    E[x(t)^2] ~ (alpha / mu) * t ln t,                                                          alpha = 2 delta
    E[x(t)^2] ~ 2 delta Gamma(2 delta - alpha) / (mu Gamma(2 + 2 delta - alpha)) t^(1+2delta-alpha), alpha < 2 delta

with mu = tau_r + alpha/(alpha - 1) the mean rest + metaorder cycle time.

  * Exact event-driven simulation (inverse-CDF Pareto flights, exponential rests); no time step.
  * The asymptotic laws are never used to generate data and nothing is fitted.
  * Uncertainty comes from 10 independent batches.

Run:       python sk_msd_verification.py                 (simulate ~2 min, then draw)
           python sk_msd_verification.py --from-results  (redraw from results/msd_results.json)
Produces:  results/msd_results.json, figs/fig_msd_regimes.{pdf,png}, figs/fig_msd_rho_eff.{pdf,png}
"""

import argparse, json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.special import gamma as Gamma

# ============================================================================
# 0. PARAMETERS
# ============================================================================
# delta = 1/2 gives 2 delta = 1 < alpha for every alpha in (1, 2), so the boundary is unreachable under the
# square-root law. delta = 3/4 is used purely as a device that puts alpha = 2 delta = 1.5 inside (1, 2).
DELTA = 0.75
TAU_R_MEAN = 1.0
ALPHAS = {"sub": 1.2, "crit": 1.5, "super": 1.8}   # 2*delta = 1.5

T_MIN, T_MAX, N_OBS = 2.0, 2.0e4, 50
OBS_TIMES = np.geomspace(T_MIN, T_MAX, N_OBS)

N_TOTAL_MAIN = 200_000
N_BATCHES = 10
BATCH_N = N_TOTAL_MAIN // N_BATCHES
SEED_BASE = 20260828

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(ROOT, "figs")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================================
# 1. EVENT-DRIVEN MONTE CARLO (exact, vectorised across trajectories)
# ============================================================================
def simulate_msd_squared(alpha, delta, tau_r_mean, N, obs_times, rng, max_cycles=2_000_000):
    """
    N i.i.d. copies of   rest ~ Exp(tau_r_mean) -> flight ~ Pareto(alpha, xmin=1) -> repeat,
    with displacement eps * r^delta at elapsed flight time r. Returns x(t_j)^2, shape (N, n_obs).
    """
    obs_times = np.asarray(obs_times, dtype=np.float64)
    n_obs = obs_times.size

    x_base = np.zeros(N)
    t_now = np.zeros(N)
    X2 = np.full((N, n_obs), np.nan)
    next_idx = np.zeros(N, dtype=np.int64)
    obs_padded = np.concatenate([obs_times, [np.inf]])

    it, n_done = 0, 0
    while it < max_cycles and n_done < N:
        it += 1
        tau_r = rng.exponential(tau_r_mean, size=N)
        T_m = (1.0 - rng.random(N)) ** (-1.0 / alpha)            # inverse-CDF Pareto
        eps = np.where(rng.random(N) < 0.5, -1.0, 1.0)

        rest_start = t_now
        flight_start = t_now + tau_r
        flight_end = flight_start + T_m

        while True:   # observation times inside the rest
            cur = obs_padded[next_idx]
            hit = (cur >= rest_start) & (cur < flight_start)
            if not hit.any():
                break
            rows = np.nonzero(hit)[0]
            X2[rows, next_idx[rows]] = x_base[rows] ** 2
            next_idx[rows] += 1

        while True:   # observation times inside the flight
            cur = obs_padded[next_idx]
            hit = (cur >= flight_start) & (cur < flight_end)
            if not hit.any():
                break
            rows = np.nonzero(hit)[0]
            x_val = x_base[rows] + eps[rows] * np.power(cur[rows] - flight_start[rows], delta)
            X2[rows, next_idx[rows]] = x_val ** 2
            next_idx[rows] += 1

        x_base = x_base + eps * np.power(T_m, delta)
        t_now = flight_end
        n_done = (next_idx >= n_obs).sum()

    if n_done < N:
        raise RuntimeError(f"hit max_cycles={max_cycles} with {N - n_done} trajectories unfinished")
    return X2


def run_batches(alpha, delta, tau_r, n_batches, batch_n, obs_times, seed0):
    msd_batches = np.zeros((n_batches, len(obs_times)))
    for b in range(n_batches):
        rng = np.random.default_rng(seed0 + 1000 * b)
        msd_batches[b] = simulate_msd_squared(alpha, delta, tau_r, batch_n, obs_times, rng).mean(axis=0)
    return msd_batches


# ============================================================================
# 2. LEADING-ORDER THEORY (Eq. MSD_three_regimes; comparison only)
# ============================================================================
def mu_cycle(alpha, tau_r=TAU_R_MEAN):
    return tau_r + alpha / (alpha - 1.0)


def theory_msd(tag, t, delta=DELTA, tau_r=TAU_R_MEAN):
    alpha = ALPHAS[tag]
    mu = mu_cycle(alpha, tau_r)
    if tag == "super":                                       # alpha > 2 delta
        return alpha / ((alpha - 2 * delta) * mu) * t
    if tag == "crit":                                        # alpha = 2 delta
        return alpha / mu * t * np.log(t)
    rho = 1 + 2 * delta - alpha                              # alpha < 2 delta
    return 2 * delta * Gamma(2 * delta - alpha) / (mu * Gamma(2 + 2 * delta - alpha)) * t ** rho


def rho_eff_centered(t, msd):
    """Local exponent d ln MSD / d ln t by centred differences (NaN at the end points)."""
    lt, lm = np.log(t), np.log(msd)
    rho = np.full_like(t, np.nan)
    rho[1:-1] = (lm[2:] - lm[:-2]) / (lt[2:] - lt[:-2])
    return rho


# ============================================================================
# 3. FIGURES
# ============================================================================
FIG_W = 6.3   # inches, so \includegraphics[width=\textwidth] keeps type at design size

PH_BLUE, PH_ORANGE, INK, GREY, LIGHT = "#0072B2", "#E67800", "#1A1A1A", "#6E6E6E", "#DADADA"
COL = {"sub": PH_ORANGE, "crit": INK, "super": PH_BLUE}
MRK = {"sub": "o", "crit": "s", "super": "^"}
ORDER = ("sub", "crit", "super")
REGIME = {"sub": "superdiffusive", "crit": "boundary", "super": "diffusive"}
LAW = {"sub": r"\propto t^{\,1+2\delta-\alpha}", "crit": r"\propto t\ln t", "super": r"\propto t"}


def set_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
        "font.size": 9, "axes.labelsize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "legend.frameon": False, "legend.handlelength": 2.2, "lines.linewidth": 1.2,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def faint_grid(ax):
    ax.grid(True, which="major", color=LIGHT, lw=0.5)
    ax.set_axisbelow(True)


def regime_label(tag):
    rel = {"sub": "<", "crit": "=", "super": ">"}[tag]
    return rf"$\alpha={ALPHAS[tag]:g}\;{rel}\;2\delta$"


def save_fig(fig, stem):
    fig.savefig(f"{FIG_DIR}/{stem}.png")
    fig.savefig(f"{FIG_DIR}/{stem}.pdf")
    plt.close(fig)


def fig_regimes(t, res):
    """E[x^2]/t on a log time axis: t^rho (rho > 1) is convex, t ln t a straight line, diffusion flat."""
    fig, ax = plt.subplots(figsize=(FIG_W, 3.7), layout="constrained")
    for tag in ORDER:
        y = np.array(res[tag]["msd_mean"]) / t
        ax.plot(t, theory_msd(tag, t) / t, "--", color=COL[tag], lw=1.1, zorder=2)
        ax.plot(t, y, color=COL[tag], lw=0.8, marker=MRK[tag], ms=2.8, mew=0.3, mec="white", zorder=3)

    ax.set_xscale("log"); ax.set_xlim(T_MIN * 0.8, T_MAX * 1.25); ax.set_ylim(0, 12.5)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\mathbb{E}[x(t)^2]\,/\,t$")
    faint_grid(ax)

    handles = [Line2D([], [], color=COL[k], marker=MRK[k], ms=3.2, lw=1.0, mew=0.3, mec="white") for k in ORDER]
    handles += [Line2D([], [], ls=""),
                Line2D([], [], color=GREY, marker="o", ms=3.2, lw=0.8, mew=0.3, mec="white"),
                Line2D([], [], color=GREY, lw=1.1, ls="--")]
    labels = [f"{regime_label(k)}  {REGIME[k]}: $\\mathbb{{E}}[x^2]{LAW[k]}$" for k in ORDER]
    labels += [" ", "Monte Carlo", "leading-order asymptotics"]
    ax.legend(handles, labels, loc="upper left", borderaxespad=0.5)
    save_fig(fig, "fig_msd_regimes")


def fig_rho_eff(t, res):
    fig, ax = plt.subplots(figsize=(FIG_W, 3.7), layout="constrained")
    nb = len(res["crit"]["msd_batches"])
    tcrit = float(stats.t.ppf(0.975, nb - 1))
    rho_sub = 1 + 2 * DELTA - ALPHAS["sub"]

    for tag in ORDER:   # local exponent; band = 95% Student-t interval over batches
        mb = np.array(res[tag]["msd_batches"])
        rho = rho_eff_centered(t, mb.mean(axis=0))
        rho_b = np.array([rho_eff_centered(t, row) for row in mb])
        se = rho_b.std(axis=0, ddof=1) / np.sqrt(nb)
        ax.fill_between(t, rho - tcrit * se, rho + tcrit * se, color=COL[tag], alpha=0.22, lw=0)
        ax.plot(t, rho, color=COL[tag], lw=1.5)

    for val, tag, txt, va in [(rho_sub, "sub", rf"$\rho=1+2\delta-\alpha={rho_sub:.2f}$", "bottom"),
                              (1.0, "super", r"$\rho=1$", "top")]:
        ax.axhline(val, color=COL[tag], ls="--", lw=0.9)
        ax.text(t[-1], val + (0.006 if va == "bottom" else -0.006), txt, color=COL[tag], fontsize=8,
                va=va, ha="right")

    sel = t > np.exp(1.2)   # local exponent of t ln t is 1 + 1/ln t
    ax.plot(t[sel], 1 + 1 / np.log(t[sel]), color=COL["crit"], ls="--", lw=1.1)

    ax.set_xscale("log"); ax.set_xlim(T_MIN * 0.8, T_MAX * 1.25); ax.set_ylim(0.92, 1.85)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\rho_{\rm eff}(t)=d\ln\mathbb{E}[x(t)^2]\,/\,d\ln t$")
    faint_grid(ax)
    handles = [Line2D([], [], color=COL[k], lw=1.5) for k in ORDER]
    handles += [Patch(facecolor=GREY, alpha=0.3, lw=0), Line2D([], [], color=COL["crit"], ls="--", lw=1.1)]
    labels = [f"{regime_label(k)}  {REGIME[k]}" for k in ORDER]
    labels += [rf"95% confidence band ({nb} batches)", r"$1+1/\ln t$, local exponent of $t\ln t$"]
    ax.legend(handles, labels, loc="upper right", borderaxespad=0.5)
    save_fig(fig, "fig_msd_rho_eff")


def make_figures(t, res):
    set_style()
    fig_regimes(t, res)
    fig_rho_eff(t, res)
    for tag in ORDER:   # numbers quoted in the text
        m = np.array(res[tag]["msd_mean"]); th = theory_msd(tag, t)
        rho = rho_eff_centered(t, m)
        print(f"[{tag}] MSD/t at t_max: MC={m[-1]/t[-1]:.3f} theory={th[-1]/t[-1]:.3f} | "
              f"MC/theory t=1e3: {np.interp(1e3, t, m/th):.3f}, t_max: {m[-1]/th[-1]:.3f} | "
              f"rho_eff(t={t[-2]:.0f}) = {rho[-2]:.3f}")
    print(f"[crit] 1+1/ln t at t={t[-2]:.0f}: {1 + 1/np.log(t[-2]):.3f}")
    print(f"Figures written to {FIG_DIR}")


# ============================================================================
# 4. DRIVER
# ============================================================================
def main():
    t = OBS_TIMES
    res = {}
    for i, (tag, alpha) in enumerate(ALPHAS.items()):
        t0 = time.time()
        mb = run_batches(alpha, DELTA, TAU_R_MEAN, N_BATCHES, BATCH_N, t, SEED_BASE + 10_000 * i)
        res[tag] = dict(alpha=alpha, msd_batches=mb.tolist(), msd_mean=mb.mean(axis=0).tolist(),
                        msd_se=(mb.std(axis=0, ddof=1) / np.sqrt(N_BATCHES)).tolist())
        print(f"[{tag}] alpha={alpha}  paths={N_TOTAL_MAIN}  {time.time() - t0:.1f}s")
    with open(f"{RESULTS_DIR}/msd_results.json", "w") as f:
        json.dump({"obs_times": t.tolist(), "delta": DELTA, "tau_r": TAU_R_MEAN, "results": res}, f)
    make_figures(t, res)


def from_results():
    with open(f"{RESULTS_DIR}/msd_results.json") as f:
        d = json.load(f)
    make_figures(np.array(d["obs_times"]), d["results"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SK MSD verification (two figures)")
    p.add_argument("--from-results", action="store_true", help="redraw from results/msd_results.json")
    from_results() if p.parse_args().from_results else main()
