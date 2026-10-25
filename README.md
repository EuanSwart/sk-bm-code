# Deriving Brownian Prices Despite Predictable Order Flow — Python and MATLAB Code

Numerical code for the Statistics Honours project *Deriving Brownian Prices Despite Predictable Order Flow*
(E. Swart and D. Wood; supervisor A/Prof T. Gebbie; Department of Statistical Sciences, University of Cape Town, 2026).
It reproduces every figure, table entry and quoted number in the report that is computed rather than derived.

## Current situation: v1.0.0

- Monte Carlo verification of the three mean-squared displacement (MSD) regimes, including the marginal
  $t\log t$ boundary $\alpha=2\delta$.
- Exact MSD for any resting-time law by numerical Laplace (Talbot) inversion: exponential, Mittag-Leffler
  and tempered Mittag-Leffler rests.
- Monte Carlo check of the Mittag-Leffler extension against the exact inversion.
- Data for the two TikZ figures of the Mittag-Leffler extension.
- 3-D pole maps of the Mellin–Barnes kernels behind the calculation discrepancies.

## Future situation: possible extensions

- Monte Carlo of the signature-space MSD (Extension II), which is currently verified analytically only.
- Multi-trader simulations, and fits to empirical (e.g. JSE) variograms.

## Research and software boundary

This is research code that accompanies a report. The analytical results are derived in the report, and
the code only checks them numerically. It is not a trading or forecasting tool.

## Report script map

| Report item | Script | Runtime |
|---|---|---|
| Figs. `fig:msd-regimes`, `fig:msd-rho-eff` and the numbers in Sec. 5.5 (`sec:msd-numerics`) | `python/sk_msd_verification.py` | ~2 min |
| Sec. 6.4–6.5 and Discussion: $t_c$, $g_\beta(1)$, $\tau_{\rm eff}$, plateaus, the 23% gap at $t=10^3$ | `python/ml_exact.py` | seconds |
| App. D.5 (`ssML:numerics`): sampler check, exact vs Monte Carlo $\mathbb{E}[x(t)^2]$ at 31 times | `python/ml_monte_carlo.py` | ~5 min (`--quick`: ~1.5 min) |
| Figs. `fig:ML-raster` (bar coordinates), `fig:ML-regimes` (curve coordinates) | `python/ml_figure_data.py` | ~20 s |
| Figs. `fig:discrep-mb-overview`, `fig:discrep-I-pos`, `fig:discrep-I-neg`, `fig:discrep-I-collision`, `fig:discrep-Q2-*` | `matlab/pole_maps.m` | ~1 min |

The phase diagram, overview and dependency diagrams are drawn directly in TikZ and need no code.
Table `tab:ML-laws` is analytical.

## Repository structure

```
python/    Monte Carlo, exact Laplace inversion, and TikZ figure data
matlab/    Mellin–Barnes pole maps
figs/      Generated figures (the report's \includegraphics paths point here)
results/   Generated data (JSON / text), so figures can be redrawn without re-simulating
```

## Requirements

- Python 3.10+ with `numpy`, `scipy`, `mpmath` and `matplotlib` (`requirements.txt`).
- MATLAB R2019b or later, base product only, for `pole_maps.m`. GNU Octave 8 also runs it for previews,
  but prints the LaTeX labels as raw text.

## Quick start

```bash
pip install -r requirements.txt
cd python
python sk_msd_verification.py            # add --from-results to redraw from results/msd_results.json
python ml_exact.py
python ml_monte_carlo.py                 # add --quick for a smoke test
python ml_figure_data.py
```

```matlab
cd matlab
pole_maps          % writes the seven PNGs into ../figs
```

## Model and parameters

A single trader alternates rests and metaorders. A metaorder (flight) lasts $L=U^{-1/\alpha}$
(Pareto, $L\ge 1$) and moves the price by $\pm r^{\delta}$ after elapsed time $r$. Rests are exponential
with mean $\tau_r$ (Sato–Kanazawa), Mittag-Leffler with index $\beta$, or tempered Mittag-Leffler with
rate $\lambda$. Simulations are event-driven with exact event times, so there is no time step.

| Experiment | $\alpha$ | $\delta$ | $\tau_r$ | $\beta$ | $\lambda$ | Paths |
|---|---|---|---|---|---|---|
| Three regimes (Sec. 5.5) | 1.2, 1.5, 1.8 | 3/4 | 1 | – | – | $2\times10^5$ per $\alpha$, 10 batches |
| Regime figure (Sec. 6) | 3/2 | 1/2 | 1 | 0.6 | $5\times10^{-4}$ | exact inversion |
| Monte Carlo check (App. D.5) | 3/2 | 1/2 | 1 | 0.6 | $10^{-4}$ | $10^3$ / $2\times10^4$ / $5\times10^3$ per batch, 10 batches |

$\delta=3/4$ in the first row is a device that puts the boundary $\alpha=2\delta$ inside $(1,2)$. Under the
square-root law ($\delta=1/2$) the boundary cannot be reached.

## Version-control policy

- The `.py` and `.m` files are the editable sources. Everything in `figs/` and `results/` is generated and
  can be rebuilt from them.
- All random seeds are fixed. The App. D.5 Monte Carlo was originally run with an equivalent C program.
  The Python port uses the same algorithm with its own fixed seeds, so its Monte Carlo values differ from
  the report only within sampling error. The exact values, the sampler check and every other quoted number
  reproduce exactly.
- Releases follow semantic versioning (`VERSION`, `CHANGELOG.md`). Tag the release that matches the
  submitted report.

## Citation and attribution

Cite the software with `CITATION.cff` and the tagged release, and the report itself as

> E. Swart and D. Wood (2026). *Deriving Brownian Prices Despite Predictable Order Flow*. Statistics Honours
> Project, Department of Statistical Sciences, University of Cape Town. Supervisor: A/Prof T. Gebbie.

The model being reconstructed is that of Sato and Kanazawa (2025), *Why do financial prices exhibit
Brownian motion despite predictable order flow?*

## Licences, AI disclosure, and disclaimer

- **Licence:** source code is released under the BSD 3-Clause License (`LICENSE`).
- **AI disclosure:** generative AI tools (Claude, Anthropic) were used to help write, port and test the code.
  All code and outputs were reviewed and checked against the report by the authors.
- **Disclaimer:** the software is provided as is, without warranty of any kind. It is research and teaching
  code and is not financial advice.

## About

Statistics Honours project, Department of Statistical Sciences, University of Cape Town, 2026.
Authors: Euan Swart and Dylan Wood. Supervisor: A/Prof Tim Gebbie.
