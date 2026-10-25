"""
Exact single-trader MSD for any resting-time law, by numerical Laplace inversion (Talbot).

Report: Section 6 (Extension I) and Appendix D, Eq. (ML-agnostic):

    L{E[x(t)^2]}(s) = N_F(s) / (R(s) + 1 - psi(0,s)),     R(s) = 1/psi_r(s) - 1,

with closed-form flight functions (Pareto flights, xmin = 1, displacement t^delta)

    N_F(s)    = int_0^1 e^{-st} t^{2d} dt + E(alpha-2d, s) + alpha E(alpha+1-2d, s) / s
    psi(0,s)  = alpha E(alpha+1, s),        E(nu, s) = int_1^inf e^{-st} t^{-nu} dt = s^{nu-1} Gamma(1-nu, s).

No small-s expansion is used, so the result is exact for the model. With R = R_exp this is also the
exact MSD of the base Sato-Kanazawa model for any (alpha, delta).

Run   python ml_exact.py   to print every number quoted in Section 6 and the Discussion.
"""
import mpmath as mp
from scipy.special import gammainc

mp.mp.dps = 30

# Parameters of Section 6 / Appendix D
ALPHA, DELTA, TAU_R, BETA = mp.mpf("1.5"), mp.mpf("0.5"), mp.mpf(1), mp.mpf("0.6")


def E(nu, s):
    """Generalised exponential integral int_1^inf e^{-st} t^{-nu} dt."""
    return s ** (nu - 1) * mp.gammainc(1 - nu, s)


def N_F(s, al=ALPHA, de=DELTA):
    a = 1 + 2 * de
    head = (mp.gamma(a) - mp.gammainc(a, s)) * s ** (-a)             # int_0^1 e^{-st} t^{2d} dt
    return head + E(al - 2 * de, s) + al * E(al + 1 - 2 * de, s) / s


def psi0(s, al=ALPHA):
    return al * E(al + 1, s)


# Rest-law functions R(s) = 1/psi_r(s) - 1  (Table tab:ML-laws)
def R_exp(tau=TAU_R):
    return lambda s: tau * s


def R_ml(tau=TAU_R, beta=BETA):
    return lambda s: (tau * s) ** beta


def R_tmp(lam, tau=TAU_R, beta=BETA):
    lam = mp.mpf(lam)
    kap = 1 + (tau * lam) ** beta
    return lambda s: ((tau * (s + lam)) ** beta - (tau * lam) ** beta) / kap


def tau_eff(lam, tau=TAU_R, beta=BETA):
    lam = mp.mpf(lam)
    return beta * tau ** beta * lam ** (beta - 1) / (1 + (tau * lam) ** beta)


def msd(t, R, al=ALPHA, de=DELTA):
    """Exact E[x(t)^2] for rest-law function R."""
    f = lambda s: N_F(s, al, de) / (R(s) + 1 - psi0(s, al))
    return mp.invertlaplace(f, mp.mpf(t), method="talbot")


def g_beta(u, beta=float(BETA), nmax=2000):
    """Crossover function g_beta(u) = sum_{n>=1} P(n beta, u)  (Eq. ML-g)."""
    return sum(gammainc(n * beta, u) for n in range(1, nmax))


if __name__ == "__main__":
    al, de, tr, be = ALPHA, DELTA, TAU_R, BETA
    D, tm = al / (al - 2 * de), al / (al - 1)
    print(f"D = {mp.nstr(D, 4)},  tau_m = {mp.nstr(tm, 4)}")

    # crossover time t_c = (tau_m / tau_r^beta)^(1/(1-beta))
    for b in ("0.6", "0.9", "0.95"):
        b = mp.mpf(b)
        print(f"t_c(beta={mp.nstr(b, 3)}) = {mp.nstr((tm / tr ** b) ** (1 / (1 - b)), 3)}")

    # g_beta(1) against its small-u and linear forms
    b = float(be)
    print(f"g_beta(1) = {g_beta(1.0):.3f}  small-u {1 / float(mp.gamma(1 + be)):.3f}  "
          f"linear term u/beta {1 / b:.3f}  ratio to small-u {g_beta(1.0) * float(mp.gamma(1 + be)):.2f}")

    # tempered law used in Figure fig:ML-regimes (lambda = 5e-4) and App. D.5 (lambda = 1e-4)
    for lam in ("5e-4", "1e-4"):
        te = tau_eff(lam)
        print(f"lambda={lam}: tau_eff = {mp.nstr(te, 4)},  plateau D/(tau_eff+tau_m) = {mp.nstr(D / (te + tm), 4)},"
              f"  factor (tau_r+tau_m)/(tau_eff+tau_m) = {mp.nstr((tr + tm) / (te + tm), 3)}")
    print(f"exact E[x^2]/t at t=1e6, tempered lambda=5e-4: {mp.nstr(msd(1e6, R_tmp('5e-4')) / 1e6, 4)}")
    print(f"exponential plateau D/(tau_r+tau_m) = {mp.nstr(D / (tr + tm), 3)}")

    # how far the Mittag-Leffler asymptote sits from the exact curve (Discussion)
    for t in (1e3, 1e6):
        ex = msd(t, R_ml())
        asym = D * mp.mpf(t) ** be / (tr ** be * mp.gamma(1 + be))
        print(f"ML asymptote / exact at t={t:.0e}: {mp.nstr(asym / ex, 4)}")
