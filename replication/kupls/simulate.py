"""The Monte Carlo study of Section 10."""
import json
import sys
import numpy as np

from .core import Design, Grid, stat_T, stat_S, crit

# ---


def one_cell(design, grid, alpha, n, R, m, level, rng, eps_law, delta):
    """Null and alternative replications for one (alpha, n) cell.

    Returns raw size under the plug-in asymptotic calibration (the quantity
    the manuscript's claim is about) and SIZE-CORRECTED power, which uses the
    empirical null quantile as the critical value. The correction matters:
    T_n over-rejects badly under heavy tails, so its raw power would flatter
    it for the wrong reason.
    """
    b_true = design.beta_c
    b_null = b_true.copy()
    # LOCAL alternative: b_1 (1 + c/sqrt n). A fixed departure is rejected with
    # probability one by both tests at every n and separates nothing; shrinking
    # it at the parametric rate keeps power comparable across the n-grid.
    b_null[0] *= 1.0 + delta / np.sqrt(n)

    T0, S0, T1, S1 = [], [], [], []
    rejT = rejS = 0
    fails = 0
    for _ in range(R):
        xi, Y, _ = design.draw(n, alpha, rng, eps_law)
        try:
            T, wT = stat_T(xi, Y, b_true, m)
            S, wS = stat_S(xi, Y, b_true, grid)
            Ta, _ = stat_T(xi, Y, b_null, m)
            Sa, _ = stat_S(xi, Y, b_null, grid)
        except np.linalg.LinAlgError:
            fails += 1
            continue
        if not all(np.isfinite(v) for v in (T, S, Ta, Sa)):
            fails += 1
            continue
        T0.append(T); S0.append(S); T1.append(Ta); S1.append(Sa)
        rejT += T > crit(wT, level, rng)
        rejS += S > crit(wS, level, rng)

    eff = len(T0)
    cT = float(np.quantile(T0, 1.0 - level))
    cS = float(np.quantile(S0, 1.0 - level))
    return dict(
        alpha=alpha, n=n, eps=eps_law, reps=eff, failures=fails,
        size_T=rejT / eff, size_S=rejS / eff,
        pow_T=float(np.mean(np.array(T1) > cT)),
        pow_S=float(np.mean(np.array(S1) > cS)),
        med_T=float(np.median(T0)), med_S=float(np.median(S0)),
        q99_T=float(np.quantile(T0, 0.99)), q99_S=float(np.quantile(S0, 0.99)),
    )


def run(alphas, ns, R, m=3, level=0.05, seed=20260815,
        eps_law="gauss", delta=1.0):
    design, grid = Design(), Grid()
    out = []
    print(f"--- eps ~ {eps_law}, H1: b_1 -> b_1 (1 + {delta}/sqrt n), "
          f"nominal level {level}, m={m}, R={R} ---", flush=True)
    print(f"{'alpha':>5} {'n':>5} {'size_T':>7} {'size_S':>7} "
          f"{'pow_T':>6} {'pow_S':>6} {'med_T':>9} {'med_S':>8}", flush=True)
    for alpha in alphas:
        for n in ns:
            # Deterministic, reproducible seeding. Python's hash() is salted
            # per process for str/tuple inputs, so deriving a seed from it
            # would make the table unreproducible -- exactly the claim the
            # paper's availability statement makes. Use explicit entropy.
            rng = np.random.default_rng(
                [seed, int(round(alpha * 10)), n,
                 {"gauss": 0, "cauchy": 1}[eps_law]])
            row = one_cell(design, grid, alpha, n, R, m, level, rng,
                           eps_law, delta)
            out.append(row)
            print(f"{alpha:5.1f} {n:5d} {row['size_T']:7.3f} "
                  f"{row['size_S']:7.3f} {row['pow_T']:6.3f} "
                  f"{row['pow_S']:6.3f} {row['med_T']:9.3g} "
                  f"{row['med_S']:8.3g}", flush=True)
    return out




def main():
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    res = run([1.2, 1.5, 1.8, 2.0], [200, 500, 1000, 2000], R)
    res += run([1.5], [200, 500, 1000, 2000], R, eps_law="cauchy")
    with open("results/simulation.json", "w") as f:
        json.dump(res, f, indent=2)
    print("\nwrote results/simulation.json")


if __name__ == "__main__":
    main()
