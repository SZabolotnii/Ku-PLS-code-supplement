"""A3 -- the two inequalities behind Prop. (decay exponent), checked before they
went into the manuscript (2026-08-22).

  Claim A (upper):  d_j(theta) <= 1 for independent coordinates, theta >= 0.
                    Proof uses ONLY independence of Z_j from S_j = sum_{k!=j} Z_k^2
                    (monotone-covariance inequality), so it must also hold on an
                    ASYMMETRIC independent law, where A_theta is not diagonal.
  Claim B (lower):  inf_j d_j >= E[(M + ||Z||^2)^{-theta/2}] / (2 E||Z||^{-theta}),
                    M = 2 * kurtosis * sigma_1^2.
  Refutation row:   a DEPENDENT construction in which Z_2^2 is large exactly
                    when Z_1^2 is small must push d_1 above 1.  If the statistic
                    could not see that, a clean table would mean nothing.

Batch-mean standard errors (40 x 25k draws), KNOWN sigma_j in the denominator,
max-z over all 200 coordinates.  With 2400 tests the null max z sits near 3.5.
Output: ../results/a3_dj_bounds.out
"""
import numpy as np
rng = np.random.default_rng(20260822)
J, B, nb = 200, 40, 25_000          # 40 batches x 25k = 1e6 draws
sig = 1.0 / np.arange(1, J + 1)

def batches(gen):
    for _ in range(B):
        yield gen(nb)

def run(name, gen, k4):
    print(f"\n== {name} ==")
    for th in (0.5, 1.0, 1.5, 2.0):
        D, L = [], []
        for Z in batches(gen):
            n2 = np.sum(Z**2, 1); w = n2 ** (-th/2)
            D.append((w[:, None] * Z**2).mean(0) / (sig**2 * w.mean()))
            M = 2 * k4 * sig[0]**2
            L.append(np.mean((M + n2) ** (-th/2)) / (2 * w.mean()))
        D = np.array(D); d = D.mean(0); se = D.std(0, ddof=1) / np.sqrt(B)
        z = (d - 1) / se
        bound = float(np.mean(L))
        print(f"  theta={th:3.1f}  max z_j={z.max():5.2f}  (#z>4: {(z>4).sum()})   "
              f"d_1={d[0]:.4f}  min d_j={d.min():.4f}  lower bound={bound:.4f}  "
              f"[{'ok' if bound < d.min() else 'VIOLATED'}]")

run("Gaussian (tab:shrink design)", lambda n: rng.standard_normal((n, J)) * sig, 3.0)
run("Laplace", lambda n: rng.laplace(0, 1/np.sqrt(2), (n, J)) * sig, 6.0)
run("centred exponential (asymmetric, independent)", lambda n: (rng.exponential(1.0, (n, J)) - 1.0) * sig, 9.0)

print("\n== refutation design (dependent), same statistic ==")
D = []
for _ in range(B):
    Z1 = rng.standard_normal(nb); Z2 = np.sign(rng.standard_normal(nb)) / np.sqrt(Z1**2 + 0.05)
    Z = np.stack([Z1, Z2], 1); w = np.sum(Z**2, 1) ** (-0.5)
    D.append((w[:, None] * Z**2).mean(0) / (Z.var(0) * w.mean()))
D = np.array(D); d = D.mean(0); se = D.std(0, ddof=1)/np.sqrt(B)
print(f"  theta=1: z_1 = {(d[0]-1)/se[0]:.1f}  (d_1={d[0]:.3f})  -> the statistic sees a real violation at once")
