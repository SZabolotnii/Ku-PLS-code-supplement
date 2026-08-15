"""Emit the manuscript's LaTeX tables straight from the stored results.

This is the only path from a computation to the paper: no number in Table 1 or
Table 2 is transcribed by hand.
"""
import json
import pathlib

RES = pathlib.Path("results")
OUT = pathlib.Path("results")


def _med(v):
    if v >= 1000:
        return f"{v:,.0f}".replace(",", r"\,")
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def simulation_table():
    rows = json.load(open(RES / "simulation.json"))
    main = [r for r in rows if r["eps"] == "gauss"]
    cauchy = [r for r in rows if r["eps"] == "cauchy"]
    R = main[0]["reps"]
    L = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Empirical size and size-corrected power at the nominal $5\%$"
        r" level, $\alpha$-stable functional predictor, Gaussian errors. $T_n$"
        r" is the BCT moment statistic, $S_n$ the ECF statistic of"
        r" \cref{thm:inference}; each is calibrated by its own plug-in"
        r" weighted-$\chi^2$ spectrum. Power is against the local alternative"
        r" $b_1\mapsto b_1(1+n^{-1/2})$ and is size-corrected at the empirical"
        rf" null quantile. $\operatorname{{med}}$ is the median of the null"
        rf" statistic. {R} replications per cell.}}",
        r"\label{tab:sim}",
        r"\begin{tabular}{cc rr rr rr}", r"\toprule",
        r"& & \multicolumn{2}{c}{size} & \multicolumn{2}{c}{power}"
        r" & \multicolumn{2}{c}{$\operatorname{med}$ (null)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"$\alpha$ & $n$ & $T_n$ & $S_n$ & $T_n$ & $S_n$ & $T_n$ & $S_n$ \\",
        r"\midrule",
    ]
    prev = None
    for r in main:
        if prev is not None and r["alpha"] != prev:
            L.append(r"\addlinespace")
        prev = r["alpha"]
        L.append(f"{r['alpha']:.1f} & {r['n']} & {r['size_T']:.3f} & "
                 f"{r['size_S']:.3f} & {r['pow_T']:.3f} & {r['pow_S']:.3f} & "
                 f"{_med(r['med_T'])} & {_med(r['med_S'])} \\\\")
    if cauchy:
        L += [r"\midrule",
              r"\multicolumn{8}{l}{\emph{Cauchy errors} "
              r"($\E|\varepsilon|=\infty$), $\alpha=1.5$:}\\"]
        for r in cauchy:
            L.append(f"{r['alpha']:.1f} & {r['n']} & {r['size_T']:.3f} & "
                     f"{r['size_S']:.3f} & {r['pow_T']:.3f} & "
                     f"{r['pow_S']:.3f} & {_med(r['med_T'])} & "
                     f"{_med(r['med_S'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_sim.tex").write_text("\n".join(L) + "\n")


def empirical_table():
    d = json.load(open(RES / "empirical.json"))
    base = [p for p in d["panels"] if p["m"] == 3]
    tails = d["tails"]

    def stars(p):
        return "^{***}" if p < .01 else "^{**}" if p < .05 else "^{*}" if p < .1 else ""

    L = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Yield-curve shocks and equity returns, "
        rf"{d['start']} to {d['end']} ({d['n']} trading days). $X_t$ is the "
        r"curve of daily yield changes across eleven maturities (1M--30Y) in "
        r"basis points; $Y_t$ is the S\&P~500 daily log return in percent. "
        r"Both statistics test $H_0:\beta=0$ with $m=3$ conjugate-gradient "
        r"steps; $p$-values come from each statistic's own plug-in "
        r"weighted-$\chi^2$ spectrum. Hill tail-index estimates use the "
        r"largest $5\%$ of observations. BCT require a tail index above~4. "
        r"$^{*}$, $^{**}$, $^{***}$ denote $p<0.1$, $0.05$, $0.01$.}",
        r"\label{tab:empirical}",
        r"\begin{tabular}{l r rr rr}", r"\toprule",
        r"& & \multicolumn{2}{c}{$T_n$ (moment)} "
        r"& \multicolumn{2}{c}{$S_n$ (ECF)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Panel & $n$ & statistic & $p$ & statistic & $p$ \\", r"\midrule",
    ]
    names = {"A contemporaneous": r"A. $Y_t$ on $X_t$",
             "B predictive t+1": r"B. $Y_{t+1}$ on $X_t$"}
    for p in base:
        L.append(f"{names.get(p['panel'], p['panel'])} & {p['n']} & "
                 f"{p['T']:.1f} & ${p['p_T']:.4f}{stars(p['p_T'])}$ & "
                 f"{p['S']:.4f} & ${p['p_S']:.4f}{stars(p['p_S'])}$ \\\\")
    e = d["ex2020"]
    L.append(r"\addlinespace")
    L.append(rf"\quad B, excluding 2020 & {e['n']} & {e['T']:.1f} & "
             f"${e['p_T']:.4f}{stars(e['p_T'])}$ & {e['S']:.4f} & "
             f"${e['p_S']:.4f}{stars(e['p_S'])}$ \\\\")
    L += [r"\midrule",
          r"\multicolumn{6}{l}{\emph{Hill tail-index estimates} "
          r"$\hat\alpha$ (top $5\%$):}\\",
          rf"\multicolumn{{6}}{{l}}{{\quad $\lVert X_t\rVert$: "
          rf"{tails['||X_t||']['5%']:.2f}\qquad $|Y_t|$: "
          rf"{tails['|Y_t|']['5%']:.2f}}}\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "table_empirical.tex").write_text("\n".join(L) + "\n")


def main():
    if (RES / "simulation.json").exists():
        simulation_table()
        print("wrote results/table_sim.tex")
    if (RES / "empirical.json").exists():
        empirical_table()
        print("wrote results/table_empirical.tex")


if __name__ == "__main__":
    main()
