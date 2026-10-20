#!/usr/bin/env python3
"""Check the sign convention of the Diff (Intelligence Gain) column in a
permutation_p_values_summary_*.csv drop.

Run this against the CSVs before and after the pipeline fix:

    python check_diff_sign.py reports/26_09_03/*.csv

THE INVARIANT
-------------
IG is defined as CE_small - CE_big, so for any token series H, S, B:

    r(H, S-B) = [sd_S * r(H,S)  -  sd_B * r(H,B)] / sd_{S-B}

The denominator and both standard deviations are positive. Therefore, whenever
r(H,S) and r(H,B) have OPPOSITE signs, both terms in the numerator carry the
sign of r(H,S), and so:

    sign( r(H, S-B) )  ==  sign( r(H,S) )        [forced, no data needed]

Rows where r(H,S) and r(H,B) share a sign are unconstrained and are skipped.

A row that fails this is not "surprising" -- it is arithmetically impossible for
the stated definition of IG, and means the Diff column was computed against
CE_big - CE_small (or with loss_small / loss_big swapped upstream).

Exit code 0 if every testable row is consistent, 1 otherwise.
"""
import sys, glob
import numpy as np
import pandas as pd

FAMILY = {"Qwen": "Qwen", "ernie": "ERNIE", "gemma": "Gemma", "liquid": "Liquid"}


def load(paths):
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["source"] = p.split("/")[-1]
        # metric name from the filename suffix, e.g. ..._summary_max_20pct.csv
        stem = d["source"].iloc[0].replace("permutation_p_values_summary_", "").replace(".csv", "")
        d["metric"] = stem
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["family"] = d["Model"].map(FAMILY).fillna(d["Model"])
    d["variant"] = np.where(d["HC_File"].str.contains("hat"), "hat", "standard")
    return d


def check(d):
    rs, rb, rd = (d["Observed_Corr_Small"], d["Observed_Corr_Big"], d["Observed_Corr_Diff"])
    d = d.copy()
    d["testable"] = np.sign(rs) * np.sign(rb) < 0
    d["consistent"] = np.where(d["testable"], np.sign(rd) == np.sign(rs), np.nan)
    return d


def main(paths):
    if not paths:
        print(__doc__)
        return 2
    d = check(load(paths))
    t = d[d["testable"]]
    n_bad = int((~t["consistent"].astype(bool)).sum())

    print(f"rows total          : {len(d)}")
    print(f"rows testable       : {len(t)}   (r_small and r_big have opposite signs)")
    print(f"rows INCONSISTENT   : {n_bad}\n")

    piv = (t.assign(bad=~t["consistent"].astype(bool))
             .pivot_table(index="family", columns="metric", values="bad",
                          aggfunc=["sum", "count"], observed=True))
    print("inconsistent / testable, by family x metric:")
    fams = sorted(t["family"].unique())
    mets = sorted(t["metric"].unique())
    print(f"  {'family':8s} " + "  ".join(f"{m:>14s}" for m in mets))
    for f in fams:
        cells = []
        for m in mets:
            sub = t[(t.family == f) & (t.metric == m)]
            bad = int((~sub["consistent"].astype(bool)).sum())
            cells.append(f"{bad:>6d} /{len(sub):>6d}")
        print(f"  {f:8s} " + "  ".join(f"{c:>14s}" for c in cells))

    if n_bad:
        print("\nworst offenders (largest |r_diff| among inconsistent rows):")
        bad = t[~t["consistent"].astype(bool)].reindex(
            t[~t["consistent"].astype(bool)]["Observed_Corr_Diff"].abs()
             .sort_values(ascending=False).index).head(10)
        cols = ["family", "Dataset", "Estimator", "variant", "metric",
                "Observed_Corr_Big", "Observed_Corr_Small", "Observed_Corr_Diff"]
        print(bad[cols].to_string(index=False,
              formatters={c: (lambda v: f"{v:+.4f}") for c in cols[-3:]}))
        print("\nFAIL: the Diff column does not follow IG = CE_small - CE_big "
              "for the families listed above.")
        return 1

    print("\nOK: every testable row is consistent with IG = CE_small - CE_big.")
    return 0


if __name__ == "__main__":
    args = []
    for a in sys.argv[1:]:
        args.extend(glob.glob(a))
    sys.exit(main(args))
