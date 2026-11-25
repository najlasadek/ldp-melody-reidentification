"""
17_run_normalised_sweeps.py
===========================
Full epsilon sweeps on all four corpora with the LENGTH-NORMALISED distance
(the distance used for the main results of the camera-ready paper).

Normalised distance: the paper's distance divided by (n-1), n = min(|a|, |b|):
    d_norm(a, b) = d_IH(a, b) / (n-1) + alpha * penalty
i.e. the fraction of mismatched intervals plus the chromaticity penalty.

Attackers (penalty terms identical to src/evaluation.py and
adaptive_aware/src/adaptive_evaluation.py):
  agnostic          : |chi(m') - chi(c)|
  aware heuristic   : M-Scale chi(c);  M-Random |chi(m') - (535/1716 + 36/143 chi(c))|
  aware adaptive    : M-Scale chi(c);  M-Random |chi(m') - (P_dc(p) + (P_cc(p)-P_dc(p)) chi(c))|
Same mechanisms, seeds (seed_random = trial*10_000 + idx, seed_scale = +5_000)
and tie-breaking (lowest index wins) as the original code. The near-diatonic
subset (chi < 0.05) is evaluated as its own candidate database, as before.

Modes
-----
  --validate : run WITHOUT normalisation and check every hit count against the
               stored results (results/*_eps_sweep.json and
               ../adaptive_aware/results/adaptive_*.json). Must pass before the
               normalised results are used.
  (default)  : run WITH normalisation and write
               results_normalised/<corpus>_eps_sweep.json
               results_normalised/adaptive_<corpus>.json
               in the same format as the original result files.

Usage (from experiments_extended/):
    python scripts/17_run_normalised_sweeps.py --validate
    python scripts/17_run_normalised_sweeps.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from math import exp, log
from multiprocessing import Pool
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation import _PROP1_A, _PROP1_B, _wilson_ci  # noqa: E402
from src.mechanisms import m_random, m_scale  # noqa: E402
from src.metrics import ALPHA, chromatic_fraction, interval_hamming  # noqa: E402

CORPORA = {
    "europa":   ("europa_200.json",              "adaptive_europa_200.json",   "europa_200"),
    "meertens": ("processed/meertens_500.json",  "adaptive_meertens_500.json", "meertens_500"),
    "symbtr":   ("processed/symbtr_200.json",    "adaptive_symbtr_200.json",   "symbtr_200"),
    "jazz":     ("processed/jazz_200.json",      "adaptive_jazz_200.json",     "jazz_200"),
}
EXP_EPS = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]   # reported range ln3 .. ln16
NUM_TRIALS = 30
OUT_DIR = BASE_DIR / "results_normalised"


def keep_prob(epsilon: float) -> float:
    e = exp(epsilon)
    return e / (e + 6)


def adaptive_coeffs(epsilon: float) -> tuple[float, float]:
    p = keep_prob(epsilon)
    return (60 - 35 * p) / 132, (84 * p - 12) / 132


def top1(perturbed, database, chis, penalty, normalise):
    """Index of the nearest candidate (ties -> lowest index)."""
    chi_p = chromatic_fraction(perturbed)
    best_i, best_d = -1, None
    for i, (cand, chi_c) in enumerate(zip(database, chis)):
        n = min(len(perturbed), len(cand))
        d = interval_hamming(perturbed, cand) + ALPHA * (n - 1) * penalty(chi_p, chi_c)
        if normalise:
            d = d / (n - 1)
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return best_i


def evaluate(dataset, epsilon, normalise, with_adaptive):
    """Hit counts for all attackers on one candidate database."""
    chis = [chromatic_fraction(m) for m in dataset]
    a, b = adaptive_coeffs(epsilon)
    pen = {
        "agnostic":       lambda cp, cc: abs(cp - cc),
        "aware_scale":    lambda cp, cc: cc,
        "heur_random":    lambda cp, cc: abs(cp - (_PROP1_A + _PROP1_B * cc)),
        "adaptive_random": lambda cp, cc: abs(cp - (a + b * cc)),
    }
    hits = {k: 0 for k in ["ag_r", "ag_s", "aw_r", "aw_s", "ad_r"]}
    for trial in range(NUM_TRIALS):
        for idx, melody in enumerate(dataset):
            pr = m_random(melody, epsilon, seed=trial * 10_000 + idx)
            ps = m_scale(melody, epsilon, seed=trial * 10_000 + idx + 5_000)
            hits["ag_r"] += top1(pr, dataset, chis, pen["agnostic"], normalise) == idx
            hits["ag_s"] += top1(ps, dataset, chis, pen["agnostic"], normalise) == idx
            hits["aw_r"] += top1(pr, dataset, chis, pen["heur_random"], normalise) == idx
            hits["aw_s"] += top1(ps, dataset, chis, pen["aware_scale"], normalise) == idx
            if with_adaptive:
                hits["ad_r"] += top1(pr, dataset, chis, pen["adaptive_random"], normalise) == idx
    hits["n"] = NUM_TRIALS * len(dataset)
    return hits


def task(args):
    corpus, k, normalise = args
    t0 = time.time()
    data = json.load(open(BASE_DIR / "data" / CORPORA[corpus][0]))
    nd = [m for m in data if chromatic_fraction(m) < 0.05]
    eps = log(k)
    full = evaluate(data, eps, normalise, with_adaptive=True)
    sub = evaluate(nd, eps, normalise, with_adaptive=False) if nd else None
    return corpus, k, normalise, full, sub, len(data), len(nd), time.time() - t0


def ratio(s, r):
    return s / r if r > 0 else float("inf")


def stored_counts(corpus, k):
    """Hit counts implied by the stored (un-normalised) result files."""
    sweep = json.load(open(BASE_DIR / "results" / f"{corpus}_eps_sweep.json"))
    adap = json.load(open(BASE_DIR.parent / "adaptive_aware" / "results" / CORPORA[corpus][1]))
    e = next(x for x in sweep["epsilon_sweep"] if abs(x["exp_epsilon"] - k) < 1e-6)
    a = next(x for x in adap["epsilon_sweep"] if abs(x["exp_epsilon"] - k) < 1e-6)
    n = sweep["N"] * sweep["num_trials"]
    nn = e["near_diatonic_N"] * sweep["num_trials"]
    c = lambda v, m: round(v * m)
    return {
        "ag_r": c(e["agnostic"]["acc1_random"], n), "ag_s": c(e["agnostic"]["acc1_scale"], n),
        "aw_r": c(e["aware"]["aware_acc1_random"], n), "aw_s": c(e["aware"]["aware_acc1_scale"], n),
        "ad_r": c(a["adaptive"]["acc1_random"], n),
        "hr_r": c(a["heuristic"]["acc1_random"], n),
        "nd_aw_r": c(e["near_diatonic"]["aware"]["aware_acc1_random"], nn),
        "nd_aw_s": c(e["near_diatonic"]["aware"]["aware_acc1_scale"], nn),
        "nd_ag_r": c(e["near_diatonic"]["agnostic"]["acc1_random"], nn),
        "nd_ag_s": c(e["near_diatonic"]["agnostic"]["acc1_scale"], nn),
    }


def build_outputs(corpus, rows):
    """Write results in the same format as the original result files."""
    rows = sorted(rows, key=lambda r: r[1])
    sweep, adap = [], []
    for _, k, _, f, s, n_data, n_nd, _ in rows:
        eps = log(k)
        n = f["n"]
        ag = {"epsilon": eps, "acc1_random": f["ag_r"] / n, "acc1_scale": f["ag_s"] / n,
              "ci1_random": _wilson_ci(f["ag_r"], n), "ci1_scale": _wilson_ci(f["ag_s"], n),
              "hits_random": f["ag_r"], "hits_scale": f["ag_s"], "n_queries": n}
        aw = {"epsilon": eps, "aware_acc1_random": f["aw_r"] / n, "aware_acc1_scale": f["aw_s"] / n,
              "aware_ci1_random": _wilson_ci(f["aw_r"], n), "aware_ci1_scale": _wilson_ci(f["aw_s"], n),
              "hits_random": f["aw_r"], "hits_scale": f["aw_s"], "n_queries": n}
        entry = {"epsilon": eps, "exp_epsilon": exp(eps), "agnostic": ag, "aware": aw,
                 "agnostic_ratio": ratio(f["ag_s"], f["ag_r"]),
                 "aware_ratio": ratio(f["aw_s"], f["aw_r"]),
                 "near_diatonic_N": n_nd}
        if s:
            m = s["n"]
            entry["near_diatonic"] = {
                "agnostic": {"acc1_random": s["ag_r"] / m, "acc1_scale": s["ag_s"] / m,
                             "hits_random": s["ag_r"], "hits_scale": s["ag_s"], "n_queries": m},
                "aware": {"aware_acc1_random": s["aw_r"] / m, "aware_acc1_scale": s["aw_s"] / m,
                          "hits_random": s["aw_r"], "hits_scale": s["aw_s"], "n_queries": m},
                "aware_ratio": ratio(s["aw_s"], s["aw_r"]),
                "agnostic_ratio": ratio(s["ag_s"], s["ag_r"]),
            }
        sweep.append(entry)
        a_, b_ = adaptive_coeffs(eps)
        adap.append({
            "epsilon": eps, "exp_epsilon": exp(eps),
            "heuristic": {"acc1_scale": f["aw_s"] / n, "acc1_random": f["aw_r"] / n,
                          "ratio": ratio(f["aw_s"], f["aw_r"]),
                          "ci1_scale": _wilson_ci(f["aw_s"], n), "ci1_random": _wilson_ci(f["aw_r"], n)},
            "adaptive": {"acc1_scale": f["aw_s"] / n, "acc1_random": f["ad_r"] / n,
                         "ratio": ratio(f["aw_s"], f["ad_r"]),
                         "ci1_scale": _wilson_ci(f["aw_s"], n), "ci1_random": _wilson_ci(f["ad_r"], n),
                         "hits_random": f["ad_r"],
                         "intercept": a_, "slope": b_, "p": keep_prob(eps)},
        })
    OUT_DIR.mkdir(exist_ok=True)
    meta = {"N": rows[0][5], "num_trials": NUM_TRIALS, "distance": "length-normalised: d / (n-1)"}
    json.dump({"dataset": corpus, **meta, "epsilon_sweep": sweep},
              open(OUT_DIR / f"{corpus}_eps_sweep.json", "w"), indent=2)
    json.dump({"corpus": CORPORA[corpus][2], **meta, "epsilon_sweep": adap},
              open(OUT_DIR / CORPORA[corpus][1], "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--workers", type=int, default=11)
    args = ap.parse_args()
    normalise = not args.validate
    # Meertens tasks are the slowest: schedule them first.
    order = ["meertens", "europa", "symbtr", "jazz"]
    tasks = [(c, k, normalise) for c in order for k in EXP_EPS]
    rows = {c: [] for c in CORPORA}
    ok = True
    with Pool(args.workers) as pool:
        for res in pool.imap_unordered(task, tasks):
            corpus, k, _, f, s, *_rest, secs = res
            rows[corpus].append(res)
            if args.validate:
                want = stored_counts(corpus, k)
                got = {"ag_r": f["ag_r"], "ag_s": f["ag_s"], "aw_r": f["aw_r"], "aw_s": f["aw_s"],
                       "ad_r": f["ad_r"], "hr_r": f["aw_r"],
                       "nd_aw_r": s["aw_r"], "nd_aw_s": s["aw_s"],
                       "nd_ag_r": s["ag_r"], "nd_ag_s": s["ag_s"]}
                bad = {key: (got[key], want[key]) for key in want if got[key] != want[key]}
                ok &= not bad
                print(f"{corpus:8s} ln{k:<2d} {'OK  ' if not bad else 'FAIL'} {bad if bad else ''} ({secs:.0f}s)",
                      flush=True)
            else:
                print(f"{corpus:8s} ln{k:<2d} aware {f['aw_s']} vs {f['aw_r']}  "
                      f"agnostic {f['ag_s']} vs {f['ag_r']}  adaptive R {f['ad_r']} ({secs:.0f}s)", flush=True)
    if args.validate:
        print("\n[PASS] all counts match the stored results" if ok
              else "\n[FAIL] some counts differ from the stored results")
        sys.exit(0 if ok else 1)
    for c in CORPORA:
        build_outputs(c, rows[c])
    print(f"\nSaved results to {OUT_DIR}")


if __name__ == "__main__":
    main()
