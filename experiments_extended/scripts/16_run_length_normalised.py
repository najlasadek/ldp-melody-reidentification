"""
16_run_length_normalised.py
===========================
Robustness check: re-run the agnostic and aware heuristic attackers with a
LENGTH-NORMALISED distance.

The paper's distance is
    d(a, b) = d_IH(a, b) + alpha * (n-1) * penalty,      n = min(|a|, |b|)
where d_IH counts mismatched intervals over the first n-1 intervals. Because
d_IH is a count, shorter candidates get systematically smaller distances.
The normalised distance divides the whole expression by (n-1):
    d_norm(a, b) = d_IH(a, b) / (n-1) + alpha * penalty
i.e. the fraction of mismatched intervals plus the same chromaticity penalty.
The relative weight of the two terms is unchanged.

Everything else is identical to src/evaluation.py: the same mechanisms, the same
seeds (seed_random = trial*10_000 + idx, seed_scale = seed_random + 5_000), the
same penalty terms, and the same tie-breaking (lowest index wins).

Also records retrieval diagnostics: number of distinct candidates ever returned
as the top-1 prediction, and the share of predictions taken by the single most
frequent candidate.

Usage (from experiments_extended/):
    python scripts/16_run_length_normalised.py --check      # verify against stored results
    python scripts/16_run_length_normalised.py europa       # one corpus
    python scripts/16_run_length_normalised.py all          # all four corpora

Results saved to: results/length_normalised_<corpus>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from math import exp, log
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.evaluation import _PROP1_A, _PROP1_B, _wilson_ci  # noqa: E402
from src.mechanisms import m_random, m_scale  # noqa: E402
from src.metrics import ALPHA, chromatic_fraction, interval_hamming  # noqa: E402

CORPORA = {
    "europa":   BASE_DIR / "data" / "europa_200.json",
    "meertens": BASE_DIR / "data" / "processed" / "meertens_500.json",
    "symbtr":   BASE_DIR / "data" / "processed" / "symbtr_200.json",
    "jazz":     BASE_DIR / "data" / "processed" / "jazz_200.json",
}

# ln 3, ln 7, ln 16 span the reported range; ln 5 and ln 8 are the SymbTr and
# Europa peaks in the paper.
EPS_VALUES = [log(3), log(5), log(7), log(8), log(16)]
NUM_TRIALS = 30


def distances(perturbed, database, chis, kind, normalise):
    """Distances from one perturbed query to every candidate.

    The un-normalised formulas match src/evaluation.py term for term
    (combined_distance, _aware_dist_scale, _aware_dist_random).
    """
    chi_p = chromatic_fraction(perturbed)
    out = []
    for cand, chi_c in zip(database, chis):
        n = min(len(perturbed), len(cand))
        iv = interval_hamming(perturbed, cand)
        if kind == "agnostic":
            pen = abs(chi_p - chi_c)
        elif kind == "aware_scale":
            pen = chi_c
        elif kind == "aware_random":
            pen = abs(chi_p - (_PROP1_A + _PROP1_B * chi_c))
        else:
            raise ValueError(kind)
        d = iv + ALPHA * (n - 1) * pen
        if normalise:
            d = d / (n - 1)
        out.append(d)
    return out


def top1(dists):
    """Index of the nearest candidate; ties go to the lowest index, matching
    the stable sort used in src/evaluation.py."""
    best_i, best_d = 0, dists[0]
    for i in range(1, len(dists)):
        if dists[i] < best_d:
            best_i, best_d = i, dists[i]
    return best_i


def run(dataset, epsilon, normalise, num_trials=NUM_TRIALS):
    chis = [chromatic_fraction(m) for m in dataset]
    preds = {k: [] for k in ["agnostic_random", "agnostic_scale",
                             "aware_random", "aware_scale"]}
    hits = {k: 0 for k in preds}
    for trial in range(num_trials):
        for idx, melody in enumerate(dataset):
            seed_r = trial * 10_000 + idx
            seed_s = trial * 10_000 + idx + 5_000
            pr = m_random(melody, epsilon, seed=seed_r)
            ps = m_scale(melody, epsilon, seed=seed_s)
            for key, pert, kind in [
                ("agnostic_random", pr, "agnostic"),
                ("agnostic_scale",  ps, "agnostic"),
                ("aware_random",    pr, "aware_random"),
                ("aware_scale",     ps, "aware_scale"),
            ]:
                p = top1(distances(pert, dataset, chis, kind, normalise))
                preds[key].append(p)
                hits[key] += int(p == idx)
    n = num_trials * len(dataset)
    out = {"epsilon": epsilon, "exp_epsilon": exp(epsilon), "n_queries": n}
    for key in preds:
        cnt = Counter(preds[key])
        out[key] = {
            "hits": hits[key],
            "acc1": hits[key] / n,
            "ci1": _wilson_ci(hits[key], n),
            "distinct_predicted": len(cnt),
            "top_candidate_share": cnt.most_common(1)[0][1] / n,
        }
    for att in ["agnostic", "aware"]:
        r, s = out[f"{att}_random"]["hits"], out[f"{att}_scale"]["hits"]
        out[f"{att}_ratio"] = s / r if r > 0 else None
    return out


def load(name):
    with open(CORPORA[name]) as fh:
        return json.load(fh)


def check():
    """Without normalisation, the script must reproduce the stored Europa
    results at ln 8 exactly (paper Table 4)."""
    res = run(load("europa"), log(8), normalise=False)
    got = (res["aware_scale"]["hits"], res["aware_random"]["hits"],
           res["agnostic_scale"]["hits"], res["agnostic_random"]["hits"])
    stored = json.load(open(BASE_DIR / "results" / "europa_eps_sweep.json"))
    e = next(x for x in stored["epsilon_sweep"] if abs(x["exp_epsilon"] - 8) < 1e-6)
    q = res["n_queries"]
    want = (round(e["aware"]["aware_acc1_scale"] * q), round(e["aware"]["aware_acc1_random"] * q),
            round(e["agnostic"]["acc1_scale"] * q), round(e["agnostic"]["acc1_random"] * q))
    print(f"un-normalised ln8 (aware S, aware R, agnostic S, agnostic R): got {got}, stored {want}")
    if got != want:
        print("[FAIL] does not reproduce stored results")
        sys.exit(1)
    print("[PASS] reproduces stored results exactly")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="?", choices=list(CORPORA) + ["all"])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        check()
        return
    names = list(CORPORA) if args.corpus == "all" else [args.corpus]
    for name in names:
        data = load(name)
        print(f"{name}: N={len(data)}", flush=True)
        results = []
        for eps in EPS_VALUES:
            t0 = time.time()
            r = run(data, eps, normalise=True)
            results.append(r)
            print(f"  ln{exp(eps):.0f}: aware S={r['aware_scale']['hits']} R={r['aware_random']['hits']} "
                  f"ratio={r['aware_ratio']}  agnostic S={r['agnostic_scale']['hits']} "
                  f"R={r['agnostic_random']['hits']}  ({time.time()-t0:.0f}s)", flush=True)
        out = BASE_DIR / "results" / f"length_normalised_{name}.json"
        with open(out, "w") as fh:
            json.dump({"corpus": name, "N": len(data), "num_trials": NUM_TRIALS,
                       "distance": "d / (n-1)", "results": results}, fh, indent=2)
        print(f"Saved {out}", flush=True)


if __name__ == "__main__":
    main()
