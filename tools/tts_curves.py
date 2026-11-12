"""Test-time scaling curves for TROGraspV2.

Reads
  --pred    the vis_info file written by test.py (per-candidate q, ids, IK residuals)
  --labels  the per-candidate labels written by validation/collision_check.py
            (<sim-root>/<step>/<hand>/<output-name>.pt)
and reports, for n = 1, 2, 4, ... up to the number of candidates per scene:

  single      expected success of one random candidate
  pass@n      oracle upper bound: P(at least one of n candidates succeeds),
              unbiased estimator 1 - C(m-c, n) / C(m, n) per scene
  best-of-n   expected success of the candidate a verifier picks (lowest score)
              from a uniformly random n-subset of the scene's m candidates,
              computed exactly; with a perfect verifier it equals pass@n
  recovery    (best-of-n - single) / (pass@n - single)

for three labels: success (= lifted AND non_collision, the paper's metric),
lifted (Isaac lift test only) and non_collision (point-cloud check only).
Averages follow the repo's convention: mean over scenes within an object,
then mean over objects.

Verifier scores (lower = better):
  random                  uniform random pick (= single, for reference)
  ik_pos / ik_rot / ik_select   per-candidate IK residuals from test.py
  oracle_noncollision     PARTIAL ORACLE: the grader's own collision check,
                          ties broken by sample order; reported only as a
                          reference line, because the success label contains it
  oracle_noncollision+ik_pos    same filter, then lowest IK position residual

Example
  python tools/tts_curves.py \
      --pred   $EXP/val/xhand/tts_N16_seed0.pth \
      --labels $EXP/sim/tts_N16_seed0/xhand/sr_final_surface_pc.pt \
      --out    $EXP/analysis/xhand_tts_N16_seed0.json --plot
"""

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch

LABELS = ["success", "lifted", "non_collision"]


def load(pred_path, labels_path):
    vis_info = torch.load(pred_path, weights_only=False)
    labels = torch.load(labels_path, weights_only=False)
    by_meta = {info["meta_name"]: info for info in vis_info}

    objects = {}  # meta -> {"scene": [m], "sample": [m], "<label>": [m], "<score>": [m]}
    missing = 0
    for meta, lab in labels.items():
        if meta not in by_meta:
            missing += 1
            continue
        info = by_meta[meta]
        m = len(lab["success"])
        rec = {}
        rec["scene"] = np.asarray(lab.get("scene_ids", torch.as_tensor(info["valid_scene_ids"]).tolist()))
        rec["sample"] = np.asarray(lab.get("sample_ids", torch.as_tensor(
            info.get("sample_ids", torch.zeros(m, dtype=torch.long))).tolist()))
        for key in LABELS:
            rec[key] = np.asarray(lab[key], dtype=bool)
        for key in ["ik_pos_err", "ik_rot_err", "ik_select_err"]:
            if key in info:
                rec[key] = torch.as_tensor(info[key]).float().numpy()
        for key, values in lab.get("collision_stats", {}).items():
            rec["cc_" + key] = np.asarray(values, dtype=np.float32)
        if any(len(v) != m for v in rec.values()):
            raise ValueError(f"{meta}: per-candidate arrays have different lengths")
        objects[meta] = rec
    if missing:
        print(f"[warn] {missing} labelled objects have no prediction entry; skipped")
    return objects


def build_scores(rec):
    m = len(rec["success"])
    scores = {"random": None}
    if "ik_pos_err" in rec:
        scores["ik_pos"] = rec["ik_pos_err"].astype(np.float64)
    if "ik_rot_err" in rec:
        scores["ik_rot"] = rec["ik_rot_err"].astype(np.float64)
    if "ik_select_err" in rec:
        scores["ik_select"] = rec["ik_select_err"].astype(np.float64)
    collide = (~rec["non_collision"]).astype(np.float64)
    scores["oracle_noncollision"] = collide  # ties broken by sample id
    if "ik_pos_err" in rec:
        ik = rec["ik_pos_err"].astype(np.float64)
        scores["oracle_noncollision+ik_pos"] = collide * (1.0 + ik.max() + 1.0) + ik
    assert all(v is None or len(v) == m for v in scores.values())
    return scores


def pass_at_n(m, c, n):
    if n > m:
        return float("nan")
    if m - c < n:
        return 1.0
    return 1.0 - math.comb(m - c, n) / math.comb(m, n)


def best_of_n(label, score, sample, n):
    """Expected label of the verifier's pick from a uniformly random n-subset.

    Candidates are ranked by (score, sample id); the candidate at rank r is the
    pick of a random n-subset iff it is in the subset and all lower ranks are
    not, which has probability C(m-1-r, n-1) / C(m, n). Exact, no sampling
    noise, and with a perfect score it reduces to the unbiased pass@n.
    `score=None` means a random pick, whose expectation is the mean label.
    """
    m = len(label)
    if n > m:
        return float("nan")
    if score is None:
        return float(np.mean(label))
    order = np.lexsort((sample, score))
    denom = math.comb(m, n)
    weights = np.array([math.comb(m - 1 - r, n - 1) for r in range(m)], dtype=np.float64) / denom
    return float(np.dot(label[order].astype(np.float64), weights))


def nanmean(xs):
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def analyse(objects, ns):
    """Per-object mean over scenes, then mean over objects."""
    result = {lab: {"single": None, "pass@n": {}, "best_of_n": defaultdict(dict)} for lab in LABELS}
    score_names = None
    per_obj = {lab: {"single": [], "pass": defaultdict(list), "bon": defaultdict(lambda: defaultdict(list))}
               for lab in LABELS}

    for meta, rec in objects.items():
        scores = build_scores(rec)
        if score_names is None:
            score_names = list(scores)
        scenes = defaultdict(list)
        for i, s in enumerate(rec["scene"]):
            scenes[int(s)].append(i)
        for lab in LABELS:
            sc_single, sc_pass, sc_bon = [], defaultdict(list), defaultdict(lambda: defaultdict(list))
            for s, ids in scenes.items():
                ids = np.asarray(ids)
                y = rec[lab][ids]
                m, c = len(ids), int(y.sum())
                sc_single.append(c / m)
                for n in ns:
                    sc_pass[n].append(pass_at_n(m, c, n))
                    for name in score_names:
                        sc_bon[name][n].append(best_of_n(
                            y, None if scores[name] is None else scores[name][ids], rec["sample"][ids], n))
            per_obj[lab]["single"].append(float(np.mean(sc_single)))
            for n in ns:
                per_obj[lab]["pass"][n].append(nanmean(sc_pass[n]))
                for name in score_names:
                    per_obj[lab]["bon"][name][n].append(nanmean(sc_bon[name][n]))

    for lab in LABELS:
        single = nanmean(per_obj[lab]["single"])
        result[lab]["single"] = single
        for n in ns:
            p = nanmean(per_obj[lab]["pass"][n])
            result[lab]["pass@n"][n] = p
            for name in score_names:
                b = nanmean(per_obj[lab]["bon"][name][n])
                gap = p - single
                result[lab]["best_of_n"][name][n] = {
                    "value": b,
                    "recovery": (b - single) / gap if gap > 1e-9 else float("nan"),
                }
        result[lab]["best_of_n"] = dict(result[lab]["best_of_n"])
    return result, score_names


def failure_breakdown(objects):
    """Candidate-level and scene-level failure statistics."""
    tot = succ = coll = lift_only = 0
    scenes = all_fail = 0
    all_fail_lift_only = 0
    for rec in objects.values():
        tot += len(rec["success"])
        succ += int(rec["success"].sum())
        coll += int((~rec["non_collision"]).sum())
        lift_only += int((rec["non_collision"] & ~rec["lifted"]).sum())
        by_scene = defaultdict(list)
        for i, s in enumerate(rec["scene"]):
            by_scene[int(s)].append(i)
        for ids in by_scene.values():
            scenes += 1
            if not rec["success"][ids].any():
                all_fail += 1
                if rec["non_collision"][ids].any():
                    all_fail_lift_only += 1
    return {
        "candidates": tot,
        "success_rate_candidates": succ / max(tot, 1),
        "fail_collision": coll / max(tot, 1),
        "fail_lift_only": lift_only / max(tot, 1),
        "scenes": scenes,
        "scenes_all_N_fail": all_fail / max(scenes, 1),
        "scenes_all_N_fail_but_some_noncollision": all_fail_lift_only / max(scenes, 1),
    }


def print_report(result, score_names, ns, breakdown):
    print(json.dumps(breakdown, indent=2))
    for lab in LABELS:
        r = result[lab]
        print(f"\n=== label: {lab}   single = {r['single']:.4f}")
        header = f"{'n':>4} {'pass@n':>8} " + " ".join(f"{s[:22]:>24}" for s in score_names)
        print(header)
        for n in ns:
            cells = []
            for s in score_names:
                v = r["best_of_n"][s][n]
                cells.append(f"{v['value']:.4f} ({v['recovery']:+.2f})".rjust(24))
            print(f"{n:>4} {r['pass@n'][n]:>8.4f} " + " ".join(cells))
    print("\n(best-of-n value, recovery ratio in brackets; "
          "oracle_* rows reuse the grader's collision check = partial oracle)")


def plot(result, score_names, ns, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(LABELS), figsize=(5 * len(LABELS), 4), sharex=True)
    for ax, lab in zip(axes, LABELS):
        r = result[lab]
        ax.plot(ns, [r["pass@n"][n] for n in ns], "k-o", label="pass@n (oracle)")
        for s in score_names:
            style = ":" if s.startswith("oracle") else "-"
            ax.plot(ns, [r["best_of_n"][s][n]["value"] for n in ns], style, marker=".", label=s)
        ax.axhline(r["single"], color="gray", ls="--", lw=1, label="single")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns)
        ax.set_xticklabels([str(n) for n in ns])
        ax.set_title(lab)
        ax.set_xlabel("candidates per scene (n)")
        ax.set_ylabel("success rate")
        ax.grid(alpha=0.3)
    axes[-1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved plot to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default=None, help="write the full result as JSON")
    ap.add_argument("--plot", action="store_true", help="also save <out>.png (needs matplotlib)")
    ap.add_argument("--max-n", type=int, default=None)
    args = ap.parse_args()

    objects = load(args.pred, args.labels)
    if not objects:
        raise SystemExit("no objects with both predictions and labels")
    min_m = min(np.bincount(rec["scene"].astype(int)).max() for rec in objects.values())
    max_n = args.max_n or min_m
    ns = [n for n in (2 ** k for k in range(0, 16)) if n <= max_n]

    result, score_names = analyse(objects, ns)
    breakdown = failure_breakdown(objects)
    print_report(result, score_names, ns, breakdown)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"ns": ns, "objects": len(objects), "breakdown": breakdown,
                       "curves": result}, f, indent=2, default=float)
        print(f"saved {args.out}")
        if args.plot:
            plot(result, score_names, ns, os.path.splitext(args.out)[0] + ".png")


if __name__ == "__main__":
    main()
