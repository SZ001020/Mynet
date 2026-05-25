#!/usr/bin/env python3
"""P8-3 Analysis: Compare CTRL vs CA-A vs AB-A eval results.
Run after training completes: python analyze_step1.py
Reads eval JSONs from run directories and produces comparison table.
"""

import json, os, sys
from pathlib import Path

RUNS = "/root/autodl-tmp/runs"
CLASS_NAMES = ["road", "building", "grass", "tree", "car"]


def find_latest(pattern: str) -> str | None:
    dirs = sorted(Path(RUNS).glob(pattern), key=os.path.getmtime, reverse=True)
    return str(dirs[0]) if dirs else None


def load_eval(run_pattern: str) -> dict:
    d = find_latest(run_pattern)
    if not d:
        return {}
    jf = os.path.join(d, "eval_256_vaihingen.json")
    if not os.path.exists(jf):
        jf = os.path.join(d, "eval_256_vaihingen.json")
        if not os.path.exists(jf):
            print(f"  WARNING: no eval JSON in {d}")
            return {}
    with open(jf) as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("Plan8 Step 1 Analysis: CTRL vs CA-A vs AB-A")
    print("=" * 70)

    results = {}
    for name, pattern in [
        ("CTRL", "plan8_ctrl_*"),
        ("CA-A", "plan8_ca_a_*"),
        ("AB-A", "plan8_ab_a_*"),
    ]:
        data = load_eval(pattern)
        if data:
            results[name] = data
            miou = data.get("avg_miou", 0)
            oa = data.get("avg_oa", 0)
            pci = data.get("per_class_iou", {})
            print(f"\n{name}: OA={oa:.4f}, mIoU={miou:.4f}")
            for cls in CLASS_NAMES:
                print(f"  {cls}: {pci.get(cls, 0):.2f}")

    if len(results) < 3:
        print("\n⚠  Not all eval results found. Check training completion.")
        return

    # Comparison table
    ctrl = results["CTRL"]
    ca = results["CA-A"]
    ab = results["AB-A"]

    print("\n" + "=" * 70)
    print("Comparison (Δ vs CTRL)")
    print("-" * 70)
    print(f"{'Metric':<14} {'CTRL':>8} {'CA-A':>8} {'ΔCA':>8} {'AB-A':>8} {'ΔAB':>8}")
    print("-" * 70)

    for metric, key in [("mIoU", "avg_miou"), ("OA", "avg_oa")]:
        cv = ctrl.get(key, 0)
        cav = ca.get(key, 0)
        abv = ab.get(key, 0)
        print(f"{metric:<14} {cv:>8.2f} {cav:>8.2f} {cav-cv:>+8.2f} {abv:>8.2f} {abv-cv:>+8.2f}")

    print("-" * 70)
    for cls in CLASS_NAMES:
        cv = ctrl.get("per_class_iou", {}).get(cls, 0)
        cav = ca.get("per_class_iou", {}).get(cls, 0)
        abv = ab.get("per_class_iou", {}).get(cls, 0)
        print(f"{cls:<14} {cv:>8.2f} {cav:>8.2f} {cav-cv:>+8.2f} {abv:>8.2f} {abv-cv:>+8.2f}")

    # Decision logic
    print("\n" + "=" * 70)
    print("Decision Guidance")
    print("-" * 70)

    ctrl_miou = ctrl.get("avg_miou", 0)
    ca_miou = ca.get("avg_miou", 0)
    ab_miou = ab.get("avg_miou", 0)

    # CTRL validation
    if ctrl_miou < 75.0:
        print("⚠  CTRL mIoU < 75% — pipeline may have issues, check implementation")
    else:
        print(f"✓  CTRL mIoU={ctrl_miou:.2f}% — pipeline validated")

    # CA-A decision
    ca_delta = ca_miou - ctrl_miou
    if ca_delta > 0.5:
        print(f"✓  CA-A: +{ca_delta:.2f}pp — cross-attn is EFFECTIVE → proceed to CA-B")
    elif ca_delta > 0.3:
        print(f"~  CA-A: +{ca_delta:.2f}pp — marginal gain, CA-B still worth trying")
    elif ca_delta > -0.5:
        print(f"✗  CA-A: {ca_delta:+.2f}pp — no meaningful gain, cross-attn may be ineffective")
    else:
        print(f"✗✗ CA-A: {ca_delta:+.2f}pp — cross-attn DEGRADES, stop Chain 1")

    # AB-A decision
    ab_delta = ab_miou - ctrl_miou
    if ab_delta > 0.5:
        print(f"✓  AB-A: +{ab_delta:.2f}pp — attn bias is EFFECTIVE → proceed to AB-B")
    elif ab_delta > 0.3:
        print(f"~  AB-A: +{ab_delta:.2f}pp — marginal gain, AB-B still worth trying")
    elif ab_delta > -0.5:
        print(f"✗  AB-A: {ab_delta:+.2f}pp — no meaningful gain")
    else:
        print(f"✗✗ AB-A: {ab_delta:+.2f}pp — attn bias DEGRADES, stop Chain 2")

    print("\n" + "=" * 70)
    print("Plan7-A reference: mIoU=77.14%, OA=87.75%")
    print("=" * 70)


if __name__ == "__main__":
    main()
