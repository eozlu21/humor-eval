#!/usr/bin/env python3
"""Compare two results JSON files (simple vs reasoned) and print deltas.

Usage:
  python compare_results.py simple.json reasoned.json
"""
import json
import argparse
from typing import Any, Dict


def load(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def compute_stats(data):
    results = data['results']
    correct = sum(1 for r in results if r['is_correct'])
    total = len(results)
    return correct, total, correct / total if total else 0.0


def compare(simple_path: str, reasoned_path: str):
    s = load(simple_path)
    r = load(reasoned_path)
    sc, st, sa = compute_stats(s)
    rc, rt, ra = compute_stats(r)

    print("="*70)
    print("COMPARISON: SIMPLE vs REASONED")
    print("="*70)
    print(f"Simple  : {sc}/{st} = {sa:.3f} ({sa*100:.1f}%)")
    print(f"Reasoned: {rc}/{rt} = {ra:.3f} ({ra*100:.1f}%)")
    diff = ra - sa
    print(f"Delta accuracy (reasoned - simple): {diff:+.3f} ({diff*100:+.1f} pp)")

    # Per-item differences where one mode is correct and the other isn't
    print("\nItems where reasoned helped (simple wrong, reasoned correct):")
    helped = []
    hurt = []
    for s_res, r_res in zip(s['results'], r['results']):
        if (not s_res['is_correct']) and r_res['is_correct']:
            helped.append(s_res['contest_number'])
        if s_res['is_correct'] and (not r_res['is_correct']):
            hurt.append(s_res['contest_number'])
    print(helped if helped else '  (none)')
    print("Items where reasoning hurt (simple correct, reasoned wrong):")
    print(hurt if hurt else '  (none)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('simple_file')
    ap.add_argument('reasoned_file')
    args = ap.parse_args()
    compare(args.simple_file, args.reasoned_file)
