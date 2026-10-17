#!/usr/bin/env python3
"""Deterministic endpoint analysis for the archived N=5 mixed branch.

Fits the smallest tangent-Hessian eigenvalue linearly in omega on the first/last
n archived rows.  The zero crossings are interpreted as local-stability
spinodals, not as saddle-node certificates.
"""
import argparse, csv, json, statistics
from pathlib import Path


def analyze(table: Path, expected: dict):
    with table.open(newline='') as f:
        rows = sorted(csv.DictReader(f), key=lambda r: float(r['omega']))
    positive = [r for r in rows if float(r['hess_min']) > 0]
    fits = []
    for side in ('lower', 'upper'):
        for n in expected['fit_windows']:
            pts = rows[:n] if side == 'lower' else rows[-n:]
            x = [float(r['omega']) for r in pts]
            y = [float(r['hess_min']) for r in pts]
            slope, intercept = statistics.linear_regression(x, y)
            ybar = statistics.mean(y)
            ss_res = sum((v - (slope*u + intercept))**2 for u, v in zip(x, y))
            ss_tot = sum((v - ybar)**2 for v in y)
            r2 = 1.0 - ss_res/ss_tot
            fits.append({
                'side': side, 'nfit': n, 'omega_min': min(x), 'omega_max': max(x),
                'slope': slope, 'intercept': intercept,
                'omega_zero': -intercept/slope, 'r2': r2,
            })
    brackets = []
    for left, right in zip(rows, rows[1:]):
        l0, l1 = float(left['hess_min']), float(right['hess_min'])
        if l0*l1 < 0:
            brackets.append({
                'omega_left': float(left['omega']), 'lambda_left': l0,
                'omega_right': float(right['omega']), 'lambda_right': l1,
            })
    return {
        'interpretation': 'local-stability spinodals from lambda_min zero crossings',
        'fit_definition': 'unweighted OLS: lambda_min = slope*omega + intercept',
        'primary_nfit': expected['primary_fit_window'],
        'row_count': len(rows),
        'strictly_positive_hessian_rows': len(positive),
        'positive_hessian_omega_range': [float(positive[0]['omega']), float(positive[-1]['omega'])],
        'positive_hessian_min_gap': min(float(r['gap_to_envelope']) for r in positive),
        'raw_sign_change_brackets': brackets,
        'fits': fits,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--table', type=Path, default=Path(__file__).resolve().parents[1]/'archived/mixed_branch.csv')
    p.add_argument('--expected', type=Path, default=Path(__file__).resolve().parents[1]/'expected.json')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = analyze(a.table, json.loads(a.expected.read_text()))
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text)
    else:
        print(text, end='')

if __name__ == '__main__':
    main()
