#!/usr/bin/env python3
"""
N=5 mixed angular metastable-branch continuation.

This program generates the raw continuation table used for the mixed local minimum
of E4(X) + omega I3(X) on the unit Frobenius sphere of doubly centered matrices.
It finds a stable mixed seed near omega=0.14, continues it in both directions, and
computes the 15-dimensional tangent Hessian at every accepted point.

Endpoint estimates used in the paper are obtained from the archived table by
../analysis/analyze_spinodals.py, which fits the smallest Hessian eigenvalue itself
and interprets its zero crossings as local-stability spinodals.
"""

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from numpy.linalg import norm, eigvalsh
from scipy.linalg import null_space
from scipy.optimize import minimize

N = 5
E_L = 639.0 / 320.0
C_L = 3.0 / 10.0
E_F = 81.0 / 40.0
C_F = 9.0 / 20.0


def centered_basis(n=5):
    return null_space(np.ones((1, n), dtype=float))

B = centered_basis(N)


def y_to_x(y):
    return B @ np.asarray(y, dtype=float).reshape(4, 4) @ B.T


def x_to_y(X):
    return (B.T @ X @ B).reshape(-1)


def invariants_x(X):
    n2 = float(np.sum(X * X))
    J0 = n2 * n2
    XXt = X @ X.T
    J1 = float(np.sum(XXt * XXt))
    row2 = np.sum(X * X, axis=1)
    col2 = np.sum(X * X, axis=0)
    JL = float(np.sum(row2 * row2))
    JR = float(np.sum(col2 * col2))
    JLR = float(np.sum(X ** 4))
    I3 = float(np.sum(X ** 3))
    E4 = 3.0 / 16.0 * J1 + 1.0 / 5.0 * (JL + JR) - JLR
    Reff = (n2 * n2 / J1) if J1 > 0 else float('nan')
    return dict(norm2=n2, J0=J0, J1=J1, JL=JL, JR=JR, JLR=JLR,
                I3=I3, E4=E4, Reff=Reff)


def e4_grad_x(X):
    XXtX = (X @ X.T) @ X
    row2 = np.sum(X * X, axis=1)[:, None]
    col2 = np.sum(X * X, axis=0)[None, :]
    return ((3.0 / 4.0) * XXtX
            + (4.0 / 5.0) * (row2 + col2) * X
            - 4.0 * (X ** 3))


def i3_grad_x(X):
    return 3.0 * X * X


def angular_obj_grad(z, omega):
    z = np.asarray(z, dtype=float)
    nz = norm(z)
    if (not np.isfinite(nz)) or nz < 1e-14:
        return 1e100, np.zeros_like(z)
    y = z / nz
    X = y_to_x(y)
    inv = invariants_x(X)
    f = inv['E4'] + omega * inv['I3']
    gy = x_to_y(e4_grad_x(X) + omega * i3_grad_x(X))
    gz = (gy - y * float(np.dot(y, gy))) / nz
    return float(f), gz


def angular_envelope(omega):
    return min((E_L - 2.0) - abs(omega) * C_L,
               (E_F - 2.0) - abs(omega) * C_F)


def classify_unit(inv, tol=5e-5):
    if abs(inv['J1'] - 0.25) < tol and abs(abs(inv['I3']) - C_L) < 5*tol:
        return 'locking'
    if abs(inv['J1'] - 1.0) < tol and abs(abs(inv['I3']) - C_F) < 5*tol:
        return 'rank_one'
    return 'mixed'


def tangent_hessian(y, omega, h=2e-5):
    y = np.asarray(y, dtype=float)
    y /= norm(y)
    T = null_space(y.reshape(1, -1))
    d = T.shape[1]
    H = np.zeros((d, d))

    def loc(z):
        yy = y + T @ z
        yy /= norm(yy)
        X = y_to_x(yy)
        inv = invariants_x(X)
        return inv['E4'] + omega * inv['I3']

    z0 = np.zeros(d)
    f0 = loc(z0)
    for i in range(d):
        ei = np.zeros(d); ei[i] = h
        H[i, i] = (loc(ei) - 2*f0 + loc(-ei)) / (h*h)
    for i in range(d):
        for j in range(i+1, d):
            ei = np.zeros(d); ej = np.zeros(d)
            ei[i] = h; ej[j] = h
            H[i, j] = H[j, i] = (
                loc(ei+ej) - loc(ei-ej) - loc(-ei+ej) + loc(-ei-ej)
            ) / (4*h*h)
    return eigvalsh((H + H.T)/2.0)


def correct_from(y0, omega, maxiter=6000):
    res = minimize(lambda z: angular_obj_grad(z, omega),
                   np.asarray(y0, dtype=float), jac=True, method='L-BFGS-B',
                   options=dict(maxiter=maxiter, ftol=1e-15, gtol=5e-12, maxls=80))
    y = res.x / norm(res.x)
    inv = invariants_x(y_to_x(y))
    obj = inv['E4'] + omega * inv['I3']
    gnorm = norm(angular_obj_grad(y, omega)[1])
    return y, inv, float(obj), float(gnorm), bool(res.success), int(res.nit)


def random_start_task(task):
    omega, seed, maxiter = task
    rng = np.random.default_rng(seed)
    y0 = rng.normal(size=16)
    y0 /= norm(y0)
    y, inv, obj, gnorm, success, nit = correct_from(y0, omega, maxiter=maxiter)
    return dict(seed=int(seed), y=y, inv=inv, obj=obj, gnorm=gnorm,
                success=success, nit=nit, cls=classify_unit(inv))


def find_seed(omega, base_seed, starts, workers, maxiter, gap_cut=0.03):
    tasks = [(omega, base_seed+i, maxiter) for i in range(starts)]
    candidates = []
    if workers <= 1:
        iterable = map(random_start_task, tasks)
        for r in iterable:
            if r['cls'] == 'mixed' and r['obj'] < angular_envelope(omega) + gap_cut:
                candidates.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(random_start_task, t) for t in tasks]
            for fu in as_completed(futs):
                r = fu.result()
                if r['cls'] == 'mixed' and r['obj'] < angular_envelope(omega) + gap_cut:
                    candidates.append(r)
    candidates.sort(key=lambda r: r['obj'])
    for r in candidates:
        ev = tangent_hessian(r['y'], omega)
        if ev[0] > 1e-5:
            r['hess_min'] = float(ev[0])
            return r
    raise RuntimeError(
        f'No stable mixed seed found at omega={omega}. '
        f'Increase --seed-starts or try omega_seed=0.135/0.145.'
    )


def make_row(omega, y, inv, obj, gnorm, hmin, nit, status='stable'):
    return dict(
        omega=float(omega), obj=float(obj), envelope=float(angular_envelope(omega)),
        gap_to_envelope=float(obj-angular_envelope(omega)), cls=classify_unit(inv),
        I3=float(inv['I3']), J1=float(inv['J1']), JL=float(inv['JL']),
        JR=float(inv['JR']), JLR=float(inv['JLR']), Reff=float(inv['Reff']),
        gnorm=float(gnorm), hess_min=float(hmin), nit=int(nit), status=status,
        y=[float(v) for v in y]
    )


def continue_direction(seed_row, direction, step0, min_step, omega_limit,
                       maxiter, hess_h, max_points=2000):
    assert direction in (-1, +1)
    rows = [seed_row]
    y = np.asarray(seed_row['y'], dtype=float)
    omega = float(seed_row['omega'])
    step = float(step0)
    last_failure = None

    for _ in range(max_points):
        target = omega + direction*step
        if direction < 0 and target < omega_limit:
            break
        if direction > 0 and target > omega_limit:
            break
        yc, inv, obj, gnorm, success, nit = correct_from(y, target, maxiter=maxiter)
        cls = classify_unit(inv)
        # Branch identity is primarily enforced through mixed classification and
        # smooth invariants. If optimizer jumps to a canonical basin, reduce step.
        close_sig = (abs(inv['I3'] - rows[-1]['I3']) < max(0.02, 30*step)
                     and abs(inv['J1'] - rows[-1]['J1']) < max(0.05, 60*step))
        if cls != 'mixed' or not close_sig or gnorm > 2e-6:
            last_failure = dict(omega=float(target), reason='jump_or_bad_stationarity',
                                cls=cls, gnorm=float(gnorm), step=float(step))
            step *= 0.5
            if step < min_step:
                break
            continue
        ev = tangent_hessian(yc, target, h=hess_h)
        hmin = float(ev[0])
        if hmin < -2e-5:
            last_failure = dict(omega=float(target), reason='negative_hessian',
                                hess_min=hmin, step=float(step))
            step *= 0.5
            if step < min_step:
                break
            continue
        row = make_row(target, yc, inv, obj, gnorm, hmin, nit,
                       status='stable' if hmin > 0 else 'near_spinodal')
        rows.append(row)
        y = yc
        omega = target
        # Make progress in the well-conditioned middle of the branch, but reduce
        # steps automatically as the soft eigenvalue approaches zero.
        if hmin < 0.0015:
            step = max(min_step, min(step, step0/8))
        elif hmin < 0.004:
            step = max(min_step, min(step, step0/4))
        elif hmin < 0.008:
            step = max(min_step, min(step, step0/2))
        else:
            step = min(step0, step*1.25)
    return rows, last_failure


def write_csv(path, rows):
    cols = ['omega','obj','envelope','gap_to_envelope','cls','I3','J1','JL','JR',
            'JLR','Reff','gnorm','hess_min','nit','status']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k) for k in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--omega-seed', type=float, default=0.14)
    ap.add_argument('--seed-starts', type=int, default=2048)
    ap.add_argument('--seed', type=int, default=42424242)
    ap.add_argument('--workers', type=int, default=max(1, min(12, (os.cpu_count() or 4)-1)))
    ap.add_argument('--step', type=float, default=0.001)
    ap.add_argument('--min-step', type=float, default=2e-6)
    ap.add_argument('--omega-lower-limit', type=float, default=0.07)
    ap.add_argument('--omega-upper-limit', type=float, default=0.18)
    ap.add_argument('--maxiter', type=int, default=6000)
    ap.add_argument('--hess-h', type=float, default=2e-5)
    ap.add_argument('--outdir', default='n5_mixed_branch_continuation_v1_output')
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    print(f'[seed search] omega={args.omega_seed} starts={args.seed_starts}')
    s = find_seed(args.omega_seed, args.seed, args.seed_starts, args.workers,
                  args.maxiter)
    seed_row = make_row(args.omega_seed, s['y'], s['inv'], s['obj'], s['gnorm'],
                        s['hess_min'], s['nit'])
    print('[seed] ', {k:seed_row[k] for k in ['omega','obj','gap_to_envelope','I3','J1','Reff','hess_min']})

    down, fail_lo = continue_direction(
        seed_row, -1, args.step, args.min_step, args.omega_lower_limit,
        args.maxiter, args.hess_h)
    up, fail_hi = continue_direction(
        seed_row, +1, args.step, args.min_step, args.omega_upper_limit,
        args.maxiter, args.hess_h)

    combined = list(reversed(down[1:])) + up
    # de-duplicate in case of numerical recurrence.
    uniq = {}
    for r in combined:
        uniq[round(r['omega'], 12)] = r
    combined = [uniq[k] for k in sorted(uniq)]
    write_csv(out/'mixed_branch.csv', combined)

    summary = {
        'omega_seed': args.omega_seed,
        'seed_random_id': s['seed'],
        'n_points': len(combined),
        'omega_stable_min': min(r['omega'] for r in combined),
        'omega_stable_max': max(r['omega'] for r in combined),
        'min_hessian_observed': min(r['hess_min'] for r in combined),
        'min_global_gap_observed': min(r['gap_to_envelope'] for r in combined),
        'lower_failure': fail_lo,
        'upper_failure': fail_hi,
        'interpretation': (
            'Raw continuation and tangent-Hessian data. Endpoint estimates are '
            'computed separately by analyze_spinodals.py from zero crossings of '
            'the smallest Hessian eigenvalue.'
        )
    }
    with open(out/'summary.json','w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
