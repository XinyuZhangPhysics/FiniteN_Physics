#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from functools import partial
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from numpy.linalg import norm, eigvalsh
from scipy.linalg import null_space
from scipy.optimize import minimize, brentq

N = 5
OMEGA_STAR = 3.0 / 16.0
E_L = 639.0 / 320.0
C_L = 3.0 / 10.0
E_F = 81.0 / 40.0
C_F = 9.0 / 20.0
RIGIDITY_REQUIRED = 4.0 / 105.0
RIGIDITY_CANONICAL = 0.111623418771389
ALPHA_F_ONSET = 1.0 / 40.0


def centered_basis(n=5):
    one = np.ones((n, 1), dtype=float)
    return null_space(one.T)

B = centered_basis(N)
P5 = np.eye(N) - np.ones((N, N)) / N


def y_to_x(y):
    Y = np.asarray(y, dtype=float).reshape(4, 4)
    return B @ Y @ B.T


def x_to_y(X):
    return (B.T @ X @ B).reshape(-1)


def locking_y(sign=1.0):
    return x_to_y(sign * P5 / 2.0)


def one_vs_rest_vector():
    u = np.array([4.0, -1.0, -1.0, -1.0, -1.0])
    u /= norm(u)
    return u


def rankone_y(sign=1.0):
    u = one_vs_rest_vector()
    return x_to_y(sign * np.outer(u, u))

YL = locking_y(+1.0)
YF = rankone_y(+1.0)


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
    V4 = 2.0 * J0 + E4
    Reff = (n2 * n2 / J1) if J1 > 0 else float('nan')
    return {
        'norm2': n2,
        'J0': J0,
        'J1': J1,
        'JL': JL,
        'JR': JR,
        'JLR': JLR,
        'I3': I3,
        'E4': E4,
        'V4': V4,
        'Reff': Reff,
    }


def v4_grad_x(X):
    n2 = float(np.sum(X * X))
    XXtX = (X @ X.T) @ X
    row2 = np.sum(X * X, axis=1)[:, None]
    col2 = np.sum(X * X, axis=0)[None, :]
    gE = (3.0 / 4.0) * XXtX + (4.0 / 5.0) * (row2 + col2) * X - 4.0 * (X ** 3)
    return 8.0 * n2 * X + gE


def i3_grad_x(X):
    return 3.0 * (X ** 2)


def angular_obj_grad(z, omega):
    z = np.asarray(z, dtype=float)
    nz = norm(z)
    if not np.isfinite(nz) or nz < 1e-14:
        return 1e100, np.zeros_like(z)
    y = z / nz
    X = y_to_x(y)
    inv = invariants_x(X)
    f = inv['E4'] + omega * inv['I3']
    gX = (v4_grad_x(X) - 8.0 * X) + omega * i3_grad_x(X)
    gy = x_to_y(gX)
    gz = (gy - y * float(np.dot(y, gy))) / nz
    return float(f), gz


def radial_obj_grad(y, alpha, cubic_sign=+1.0):
    y = np.asarray(y, dtype=float)
    X = y_to_x(y)
    inv = invariants_x(X)
    f = alpha * inv['norm2'] + cubic_sign * inv['I3'] + inv['V4']
    gX = 2.0 * alpha * X + cubic_sign * i3_grad_x(X) + v4_grad_x(X)
    gy = x_to_y(gX)
    return float(f), gy


def classify_unit(inv, tol=5e-5):
    if abs(inv['J1'] - 0.25) < tol and abs(abs(inv['I3']) - C_L) < 5 * tol:
        return 'locking'
    if abs(inv['J1'] - 1.0) < tol and abs(abs(inv['I3']) - C_F) < 5 * tol:
        return 'rank_one'
    return 'mixed'


def classify_radial(inv, tol=1e-7):
    if inv['norm2'] < tol:
        return 'paramagnet'
    n2 = inv['norm2']
    j1hat = inv['J1'] / (n2 * n2)
    i3hat = abs(inv['I3']) / (n2 ** 1.5)
    if abs(j1hat - 0.25) < 5e-4 and abs(i3hat - C_L) < 1e-3:
        return 'locking'
    if abs(j1hat - 1.0) < 5e-4 and abs(i3hat - C_F) < 1e-3:
        return 'rank_one'
    return 'mixed'


def angular_envelope(omega):
    return min((E_L - 2.0) - abs(omega) * C_L, (E_F - 2.0) - abs(omega) * C_F)


def radial_stationary_rho(alpha, c, e):
    disc = 9.0 * c * c - 32.0 * e * alpha
    if disc < 0:
        return None
    rho = (3.0 * c + math.sqrt(max(disc, 0.0))) / (8.0 * e)
    return rho if rho > 0 else None


def radial_branch(alpha, c, e):
    rho = radial_stationary_rho(alpha, c, e)
    candidates = [(0.0, 0.0)]
    if rho is not None:
        f = alpha * rho * rho - c * rho ** 3 + e * rho ** 4
        candidates.append((f, rho))
    return min(candidates, key=lambda t: t[0])


def radial_envelope(alpha):
    fL, rL = radial_branch(alpha, C_L, E_L)
    fF, rF = radial_branch(alpha, C_F, E_F)
    if fF <= fL:
        return fF, 'rank_one' if rF > 0 else 'paramagnet', rF
    return fL, 'locking' if rL > 0 else 'paramagnet', rL


def canonical_fl_crossing():
    def d(alpha):
        fL, _ = radial_branch(alpha, C_L, E_L)
        fF, _ = radial_branch(alpha, C_F, E_F)
        return fF - fL
    xs = np.linspace(-200.0, -1e-3, 2000)
    vals = [d(x) for x in xs]
    for a, b, fa, fb in zip(xs[:-1], xs[1:], vals[:-1], vals[1:]):
        if fa == 0 or fa * fb < 0:
            root = brentq(d, a, b, xtol=1e-13, rtol=1e-13)
            fL, rL = radial_branch(root, C_L, E_L)
            fF, rF = radial_branch(root, C_F, E_F)
            return root, rF, rL
    raise RuntimeError('Could not bracket canonical F-L crossing')


def run_angular_start(task):
    omega, seed, maxiter = task
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=16)
    z0 /= norm(z0)
    f = lambda z: angular_obj_grad(z, omega)
    res = minimize(f, z0, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 2e-10, 'maxls': 50})
    z = res.x / norm(res.x)
    X = y_to_x(z)
    inv = invariants_x(X)
    obj = inv['E4'] + omega * inv['I3']
    gnorm = norm(angular_obj_grad(z, omega)[1])
    return {'omega': omega, 'seed': seed, 'success': bool(res.success), 'nit': int(res.nit), 'obj': float(obj), 'gnorm': float(gnorm), 'y': z, **inv}


def optimize_angular(omega, starts, seed0, maxiter, workers, include_canonical=True):
    tasks = [(omega, seed0 + i, maxiter) for i in range(starts)]
    rows = []
    if workers <= 1:
        rows = [run_angular_start(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_angular_start, t) for t in tasks]
            for fu in as_completed(futures):
                rows.append(fu.result())
    if include_canonical:
        for name, y0 in [('locking_seed', -YL), ('rankone_seed', -YF)]:
            f = lambda z: angular_obj_grad(z, omega)
            res = minimize(f, y0.copy(), jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50})
            z = res.x / norm(res.x)
            inv = invariants_x(y_to_x(z))
            rows.append({'omega': omega, 'seed': name, 'success': bool(res.success), 'nit': int(res.nit), 'obj': float(inv['E4'] + omega * inv['I3']), 'gnorm': float(norm(angular_obj_grad(z, omega)[1])), 'y': z, **inv})
    rows.sort(key=lambda r: r['obj'])
    return rows


def cluster_angular_rows(rows, tol=3e-7):
    clusters = []
    for r in rows:
        sig = np.array([r['obj'], r['I3'], r['J1'], r['JL'], r['JR'], r['JLR']])
        placed = False
        for c in clusters:
            ref = c['sig']
            if norm(sig - ref) <= tol * (1.0 + norm(ref)):
                c['count'] += 1
                if r['obj'] < c['best']['obj']:
                    c['best'] = r
                    c['sig'] = sig
                placed = True
                break
        if not placed:
            clusters.append({'sig': sig, 'count': 1, 'best': r})
    clusters.sort(key=lambda c: c['best']['obj'])
    return clusters


def tangent_hessian_angular(y, omega, h=2e-5):
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
    f0 = loc(np.zeros(d))
    for i in range(d):
        ei = np.zeros(d); ei[i] = h
        H[i, i] = (loc(ei) - 2 * f0 + loc(-ei)) / (h * h)
    for i in range(d):
        for j in range(i + 1, d):
            ei = np.zeros(d); ej = np.zeros(d)
            ei[i] = h; ej[j] = h
            val = (loc(ei + ej) - loc(ei - ej) - loc(-ei + ej) + loc(-ei - ej)) / (4 * h * h)
            H[i, j] = H[j, i] = val
    return eigvalsh((H + H.T) / 2.0)


def run_radial_start(task):
    alpha, seed, maxiter, init_scale = task
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=16)
    direction /= norm(direction)
    rho = math.exp(rng.uniform(math.log(0.03), math.log(max(0.06, init_scale))))
    y0 = rho * direction
    f = lambda y: radial_obj_grad(y, alpha, +1.0)
    res = minimize(f, y0, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 2e-9, 'maxls': 60})
    y = res.x
    X = y_to_x(y)
    inv = invariants_x(X)
    obj = alpha * inv['norm2'] + inv['I3'] + inv['V4']
    gnorm = norm(radial_obj_grad(y, alpha, +1.0)[1])
    return {'alpha': alpha, 'seed': seed, 'success': bool(res.success), 'nit': int(res.nit), 'obj': float(obj), 'gnorm': float(gnorm), 'y': y, **inv}


def optimize_radial(alpha, starts, seed0, maxiter, workers):
    init_scale = max(2.0, 1.5 * math.sqrt(max(0.0, -alpha)) + 1.0)
    tasks = [(alpha, seed0 + i, maxiter, init_scale) for i in range(starts)]
    rows = []
    if workers <= 1:
        rows = [run_radial_start(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_radial_start, t) for t in tasks]
            for fu in as_completed(futures):
                rows.append(fu.result())
    # Exact canonical seeds and paramagnet.
    rows.append({'alpha': alpha, 'seed': 'paramagnet_seed', 'success': True, 'nit': 0, 'obj': 0.0, 'gnorm': 0.0, 'y': np.zeros(16), **invariants_x(np.zeros((5, 5)))})
    for name, yunit, c, e in [('locking_seed', -YL, C_L, E_L), ('rankone_seed', -YF, C_F, E_F)]:
        rho = radial_stationary_rho(alpha, c, e)
        if rho is None:
            continue
        y0 = rho * yunit
        f = lambda y: radial_obj_grad(y, alpha, +1.0)
        res = minimize(f, y0, jac=True, method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 60})
        y = res.x
        inv = invariants_x(y_to_x(y))
        rows.append({'alpha': alpha, 'seed': name, 'success': bool(res.success), 'nit': int(res.nit), 'obj': float(alpha * inv['norm2'] + inv['I3'] + inv['V4']), 'gnorm': float(norm(radial_obj_grad(y, alpha, +1.0)[1])), 'y': y, **inv})
    rows.sort(key=lambda r: r['obj'])
    return rows


def chol_from_params(p):
    L = np.zeros((4, 4), dtype=float)
    idx = 0
    for i in range(4):
        for j in range(i + 1):
            L[i, j] = p[idx]
            idx += 1
    A = L @ L.T
    na = norm(A, 'fro')
    if na < 1e-15:
        A = np.eye(4) / 2.0
        na = norm(A, 'fro')
    A /= na
    return A


def psd_x_from_params(p):
    A = chol_from_params(p)
    return B @ A @ B.T


def maxwell_value_from_x(X):
    inv = invariants_x(X)
    return inv['E4'] - 3.0 / 16.0 * inv['I3'] + 19.0 / 320.0


def psi0u_unit(inv):
    return 0.5 * inv['J1'] - (5.0 / 12.0) * (inv['JL'] + inv['JR']) + 1.0 / 24.0


def rigidity_ratio_from_x(X):
    inv = invariants_x(X)
    # X is unit norm by construction.
    p = (1.0 - inv['J1']) / 2.0
    denom = p * (3.0 / 8.0 - p)
    if denom <= 1e-12:
        return 1e6, p, float('nan')
    S = 13.0 * (inv['JL'] + inv['JR']) - 40.0 * inv['JLR']
    delta = 3.0 / 10.0 * psi0u_unit(inv) + S / 40.0
    return float(delta / denom), float(p), float(delta)


def psd_params_from_x(X, jitter=1e-12):
    A = B.T @ X @ B
    A = (A + A.T) / 2.0
    # The parameterization normalizes LL^T, so an overall scale is irrelevant.
    evals = np.linalg.eigvalsh(A)
    if evals[0] <= 0:
        A = A + (jitter - evals[0]) * np.eye(4)
    L = np.linalg.cholesky(A)
    vals = []
    for i in range(4):
        for j in range(i + 1):
            vals.append(L[i, j])
    return np.array(vals, dtype=float)


def canonical_rigidity_x():
    coeff = [3.0, 1.0, 5.0, -1.0]
    roots = np.roots(coeff)
    t = float([r.real for r in roots if abs(r.imag) < 1e-12 and 0.0 < r.real < 1.0][0])
    u = one_vs_rest_vector()
    X = np.outer(u, u) + t * (P5 - np.outer(u, u))
    X /= norm(X, 'fro')
    return X, t


def psd_random_param(rng):
    return rng.normal(size=10)


def random_psd_screen(samples, seed, batch=10000):
    rng = np.random.default_rng(seed)
    best_m = (float('inf'), None)
    best_r = (float('inf'), None)
    for _ in range(0, samples, batch):
        m = min(batch, samples - _)
        for _j in range(m):
            p = psd_random_param(rng)
            X = psd_x_from_params(p)
            mv = maxwell_value_from_x(X)
            if mv < best_m[0]:
                best_m = (mv, p.copy())
            rr, pp, dd = rigidity_ratio_from_x(X)
            if rr < best_r[0]:
                best_r = (rr, p.copy())
    return best_m, best_r


def optimize_psd_objective(kind, seeds, maxiter):
    out = []
    for p0 in seeds:
        if kind == 'maxwell':
            fun = lambda p: maxwell_value_from_x(psd_x_from_params(p))
        elif kind == 'rigidity':
            fun = lambda p: rigidity_ratio_from_x(psd_x_from_params(p))[0]
        else:
            raise ValueError(kind)
        res = minimize(fun, p0, method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
        X = psd_x_from_params(res.x)
        inv = invariants_x(X)
        if kind == 'maxwell':
            val = maxwell_value_from_x(X)
            extra = {'p_rigidity': (1.0 - inv['J1']) / 2.0}
        else:
            val, pp, dd = rigidity_ratio_from_x(X)
            extra = {'p_rigidity': pp, 'delta': dd}
        out.append({'value': float(val), 'success': bool(res.success), 'nit': int(res.nit), 'params': res.x, **inv, **extra})
    out.sort(key=lambda r: r['value'])
    return out


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})


def build_omega_grid(preset):
    if preset == 'quick':
        return np.unique(np.concatenate([np.linspace(0, 0.30, 13), np.linspace(0.17, 0.205, 8)]))
    if preset == 'standard':
        return np.unique(np.concatenate([np.linspace(0, 0.30, 31), np.linspace(0.16, 0.215, 23), [OMEGA_STAR]]))
    return np.unique(np.concatenate([np.linspace(0, 0.32, 65), np.linspace(0.15, 0.225, 61), np.linspace(0.182, 0.193, 45), [OMEGA_STAR]]))


def build_alpha_grid(preset):
    root, _, _ = canonical_fl_crossing()
    if preset == 'quick':
        return np.unique(np.array([0.04, 0.03, 0.026, 0.025, 0.024, 0.02, 0.0, -1.0, -10.0, -50.0, -100.0, root - 0.5, root, root + 0.5, -125.0]))
    if preset == 'standard':
        return np.unique(np.concatenate([np.linspace(0.04, -0.1, 18), np.linspace(-1, -20, 10), np.linspace(-30, -100, 8), np.linspace(root - 2.0, root + 2.0, 17), [root, -125.0]]))
    return np.unique(np.concatenate([np.linspace(0.04, -0.12, 41), np.linspace(-0.2, -10, 25), np.linspace(-15, -90, 16), np.linspace(root - 4.0, root + 4.0, 65), [root, -130.0]]))


def preset_params(preset):
    if preset == 'quick':
        return dict(angular_starts=32, radial_starts=24, psd_samples=2000, psd_polish=12, maxiter=1200, psd_maxiter=2500)
    if preset == 'standard':
        return dict(angular_starts=128, radial_starts=96, psd_samples=30000, psd_polish=40, maxiter=2000, psd_maxiter=5000)
    return dict(angular_starts=384, radial_starts=256, psd_samples=200000, psd_polish=96, maxiter=3000, psd_maxiter=9000)


def save_json(path, obj):
    def clean(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, (np.floating, np.integer)): return x.item()
        if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        return x
    with open(path, 'w') as f:
        json.dump(clean(obj), f, indent=2, sort_keys=True)


def main():
    ap = argparse.ArgumentParser(description='Publication-grade N=5 finite-N geometry numerical audit')
    ap.add_argument('--mode', choices=['all', 'angular', 'radial', 'maxwell'], default='all')
    ap.add_argument('--preset', choices=['quick', 'standard', 'publication'], default='standard')
    ap.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 4) - 1)))
    ap.add_argument('--seed', type=int, default=20261015)
    ap.add_argument('--outdir', type=str, default='n5_publication_audit_v1_output')
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pp = preset_params(args.preset)
    summary = {'preset': args.preset, 'seed': args.seed, 'workers': args.workers, 'constants': {
        'omega_star': OMEGA_STAR, 'eL': E_L, 'cL': C_L, 'eF': E_F, 'cF': C_F,
        'rigidity_required': RIGIDITY_REQUIRED, 'rigidity_canonical': RIGIDITY_CANONICAL,
        'alpha_F_onset': ALPHA_F_ONSET,
    }}

    root, rhoF, rhoL = canonical_fl_crossing()
    summary['canonical_radial_crossing'] = {'alpha_FL': root, 'rho_F': rhoF, 'rho_L': rhoL, 'omega_F': 1.0/rhoF, 'omega_L': 1.0/rhoL}
    print(f'[canonical] alpha_FL={root:.15g}, rhoF={rhoF:.12g}, rhoL={rhoL:.12g}')

    if args.mode in ('all', 'angular'):
        angular_rows = []
        catalog_rows = []
        worst_gap = 0.0
        mixed_global = []
        metastable = []
        omegas = build_omega_grid(args.preset)
        for k, omega in enumerate(omegas):
            rows = optimize_angular(float(omega), pp['angular_starts'], args.seed + 100000*k, pp['maxiter'], args.workers)
            best = rows[0]
            env = angular_envelope(float(omega))
            gap = best['obj'] - env
            worst_gap = min(worst_gap, gap)
            cls = classify_unit(best)
            if cls == 'mixed' and gap < 5e-8:
                mixed_global.append(float(omega))
            clusters = cluster_angular_rows(rows)
            # Hessian-audit low-lying noncanonical clusters in metastable window.
            for ci, c in enumerate(clusters[:12]):
                b = c['best']
                ccls = classify_unit(b)
                hmin = None
                if 0.105 <= omega <= 0.165 and ccls == 'mixed' and b['obj'] < env + 0.02:
                    ev = tangent_hessian_angular(b['y'], float(omega))
                    hmin = float(ev[0])
                    if hmin > -2e-5:
                        metastable.append({'omega': float(omega), 'obj': b['obj'], 'gap_to_envelope': b['obj']-env, 'I3': b['I3'], 'J1': b['J1'], 'Reff': b['Reff'], 'hess_min': hmin, 'count': c['count']})
                catalog_rows.append({'omega': float(omega), 'cluster': ci, 'count': c['count'], 'obj': b['obj'], 'gap_to_envelope': b['obj']-env, 'class': ccls, 'I3': b['I3'], 'J1': b['J1'], 'JL': b['JL'], 'JR': b['JR'], 'JLR': b['JLR'], 'Reff': b['Reff'], 'gnorm': b['gnorm'], 'hess_min': hmin})
            angular_rows.append({'omega': float(omega), 'best_obj': best['obj'], 'canonical_envelope': env, 'gap_to_envelope': gap, 'class': cls, 'I3': best['I3'], 'J1': best['J1'], 'JL': best['JL'], 'JR': best['JR'], 'JLR': best['JLR'], 'Reff': best['Reff'], 'gnorm': best['gnorm'], 'n_clusters': len(clusters)})
            print(f'[angular {k+1:3d}/{len(omegas)}] omega={omega:.9f} gap={gap:+.3e} class={cls} clusters={len(clusters)}')
        write_csv(outdir/'angular_scan.csv', angular_rows, list(angular_rows[0].keys()))
        write_csv(outdir/'angular_minima_catalog.csv', catalog_rows, list(catalog_rows[0].keys()))
        summary['angular'] = {'n_omega': len(omegas), 'worst_negative_gap': worst_gap, 'mixed_global_omegas': mixed_global, 'metastable_mixed': metastable, 'pass_no_below_envelope': worst_gap > -5e-8, 'pass_no_mixed_global': len(mixed_global)==0}

    if args.mode in ('all', 'radial'):
        radial_rows = []
        worst_gap = 0.0
        mixed_global = []
        alphas = build_alpha_grid(args.preset)
        for k, alpha in enumerate(alphas):
            rows = optimize_radial(float(alpha), pp['radial_starts'], args.seed + 200000*k, pp['maxiter'], args.workers)
            best = rows[0]
            env, envcls, envrho = radial_envelope(float(alpha))
            gap = best['obj'] - env
            worst_gap = min(worst_gap, gap)
            cls = classify_radial(best)
            if cls == 'mixed' and gap < 5e-7:
                mixed_global.append(float(alpha))
            radial_rows.append({'alpha': float(alpha), 'best_obj': best['obj'], 'canonical_envelope': env, 'gap_to_envelope': gap, 'class': cls, 'canonical_class': envcls, 'rho': math.sqrt(max(best['norm2'],0.0)), 'canonical_rho': envrho, 'I3': best['I3'], 'J1': best['J1'], 'Reff': best['Reff'], 'gnorm': best['gnorm']})
            print(f'[radial {k+1:3d}/{len(alphas)}] alpha={alpha:.9g} gap={gap:+.3e} class={cls}/{envcls}')
        write_csv(outdir/'radial_scan.csv', radial_rows, list(radial_rows[0].keys()))
        summary['radial'] = {'n_alpha': len(alphas), 'worst_negative_gap': worst_gap, 'mixed_global_alphas': mixed_global, 'pass_no_below_envelope': worst_gap > -2e-6, 'pass_no_mixed_global': len(mixed_global)==0}

    if args.mode in ('all', 'maxwell'):
        best_m, best_r = random_psd_screen(pp['psd_samples'], args.seed + 300000)
        rng = np.random.default_rng(args.seed + 300001)
        Xrig, trig = canonical_rigidity_x()
        seed_lock = psd_params_from_x(P5 / 2.0)
        seed_rank = psd_params_from_x(np.outer(one_vs_rest_vector(), one_vs_rest_vector()))
        seed_rig = psd_params_from_x(Xrig)
        extra_m = max(0, pp['psd_polish'] - 3)
        extra_r = max(0, pp['psd_polish'] - 2)
        seeds_m = [best_m[1], seed_lock, seed_rank] + [psd_random_param(rng) for _ in range(extra_m)]
        seeds_r = [best_r[1], seed_rig] + [psd_random_param(rng) for _ in range(extra_r)]
        maxwell = optimize_psd_objective('maxwell', seeds_m, pp['psd_maxiter'])
        rigidity = optimize_psd_objective('rigidity', seeds_r, pp['psd_maxiter'])
        m0 = maxwell[0]
        r0 = rigidity[0]
        print(f"[maxwell] min={m0['value']:.15g} I3={m0['I3']:.12g} J1={m0['J1']:.12g}")
        print(f"[rigidity] min={r0['value']:.15g} p={r0['p_rigidity']:.12g} required={RIGIDITY_REQUIRED:.12g}")
        save_json(outdir/'maxwell_results.json', {'random_screen_best': best_m[0], 'polished': maxwell[:20]})
        save_json(outdir/'rigidity_results.json', {'random_screen_best': best_r[0], 'polished': rigidity[:20]})
        summary['maxwell'] = {'random_screen_min': best_m[0], 'polished_min': m0['value'], 'pass_nonnegative': m0['value'] > -5e-9}
        summary['rigidity'] = {'random_screen_min': best_r[0], 'polished_min': r0['value'], 'required': RIGIDITY_REQUIRED, 'canonical_value': RIGIDITY_CANONICAL, 'canonical_t': trig, 'pass_required_margin': r0['value'] > RIGIDITY_REQUIRED + 1e-4, 'distance_to_canonical_value': r0['value'] - RIGIDITY_CANONICAL}

    save_json(outdir/'summary.json', summary)
    overall = []
    for section in ('angular','radial','maxwell','rigidity'):
        if section in summary:
            overall.extend(v for k,v in summary[section].items() if k.startswith('pass_'))
    summary['overall_pass'] = bool(all(overall)) if overall else None
    save_json(outdir/'summary.json', summary)
    print('\n=== AUDIT SUMMARY ===')
    print(json.dumps(summary, indent=2, default=lambda x: x.tolist() if isinstance(x,np.ndarray) else x))
    if summary['overall_pass'] is False:
        sys.exit(2)


if __name__ == '__main__':
    main()
