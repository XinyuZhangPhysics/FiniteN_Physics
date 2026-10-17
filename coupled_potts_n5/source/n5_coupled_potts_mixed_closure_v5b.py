#!/usr/bin/env python3
import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mpmath as mp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, linear_sum_assignment, minimize, root
from scipy.special import logsumexp


def parse_grid(spec):
    a, b, n = spec.split(":")
    return np.linspace(float(a), float(b), int(n))


def cells_logabc(N, abc):
    n = N - 1
    a, b, c = map(float, abc)
    M = max(a, b, c, 0.0)
    ea = math.exp(a - M)
    eb = math.exp(b - M)
    ec = math.exp(c - M)
    e0 = math.exp(-M)
    D = ea + 2.0 * n * eb + n * ec + n * (n - 1) * e0
    return np.array([ea / D, eb / D, ec / D, e0 / D], dtype=float)


def cells_F(N, ab):
    return cells_logabc(N, [ab[0], ab[1], 0.0])


def cells_L(N, c):
    return cells_logabc(N, [c, 0.0, c])


def cells_to_matrix(N, cells):
    x, y, z, w = map(float, cells)
    P = np.full((N, N), w, dtype=float)
    P[0, :] = y
    P[:, 0] = y
    P[0, 0] = x
    for i in range(1, N):
        P[i, i] = z
    P = np.maximum(P, 1e-300)
    P /= P.sum()
    return P


def free_energy_cells(N, cells, j, t):
    n = N - 1
    x, y, z, w = map(float, cells)
    counts = np.array([1.0, 2.0 * n, float(n), float(n * (n - 1))])
    p = np.array([x, y, z, w], dtype=float)
    r1 = x + n * y
    r0 = y + z + (n - 1) * w
    ent = np.sum(np.where(p > 0, counts * p * np.log(p), 0.0))
    return float(
        t * ent
        - np.sum(counts * p * p)
        + (1.0 - j) * (r1 * r1 + n * r0 * r0)
    )


def stationary_abc(N, abc, j, t):
    n = N - 1
    a, b, c = map(float, abc)
    x, y, z, w = cells_logabc(N, abc)
    dr = x + n * y - y - z - (n - 1) * w
    return np.array([
        t * a - 2.0 * (x - w) + 2.0 * (1.0 - j) * dr,
        t * b - 2.0 * (y - w) + (1.0 - j) * dr,
        t * c - 2.0 * (z - w),
    ], dtype=float)


def stationary_F(N, ab, j, t):
    return stationary_abc(N, [ab[0], ab[1], 0.0], j, t)[:2]


def stationary_L(N, c, j, t):
    return float(stationary_abc(N, [c, 0.0, c], j, t)[2])


def fm_equations(V, N, j):
    aF, bF, aM, bM, cMlog, t = map(float, V)
    F_cells = cells_F(N, [aF, bF])
    M_cells = cells_logabc(N, [aM, bM, cMlog])
    return np.r_[
        stationary_F(N, [aF, bF], j, t),
        stationary_abc(N, [aM, bM, cMlog], j, t),
        free_energy_cells(N, F_cells, j, t) - free_energy_cells(N, M_cells, j, t),
    ]


def ml_equations(V, N, j):
    aM, bM, cMlog, cL, t = map(float, V)
    M_cells = cells_logabc(N, [aM, bM, cMlog])
    L_cells = cells_L(N, cL)
    return np.r_[
        stationary_abc(N, [aM, bM, cMlog], j, t),
        stationary_L(N, cL, j, t),
        free_energy_cells(N, M_cells, j, t) - free_energy_cells(N, L_cells, j, t),
    ]


def triple_equations(V, N):
    aF, bF, aM, bM, cMlog, cL, j, t = map(float, V)
    F_cells = cells_F(N, [aF, bF])
    M_cells = cells_logabc(N, [aM, bM, cMlog])
    L_cells = cells_L(N, cL)
    return np.r_[
        stationary_F(N, [aF, bF], j, t),
        stationary_abc(N, [aM, bM, cMlog], j, t),
        stationary_L(N, cL, j, t),
        free_energy_cells(N, F_cells, j, t) - free_energy_cells(N, M_cells, j, t),
        free_energy_cells(N, L_cells, j, t) - free_energy_cells(N, M_cells, j, t),
    ]


def solve_triple(N):
    if N != 5:
        raise ValueError("Default triple seed is specialized to N=5.")
    V0 = np.array([
        5.3530, -67.47,
        10.0373, -55.61, 7.2165,
        30.04,
        0.02460, 0.013314,
    ])
    sol = root(lambda V: triple_equations(V, N), V0, method="hybr", tol=1e-11)
    if not sol.success:
        raise RuntimeError("Triple solve failed: " + sol.message)
    V = sol.x
    aF, bF, aM, bM, cM, cL, j, t = map(float, V)
    return {
        "vector": V,
        "j": j,
        "t": t,
        "ratio": t / j,
        "F_cells": cells_F(N, [aF, bF]),
        "M_cells": cells_logabc(N, [aM, bM, cM]),
        "L_cells": cells_L(N, cL),
        "residual_inf": float(np.linalg.norm(triple_equations(V, N), np.inf)),
    }


def high_precision_triple(N, triple, dps=80):
    mp.mp.dps = dps
    n = N - 1

    def cells(a, b, c):
        D = mp.e**a + 2*n*mp.e**b + n*mp.e**c + n*(n-1)
        w = 1 / D
        return mp.e**a*w, mp.e**b*w, mp.e**c*w, w

    def stat(a, b, c, j, t):
        x, y, z, w = cells(a, b, c)
        dr = x+n*y-y-z-(n-1)*w
        return (
            t*a - 2*(x-w) + 2*(1-j)*dr,
            t*b - 2*(y-w) + (1-j)*dr,
            t*c - 2*(z-w),
        )

    counts = (1, 2*n, n, n*(n-1))

    def ff(cel, j, t):
        x, y, z, w = cel
        p = (x, y, z, w)
        r1 = x+n*y
        r0 = y+z+(n-1)*w
        ent = sum(counts[k]*p[k]*mp.log(p[k]) for k in range(4))
        sq = sum(counts[k]*p[k]*p[k] for k in range(4))
        return t*ent - sq + (1-j)*(r1*r1+n*r0*r0)

    v = triple["vector"]
    x0 = tuple(mp.mpf(str(float(q))) for q in v)

    def eqs(aF, bF, aM, bM, cM, cL, j, t):
        sF = stat(aF, bF, mp.mpf("0"), j, t)
        sM = stat(aM, bM, cM, j, t)
        sL = stat(cL, mp.mpf("0"), cL, j, t)
        cF = cells(aF, bF, mp.mpf("0"))
        cM0 = cells(aM, bM, cM)
        cL0 = cells(cL, mp.mpf("0"), cL)
        return (
            sF[0], sF[1],
            sM[0], sM[1], sM[2],
            sL[2],
            ff(cF,j,t)-ff(cM0,j,t),
            ff(cL0,j,t)-ff(cM0,j,t),
        )

    sol = mp.findroot(eqs, x0, tol=mp.mpf("1e-60"), maxsteps=100)
    names = ["aF","bF","aM","bM","cM","cL","j","t"]
    out = {names[k]: mp.nstr(sol[k], 60) for k in range(8)}
    out["ratio"] = mp.nstr(sol[7]/sol[6], 60)
    out["dps"] = dps
    return out


def continue_boundaries(N, jgrid, triple):
    Vt = triple["vector"]
    seed_fm = np.array([Vt[0],Vt[1],Vt[2],Vt[3],Vt[4],Vt[7]], dtype=float)
    seed_ml = np.array([Vt[2],Vt[3],Vt[4],Vt[5],Vt[7]], dtype=float)

    rows = []
    for j in sorted(jgrid, reverse=True):
        sfm = root(lambda V: fm_equations(V, N, float(j)), seed_fm, method="hybr", tol=1e-11)
        sml = root(lambda V: ml_equations(V, N, float(j)), seed_ml, method="hybr", tol=1e-11)

        row = {"j": float(j)}

        if sfm.success and sfm.x[-1] > 0:
            seed_fm = sfm.x.copy()
            row.update({
                "fm_success": True,
                "t_FM": float(sfm.x[-1]),
                "ratio_FM": float(sfm.x[-1] / j),
                "FM_residual": float(np.linalg.norm(fm_equations(sfm.x, N, float(j)), np.inf)),
                "FM_aF": float(sfm.x[0]),
                "FM_bF": float(sfm.x[1]),
                "FM_aM": float(sfm.x[2]),
                "FM_bM": float(sfm.x[3]),
                "FM_cM": float(sfm.x[4]),
            })
        else:
            row["fm_success"] = False

        if sml.success and sml.x[-1] > 0:
            seed_ml = sml.x.copy()
            row.update({
                "ml_success": True,
                "t_ML": float(sml.x[-1]),
                "ratio_ML": float(sml.x[-1] / j),
                "ML_residual": float(np.linalg.norm(ml_equations(sml.x, N, float(j)), np.inf)),
                "ML_aM": float(sml.x[0]),
                "ML_bM": float(sml.x[1]),
                "ML_cM": float(sml.x[2]),
                "ML_cL": float(sml.x[3]),
            })
        else:
            row["ml_success"] = False

        rows.append(row)

    return pd.DataFrame(rows).sort_values("j")


def hessian_inertia_mp(N, cells, j, t, dps=80):
    positive = [float(q) for q in cells if float(q) > 0.0]
    if not positive:
        raise ValueError("All cell probabilities vanished.")
    adaptive_dps = max(dps, int(max(0.0, -math.log10(min(positive)))) + 80)
    mp.mp.dps = adaptive_dps
    x, y, z, w = [mp.mpf(str(float(q))) for q in cells]
    P = [[w for _ in range(N)] for __ in range(N)]
    for k in range(N):
        P[0][k] = y
        P[k][0] = y
    P[0][0] = x
    for i in range(1, N):
        P[i][i] = z

    m = N*N
    H = mp.matrix(m, m)
    aa = mp.mpf(str(float(1.0-j)))
    tt = mp.mpf(str(float(t)))

    for p in range(m):
        i, k = divmod(p, N)
        H[p,p] = tt/P[i][k] - 2

    for p in range(m):
        i, k = divmod(p, N)
        for q in range(m):
            ii, kk = divmod(q, N)
            if i == ii:
                H[p,q] += aa
            if k == kk:
                H[p,q] += aa

    B = mp.matrix(m, m-1)
    for k in range(m-1):
        B[k,k] = 1
        B[m-1,k] = -1

    K = B.T * H * B
    vals = mp.eigsy(K, eigvals_only=True)
    neg = sum(1 for i in range(len(vals)) if vals[i] < 0)
    zero = sum(1 for i in range(len(vals)) if abs(vals[i]) < mp.mpf("1e-40"))
    return {
        "negative_count": int(neg),
        "zero_count": int(zero),
        "min_congruence_eigenvalue": mp.nstr(vals[0], 45),
        "mp_dps": int(adaptive_dps),
    }


def branch_label_from_cells(N, cells):
    P = cells_to_matrix(N, cells)
    d = diagnostics(P)
    if d["primary2"] < 1e-10 and d["R_eff"] < 1e-8:
        return "P"
    if d["primary2"] < 1e-8 and abs(d["R_eff"] - (N - 1)) < 2e-4:
        return "L"
    if d["primary2"] >= 1e-10 and abs(d["R_eff"] - 1.0) < 2e-4:
        return "F"
    return "M"


def _candidate_roots(fun, seeds, bounds, residual_tol=2e-8):
    accepted = []
    lo, hi = bounds
    for seed in seeds:
        x0 = np.asarray(seed, dtype=float)
        if not np.all(np.isfinite(x0)):
            continue
        attempts = []
        for method in ("hybr", "lm"):
            try:
                r = root(fun, x0, method=method, tol=1e-11)
                attempts.append(np.asarray(r.x, dtype=float))
            except Exception:
                pass
        try:
            r = least_squares(
                fun, np.clip(x0, lo, hi), bounds=(lo, hi),
                xtol=1e-13, ftol=1e-13, gtol=1e-13,
                max_nfev=8000, x_scale="jac",
            )
            attempts.append(np.asarray(r.x, dtype=float))
        except Exception:
            pass
        try:
            r = minimize(
                lambda x: 0.5 * float(np.dot(fun(x), fun(x))),
                np.clip(x0, lo, hi), method="L-BFGS-B",
                bounds=[(lo, hi)] * len(x0),
                options={"maxiter": 5000, "ftol": 1e-20, "gtol": 1e-12, "maxls": 80},
            )
            attempts.append(np.asarray(r.x, dtype=float))
        except Exception:
            pass

        for x in attempts:
            if not np.all(np.isfinite(x)):
                continue
            resid = float(np.linalg.norm(fun(x), np.inf))
            if resid > residual_tol:
                continue
            if all(np.linalg.norm(x - y, np.inf) > 1e-6 for y, _ in accepted):
                accepted.append((x, resid))
    return accepted


def solve_branch_at_t(N, j, t, seedF, seedM, seedL):
    # Multiple starts are essential near the first-order boundaries: a Newton
    # solve from one continuation seed can fail even when the branch exists.
    Fseeds = [
        np.asarray(seedF, dtype=float),
        np.array([3.0, -8.0]),
        np.array([5.0, -30.0]),
        np.array([7.0, -70.0]),
        np.array([10.0, -150.0]),
    ]
    Mseeds = [
        np.asarray(seedM, dtype=float),
        np.array([5.0, -20.0, 3.0]),
        np.array([7.0, -40.0, 5.0]),
        np.array([9.0, -80.0, 7.0]),
        np.array([11.0, -160.0, 8.0]),
    ]
    Lseeds = [
        np.array([float(seedL)]),
        np.array([5.0]),
        np.array([10.0]),
        np.array([20.0]),
        np.array([35.0]),
    ]

    raw = []
    for x, resid in _candidate_roots(
        lambda ab: stationary_F(N, ab, j, t), Fseeds, (-400.0, 400.0)
    ):
        cells = cells_F(N, x)
        raw.append((branch_label_from_cells(N, cells), cells, x, resid, "F-family"))

    for x, resid in _candidate_roots(
        lambda abc: stationary_abc(N, abc, j, t), Mseeds, (-500.0, 500.0)
    ):
        cells = cells_logabc(N, x)
        raw.append((branch_label_from_cells(N, cells), cells, x, resid, "M-family"))

    for x, resid in _candidate_roots(
        lambda cc: np.array([stationary_L(N, cc[0], j, t)]),
        Lseeds, (-400.0, 400.0)
    ):
        cells = cells_L(N, x[0])
        raw.append((branch_label_from_cells(N, cells), cells, x, resid, "L-family"))

    # The paramagnet is an exact stationary branch for all (j,t).
    q = 1.0 / (N * N)
    Pcells = np.array([q, q, q, q], dtype=float)
    branches = {"P": (Pcells, np.zeros(1), 0.0, "exact")}

    # Keep the lowest-energy correctly classified root for each ordered branch.
    for label, cells, pars, resid, family in raw:
        if label == "P":
            continue
        E = free_energy_cells(N, cells, j, t)
        if label not in branches or E < free_energy_cells(N, branches[label][0], j, t):
            branches[label] = (cells, pars, resid, family)

    meta = {
        "found_labels": sorted(branches.keys()),
        "ordered_branch_count": int(sum(k in branches for k in ("F", "M", "L"))),
        "raw_root_count": int(len(raw)),
    }
    return branches, meta


def full_objective_grad(y, N, j, t):
    zz = np.concatenate([np.asarray(y, dtype=float), np.array([0.0])])
    lp = zz - logsumexp(zz)
    p = np.exp(lp)
    P = p.reshape(N, N)
    r = P.sum(axis=1)
    c = P.sum(axis=0)
    f = (
        t*np.dot(p, lp)
        - np.dot(p,p)
        + 0.5*(1.0-j)*(np.dot(r,r)+np.dot(c,c))
    )
    g = (
        t*(lp.reshape(N,N)+1.0)
        - 2.0*P
        + (1.0-j)*(r[:,None]+c[None,:])
    ).ravel()
    avg = float(np.dot(p,g))
    return float(f), (p*(g-avg))[:-1]


def logits_from_P(P):
    p = np.maximum(P.ravel(), 1e-300)
    p /= p.sum()
    lp = np.log(p)
    return lp[:-1]-lp[-1]


def optimize_full(P0, N, j, t):
    y0 = logits_from_P(P0)
    res = minimize(
        lambda y: full_objective_grad(y,N,j,t),
        y0, jac=True, method="L-BFGS-B",
        bounds=[(-180.0,180.0)]*len(y0),
        options={"maxiter":3000,"ftol":1e-15,"gtol":5e-11,"maxls":80},
    )
    zz = np.concatenate([res.x, np.array([0.0])])
    lp = zz-logsumexp(zz)
    P = np.exp(lp).reshape(N,N)
    f, gr = full_objective_grad(res.x,N,j,t)
    return {"P":P,"energy":f,"grad_inf":float(np.max(np.abs(gr))),"success":bool(res.success)}


def canonicalize_fourcell(P):
    N = P.shape[0]
    i0,j0 = np.unravel_index(np.argmax(P), P.shape)
    remr = [i for i in range(N) if i != i0]
    remc = [j for j in range(N) if j != j0]
    rr,cc = linear_sum_assignment(-P[np.ix_(remr,remc)])
    ro = [i0] + [remr[i] for i in rr]
    co = [j0] + [remc[j] for j in cc]
    Q = P[np.ix_(ro,co)]
    ys = np.r_[Q[0,1:],Q[1:,0]]
    zs = np.diag(Q)[1:]
    mask = np.ones((N-1,N-1),dtype=bool)
    np.fill_diagonal(mask,False)
    ws = Q[1:,1:][mask]
    cells = np.array([Q[0,0],ys.mean(),zs.mean(),ws.mean()])
    defect = float(max(ys.std(),zs.std(),ws.std()))
    return Q,cells,defect


def diagnostics(P):
    N=P.shape[0]
    H=np.eye(N)-np.ones((N,N))/N
    r=P.sum(1); c=P.sum(0)
    X=H@P@H
    s2=np.linalg.svd(X,compute_uv=False)**2
    ov=float(s2.sum()); j1=float(np.dot(s2,s2))
    return {
        "primary2":float(np.sum((r-1/N)**2)+np.sum((c-1/N)**2)),
        "R_eff":float(ov*ov/j1) if j1>1e-30 else 0.0,
    }


def audit_one(task):
    N,j,t,seedF,seedM,seedL,starts,seed = task
    rng=np.random.default_rng(seed)
    try:
        branches, branch_meta = solve_branch_at_t(
            N,j,t,np.array(seedF),np.array(seedM),float(seedL)
        )
    except Exception as exc:
        return {
            "j":j,"t":t,"ratio":t/j,"success":False,
            "branch_solver_error":repr(exc),
        }

    energies={k:free_energy_cells(N,v[0],j,t) for k,v in branches.items()}
    pred=min(energies,key=energies.get)
    predE=energies[pred]

    seeds=[]
    for lab,v in branches.items():
        cells=v[0]
        P=cells_to_matrix(N,cells)
        seeds.append((lab,P))
        for k in range(1,N):
            pr=np.arange(N); pc=np.arange(N)
            pr[[0,k]]=pr[[k,0]]
            seeds.append((lab,P[pr][:,pc]))

    alpha_cycle=[0.08,0.15,0.3,0.6,1.0,2.0]
    while len(seeds)<starts:
        alpha=alpha_cycle[len(seeds)%len(alpha_cycle)]
        P=rng.dirichlet(np.full(N*N,alpha)).reshape(N,N)
        seeds.append(("random",P))

    best=None
    for source,P0 in seeds:
        try:
            q=optimize_full(P0,N,j,t)
        except Exception:
            continue
        if best is None or q["energy"]<best["energy"]:
            best=q|{"source":source}

    if best is None:
        return {"j":j,"t":t,"success":False}

    Q,cells,defect=canonicalize_fourcell(best["P"])
    groupE=free_energy_cells(N,cells,j,t)
    d=diagnostics(best["P"])
    return {
        "j":j,"t":t,"ratio":t/j,
        "predicted_label":pred,
        "predicted_energy":predE,
        "stationary_branches_found":",".join(branch_meta["found_labels"]),
        "ordered_branch_count":branch_meta["ordered_branch_count"],
        "raw_root_count":branch_meta["raw_root_count"],
        "full_energy":best["energy"],
        "delta_full_minus_correct_envelope":best["energy"]-predE,
        "canonical_group_energy_minus_full":groupE-best["energy"],
        "fourcell_symmetry_defect":defect,
        "best_source":best["source"],
        "full_grad_inf":best["grad_inf"],
        "full_primary2":d["primary2"],
        "full_R_eff":d["R_eff"],
        "canon_x":cells[0],"canon_y":cells[1],"canon_z":cells[2],"canon_w":cells[3],
        "success":True,
    }


def make_audit_tasks(N, boundaries, triple, starts, seed_blocks, seed):
    rows=boundaries[(boundaries.fm_success==True)&(boundaries.ml_success==True)].copy()
    target_js=np.linspace(rows.j.min(), rows.j.max(), min(8,len(rows)))
    tasks=[]
    for j0 in target_js:
        r=rows.iloc[(rows.j-j0).abs().argmin()]
        j=float(r.j); tf=float(r.t_FM); tl=float(r.t_ML)
        seedF=[r.FM_aF,r.FM_bF]
        seedM=[r.FM_aM,r.FM_bM,r.FM_cM]
        seedL=float(r.ML_cL)
        eps=max(1e-6,0.08*(tl-tf))
        ts=[
            max(1e-8,tf-eps),
            tf+eps,
            0.5*(tf+tl),
            tl-eps,
            tl+eps,
        ]
        for ib in range(seed_blocks):
            for k,t in enumerate(ts):
                tasks.append((
                    N,j,float(t),seedF,seedM,seedL,starts,
                    seed+100000*ib+1009*k+int(1e6*j),
                ))
    return tasks


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--N",type=int,default=5)
    ap.add_argument("--j-grid",default="0.004:0.0244:55")
    ap.add_argument("--hessian-samples",type=int,default=12)
    ap.add_argument("--audit-starts",type=int,default=128)
    ap.add_argument("--audit-seeds",type=int,default=3)
    ap.add_argument("--workers",type=int,default=10)
    ap.add_argument("--seed",type=int,default=20261015)
    ap.add_argument("--skip-audit",action="store_true")
    ap.add_argument("--out",default="n5_coupled_potts_mixed_closure_v5b")
    args=ap.parse_args()

    if args.N!=5:
        raise ValueError("v5 is currently specialized to N=5.")

    outdir=Path(args.out)
    outdir.mkdir(parents=True,exist_ok=True)

    triple=solve_triple(args.N)
    hp=high_precision_triple(args.N,triple)
    triple_json={
        "double":{
            "j":triple["j"],"t":triple["t"],"ratio":triple["ratio"],
            "F_cells":triple["F_cells"].tolist(),
            "M_cells":triple["M_cells"].tolist(),
            "L_cells":triple["L_cells"].tolist(),
            "residual_inf":triple["residual_inf"],
        },
        "high_precision":hp,
    }
    (outdir/"triple_point.json").write_text(json.dumps(triple_json,indent=2),encoding="utf-8")

    jgrid=parse_grid(args.j_grid)
    jgrid=jgrid[jgrid<triple["j"]]
    boundaries=continue_boundaries(args.N,jgrid,triple)
    boundaries.to_csv(outdir/"mixed_maxwell_boundaries.csv",index=False)

    good=boundaries[(boundaries.fm_success==True)&(boundaries.ml_success==True)].copy()
    hs=[]
    if len(good):
        take=np.linspace(0,len(good)-1,min(args.hessian_samples,len(good))).round().astype(int)
        for idx in np.unique(take):
            r=good.iloc[idx]
            for side,tcol,abc in [
                ("FM_M","t_FM",[r.FM_aM,r.FM_bM,r.FM_cM]),
                ("mid_M",None,[
                    0.5*(r.FM_aM+r.ML_aM),
                    0.5*(r.FM_bM+r.ML_bM),
                    0.5*(r.FM_cM+r.ML_cM),
                ]),
                ("ML_M","t_ML",[r.ML_aM,r.ML_bM,r.ML_cM]),
            ]:
                t=float(r[tcol]) if tcol else 0.5*(float(r.t_FM)+float(r.t_ML))
                sol=root(lambda v:stationary_abc(args.N,v,float(r.j),t),np.array(abc),tol=1e-11)
                if not sol.success:
                    continue
                cells=cells_logabc(args.N,sol.x)
                inertia=hessian_inertia_mp(args.N,cells,float(r.j),t,80)
                hs.append({
                    "j":float(r.j),"t":t,"ratio":t/float(r.j),
                    "sample":side,
                    "x":cells[0],"y":cells[1],"z":cells[2],"w":cells[3],
                    **inertia,
                })
    pd.DataFrame(hs).to_csv(outdir/"mixed_high_precision_hessian.csv",index=False)

    audit=None
    if not args.skip_audit and len(good):
        tasks=make_audit_tasks(
            args.N,good,triple,args.audit_starts,args.audit_seeds,args.seed
        )
        rows=[]
        if args.workers<=1:
            for task in tasks:
                rows.append(audit_one(task))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs=[ex.submit(audit_one,task) for task in tasks]
                for fut in as_completed(futs):
                    try:
                        rows.append(fut.result())
                    except Exception as exc:
                        rows.append({"success":False,"worker_error":repr(exc)})
        audit=pd.DataFrame(rows).sort_values(["j","t"])
        audit.to_csv(outdir/"corrected_fullspace_audit.csv",index=False)

    summary={
        "N":args.N,
        "triple":{
            "j":triple["j"],"t":triple["t"],"ratio":triple["ratio"],
        },
        "hard_matching_Potts_ratio":(args.N-2)/((args.N-1)*math.log(args.N-1)),
        "boundary_rows":int(len(boundaries)),
        "successful_boundary_rows":int(len(good)),
        "hessian_rows":int(len(hs)),
        "hessian_negative_count_total":int(sum(x["negative_count"] for x in hs)),
    }
    if len(good):
        summary["boundary_ratio_ranges"]={
            "FM_min":float(good.ratio_FM.min()),
            "FM_max":float(good.ratio_FM.max()),
            "ML_min":float(good.ratio_ML.min()),
            "ML_max":float(good.ratio_ML.max()),
        }
    if audit is not None and len(audit):
        q=audit[audit.get("success", False)==True].copy()
        d=q["delta_full_minus_correct_envelope"].dropna() if "delta_full_minus_correct_envelope" in q else pd.Series(dtype=float)
        summary["audit"]={
            "rows":int(len(audit)),
            "successful_rows":int(len(q)),
            "worst_delta_full_minus_correct_envelope":float(d.min()) if len(d) else None,
            "violations_below_minus_1e-9":int(np.sum(d<-1e-9)) if len(d) else None,
            "max_fourcell_symmetry_defect":float(q.fourcell_symmetry_defect.max()) if len(q) and "fourcell_symmetry_defect" in q else None,
            "max_abs_group_energy_minus_full":float(q.canonical_group_energy_minus_full.abs().max()) if len(q) and "canonical_group_energy_minus_full" in q else None,
            "branch_solver_failures":int(np.sum(audit.success==False)) if "success" in audit else int(len(audit)),
        }

    (outdir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print(f"\nWrote outputs to: {outdir.resolve()}")


if __name__=="__main__":
    main()
