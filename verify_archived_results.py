#!/usr/bin/env python3
"""Verify the frozen numerical results used in the manuscript.

Standard-library only.  This script does not rerun optimizers.  It checks the
archived output tables against the numerical statements quoted in the paper.
"""
from __future__ import annotations
import csv, json, math, sys
from decimal import Decimal as D, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FAIL = []

def readj(p):
    return json.loads((ROOT/p).read_text())

def readcsv(p):
    with (ROOT/p).open(newline='') as f:
        return list(csv.DictReader(f))

def need(ok, msg):
    if not ok:
        raise AssertionError(msg)

def close(a, b, tol):
    return math.isfinite(float(a)) and abs(float(a)-float(b)) <= float(tol)

def check(name, fn):
    try:
        fn(); print('PASS ', name)
    except Exception as exc:
        FAIL.append(name); print('FAIL ', name, '-', exc)

LAND = Path('landau_n5_global')
MIX = Path('landau_n5_mixed_branch')
POTTS = Path('coupled_potts_n5')
LAND_RUNS = ['seed_20261015','seed_31415926','seed_27182818']


def landau_counts_and_sequence():
    e = readj(LAND/'expected.json')
    angular_random = radial_random = psd = 0
    for run in LAND_RUNS:
        base = LAND/'archived'/run
        a = readcsv(base/'angular_scan.csv')
        r = readcsv(base/'radial_scan.csv')
        s = readj(base/'summary.json')
        need(len(a) == 150 == s['angular']['n_omega'], run+' angular grid')
        need(len(r) == 148 == s['radial']['n_alpha'], run+' radial grid')
        angular_random += 150*384
        radial_random += 148*256
        psd += 200000
        seq=[]
        for row in sorted(r, key=lambda x:-float(x['alpha'])):
            if abs(float(row['alpha'])-e['alpha_FL']) < 1e-8 or abs(float(row['alpha'])-e['alpha_onset_F']) < 1e-8:
                continue
            if not seq or seq[-1] != row['class']:
                seq.append(row['class'])
        need(seq == e['phase_sequence'], run+' phase sequence '+repr(seq))
    need(angular_random == e['angular_random_starts'], 'angular start total')
    need(radial_random == e['radial_random_starts'], 'radial start total')
    need(psd == e['psd_shared_samples'], 'PSD draw total')


def landau_envelope_and_basins():
    e = readj(LAND/'expected.json')
    for run, target in zip(LAND_RUNS, e['angular_worst_gaps']):
        base = LAND/'archived'/run
        rows = readcsv(base/'angular_scan.csv')
        gap = min(float(x['gap_to_envelope']) for x in rows)
        need(close(gap, target, e['tolerances']['rounded_angular_gap_abs']), run+' worst angular gap')
        need(all(x['class']!='mixed' and float(x['gap_to_envelope']) >= -e['tolerances']['angular_envelope'] for x in rows), run+' angular preemption')
        catalog = readcsv(base/'angular_minima_catalog.csv')
        for lab in ('locking','rank_one'):
            need(any(abs(float(x['omega'])-e['omega_star'])<1e-15 and x['class']==lab and int(x['count'])>1 and abs(float(x['gap_to_envelope']))<e['tolerances']['basin_energy_abs'] for x in catalog), run+' '+lab+' basin at omega*')
        rad = readcsv(base/'radial_scan.csv')
        need(all(x['class']!='mixed' and float(x['gap_to_envelope']) >= -e['tolerances']['radial_envelope'] for x in rad), run+' radial preemption')
        s = readj(base/'summary.json')
        need(close(s['canonical_radial_crossing']['alpha_FL'], e['alpha_FL'], e['tolerances']['alpha_FL_abs']), run+' alpha_FL')
        need(close(s['constants']['alpha_F_onset'], e['alpha_onset_F'], 1e-15), run+' alpha onset')


def landau_psd_support():
    for run in LAND_RUNS:
        base = LAND/'archived'/run
        m = readj(base/'maxwell_results.json')
        r = readj(base/'rigidity_results.json')
        need(min(x['value'] for x in m['polished']) > -5e-9, run+' Maxwell numerical minimum')
        need(min(x['value'] for x in r['polished']) > 4/105 + 1e-4, run+' rigidity numerical margin')


def mixed_branch():
    e = readj(MIX/'expected.json')
    rows = sorted(readcsv(MIX/'archived/mixed_branch.csv'), key=lambda x:float(x['omega']))
    need(len(rows)==118, 'mixed row count')
    stable = [x for x in rows if float(x['hess_min'])>0]
    need(len(stable)==114, 'strictly positive Hessian rows')
    need(close(min(float(x['gap_to_envelope']) for x in stable), e['stable_min_gap'], e['tolerances']['gap_abs']), 'metastable gap')
    signs=[]
    for a,b in zip(rows,rows[1:]):
        if float(a['hess_min'])*float(b['hess_min'])<0:
            signs.append((float(a['omega']),float(b['omega'])))
    need(len(signs)==2, 'two Hessian sign-change brackets')
    for got, exp in zip(signs, e['brackets']):
        need(abs(got[0]-exp[0])<2e-7 and abs(got[1]-exp[1])<2e-7, 'spinodal bracket')
    at14=[x for x in rows if abs(float(x['omega'])-.14)<1e-15]
    need(len(at14)==1, 'omega=.14 row')
    for k,v in e['diagnostic_014'].items():
        need(close(at14[0][k], v, e['tolerances']['diagnostic_abs']), 'omega=.14 '+k)


def mixed_endpoint_analysis():
    sys.path.insert(0, str((ROOT/MIX/'analysis').resolve()))
    from analyze_spinodals import analyze
    e = readj(MIX/'expected.json')
    fresh = analyze(ROOT/MIX/'archived/mixed_branch.csv', e)
    need(fresh['row_count']==118, 'analysis row count')
    for side, target in zip(('lower','upper'), e['endpoint_estimates']):
        f = next(x for x in fresh['fits'] if x['side']==side and x['nfit']==e['primary_fit_window'])
        need(abs(f['omega_zero']-target) <= e['tolerances']['endpoint_abs'], side+' endpoint estimate')
        need(f['r2'] > e['minimum_fit_r2'], side+' primary fit R^2')
        roots=[x['omega_zero'] for x in fresh['fits'] if x['side']==side]
        need(max(roots)-min(roots) < e['tolerances']['fit_root_window_spread'], side+' fit-window stability')


def potts_boundaries_and_hessian():
    e = readj(POTTS/'expected.json')
    rows = readcsv(POTTS/'archived/mixed_maxwell_boundaries.csv')
    s = readj(POTTS/'archived/summary.json')
    need(len(rows)==e['boundary_rows']==s['boundary_rows']==s['successful_boundary_rows'], 'Maxwell row count')
    need(all(x['fm_success']=='True' and x['ml_success']=='True' for x in rows), 'Maxwell success')
    need(all(float(x['FM_residual'])<1e-8 and float(x['ML_residual'])<1e-8 for x in rows), 'Maxwell residuals')
    h = readcsv(POTTS/'archived/mixed_high_precision_hessian.csv')
    need(len(h)==e['hessian_rows'], 'Hessian row count')
    need(sum(int(x['negative_count']) for x in h)==0, 'negative Hessian modes')
    need(all(int(x['zero_count'])==0 and int(x['mp_dps'])>=80 and D(x['min_congruence_eigenvalue'])>0 for x in h), 'positive arbitrary-precision inertia')


def potts_fullspace():
    e = readj(POTTS/'expected.json')
    rows = readcsv(POTTS/'archived/corrected_fullspace_audit.csv')
    s = readj(POTTS/'archived/summary.json')['audit']
    need(len(rows)==e['audit_rows']==s['rows']==s['successful_rows'], 'full-simplex audit count')
    need(all(x['success']=='True' for x in rows) and s['branch_solver_failures']==0, 'full-simplex solver success')
    v=[float(x['delta_full_minus_correct_envelope']) for x in rows]
    need(all(x>=e['no_preemption_threshold'] for x in v), 'no lower full-simplex competitor')
    need(close(min(v), e['worst_difference'], e['tolerances']['rounded_energy_abs']), 'worst full-minus-envelope difference')
    mismatch=max(abs(float(x['canonical_group_energy_minus_full'])) for x in rows)
    defect=max(float(x['fourcell_symmetry_defect']) for x in rows)
    need(close(mismatch,e['max_canonical_energy_mismatch'],e['tolerances']['rounded_energy_abs']), 'canonical energy mismatch')
    need(close(defect,e['max_fourcell_symmetry_defect'],e['tolerances']['symmetry_rounded_abs']), 'four-cell symmetry defect')


def potts_triple():
    e=readj(POTTS/'expected.json'); raw=readj(POTTS/'archived/triple_point.json'); h=raw['high_precision']; tol=e['tolerances']
    with localcontext() as ctx:
        ctx.prec=80
        v={k:D(h[k]) for k in ('aF','bF','aM','bM','cM','cL','j','t')}; j=v['j']; t=v['t']; n=D(4)
        def cells(a,b,c):
            ea,eb,ec=a.exp(),b.exp(),c.exp(); den=ea+2*n*eb+n*ec+n*(n-1); return ea/den,eb/den,ec/den,1/den
        def stat(a,b,c):
            x,y,z,w=cells(a,b,c); dr=x+n*y-y-z-(n-1)*w
            return t*a-2*(x-w)+2*(1-j)*dr, t*b-2*(y-w)+(1-j)*dr, t*c-2*(z-w)
        def energy(a,b,c):
            x,y,z,w=cells(a,b,c); counts=(D(1),2*n,n,n*(n-1)); p=(x,y,z,w); r1=x+n*y; r0=y+z+(n-1)*w
            return t*sum(k*q*q.ln() for k,q in zip(counts,p))-sum(k*q*q for k,q in zip(counts,p))+(1-j)*(r1*r1+n*r0*r0)
        sf=stat(v['aF'],v['bF'],D(0)); sm=stat(v['aM'],v['bM'],v['cM']); sl=stat(v['cL'],D(0),v['cL'])
        need(max(abs(q) for q in (*sf[:2],*sm,sl[2]))<D(tol['triple_stationarity_residual']), 'triple stationarity residual')
        ef=energy(v['aF'],v['bF'],D(0)); em=energy(v['aM'],v['bM'],v['cM']); el=energy(v['cL'],D(0),v['cL'])
        need(max(abs(ef-em),abs(el-em))<D(tol['triple_coexistence_residual']), 'triple coexistence residual')
        need(abs(t/j-D(h['ratio']))<D(tol['triple_ratio_residual']), 'triple ratio residual')
        for k,target in e['triple'].items(): need(abs(D(h[k])-D(target))<D(tol['triple_anchor_abs']), 'triple '+k)


def source_hashes():
    import hashlib
    lines=(ROOT/'SOURCE_SHA256.txt').read_text().splitlines()
    for line in lines:
        digest, rel=line.split(None,1)
        got=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
        need(got==digest, 'source hash '+rel)

if __name__=='__main__':
    checks=[
        ('source hashes',source_hashes),
        ('Landau counts and P->Q_F->Q_L sequence',landau_counts_and_sequence),
        ('Landau full-space envelopes and canonical basins',landau_envelope_and_basins),
        ('Landau PSD Maxwell/rigidity numerical support',landau_psd_support),
        ('mixed metastable branch table',mixed_branch),
        ('mixed spinodal endpoint analysis',mixed_endpoint_analysis),
        ('coupled-Potts Maxwell continuation and Hessian inertia',potts_boundaries_and_hessian),
        ('coupled-Potts full-simplex audit',potts_fullspace),
        ('coupled-Potts high-precision triple point',potts_triple),
    ]
    for name,fn in checks: check(name,fn)
    print('PUBLIC_REPRODUCIBILITY_STATUS = '+('PASS' if not FAIL else 'FAIL'))
    raise SystemExit(1 if FAIL else 0)
