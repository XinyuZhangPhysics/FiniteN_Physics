# Numerical claim map

| Manuscript content | Epistemic status | Public source/data |
|---|---|---|
| Sec. 4 / App. D: `N=5` full-space angular audit; canonical locking/rank-one basins | numerical evidence | `landau_n5_global/source/n5_publication_numerical_audit_v1.py`; `landau_n5_global/archived/seed_*` |
| Sec. 4 / App. D: radial `P -> Q_F -> Q_L` sequence and canonical `alpha_FL` reconstruction | numerical + exact canonical anchors | same as above |
| App. D: PSD Maxwell and rigidity stress tests | supporting numerical evidence; not proofs of the open inequalities | same as above, `maxwell_results.json`, `rigidity_results.json` |
| Sec. 4 / App. D: mixed local minimum, positive gap, local-stability spinodals | numerical local-branch result | `landau_n5_mixed_branch/source/n5_mixed_branch_continuation_v1.py`; archived `mixed_branch.csv`; `analysis/analyze_spinodals.py` |
| Sec. 5 / App. G: F/M and M/L Maxwell curves | numerical continuation within the exact four-cell stationary family | `coupled_potts_n5/source/n5_coupled_potts_mixed_closure_v5b.py`; `mixed_maxwell_boundaries.csv` |
| Sec. 5 / App. G: mixed-branch local stability in the full simplex | arbitrary-precision numerical Hessian test | `mixed_high_precision_hessian.csv` |
| Sec. 5 / App. G: no lower full-simplex competitor on the audited parameter set | multistart numerical audit | `corrected_fullspace_audit.csv` |
| Sec. 5 / App. G: triple point | high-precision numerical root and residual check | `triple_point.json` |

## Analytic results intentionally absent from the code archive

The exact invariant basis, sharp quartic and cubic inequalities, equality-orbit
classifications, two-copy wall, three-copy support walls, collision/forest/matching
results, structural-stability arguments, spatial variational bounds, and exact-sector
general-`r` formulas are established analytically in the manuscript.  They therefore
do not require a numerical source program in this public archive.
