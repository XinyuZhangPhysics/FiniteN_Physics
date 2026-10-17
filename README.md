# Finite-state selection of overlap geometry and support topology
## Public numerical reproducibility package

This archive contains the source programs and frozen numerical outputs on which the
paper's non-analytic numerical statements depend.  It is intentionally smaller than
the internal research archive: exact analytic derivations, exploratory programs,
superseded routes, and computations that failed for non-scientific reasons are not
included.

### What is included

1. **`landau_n5_global/`** — three independent full-space `N=5` Landau audits.
   These data support the numerical `P -> Q_F -> Q_L` sequence, the absence of an
   observed noncanonical global preemption on the audited grids, and the numerical
   PSD Maxwell/rigidity stress tests reported in Sec. 4 and Appendix D.

2. **`landau_n5_mixed_branch/`** — the mixed metastable angular branch.
   The archived continuation table supports the local-stability spinodals and the
   positive free-energy gap to the canonical envelope.  The endpoint analysis fits
   the smallest tangent-Hessian eigenvalue linearly and interprets its zero crossings
   as **local-stability spinodals**, not as saddle-node theorems.

3. **`coupled_potts_n5/`** — the finite-coupling `N=5` coupled-Potts calculation.
   The archived outputs contain the 55-row F/M and M/L Maxwell continuations, 45
   arbitrary-precision tangent-Hessian tests, 120 full-simplex multistart audits, and
   the high-precision triple point used in Sec. 5 and Appendix G.

### What is deliberately excluded

- algebraic and representation-theoretic derivations that are proved directly in the
  manuscript and appendices;
- exploratory SOS, Groebner, interval, and symbolic programs that do not supply a
  numerical statement used by the paper;
- failed calculations whose failure was caused by memory, wall time, software, or
  workflow limitations rather than by a scientific result;
- superseded interpretations (in particular the old `lambda_min^2` fold reading of
  the mixed branch);
- private research records, candidate-discovery logs, local filesystem provenance,
  and historical packaging metadata.

This separation is intentional: the public package documents the calculations that
support the paper, not the full history of how the project was developed.

## Quick verification

The archived results can be checked without NumPy/SciPy and without rerunning any
optimizer:

```bash
python3 verify_archived_results.py
```

Expected final line:

```text
PUBLIC_REPRODUCIBILITY_STATUS = PASS
```

The verifier checks the source hashes and the manuscript-level numerical statements,
including start counts, phase sequence, envelope gaps, mixed-branch spinodal fits,
coupled-Potts Maxwell rows, arbitrary-precision Hessian inertia, full-simplex
no-preemption tolerances, and the triple-point residuals.

## Full reruns

A tested reproduction environment is listed in `requirements.txt`.  For example:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then run one or more of:

```bash
./run_landau_global.sh
./run_landau_mixed_branch.sh
./run_coupled_potts.sh
```

The full audits are computationally expensive.  `WORKERS` can be overridden, e.g.
`WORKERS=4 ./run_landau_global.sh`.  The frozen outputs are the data quoted by the
paper; fresh optimizer trajectories can exhibit harmless floating-point variation, so
scientific comparisons should use the tolerances encoded in the `expected.json`
files rather than byte-for-byte equality.

## Reproduction environment

The dependency pins in `requirements.txt` reproduce a present-day environment in
which the sources were audited.  The original execution-time package versions were
not recorded and are therefore not reconstructed here.  This distinction does not
affect the frozen numerical evidence: all quoted manuscript quantities are checked
from the archived output files by the standard-library verifier.

## File integrity

- `SOURCE_SHA256.txt` binds the three numerical source programs.
- `SHA256SUMS.txt` binds every public-release file except itself.
- `MANIFEST.json` gives the role of every top-level calculation family.

See `CLAIM_MAP.md` for the correspondence between manuscript claims and files.
