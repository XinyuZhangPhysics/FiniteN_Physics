# Public reproducibility release audit

## Scientific inclusion criterion

A program is included only if a numerical statement in the manuscript depends on its
output or if it deterministically analyzes such an output.  This criterion selects
three calculation families:

1. N=5 Landau full-space global audit and PSD stress tests;
2. N=5 Landau mixed metastable-branch continuation and spinodal analysis;
3. N=5 coupled-Potts Maxwell continuation, high-precision Hessian tests, full-simplex
   multistart audit, and triple-point solve.

## Exclusion criterion

The release omits programs whose role is fully replaced by a strict analytic proof in
the manuscript.  It also omits exploratory or superseded calculations and calculations
that terminated because of computational resources, software environment, or workflow
limitations without producing a scientific conclusion used in the paper.

In particular, no dormant SOS/Groebner/interval experiments, old fold interpretations,
private project-history files, candidate-search inventories, or local-path provenance
are present in the public archive.

## Verification performed for this release

- archived-result verifier: PASS;
- public source hash check: PASS;
- Python syntax compilation for all public Python files: PASS;
- small-scale smoke run of the Landau global-audit source: PASS;
- small-scale smoke run of the coupled-Potts source: PASS;
- deterministic mixed-branch endpoint analysis of the archived table: PASS;
- shell-script syntax checks: PASS.

The publication-scale optimizations were not rerun during packaging; the frozen outputs
are the data used by the manuscript and are independently checked by
`verify_archived_results.py`.

## GitHub public-release preflight (2026-10-17)

- All three scientific source families, endpoint analysis, archived results,
  expected values, run configurations, and provenance files are present.
- Frozen-result verifier: `PUBLIC_REPRODUCIBILITY_STATUS = PASS`.
- Syntax checks: all five public Python files and all three shell scripts pass.
- Original source and release-file checksum manifests passed before cleanup.
- Content and filename audit found no credentials, private keys, absolute local
  paths, environment files, editor workspaces, temporary logs, crash dumps,
  unrelated large caches, or internal/legacy/candidate-discovery/failed-run files.
  Documentation references to excluded research history describe the package scope.
- Removed one generated Python bytecode file from the endpoint-analysis cache.
  Added `.gitignore` for local artifacts and refreshed `SHA256SUMS.txt`.
- Scientific sources, analysis, frozen outputs, validation data, thresholds, and
  `SOURCE_SHA256.txt` are unchanged. Publication-scale computations were not rerun.
- No software license was selected or added.
