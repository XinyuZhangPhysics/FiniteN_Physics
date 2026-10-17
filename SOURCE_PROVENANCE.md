# Source provenance

The public archive was assembled from the frozen internal reproducibility package
prepared for the manuscript.

Two numerical sources are byte-identical to the frozen scientific sources:

- `landau_n5_global/source/n5_publication_numerical_audit_v1.py`
  - frozen SHA-256: `4b127df071249852caaae2ba5da1e74072a1f03c6e58961f20ead06f18af0061`
- `coupled_potts_n5/source/n5_coupled_potts_mixed_closure_v5b.py`
  - frozen SHA-256: `6c0ce6f1402673ea23dbe88921c16a01e456bb89f3b75850ef6cb5a33b6b1c3f`

The mixed-branch source is a publication-facing cleanup of the frozen source whose
SHA-256 was
`e01535144b382950b3b3b8404ef9795316456f999c243f1cf4e9b4cd6dc4f254`.
The continuation kernel, optimization settings, branch-identity tests, tangent-Hessian
calculation, accepted-row logic, and CSV output are unchanged.  Only the superseded
`lambda_min^2` fold-fit postprocessing and its explanatory text were removed.  The
paper's endpoint estimates are instead produced by
`landau_n5_mixed_branch/analysis/analyze_spinodals.py`, which analyzes the archived
raw Hessian eigenvalues directly.

`SOURCE_SHA256.txt` records the hashes of the public-release source files.
