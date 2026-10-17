# Reproduction environment

The numerical sources are Python programs.  A present-day tested environment used
when the archive was prepared was:

- Python 3.14.x
- NumPy 2.5.4
- SciPy 1.18.1
- pandas 3.0.6
- mpmath 1.4.1

The full package is most naturally run on a multicore machine.  The worker counts in
`RUN_CONFIGS.json` reproduce the original calculation protocol, but smaller worker
counts can be used for portability.

The original execution-time Python package versions were not recorded.  The versions
above are therefore reproduction pins, not reconstructed historical metadata.  The
frozen output tables are checked independently by `verify_archived_results.py`, which
uses only the Python standard library.
