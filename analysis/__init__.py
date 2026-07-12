# Top-level `analysis/` package.
#
# Post-hoc analysis code that is invoked directly with `python -m analysis.<name>.eval`
# rather than through `evals/main.py`. This package is intentionally OUTSIDE `evals/`
# so it does not participate in the `eval_name` dispatch machinery (`evals/scaffold.py`).
