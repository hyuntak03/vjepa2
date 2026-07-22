"""Parabolic-motion predictor-surprise eval for V-JEPA2.

A self-contained, config-driven physical-plausibility probe that lives under
evals/analysis_vlm without modifying any existing code. It reuses the pretrained
V-JEPA2 encoder+predictor and the low-level forward helpers from
`analysis.intphys2` (imported, never modified).

Modules
-------
dataset.py : load scenes (possible/higher/frozen), splice the shared context.
forward.py : masked-context encode -> predictor -> per-variant target L1 (surprise).
scoring.py : argmin (min-L1 ranking) + pairwise (impossible>possible) scorers.
eval.py    : entry point (`python -m evals.analysis_vlm.parabolic.eval --config ...`).
"""
