# IntPhys2 pairwise-surprise evaluation harness.
#
# Reproduces the prediction-based evaluation protocol of "IntPhys 2" (Bordes et al., 2025;
# arXiv:2506.09849), Appendix D. Uses a frozen V-JEPA(2) encoder + predictor to score each
# video by its per-window prediction error ("surprise"), then compares surprise inside each
# scene's (Possible, Impossible) quadruplet to compute pairwise accuracy.
#
# Design axes (all YAML-driven, see `configs/analysis/intphys2/`):
#   data         : IntPhys2 split root + metadata.csv, target framerate, spatial resize
#   model        : V-JEPA(2) family + which state_dict keys become the context/target encoders
#   surprise     : sliding-window size / context length / stride / distance / aggregation
#   evaluation   : pairwise vs single-video (AUROC), breakdown axes (condition, Difficulty, ...)
#
# Entry point:
#   python -m analysis.intphys2.eval --config configs/analysis/intphys2/vjepa2_vitl_debug.yaml
