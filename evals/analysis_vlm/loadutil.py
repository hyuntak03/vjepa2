# -----------------------------------------------------------------------------
# Model-weight location resolver shared by the VLM encoder backends.
#
# Lets the config point at weights flexibly (config: model_kwargs.wrapper_kwargs):
#   pretrained : HF repo id (e.g. "Qwen/Qwen3-VL-4B-Instruct") OR a local dir
#   cache_dir  : HF cache root to resolve a repo id offline-first
#                (e.g. "/data/dataset/LLaVA-Video-100K-Subset/")
# Falls back to model_kwargs.checkpoint if `pretrained` is not given.
# -----------------------------------------------------------------------------

import glob
import logging
import os

logger = logging.getLogger()


def find_snapshot(path):
    """Accept a snapshot dir, an HF-cache 'models--...' root, or any dir with config.json."""
    path = os.path.expanduser(path)
    if os.path.exists(os.path.join(path, "config.json")):
        return path
    cands = sorted(glob.glob(os.path.join(path, "snapshots", "*")))
    for c in cands:
        if os.path.exists(os.path.join(c, "config.json")):
            return c
    raise FileNotFoundError(f"no snapshot with config.json found under {path}")


def resolve_model_dir(checkpoint, wrapper_kwargs):
    """Return a local directory containing config.json + safetensors shards.

    Resolution order:
      1) wrapper_kwargs.pretrained (or checkpoint) as a local dir       -> find_snapshot
      2) otherwise treat it as an HF repo id and resolve under cache_dir
         (offline first; downloads only if not already cached)
    """
    wk = wrapper_kwargs or {}
    cand = wk.get("pretrained") or checkpoint
    cache_dir = wk.get("cache_dir")
    if not cand:
        raise ValueError("set model_kwargs.checkpoint or wrapper_kwargs.pretrained")

    expanded = os.path.expanduser(str(cand))
    if os.path.isdir(expanded):
        return find_snapshot(expanded)

    # treat as HF repo id, resolve under cache_dir (offline-first)
    from huggingface_hub import snapshot_download

    try:
        d = snapshot_download(cand, cache_dir=cache_dir, local_files_only=True)
        logger.info(f"resolved '{cand}' from cache_dir={cache_dir} -> {d}")
        return d
    except Exception as e:
        logger.warning(f"'{cand}' not cached under {cache_dir} ({e}); attempting download")
        return snapshot_download(cand, cache_dir=cache_dir)
