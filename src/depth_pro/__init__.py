"""Apple Depth Pro zero-shot metric depth predictions.

Wraps the upstream ``depth_pro`` package (installed separately from
https://github.com/apple/ml-depth-pro) for use in the CIL pipeline.

The public entry point is ``src.depth_pro.predict``, which mirrors the
interface of ``src.finetune.predict`` so downstream scripts
(``eval_preds.py``, ``eval_refine.py``) accept either model's output.
"""
