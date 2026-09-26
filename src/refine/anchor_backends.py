"""Anchor sources for the sparse-to-dense corrector.

The corrector (`src/refine/correct_dense.py`) is anchor-source-agnostic: it
takes a list of anchor pixel coordinates and a per-anchor log-depth shift
Δ_a and propagates them densely. This module defines the protocol that any
"sensor" producing those Δ_a must implement, and provides three concrete
backends:

  - `OracleBackend`     — eval-only; reads the ground-truth depth and
                           returns the patch-median log-depth at each
                           anchor. Establishes the corrector's upper bound.
  - `SecondModelBackend` — reads a different pre-trained depth model's
                           precomputed prediction at each anchor (e.g. Apple
                           Depth Pro); a cheap, architecturally decorrelated
                           source.
  - `DepthLMBackend`     — queries the DepthLM (Pixtral-12B) depth-from-
                           language VLM with a marker rendered per anchor.

All three implement the same `query_depth` protocol and slot into the eval
harness without any change to the corrector.

Returned values are *log-depths*, not raw depths, because that's the unit
the corrector internally subtracts log D0 in.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

EPS = 1e-6


class AnchorBackend(Protocol):
    """Protocol every anchor source must satisfy.

    Every source returns a log-depth at queried pixels; the corrector never
    sees the source internals.
    """

    name: str

    def query_depth(
        self,
        rgb: np.ndarray,
        anchors_yx: Sequence[tuple[int, int]],
        *,
        stem: str | None = None,
    ) -> tuple[list[float], list[float] | None]:
        """Returns (log_depths, confidences) per anchor.

        Args:
            rgb:        [H, W, 3] uint8 image.
            anchors_yx: list of (y, x) pixel coordinates.
            stem:       optional image identifier; backends that load
                        per-image cached predictions (oracle, second-model)
                        use this to find the right file.

        Returns:
            (log_depths, confidences) — both lists of length len(anchors_yx).
            `confidences` may be None for backends that don't model it.
        """
        ...


# -------------------------------------------------------------------------
# Oracle backend


class OracleBackend:
    """Eval-only: reads ground-truth depth and returns patch-median log-depth.

    Args:
        gt_root: directory containing <stem>_depth.npy files.
        patch:   anchor target = median of log GT in a (2r+1)x(2r+1) patch
                 around each anchor; patch=1 → single pixel.
                 Patch>1 emulates a trained VLM's natural per-region
                 smoothing and is resistant to edge-pixel GT outliers.
    """

    name = "oracle"

    def __init__(self, gt_root: str | Path, patch: int = 15):
        self.gt_root = Path(gt_root)
        if patch < 1 or patch % 2 == 0:
            raise ValueError(f"patch must be a positive odd integer, got {patch}")
        self.r = patch // 2

    def query_depth(
        self,
        rgb: np.ndarray,
        anchors_yx: Sequence[tuple[int, int]],
        *,
        stem: str | None = None,
    ) -> tuple[list[float], list[float] | None]:
        if stem is None:
            raise ValueError("OracleBackend requires stem (to locate the GT file)")
        gt_path = self.gt_root / f"{stem}_depth.npy"
        gt = np.load(gt_path).astype(np.float32)
        log_gt = np.log(np.maximum(gt, EPS))
        H_, W_ = gt.shape
        r = self.r
        out: list[float] = []
        for (y, x) in anchors_yx:
            y0, y1 = max(0, y - r), min(H_, y + r + 1)
            x0, x1 = max(0, x - r), min(W_, x + r + 1)
            patch_gt = gt[y0:y1, x0:x1]
            mask = patch_gt > 0
            if mask.any():
                target = float(np.median(log_gt[y0:y1, x0:x1][mask]))
            else:
                target = float(log_gt[y, x])
            out.append(target)
        return out, None


# -------------------------------------------------------------------------
# Second-model backend


class SecondModelBackend:
    """Anchors from a different (precomputed) monocular-depth model.

    The model itself is not loaded here; instead we expect its predictions
    to have been precomputed once on the val/test set and cached as
    <stem>.npy log-depth (or raw depth) arrays. This keeps the eval loop
    cheap and lets the second-model integration live in a separate
    inference script.

    Args:
        preds_dir: directory containing <stem>.npy files with the second
                   model's per-pixel depth predictions.
        in_log:    True if the cached files store log-depth; False (default)
                   for raw depth in meters.
    """

    def __init__(self, preds_dir: str | Path, *, name: str = "second_model",
                 in_log: bool = False):
        self.preds_dir = Path(preds_dir)
        self.name = name
        self.in_log = in_log

    def query_depth(
        self,
        rgb: np.ndarray,
        anchors_yx: Sequence[tuple[int, int]],
        *,
        stem: str | None = None,
    ) -> tuple[list[float], list[float] | None]:
        if stem is None:
            raise ValueError(f"{self.name} backend requires stem")
        path = self.preds_dir / f"{stem}.npy"
        arr = np.load(path).astype(np.float32)
        if self.in_log:
            log_d = arr
        else:
            log_d = np.log(np.maximum(arr, EPS))
        out = [float(log_d[y, x]) for (y, x) in anchors_yx]
        return out, None


# -------------------------------------------------------------------------
# DepthLM backend


class DepthLMBackend:
    """Anchors from DepthLM (Pixtral-12B) via marker prompting.

    DepthLM (https://arxiv.org/abs/2509.25413) takes an RGB image with a
    small red arrow rendered *pointing at* the queried pixel and emits a
    scalar metric depth as text inside a ``<answer> ... </answer>`` block,
    typically preceded by a brief ``<think> ... </think>`` trace.

    The rendering matches the official repo
    (https://github.com/facebookresearch/DepthLM_Official/utils/datasets.py)
    exactly: a 5-pixel horizontal shaft extending LEFT of the anchor, with
    a 2+2-pixel V-shaped arrowhead — the anchor pixel itself is *not
    overwritten*, so the image at the query location is preserved.

    siRMSE robustness: the CIL dataset ships without camera intrinsics, so
    we skip the focal-length unification from the DepthLM paper. siRMSE is
    scale-invariant per image; a constant log-shift in DepthLM's emitted
    depths is stripped by the corrector's mean-zero post-processing.

    The 12B model is quantised to fit on a 16 GB GPU; ``load_in_4bit`` (NF4,
    default) or ``load_in_8bit`` selects the scheme via bitsandbytes.

    Args:
        model_id:       HuggingFace repo or local path.
        device:         "cuda" or "cpu".
        load_in_4bit:   wrap the model in bitsandbytes NF4 4-bit linears (default).
        load_in_8bit:   wrap the model in bitsandbytes 8-bit linears (mutually
                        exclusive with load_in_4bit).
        marker_size:    arrow shaft length in pixels (paper uses 5).
        marker_color:   RGB of the marker (uint8). Paper uses (255, 0, 0).
        batch_size:     anchors per VLM generate call.
        max_new_tokens: generation length cap; the think+answer template fits
                        comfortably within the default.
        prompt:         the text following the [IMG] token. Default is the
                        exact training prompt from the DepthLM repo.
        log_clip:       log-depth clipping bounds.
    """

    name = "depthlm"
    DEFAULT_PROMPT = (
        "Given this image, how far is the point pointed by the red arrow "
        "from the camera? Output the thinking process in <think> </think> "
        "and final answer (the meter number only, without the unit) in "
        "<answer> </answer> tags."
    )

    def __init__(
        self,
        model_id: str = "facebook/DepthLM",
        device: str = "cuda",
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        marker_size: int = 5,
        marker_color: tuple[int, int, int] = (255, 0, 0),
        batch_size: int = 4,
        max_new_tokens: int = 48,
        prompt: str | None = None,
        log_clip: tuple[float, float] = (-7.0, 5.0),
    ):
        try:
            import torch
            from transformers import (  # type: ignore
                AutoProcessor, LlavaForConditionalGeneration,
            )
        except ImportError as e:
            raise ImportError(
                "DepthLMBackend requires transformers and torch. "
                "Run inside the monodepth-5060 conda env."
            ) from e

        # NF4 over int8: lower memory, negligible quality loss for short
        # numeric generation.
        load_kwargs: dict = {"device_map": device}
        if load_in_4bit and load_in_8bit:
            raise ValueError("Pick one of load_in_4bit / load_in_8bit.")
        if load_in_4bit or load_in_8bit:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "Quantized loading requires bitsandbytes. "
                    "pip install bitsandbytes."
                ) from e
            if load_in_4bit:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True
                )
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        quant_label = "nf4" if load_in_4bit else ("int8" if load_in_8bit else "bf16")
        print(f"[depthlm] loading {model_id} ({quant_label}) ...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id, **load_kwargs,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.device = device

        self.marker_size = marker_size
        self.marker_color = np.array(marker_color, dtype=np.uint8)
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt if prompt is not None else self.DEFAULT_PROMPT
        self.log_clip = log_clip

        self._torch = torch  # cache for query_depth

    def _render_marker(self, rgb: np.ndarray, y: int, x: int) -> np.ndarray | None:
        """Render the official DepthLM arrow pointing at (y, x).

        Matches utils/datasets.py from the DepthLM repo exactly:
          - Horizontal shaft: 5 pixels extending LEFT of (x, y).
          - Arrowhead: 2 pixels above-left and 2 pixels below-left, in a
            leftward-opening V pattern.
        The anchor pixel itself is *NOT* modified.

        Returns None if the anchor is too close to the image boundary for
        the arrow to fit — the caller should treat this anchor as NaN.
        """
        s = self.marker_size
        H, W = rgb.shape[:2]
        # Border check mirrors the repo (which falls back to a different
        # sample). We have no fallback at inference, so the caller drops it.
        if not (s <= x < W - s and s <= y < H - s):
            return None

        img = rgb.copy()
        col = self.marker_color
        # Horizontal shaft: 5 pixels left of anchor (anchor pixel itself untouched).
        for dx in range(1, s + 1):
            img[y, x - dx] = col
        # Arrowhead: V opening leftward, 2 above + 2 below the shaft.
        for dy in range(1, s // 2 + 1):
            img[y + dy, x - dy - 1] = col
            img[y - dy, x - dy - 1] = col
        return img

    def _parse_depth(self, text: str) -> float:
        """Extract the depth from the model's generated text.

        Primary form (matches the training template):
            ``<think> ... </think><answer> X.XX </answer>``
        Fallback: any "<number> meters" or bare positive float in the text.
        """
        import re
        m = re.search(r"<answer>\s*([-+]?\d+(?:\.\d+)?)\s*</answer>", text)
        if m is None:
            m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*meters?", text)
        if m is None:
            m = re.search(r"\b([-+]?\d+\.\d+)\b", text)
        if m is None:
            return float("nan")
        try:
            return float(m.group(1))
        except ValueError:
            return float("nan")

    def query_depth(
        self,
        rgb: np.ndarray,
        anchors_yx,  # noqa: ANN001 — Sequence[tuple[int, int]]
        *,
        stem: str | None = None,
    ) -> tuple[list[float], list[float] | None]:
        torch = self._torch
        from PIL import Image

        n = len(anchors_yx)
        log_depths: list[float] = [float("nan")] * n

        for i in range(0, n, self.batch_size):
            batch = anchors_yx[i : i + self.batch_size]
            rendered = [
                (self._render_marker(rgb, int(y), int(x)), j)
                for j, (y, x) in enumerate(batch)
            ]
            # Drop anchors too close to the image border (marker doesn't fit).
            keep = [(img, j) for (img, j) in rendered if img is not None]
            if not keep:
                continue
            pil_imgs = [Image.fromarray(img) for (img, _) in keep]
            keep_idx = [j for (_, j) in keep]
            # One conversation per image. Pixtral's chat template inserts [IMG]
            # in the user message; we ask for one depth per image.
            convs = [
                [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ]}]
                for _ in keep_idx
            ]
            prompts = [
                self.processor.apply_chat_template(c, add_generation_prompt=True)
                for c in convs
            ]
            inputs = self.processor(
                text=prompts, images=pil_imgs,
                return_tensors="pt", padding=True,
            ).to(self.device)

            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    # Stop as soon as the answer block closes; nothing past
                    # </answer> is needed.
                    stop_strings=["</answer>"],
                    tokenizer=self.tokenizer,
                )

            in_len = inputs["input_ids"].shape[1]
            new_tokens = out_ids[:, in_len:]
            texts = self.processor.batch_decode(
                new_tokens, skip_special_tokens=True
            )
            for local_j, t in enumerate(texts):
                d_m = self._parse_depth(t)
                if d_m is None or not np.isfinite(d_m) or d_m <= 0:
                    continue
                log_d = float(np.log(d_m))
                log_d = max(self.log_clip[0], min(self.log_clip[1], log_d))
                # Map batch-local index back through keep_idx to the original
                # anchor index in this slice, then to the global anchor index.
                log_depths[i + keep_idx[local_j]] = log_d

        # No usable confidence signal yet — DepthLM emits a single number.
        return log_depths, None


def make_backend(spec: str, **kwargs) -> AnchorBackend:
    """Factory: build a backend from a string spec.

    Used by `eval_anchor_sources.py` so users can swap backends via a
    single CLI flag. The spec format is intentionally simple; complex
    backends (e.g. a live VLM) should be instantiated explicitly.
    """
    if spec == "oracle":
        return OracleBackend(gt_root=kwargs["gt_root"],
                             patch=kwargs.get("patch", 15))
    if spec == "second":
        return SecondModelBackend(
            preds_dir=kwargs["preds_dir"],
            name=kwargs.get("name", "second_model"),
            in_log=kwargs.get("in_log", False),
        )
    if spec == "vlm" or spec == "depthlm":
        return DepthLMBackend(
            model_id=kwargs.get("model_id", "facebook/DepthLM"),
            device=kwargs.get("device", "cuda"),
            load_in_4bit=kwargs.get("load_in_4bit", True),
            load_in_8bit=kwargs.get("load_in_8bit", False),
            batch_size=kwargs.get("batch_size", 4),
        )
    raise ValueError(f"unknown backend spec {spec!r}")
