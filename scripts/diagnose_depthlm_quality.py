"""Diagnose DepthLM anchor quality on CIL val data.

Why: the per-image deltas in `outputs/diagnostics/depthlm_tail10_subset/`
show DepthLM-anchored corrections regressing siRMSE by +0.27 to +0.43 on
several images — far worse than the oracle ceiling of ~−0.006. Two
deviations from the DepthLM paper are suspect:

  1) **No focal-length unification.** Paper Finding 3: unifying funi
     doubles δ1 (0.358 → 0.694). Pixtral-12B is trained at funi=750 px.
  2) **GRPO-style prompt.** The released Pixtral chat template has no
     ``<think>/<answer>`` tags; the SFT training prompt is plain
     "How many meters is this point from the camera?".

This script measures, on N random val images × K random anchors each,
the per-pixel log-depth error |log d_VLM − log GT| under four conditions:

  (a) current   = no resize, GRPO-style prompt with <think>/<answer>
  (b) sft       = no resize, SFT prompt
  (c) resize    = funi-resize (FoV-assumed), GRPO prompt
  (d) both      = funi-resize + SFT prompt

For each condition we also report the *scale-invariant* per-anchor error
(after subtracting the per-image median log-shift between VLM and GT) —
that is the right denominator since siRMSE strips the global log-shift
per image.

Usage:
    sbatch slurm/diagnose_depthlm_quality.sbatch \\
        --n-images 25 --n-anchors 15 \\
        --out outputs/diagnostics/depthlm_quality/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EPS = 1e-6

# DepthLM/Pixtral training defaults (paper §3.4 + Appendix F).
FUNI_TARGET = 750            # Pixtral-12B unified focal length, in pixels
FOV_ASSUMED_DEG = 60.0       # assumed horizontal FoV for CIL (no intrinsics)
MARKER_SIZE_PX = 5
LOG_CLIP = (-7.0, 5.0)

# SFT prompt from the paper §3.1 (visual prompting):
PROMPT_SFT = (
    "How many meters is this point from the camera?"
)
# Current code prompt (GRPO style):
PROMPT_GRPO = (
    "Given this image, how far is the point pointed by the red arrow "
    "from the camera? Output the thinking process in <think> </think> "
    "and final answer (the meter number only, without the unit) in "
    "<answer> </answer> tags."
)


def render_marker(rgb: np.ndarray, y: int, x: int,
                  s: int = MARKER_SIZE_PX,
                  color=(255, 0, 0)) -> np.ndarray | None:
    """Render the official DepthLM left-pointing arrow (5 px shaft + V head).

    The anchor pixel itself is NOT overwritten so the image content at the
    query location is preserved. Returns None if the marker can't fit.
    """
    H, W = rgb.shape[:2]
    if not (s <= x < W - s and s <= y < H - s):
        return None
    img = rgb.copy()
    col = np.array(color, dtype=np.uint8)
    for dx in range(1, s + 1):
        img[y, x - dx] = col
    for dy in range(1, s // 2 + 1):
        img[y + dy, x - dy - 1] = col
        img[y - dy, x - dy - 1] = col
    return img


def parse_depth(text: str) -> float:
    """Extract a positive float from the model's generated text.

    Order of preference: <answer>X</answer>, "X meters", any positive float.
    """
    m = re.search(r"<answer>\s*([-+]?\d+(?:\.\d+)?)\s*</answer>", text)
    if m is None:
        m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*meters?", text, flags=re.I)
    if m is None:
        m = re.search(r"\b([-+]?\d+\.\d+)\b", text)
    if m is None:
        return float("nan")
    try:
        v = float(m.group(1))
        return v if v > 0 else float("nan")
    except ValueError:
        return float("nan")


def funi_scale_factor(W: int, fov_deg: float = FOV_ASSUMED_DEG,
                      funi: int = FUNI_TARGET) -> float:
    """Scale factor s such that resizing W by s yields the model's funi.

    fx_assumed = W / (2 tan(fov/2))   (from pinhole model)
    s          = funi / fx_assumed
    """
    fx = W / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    return float(funi / fx)


def load_image_data(train_root: Path, preds_dir: Path, stem: str):
    rgb = np.array(Image.open(train_root / f"{stem}_rgb.png").convert("RGB"),
                   dtype=np.uint8)
    gt = np.load(train_root / f"{stem}_depth.npy").astype(np.float32)
    D0 = np.load(preds_dir / f"{stem}.npy").astype(np.float32)
    return rgb, gt, D0


def sample_anchors(gt: np.ndarray, D0: np.ndarray, n: int,
                   rng: np.random.Generator, margin: int = 10
                   ) -> list[tuple[int, int]]:
    H, W = gt.shape
    valid = (gt > 0) & (D0 > 0) & np.isfinite(gt) & np.isfinite(D0)
    border = np.zeros_like(valid)
    border[margin:H - margin, margin:W - margin] = True
    valid &= border
    idx = np.flatnonzero(valid)
    if len(idx) < n:
        return []
    sel = rng.choice(idx, size=n, replace=False)
    return [(int(i // W), int(i % W)) for i in sel]


def build_inputs(processor, rgb_marked_list, prompt_text: str):
    convs = [
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]}]
        for _ in rgb_marked_list
    ]
    prompts = [
        processor.apply_chat_template(c, add_generation_prompt=True)
        for c in convs
    ]
    pil_imgs = [Image.fromarray(img) for img in rgb_marked_list]
    inputs = processor(text=prompts, images=pil_imgs,
                       return_tensors="pt", padding=True)
    return inputs


def query_batch(model, processor, tokenizer, rgb_marked_list, prompt_text: str,
                max_new_tokens: int, device: str) -> list[str]:
    import torch
    inputs = build_inputs(processor, rgb_marked_list, prompt_text).to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stop_strings=["</answer>", "</s>"],
            tokenizer=tokenizer,
        )
    in_len = inputs["input_ids"].shape[1]
    new_tokens = out_ids[:, in_len:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)


def run_condition(name: str, cond: dict, stems: list[str], n_anchors: int,
                  train_root: Path, preds_dir: Path,
                  model, processor, tokenizer, device: str,
                  batch_size: int, seed: int, verbose: bool = True
                  ) -> list[dict]:
    """Run all (image × anchor) queries for one condition.

    cond: {"prompt": str, "resize": bool}.

    Returns a flat list of records, one per (stem, anchor).
    """
    use_resize = cond["resize"]
    prompt_text = cond["prompt"]
    # GRPO prompt needs more tokens for the <think> body.
    is_grpo = "<think>" in prompt_text
    max_new_tokens = 96 if is_grpo else 24

    rng = np.random.default_rng(seed)  # same anchor seed across conditions
    records: list[dict] = []

    # Build flat (stem, anchor) list, then batch across images.
    work: list[tuple[str, tuple[int, int], np.ndarray, float, float]] = []
    for stem in stems:
        rgb, gt, D0 = load_image_data(train_root, preds_dir, stem)
        anchors = sample_anchors(gt, D0, n_anchors, rng)
        if not anchors:
            continue

        # Resize image (and proportionally the anchor coords) if requested.
        H0, W0 = rgb.shape[:2]
        if use_resize:
            s = funi_scale_factor(W0)
            new_W, new_H = int(round(W0 * s)), int(round(H0 * s))
            rgb_resized = np.array(
                Image.fromarray(rgb).resize((new_W, new_H),
                                            resample=Image.BICUBIC),
                dtype=np.uint8,
            )
            scaled_anchors = [
                (int(round(y * s)), int(round(x * s))) for (y, x) in anchors
            ]
        else:
            rgb_resized = rgb
            scaled_anchors = anchors

        for (y_q, x_q), (y0, x0) in zip(scaled_anchors, anchors):
            marked = render_marker(rgb_resized, y_q, x_q)
            if marked is None:
                continue
            # Record GT and D0 at the *original* coordinate.
            work.append((stem, (y0, x0), marked,
                         float(gt[y0, x0]), float(D0[y0, x0])))

    if verbose:
        print(f"[{name}] queries to run: {len(work)} (resize={use_resize}, "
              f"prompt={'GRPO' if is_grpo else 'SFT'}, max_new_tokens={max_new_tokens})")

    # Batch through the model.
    t0 = time.time()
    for i in range(0, len(work), batch_size):
        batch = work[i:i + batch_size]
        imgs = [w[2] for w in batch]
        try:
            texts = query_batch(model, processor, tokenizer, imgs,
                                prompt_text, max_new_tokens, device)
        except Exception as e:
            print(f"[{name}] batch {i} failed: {e}")
            texts = ["" for _ in batch]
        for (stem, (y, x), _img, gt_m, d0_m), t in zip(batch, texts):
            d_vlm = parse_depth(t)
            records.append({
                "cond": name,
                "stem": stem,
                "y": y, "x": x,
                "gt_m": gt_m,
                "d0_m": d0_m,
                "d_vlm_m": d_vlm,
                "raw": t[:200],
            })
        if verbose and ((i // batch_size) % 10 == 9):
            done = i + len(batch)
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  [{name}] {done}/{len(work)}  ({rate:.1f} q/s)")
    if verbose:
        print(f"[{name}] done in {time.time() - t0:.1f}s")
    return records


def summarise(records: list[dict]) -> dict:
    """Aggregate metrics per condition."""
    by_cond: dict[str, list[dict]] = {}
    for r in records:
        by_cond.setdefault(r["cond"], []).append(r)

    out = {}
    for cond, recs in by_cond.items():
        n = len(recs)
        valid = [r for r in recs if np.isfinite(r["d_vlm_m"]) and r["d_vlm_m"] > 0]
        n_valid = len(valid)

        if n_valid == 0:
            out[cond] = {"n": n, "n_valid": 0}
            continue

        # Per-anchor raw log errors (NOT scale-stripped).
        log_err_gt = np.array([
            abs(np.log(r["d_vlm_m"]) - np.log(max(r["gt_m"], EPS)))
            for r in valid
        ])
        log_err_d0 = np.array([
            abs(np.log(r["d_vlm_m"]) - np.log(max(r["d0_m"], EPS)))
            for r in valid
        ])

        # Per-image scale-stripped error: subtract median(log d_VLM - log GT)
        # per image before measuring the residual. That's the siRMSE-relevant
        # comparison.
        by_stem: dict[str, list[dict]] = {}
        for r in valid:
            by_stem.setdefault(r["stem"], []).append(r)
        si_errs = []
        for stem, group in by_stem.items():
            d_log = np.array([
                np.log(r["d_vlm_m"]) - np.log(max(r["gt_m"], EPS))
                for r in group
            ])
            shift = float(np.median(d_log))
            si_errs.extend(np.abs(d_log - shift).tolist())
        si_errs = np.array(si_errs) if si_errs else np.array([float("nan")])

        out[cond] = {
            "n": n,
            "n_valid": n_valid,
            "parse_rate": n_valid / max(n, 1),
            "abs_log_err_vs_gt":  {"mean": float(log_err_gt.mean()),
                                   "median": float(np.median(log_err_gt)),
                                   "p90": float(np.percentile(log_err_gt, 90))},
            "abs_log_err_vs_d0":  {"mean": float(log_err_d0.mean()),
                                   "median": float(np.median(log_err_d0)),
                                   "p90": float(np.percentile(log_err_d0, 90))},
            "scaleinv_log_err":   {"mean": float(si_errs.mean()),
                                   "median": float(np.median(si_errs)),
                                   "p90": float(np.percentile(si_errs, 90))},
            "n_images": len(by_stem),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preds-dir", default="outputs/val_preds/vits_lora_r8/")
    p.add_argument("--train-root", required=True)
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--n-images", type=int, default=25)
    p.add_argument("--n-anchors", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--model-id", default="facebook/DepthLM")
    p.add_argument("--quant", choices=["nf4", "int8", "bf16"], default="nf4")
    p.add_argument("--conditions", default="current,sft,resize,both",
                   help="Comma-separated subset of {current,sft,resize,both}.")
    args = p.parse_args()

    import torch
    from transformers import (  # type: ignore
        AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_root = Path(args.train_root)
    preds_dir = Path(args.preds_dir)

    # Pick N images deterministically.
    all_preds = sorted(preds_dir.glob("*.npy"))
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(all_preds), size=min(args.n_images, len(all_preds)),
                      replace=False)
    stems = [all_preds[i].stem for i in sorted(pick)]
    # Drop any stem missing GT/RGB.
    stems = [s for s in stems
             if (train_root / f"{s}_rgb.png").exists()
             and (train_root / f"{s}_depth.npy").exists()]
    print(f"[diag] N images: {len(stems)}  anchors/image: {args.n_anchors}")
    print(f"[diag] stems: {stems[:5]}{' ...' if len(stems) > 5 else ''}")

    # Load model once.
    load_kwargs: dict = {"device_map": "cuda"}
    if args.quant == "nf4":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif args.quant == "int8":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
    print(f"[diag] loading {args.model_id} ({args.quant}) ...")
    t0 = time.time()
    model = LlavaForConditionalGeneration.from_pretrained(args.model_id,
                                                          **load_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer
    print(f"[diag] model ready in {time.time() - t0:.1f}s")

    conditions = {
        "current": {"prompt": PROMPT_GRPO, "resize": False},
        "sft":     {"prompt": PROMPT_SFT,  "resize": False},
        "resize":  {"prompt": PROMPT_GRPO, "resize": True},
        "both":    {"prompt": PROMPT_SFT,  "resize": True},
    }
    requested = [c.strip() for c in args.conditions.split(",") if c.strip()]
    conditions = {k: v for k, v in conditions.items() if k in requested}
    print(f"[diag] running conditions: {list(conditions)}")

    all_records: list[dict] = []
    for name, cond in conditions.items():
        recs = run_condition(name, cond, stems, args.n_anchors,
                             train_root, preds_dir,
                             model, processor, tokenizer,
                             device="cuda", batch_size=args.batch_size,
                             seed=args.seed)
        all_records.extend(recs)
        # Incremental write so partial progress is recoverable.
        (out_dir / "records.json").write_text(json.dumps(all_records, indent=1))

    summary = summarise(all_records)
    summary_meta = {
        "args": vars(args),
        "n_stems": len(stems),
        "funi_target": FUNI_TARGET,
        "fov_assumed_deg": FOV_ASSUMED_DEG,
        "summary": summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_meta, indent=2))

    # Markdown table.
    lines = ["# DepthLM anchor-quality diagnostic\n"]
    lines.append(f"N images: {len(stems)}  •  anchors/image: {args.n_anchors}  "
                 f"•  funi target: {FUNI_TARGET}  •  FoV assumed: {FOV_ASSUMED_DEG}°\n")
    lines.append("\n## Per-condition error (lower = better)\n")
    lines.append("| Condition | parse % | |Δlog GT| mean | median | p90 | "
                 "**scale-inv |Δlog|** mean | median | p90 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for cond in requested:
        if cond not in summary:
            continue
        s = summary[cond]
        if s.get("n_valid", 0) == 0:
            lines.append(f"| `{cond}` | 0% | — | — | — | — | — | — |")
            continue
        eg = s["abs_log_err_vs_gt"]
        si = s["scaleinv_log_err"]
        lines.append(
            f"| `{cond}` | {s['parse_rate']*100:.0f}% "
            f"| {eg['mean']:.3f} | {eg['median']:.3f} | {eg['p90']:.3f} "
            f"| **{si['mean']:.3f}** | {si['median']:.3f} | {si['p90']:.3f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines))
    print(f"\n[diag] wrote {out_dir/'summary.json'}, {out_dir/'summary.md'}")
    print(f"[diag] wrote {out_dir/'records.json'}  ({len(all_records)} rows)")


if __name__ == "__main__":
    main()
