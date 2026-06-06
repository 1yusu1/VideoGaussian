#!/usr/bin/env python3
"""Generate text-to-video clips with Hugging Face Diffusers.

This is an optional data-generation helper for VideoGaussian experiments. It
keeps Diffusers out of the core requirements and writes a manifest so generated
clips can be traced back to prompts, seeds, and model ids.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_NEGATIVE_PROMPT = (
    "inconsistent motion, blurry, jittery, distorted, low quality, text, "
    "subtitles, watermark, fast cuts, scene cuts, moving people"
)

DEFAULT_MODEL_IDS = {
    "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "ltx": "Lightricks/LTX-Video",
    "cogvideox": "THUDM/CogVideoX-2b",
}


@dataclass(frozen=True)
class PromptSpec:
    scene: str
    prompt: str
    negative_prompt: str | None = None
    seed: int | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate local text-to-video clips for VideoGaussian experiments."
    )
    parser.add_argument(
        "--backend",
        default="wan",
        choices=sorted(DEFAULT_MODEL_IDS),
        help="Diffusers video pipeline backend.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Hugging Face model id or local Diffusers checkpoint directory.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to generate. Can be passed multiple times.",
    )
    parser.add_argument(
        "--prompts-file",
        default=None,
        help=(
            "CSV/TSV/TXT prompt file. CSV/TSV may contain scene,prompt,"
            "negative_prompt,seed columns. TXT uses one prompt per line."
        ),
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Generate only matching scene name(s) from --prompts-file. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for generated mp4 files.")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1000, help="Base seed for prompts without one.")
    parser.add_argument(
        "--negative-prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Fallback negative prompt when a prompt row does not specify one.",
    )
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Pipeline dtype. Wan keeps its VAE in fp32 for decode quality.",
    )
    parser.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Diffusers model CPU offload to reduce VRAM.",
    )
    parser.add_argument(
        "--vae-tiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable VAE tiling when the pipeline VAE supports it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned generations and write no videos.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prompts = load_prompts(args.prompt, args.prompts_file, args.negative_prompt, args.seed)
    if args.scene:
        wanted = set(args.scene)
        prompts = [spec for spec in prompts if spec.scene in wanted]
    if not prompts:
        raise SystemExit("No prompts provided. Use --prompt or --prompts-file.")

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    model_id = args.model_id or DEFAULT_MODEL_IDS[args.backend]

    print(f"Backend: {args.backend}")
    print(f"Model: {model_id}")
    print(f"Output: {out_dir}")
    print(f"Videos: {len(prompts)}")

    manifest: dict[str, Any] = {
        "backend": args.backend,
        "model_id": model_id,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "entries": [],
    }

    pipe = None
    if not args.dry_run:
        pipe = load_pipeline(
            backend=args.backend,
            model_id=model_id,
            dtype=args.dtype,
            device=args.device,
            cpu_offload=args.cpu_offload,
            vae_tiling=args.vae_tiling,
            cache_dir=args.cache_dir,
        )

    for index, spec in enumerate(prompts):
        seed = args.seed + index if spec.seed is None else spec.seed
        scene = spec.scene or f"scene_{index:03d}"
        out_path = out_dir / f"{index:03d}_{slugify(scene)}_seed{seed}.mp4"
        entry = {
            "index": index,
            "scene": scene,
            "prompt": spec.prompt,
            "negative_prompt": spec.negative_prompt or args.negative_prompt,
            "seed": seed,
            "video": str(out_path),
        }
        manifest["entries"].append(entry)

        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path}")
            continue

        print(f"[generate] {scene} seed={seed}")
        if args.dry_run:
            print(f"  prompt: {spec.prompt}")
            continue

        generate_one(pipe, args, spec, seed, out_path)

    if not args.dry_run:
        manifest_path = out_dir / "generation_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Manifest: {manifest_path}")


def load_prompts(
    inline_prompts: list[str],
    prompts_file: str | None,
    default_negative_prompt: str,
    base_seed: int,
) -> list[PromptSpec]:
    specs: list[PromptSpec] = []

    for i, prompt in enumerate(inline_prompts):
        specs.append(
            PromptSpec(
                scene=f"inline_{i:03d}",
                prompt=prompt.strip(),
                negative_prompt=default_negative_prompt,
                seed=base_seed + i,
            )
        )

    if not prompts_file:
        return specs

    path = Path(prompts_file).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Prompt file not found: {path}")

    if path.suffix.lower() in {".csv", ".tsv"}:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if not reader.fieldnames or "prompt" not in reader.fieldnames:
                raise SystemExit("CSV/TSV prompt files must contain a 'prompt' column.")
            for row_index, row in enumerate(reader):
                prompt = (row.get("prompt") or "").strip()
                if not prompt:
                    continue
                seed_text = (row.get("seed") or "").strip()
                specs.append(
                    PromptSpec(
                        scene=(row.get("scene") or f"prompt_{row_index:03d}").strip(),
                        prompt=prompt,
                        negative_prompt=(row.get("negative_prompt") or default_negative_prompt).strip(),
                        seed=int(seed_text) if seed_text else base_seed + len(specs),
                    )
                )
        return specs

    for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        prompt = line.strip()
        if not prompt or prompt.startswith("#"):
            continue
        specs.append(
            PromptSpec(
                scene=f"prompt_{line_index:03d}",
                prompt=prompt,
                negative_prompt=default_negative_prompt,
                seed=base_seed + len(specs),
            )
        )
    return specs


def load_pipeline(
    *,
    backend: str,
    model_id: str,
    dtype: str,
    device: str,
    cpu_offload: bool,
    vae_tiling: bool,
    cache_dir: str | None,
) -> Any:
    try:
        import torch
        from diffusers import AutoModel
    except ImportError as exc:
        raise SystemExit(
            "Install Diffusers first, for example: "
            "pip install git+https://github.com/huggingface/diffusers transformers accelerate"
        ) from exc

    torch_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype]
    kwargs = {"cache_dir": cache_dir} if cache_dir else {}

    if backend == "wan":
        from diffusers import WanPipeline

        vae = AutoModel.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float32,
            **kwargs,
        )
        pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch_dtype, **kwargs)
    elif backend == "ltx":
        from diffusers import LTXPipeline

        pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
    elif backend == "cogvideox":
        from diffusers import CogVideoXPipeline

        pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    if vae_tiling and getattr(pipe, "vae", None) is not None and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()
    return pipe


def generate_one(pipe: Any, args: argparse.Namespace, spec: PromptSpec, seed: int, out_path: Path) -> None:
    import torch
    from diffusers.utils import export_to_video

    generator = torch.Generator(args.device).manual_seed(seed) if args.device.startswith("cuda") else None
    call_args: dict[str, Any] = {
        "prompt": spec.prompt,
        "negative_prompt": spec.negative_prompt or args.negative_prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
    }
    if generator is not None:
        call_args["generator"] = generator

    if args.backend == "ltx":
        call_args.setdefault("decode_timestep", 0.03)
        call_args.setdefault("decode_noise_scale", 0.025)

    frames = pipe(**call_args).frames[0]
    export_to_video(frames, str(out_path), fps=args.fps)


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip().lower()).strip("_")
    return slug[:80] or "scene"


if __name__ == "__main__":
    main()
