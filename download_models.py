"""Parallel model download for InfiniteTalk runtime image.

Run inside the models image build. VARIANT env var selects the InfiniteTalk
weight to bundle (single|multi) so we don't ship both ~14GB files when only
one is used at runtime.
"""

import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

OUT = Path(os.environ.get("MODELS_DIR", "/models"))
VARIANT = os.environ.get("VARIANT", "single").lower()
if VARIANT not in ("single", "multi"):
    sys.exit(f"VARIANT must be 'single' or 'multi', got {VARIANT!r}")

# (repo_id, filename_in_repo, target_subdir, target_filename)
SHARED = [
    ("Kijai/WanVideo_comfy",
     "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
     "diffusion_models", "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"),
    ("Kijai/WanVideo_comfy",
     "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
     "loras", "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"),
    ("Kijai/WanVideo_comfy",
     "Wan2_1_VAE_bf16.safetensors",
     "vae", "Wan2_1_VAE_bf16.safetensors"),
    ("Kijai/WanVideo_comfy",
     "umt5-xxl-enc-fp8_e4m3fn.safetensors",
     "text_encoders", "umt5-xxl-enc-fp8_e4m3fn.safetensors"),
    ("Comfy-Org/Wan_2.1_ComfyUI_repackaged",
     "split_files/clip_vision/clip_vision_h.safetensors",
     "clip_vision", "clip_vision_h.safetensors"),
    ("Kijai/MelBandRoFormer_comfy",
     "MelBandRoformer_fp16.safetensors",
     "diffusion_models", "MelBandRoformer_fp16.safetensors"),
]

VARIANT_FILE = {
    "single": ("Kijai/WanVideo_comfy_fp8_scaled",
               "InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors",
               "diffusion_models",
               "Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors"),
    "multi":  ("Kijai/WanVideo_comfy_fp8_scaled",
               "InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors",
               "diffusion_models",
               "Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors"),
}


def fetch(repo_id: str, src: str, subdir: str, dst_name: str) -> str:
    target_dir = OUT / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / dst_name
    if target.exists():
        return f"skip {target}"
    cached = hf_hub_download(repo_id=repo_id, filename=src)
    # `cached` is a symlink into the HF cache. Resolve and copy the real blob
    # so the model survives after the cache layer is discarded.
    real = os.path.realpath(cached)
    shutil.copyfile(real, target)
    return f"ok   {target} ({os.path.getsize(target)} bytes)"


def main() -> None:
    jobs = SHARED + [VARIANT_FILE[VARIANT]]
    print(f"Downloading {len(jobs)} files (VARIANT={VARIANT}) -> {OUT}")
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, *j): j for j in jobs}
        for fut in as_completed(futures):
            j = futures[fut]
            try:
                print(fut.result(), flush=True)
            except Exception as e:
                failures.append((j, e))
                print(f"FAIL {j[0]}/{j[1]}: {e}", flush=True)
    if failures:
        sys.exit(f"{len(failures)} download(s) failed")


if __name__ == "__main__":
    main()
