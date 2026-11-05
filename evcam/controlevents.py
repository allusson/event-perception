"""ControlEvents pipelines (Hu et al., WACV 2026) and 2D skeleton helpers (notebooks 09, 10).

Needs the `diffusion` extra: pip install -e ".[diffusion]". Weights are downloaded from Hugging
Face into weights/controlevents/ on first use.
"""

import gc
import os
import zipfile

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # must be set before importing torch

import cv2
import numpy as np
import torch
from PIL import Image

SIZE = 512
PROMPT_HUMAN = "A moving human"
WEIGHTS = "weights/controlevents"
DATA = "data/controlevents"
SD14 = "CompVis/stable-diffusion-v1-4"
SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SKELETON_UNET = "EventDiffusion/combined_unet_all"
SKELETON_CONTROLNET = "EventDiffusion/combined_controlnet_all"
# VAE, text encoder, tokenizer and scheduler come from the SD base; fp16 files (upcast on MPS / CPU)
BASE_FILES = ["model_index.json", "scheduler/*", "tokenizer/*", "text_encoder/config.json",
              "text_encoder/model.fp16.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors"]

# 13 joints: 0 head, 1/2 shoulders, 3/4 elbows, 5/6 hips, 7/8 hands, 9/10 knees, 11/12 feet;
# odd joints and joint 5 on one side, even joints and joint 6 on the other
LIMBS = [(0, 1), (0, 2), (1, 2), (1, 3), (3, 7), (2, 4), (4, 8), (1, 5), (2, 6), (5, 6), (5, 9), (9, 11), (6, 10), (10, 12)]
ARM_LIMBS = [(1, 3), (3, 7), (2, 4), (4, 8)]          # shoulder-elbow-hand
LEG_LIMBS = [(5, 9), (9, 11), (6, 10), (10, 12)]      # hip-knee-foot
# OpenPose body keypoints for the 13 joints: nose, R/L shoulder, R/L elbow, R/L hip, R/L wrist, R/L knee, R/L ankle
OPENPOSE_TO_JOINT = [0, 2, 5, 3, 6, 8, 11, 4, 7, 9, 12, 10, 13]


def pick_device():
    """(device, dtype): CUDA in fp16, else MPS or CPU in fp32."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def fetch(repo, patterns=None, weights=WEIGHTS):
    """Download a Hugging Face model repo into weights/controlevents/<repo>; skip if already there."""
    from huggingface_hub import snapshot_download
    local = os.path.join(weights, repo)
    done = os.path.join(local, ".complete")
    if not os.path.exists(done):
        snapshot_download(repo, local_dir=local, allow_patterns=patterns)
        open(done, "w").close()
    return local


def fetch_data(data=DATA):
    """Download and unpack the ControlEvents data (skeleton_data/, normalmap_data/) into data/controlevents/."""
    from huggingface_hub import snapshot_download
    if not all(os.path.isdir(f"{data}/{d}") for d in ("skeleton_data", "normalmap_data")):
        zips = snapshot_download("YixuanHu/ControlEvents_data", repo_type="dataset",
                                 local_dir=f"{data}/_zips", allow_patterns=["*.zip"])
        for z in ("skeleton_data.zip", "normalmap_data.zip"):
            zipfile.ZipFile(f"{zips}/{z}").extractall(data)


def setup_pipe(pipe, device):
    """UniPC scheduler, no progress bar, moved to `device`."""
    from diffusers import UniPCMultistepScheduler
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def sd14_pipe(device, dtype):
    """Plain Stable Diffusion 1.4 text-to-image pipeline."""
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        fetch(SD14, BASE_FILES + ["unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors"]),
        variant="fp16", torch_dtype=dtype, safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    return setup_pipe(pipe, device)


def control_pipe(unet_repo, controlnet_repo, device, dtype):
    """SD 1.5 base with a ControlEvents UNet and ControlNet."""
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(fetch(unet_repo), torch_dtype=dtype)
    controlnet = ControlNetModel.from_pretrained(fetch(controlnet_repo), torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        fetch(SD15, BASE_FILES), unet=unet, controlnet=controlnet, variant="fp16", torch_dtype=dtype,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    return setup_pipe(pipe, device)


def generate(pipe, prompt, image=None, seed=0, num_steps=30, guidance=7.5, size=SIZE):
    """One size x size sample. The initial noise comes from a CPU generator seeded with `seed`,
    so it is the same on every device, and the same for every call with the same seed."""
    kw = {} if image is None else {"image": image}
    gen = torch.Generator("cpu").manual_seed(seed)
    return pipe(prompt, num_inference_steps=num_steps, guidance_scale=guidance, height=size, width=size,
                generator=gen, **kw).images[0]


def free(device):
    """Release cached memory after the caller dropped its references."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def openpose_detector():
    """OpenPose body detector (lllyasviel/Annotators)."""
    from controlnet_aux import OpenposeDetector
    return OpenposeDetector.from_pretrained(
        fetch("lllyasviel/Annotators", ["body_pose_model.pth", "hand_pose_model.pth", "facenet.pth"]))


def detect_joints(detector, rgb):
    """13 joints in [0, 1] image coordinates (NaN = missing) of the highest-scoring person, or None.

    rgb : (H, W, 3) uint8 image
    """
    poses = detector.detect_poses(np.asarray(rgb))
    if not poses:
        return None
    kp = max(poses, key=lambda r: r.body.total_score).body.keypoints
    return np.array([[kp[i].x, kp[i].y] if kp[i] is not None else [np.nan, np.nan] for i in OPENPOSE_TO_JOINT])


def draw_skeleton(xy, size=SIZE, image=None, color=(255, 0, 0)):
    """Red skeleton from 13 joints in [0, 1] image coordinates (NaN = missing joint).

    On white by default (the ControlEvents skeleton condition), or drawn over `image`
    ((size, size, 3) uint8) for display.
    """
    im = np.full((size, size, 3), 255, np.uint8) if image is None else np.array(image, np.uint8)
    p = lambda j: tuple(int(v) for v in xy[j] * size)
    ok = ~np.isnan(xy).any(1)
    for a, b in LIMBS:
        if ok[a] and ok[b]:
            cv2.line(im, p(a), p(b), color, 3, cv2.LINE_AA)
    for j in np.flatnonzero(ok):
        cv2.circle(im, p(j), 5, color, -1, cv2.LINE_AA)
    return Image.fromarray(im)


def canvas_box(wh, crop=None):
    """Square box (x0, y0, x1, y1) in image pixels that is mapped to the square generation canvas.

    crop None: the centred square of side min(W, H). Otherwise the square of side max(w, h)
    centred on `crop` (x0, y0, x1, y1); it may extend past the image, which is then padded.
    """
    W, H = wh
    if crop is None:
        s = min(W, H)
        x0, y0 = (W - s) // 2, (H - s) // 2
        return (x0, y0, x0 + s, y0 + s)
    x0, y0, x1, y1 = crop
    s = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    return (cx - s // 2, cy - s // 2, cx - s // 2 + s, cy - s // 2 + s)


def to_canvas(img, box, size=SIZE):
    """Cut `box` out of `img` ((H, W) or (H, W, C); zero padding outside) and resize to size x size."""
    x0, y0, x1, y1 = box
    H, W = img.shape[:2]
    out = np.zeros((y1 - y0, x1 - x0) + img.shape[2:], img.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_AREA if out.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(out, (size, size), interpolation=interp)
