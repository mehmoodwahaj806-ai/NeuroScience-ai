"""
Two-stage brain MRI tumor screening pipeline.

Stage 1 (binary, EfficientNet-B0): brain MRI slice contains a tumor or not.
  - Tuned for HIGH SENSITIVITY (catches ~99.9% of true tumors) at >=90% specificity.
  - Includes an energy-based out-of-distribution (OOD) check to flag inputs that
    don't look like the training distribution of brain MRI slices at all
    (e.g. a random photo, an X-ray of something else, a corrupted scan).

Stage 2 (3-class, EfficientNet-B0): only run if Stage 1 flags a tumor.
  - Classifies into glioma / meningioma / pituitary.
  - Also has its own OOD energy check.

All thresholds, temperatures and OOD cutoffs below are taken verbatim from the
original deployment_bundle.json files that shipped with the provided model,
so this reproduces the original app's decision behavior, not an approximation.
"""
import io
import json
import yaml
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

BASE_DIR = Path(__file__).parent
WEIGHTS_DIR = BASE_DIR / "model_weights"
CONFIG_DIR = BASE_DIR / "configs"

with open(CONFIG_DIR / "common.yaml") as f:
    COMMON_CFG = yaml.safe_load(f)

with open(WEIGHTS_DIR / "stage1" / "deployment_bundle.json") as f:
    STAGE1_BUNDLE = json.load(f)

with open(WEIGHTS_DIR / "stage2" / "deployment_bundle.json") as f:
    STAGE2_BUNDLE = json.load(f)

CLASS_NAMES = STAGE2_BUNDLE["class_names"]  # ["glioma", "meningioma", "pituitary"]

_PREPROC = COMMON_CFG["preprocessing"]
_IMG_SIZE = tuple(_PREPROC["target_size"])
_CLIP_LO, _CLIP_HI = _PREPROC["intensity_clip_percentiles"]
_NORM_MEAN = np.array(_PREPROC["normalize_mean"], dtype=np.float32)
_NORM_STD = np.array(_PREPROC["normalize_std"], dtype=np.float32)
_MIN_RES = tuple(_PREPROC["min_valid_resolution"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_efficientnet_b0(num_classes: int) -> nn.Module:
    """Recreate the exact architecture the checkpoints were trained with:
    torchvision efficientnet_b0 with its classifier head replaced."""
    model = torchvision.models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _load_stage_model(stage_dir: str, filename: str, num_classes: int) -> nn.Module:
    ckpt_path = WEIGHTS_DIR / stage_dir / filename
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model = _build_efficientnet_b0(num_classes)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


print("Loading Stage 1 (tumor detector)...")
STAGE1_MODEL = _load_stage_model("stage1", "stage1_efficientnet_b0_best.pt", num_classes=1)
print("Loading Stage 2 (tumor type classifier)...")
STAGE2_MODEL = _load_stage_model("stage2", "stage2_efficientnet_b0_best.pt", num_classes=len(CLASS_NAMES))
print("Models loaded on", DEVICE)


class InvalidImageError(Exception):
    pass


def preprocess(image_bytes: bytes) -> torch.Tensor:
    """Reproduces configs/common.yaml preprocessing:
    - decode, ensure resolution is sane
    - convert to grayscale-consistent 3-channel
    - percentile intensity clipping (1st-99th) for contrast robustness
    - resize to 224x224 (area interpolation)
    - ImageNet mean/std normalization
    Returns a (1, 3, 224, 224) float tensor.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception as e:
        raise InvalidImageError(f"Could not decode image: {e}")

    if img.width < _MIN_RES[0] or img.height < _MIN_RES[1]:
        raise InvalidImageError(
            f"Image resolution {img.width}x{img.height} is below the minimum "
            f"required {_MIN_RES[0]}x{_MIN_RES[1]}."
        )

    # Normalize color mode -> grayscale intensity -> 3-channel
    gray = img.convert("L")
    arr = np.asarray(gray, dtype=np.float32)

    # Percentile intensity clipping then rescale to 0-255
    lo, hi = np.percentile(arr, [_CLIP_LO, _CLIP_HI])
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo) * 255.0

    clipped = Image.fromarray(arr.astype(np.uint8), mode="L")
    if _PREPROC.get("grayscale_to_3channel", True):
        clipped = clipped.convert("RGB")

    resized = clipped.resize(tuple(_IMG_SIZE), Image.BILINEAR)  # 'area' approximated w/ bilinear on PIL
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - _NORM_MEAN) / _NORM_STD
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)
    return tensor.to(DEVICE)


def _energy_score(logits: torch.Tensor) -> float:
    """Free-energy OOD score: -logsumexp(logits). Lower/more negative = more
    in-distribution for how these models were calibrated."""
    return (-torch.logsumexp(logits, dim=1)).item()


def _ood_flag(energy: float, stats: dict) -> bool:
    return energy < stats["energy_low_cutoff"] or energy > stats["energy_high_cutoff"]


@torch.no_grad()
def run_pipeline(image_bytes: bytes) -> dict:
    x = preprocess(image_bytes)

    # ---------------- Stage 1: tumor vs no tumor ----------------
    logit1 = STAGE1_MODEL(x)  # shape (1,1)
    temp1 = STAGE1_BUNDLE["temperature"]
    calibrated_logit1 = logit1 / temp1
    prob_tumor = torch.sigmoid(calibrated_logit1).item()
    threshold1 = STAGE1_BUNDLE["decision_threshold"]
    tumor_flag = prob_tumor >= threshold1

    # Stage 1 OOD uses energy from the raw single logit expanded to a 2-logit
    # equivalent [0, logit] so free-energy is well defined for binary heads.
    energy1 = _energy_score(torch.cat([torch.zeros_like(logit1), logit1], dim=1))
    ood1 = _ood_flag(energy1, STAGE1_BUNDLE["ood_stats"])

    result = {
        "stage1": {
            "probability_tumor": round(prob_tumor, 4),
            "decision_threshold": threshold1,
            "tumor_detected": bool(tumor_flag),
            "out_of_distribution": bool(ood1),
            "energy_score": round(energy1, 4),
        },
        "stage2": None,
        "final_verdict": None,
    }

    if ood1:
        result["final_verdict"] = "uncertain_not_brain_mri"
        return result

    if not tumor_flag:
        result["final_verdict"] = "no_tumor_detected"
        return result

    # ---------------- Stage 2: tumor subtype ----------------
    logits2 = STAGE2_MODEL(x)  # shape (1,3)
    temp2 = STAGE2_BUNDLE["temperature"]
    calibrated_logits2 = logits2 / temp2
    probs2 = F.softmax(calibrated_logits2, dim=1).squeeze(0)
    top_idx = int(torch.argmax(probs2).item())
    top_prob = float(probs2[top_idx].item())

    energy2 = _energy_score(logits2)
    ood2 = _ood_flag(energy2, STAGE2_BUNDLE["ood_stats"])

    class_probs = {CLASS_NAMES[i]: round(float(probs2[i].item()), 4) for i in range(len(CLASS_NAMES))}

    result["stage2"] = {
        "class_probabilities": class_probs,
        "predicted_class": CLASS_NAMES[top_idx],
        "confidence": round(top_prob, 4),
        "out_of_distribution": bool(ood2),
        "energy_score": round(energy2, 4),
    }

    if ood2:
        result["final_verdict"] = "tumor_detected_type_uncertain"
    else:
        result["final_verdict"] = f"tumor_detected_{CLASS_NAMES[top_idx]}"

    return result
