"""
model_stub.py
-----------------------------------
This module provides a unified interface for prediction.
If a trained TorchScript model exists, it performs real inference.
Otherwise, it falls back to a dummy, rule-based prediction.

Use:
    from .model_stub import infer
    result = infer(image_path, sensors, job_id)
"""

from pathlib import Path
import random, io, json
from PIL import Image

# --- Try loading Torch model if available ---
try:
    import torch
    from torchvision import transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Paths
HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parents[1] / "models"
MODEL_PATH = MODELS_DIR / "rice_model_ts.pt"
CLASS_MAP_PATH = MODELS_DIR / "class_map.json"

_model = None
_idx_to_class = None
_device = None
_transform = None


def _load_model():
    """Load TorchScript model and class map if present"""
    global _model, _idx_to_class, _device, _transform

    if not TORCH_AVAILABLE:
        print("Torch not available — using dummy mode.")
        return False

    if not MODEL_PATH.exists() or not CLASS_MAP_PATH.exists():
        print("No TorchScript model found — using dummy mode.")
        return False

    try:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = torch.jit.load(str(MODEL_PATH), map_location=_device)
        _model.eval()

        with open(CLASS_MAP_PATH, "r") as f:
            class_to_idx = json.load(f)
        _idx_to_class = {int(v): k for k, v in class_to_idx.items()}

        _transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        print("✅ Torch model loaded successfully on", _device)
        return True

    except Exception as e:
        print("⚠️ Failed to load model:", e)
        return False


# Load model at import
_load_model()


# --- Utility functions ---
def _severity_from_conf(conf):
    if conf >= 0.9:
        return "high"
    elif conf >= 0.75:
        return "moderate"
    return "low"


def _advisory_text(disease, severity):
    texts = {
        "healthy": "Crop is healthy. Maintain regular irrigation and fertilizer schedules.",
        "blast": "Rice Blast detected. Use tricyclazole fungicide and improve field drainage.",
        "brown_spot": "Brown Spot detected. Apply balanced NPK fertilizer and remove infected leaves.",
        "bacterial_blight": "Bacterial Blight detected. Remove affected plants and use resistant varieties.",
        "other": "Uncertain disease. Seek expert agronomy advice."
    }
    base = texts.get(disease, texts["other"])
    suffix = {
        "high": " (High severity — immediate action needed.)",
        "moderate": " (Moderate severity — monitor and treat.)",
        "low": " (Low severity — observe further before action.)"
    }[severity]
    return base + suffix


# --- Main inference function ---
def infer(image_path: str, sensors: dict = None, job_id: str = None) -> dict:
    """
    Perform inference using real model (if available) or dummy fallback.
    """
    # ------------------ REAL MODEL ------------------
    if _model is not None and TORCH_AVAILABLE:
        try:
            img = Image.open(image_path).convert("RGB")
            x = _transform(img).unsqueeze(0).to(_device)

            with torch.no_grad():
                logits = _model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            idx = int(probs.argmax())
            disease = _idx_to_class.get(idx, "other")
            conf = float(probs[idx])
            severity = _severity_from_conf(conf)
            advisory = _advisory_text(disease, severity)

            return {
                "disease": disease,
                "confidence": round(conf, 3),
                "severity": severity,
                "advisory": advisory,
                "saliency_url": None  # placeholder for Grad-CAM
            }
        except Exception as e:
            print("⚠️ Model inference failed:", e)

    # ------------------ DUMMY FALLBACK ------------------
    rh = sensors.get("rh24") if sensors else None
    sm = sensors.get("soil_moisture") if sensors else None

    if rh and rh > 85:
        disease = "blast"
        conf = 0.86
    elif sm and sm < 20:
        disease = "brown_spot"
        conf = 0.82
    else:
        disease = random.choice(["healthy", "blast", "brown_spot", "bacterial_blight"])
        conf = random.uniform(0.7, 0.95)

    severity = _severity_from_conf(conf)
    advisory = _advisory_text(disease, severity)

    return {
        "disease": disease,
        "confidence": round(conf, 3),
        "severity": severity,
        "advisory": advisory,
        "saliency_url": None
    }


if __name__ == "__main__":
    # Quick test
    test_img = "sample_leaf.jpg"  # replace with an image path for testing
    if Path(test_img).exists():
        print(infer(test_img, {"rh24": 90, "soil_moisture": 30}))
    else:
        print("No sample image found for test.")
