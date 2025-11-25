# agrix/scripts/export_torchscript.py
import torch
import json
from torchvision import models
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "backend" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load class map
with open(MODEL_DIR / "class_map.json", "r") as f:
    class_map = json.load(f)
NUM_CLASSES = len(class_map)

# Build model architecture same as training
model = models.efficientnet_b0(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, NUM_CLASSES)

# Load trained weights
state = torch.load(MODEL_DIR / "best_weights.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

# Export to TorchScript
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)
traced_model.save(MODEL_DIR / "rice_model_ts.pt")

print("TorchScript model saved to:", MODEL_DIR / "rice_model_ts.pt")
