# agrix/backend/app/inference.py
import torch, json, io
from torchvision import transforms
from PIL import Image
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "rice_model_ts.pt"
CLASS_MAP_PATH = MODEL_DIR / "class_map.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading model on", device)
model = torch.jit.load(str(MODEL_PATH), map_location=device)
model.eval()

with open(CLASS_MAP_PATH, "r") as f:
    class_to_idx = json.load(f)
idx_to_class = {int(v):k for k,v in class_to_idx.items()}

tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def infer_bytes(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return {"disease": idx_to_class[idx], "confidence": float(probs[idx]), "probs": probs.tolist()}
