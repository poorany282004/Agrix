from PIL import Image
import numpy as np
import os

# ✅ Updated class list (9 diseases + healthy)
CLASS_NAMES = [
    'bacterialblight',
    'blast',
    'brownspot',
    'healthy',
    'leaf_blast',
    'leaf_scald',
    'narrow_brown_spot',
    'sheath_blight',
    'tungro'
]

def load_image_array(path, size=(224,224)):
    img = Image.open(path).convert('RGB').resize(size)
    arr = np.asarray(img) / 255.0
    return arr

def fallback_predict(path):
    """
    Heuristic:
    - Compute mean green channel value vs red; high green -> healthy
    - Otherwise split among disease names with some randomness
    """
    arr = load_image_array(path)
    r = arr[:,:,0]
    g = arr[:,:,1]
    b = arr[:,:,2]
    # heuristic: healthy_score = normalized (g - r)
    diff = (g - r).mean()
    healthy_score = (diff + 0.2) / 0.6  # scale roughly to 0..1
    healthy_score = max(0.0, min(1.0, healthy_score))
    # make a distribution
    other_prob = max(0.0, 1.0 - healthy_score)
    # simple deterministic split for reproducibility
    probs = [healthy_score] + [other_prob * w for w in [0.4, 0.2, 0.2, 0.2]]
    # normalize just in case
    total = sum(probs)
    probs = [p/total for p in probs]
    idx = int(np.argmax(probs))
    return {'label': CLASS_NAMES[idx], 'prob': float(round(probs[idx], 4)), 'probs': [float(round(p,4)) for p in probs]}

def predict_image(path):
    # For this simple version, always use fallback
    return fallback_predict(path)
