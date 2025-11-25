# agrix/scripts/train.py
import json
import torch
from pathlib import Path
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "merged"
OUT_MODELS = ROOT / "backend" / "models"
OUT_MODELS.mkdir(parents=True, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
BATCH = 32
EPOCHS = 6
LR = 1e-4

# Data transforms
tf_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

tf_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Datasets
train_ds = datasets.ImageFolder(str(DATA_DIR/"train"), transform=tf_train)
val_ds = datasets.ImageFolder(str(DATA_DIR/"val"), transform=tf_val)
NUM_CLASSES = len(train_ds.classes)
print("Classes:", train_ds.classes)

# Save class map
with open(OUT_MODELS/"class_map.json","w") as f:
    json.dump(train_ds.class_to_idx, f)

# Data loaders
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4)

# Model: EfficientNet-B0
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(device)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_ds)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)
    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")

    # Save best weights
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), OUT_MODELS/"best_weights.pth")
        print("Saved best weights with val_acc:", best_val_acc)
