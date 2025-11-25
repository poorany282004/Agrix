"""
train.py — Train a CNN (MobileNetV2) model for Agrix Rice Disease Detection
Author: Agrix Project Team
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

print("🔍 TensorFlow version:", tf.__version__)

# -----------------------------
# ⚙️ Configuration
# -----------------------------
DATA_DIR = 'dataset'             
IMG_SIZE = (224, 224)            
BATCH_SIZE = 16                  
EPOCHS = 10                      
MODEL_SAVE_DIR = 'saved_model'   
MODEL_FILENAME = 'agrix_model.keras'  

# -----------------------------
# 📂 Verify dataset exists
# -----------------------------
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError("❌ dataset/ folder not found!")

train_dir = os.path.join(DATA_DIR, 'train')
val_dir = os.path.join(DATA_DIR, 'val')

# -----------------------------
# 🧪 Data Generators
# -----------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_gen = ImageDataGenerator(rescale=1./255)

train_ds = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_ds = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

num_classes = len(train_ds.class_indices)
print(f"✅ Detected {num_classes} classes:", list(train_ds.class_indices.keys()))

# -----------------------------
# 🧠 Build Model using MobileNetV2
# -----------------------------
base_model = MobileNetV2(
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    weights='imagenet'
)
base_model.trainable = False  

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n🚀 Starting training...\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# -----------------------------
# 🔧 Fine-Tuning Stage
# -----------------------------
print("\n🔧 Unfreezing last MobileNetV2 layers...\n")

base_model.trainable = True  
for layer in base_model.layers[:-50]:  # train last 50 layers
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

# -----------------------------
# 💾 Save Model
# -----------------------------
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
save_path = os.path.join(MODEL_SAVE_DIR, MODEL_FILENAME)
model.save(save_path)

print(f"✅ Model saved: {save_path}")

# -----------------------------
# 🧾 Save labels
# -----------------------------
labels_path = os.path.join(MODEL_SAVE_DIR, "labels.txt")

with open(labels_path, "w") as f:
    for label in train_ds.class_indices.keys():
        f.write(label + "\n")

print(f"✅ Labels saved: {labels_path}")
