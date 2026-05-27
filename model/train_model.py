import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils import class_weight
import numpy as np
import os

IMG_SIZE = 224
BATCH_SIZE = 32

data_dir = "../data/images"

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# ------------------------------------------------
# CALCULATE CLASS WEIGHTS
# ------------------------------------------------

train_labels = []
for images, labels in train_ds:
    train_labels.extend(labels.numpy().flatten())

train_labels = np.array(train_labels).astype(int)

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.array([0, 1]),
    y=train_labels
)

class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"\n🔍 Class distribution:")
print(f"   Benign (0): {np.sum(train_labels == 0)} images")
print(f"   Melanoma (1): {np.sum(train_labels == 1)} images")
print(f"\n⚖️ Class weights: {class_weight_dict}")
print(f"   Melanoma images weighted {class_weight_dict[1]:.2f}x more\n")

# ------------------------------------------------
# DATA AUGMENTATION
# ------------------------------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# ------------------------------------------------
# BUILD MODEL (UNFROZEN!)
# ------------------------------------------------

base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# UNFREEZE the base model so it can learn melanoma features
base_model.trainable = True

# Fine-tune from layer 100 onwards (keep early layers frozen)
for layer in base_model.layers[:100]:
    layer.trainable = False

model = models.Sequential([
    data_augmentation,
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),  # Added hidden layer
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

# Use a lower learning rate for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR
    loss='binary_crossentropy',
    metrics=['accuracy', 
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)

model.summary()

# ------------------------------------------------
# TRAIN WITH CLASS WEIGHTS
# ------------------------------------------------

print("\n🚀 Starting fine-tuning with class balancing...\n")

# Add early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_recall',
    patience=5,
    restore_best_weights=True,
    mode='max'
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=30,  # Increased to 30
    class_weight=class_weight_dict,
    callbacks=[early_stopping],
    verbose=1
)

# ------------------------------------------------
# SAVE MODEL
# ------------------------------------------------

model.save("melanoma_model.keras")

print("\n✅ Fine-tuned model training complete!")
print(f"\n📊 Final metrics:")
print(f"   Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"   Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"   Training Recall: {history.history['recall'][-1]:.4f}")
print(f"   Validation Recall: {history.history['val_recall'][-1]:.4f}")
print(f"   Validation AUC: {history.history['val_auc'][-1]:.4f}")