import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib

# -----------------------------
# DATASET LOCATION
# -----------------------------

data_dir = pathlib.Path("../data")

IMG_SIZE = (224,224)
BATCH_SIZE = 32

# -----------------------------
# LOAD DATASET
# -----------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("Classes:", class_names)

# -----------------------------
# PERFORMANCE OPTIMIZATION
# -----------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# DATA AUGMENTATION
# -----------------------------

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# -----------------------------
# LOAD PRETRAINED MODEL
# -----------------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

# -----------------------------
# BUILD MODEL
# -----------------------------

inputs = keras.Input(shape=(224,224,3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

# -----------------------------
# COMPILE MODEL
# -----------------------------

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# CLASS WEIGHTS (IMPORTANT)
# -----------------------------

class_weight = {
    0:1,
    1:6
}

# -----------------------------
# TRAIN MODEL
# -----------------------------

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weight
)

# -----------------------------
# SAVE MODEL
# -----------------------------

model.save("melanoma_model.keras")

print("Training complete. Model saved.")