import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MODEL_PATH = "melanoma_model.h5"

IMG_SIZE = 224

print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

image_path = input("Enter path to skin lesion image: ")

img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
img = img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)[0][0]

if prediction > 0.5:
    risk = "HIGH RISK (Possible Melanoma)"
else:
    risk = "LOW RISK (Likely Benign)"

confidence = prediction if prediction > 0.5 else 1 - prediction

print("\n--- AI ANALYSIS RESULT ---")
print("Risk Level:", risk)
print("Confidence:", round(confidence * 100, 2), "%")

print("\nDISCLAIMER:")
print("This AI tool is for research purposes only and not a medical diagnosis.")