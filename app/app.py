import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import sys
import os

# ------------------------------------------------
# ARIZE TRACING SETUP
# ------------------------------------------------
from arize.otel import register

ARIZE_API_KEY = os.getenv("ARIZE_API_KEY", "ak-24da06a2-f5e0-4814-a1ff-e991ec311bae-Xl0TwmqTxSnljZeMbfJUeQM3hKFEZJ6V")

tracer_provider = register(
    space_id="U3BhY2U6NDIxMjA6YmNOcw==",
    api_key=ARIZE_API_KEY,
    project_name="melanoma-scan-ai",
)

# ------------------------------------------------

MODEL_PATH = "../model/melanoma_model.keras"
IMG_SIZE = 224

# ------------------------------------------------
# GRADCAM FUNCTIONS (BUILT-IN)
# ------------------------------------------------

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate GradCAM heatmap"""
    
    # Create a model that maps the input image to the activations of the last conv layer
    # as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on original image"""
    
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Superimpose the heatmap on original image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    
    return superimposed_img

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

logo = Image.open("logo.png")

st.set_page_config(
    page_title="MelanoScan AI",
    page_icon=logo,
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

body {
background: linear-gradient(135deg,#bdc2f5,#6875e9);
}

.main {
background-color:white;
border-radius:15px;
padding:2rem;
}

h1{
color:#122056;
}

h2,h3{
color:#5b65dc;
}

.stButton>button {
background-color:#5b65dc;
color:white;
border-radius:10px;
border:none;
padding:10px 20px;
}

.low-risk {
background-color:#d1fae5;
color:#065f46;
padding:20px;
border-radius:12px;
font-size:20px;
font-weight:bold;
}

.high-risk {
background-color:#fee2e2;
color:#991b1b;
padding:20px;
border-radius:12px;
font-size:20px;
font-weight:bold;
}

.card {
background:#f8f9ff;
padding:20px;
border-radius:15px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
"Go to",
[
"Home",
"Analyze Image",
"Model Information",
"Dataset",
"Results"
]
)

# ------------------------------------------------
# HOME PAGE
# ------------------------------------------------

if page == "Home":

    col1,col2 = st.columns([1,3])

    with col1:
        st.image(logo,width=140)

    with col2:
        st.title("MelanoScan AI: An AI Skin Lesion Risk Analyzer")
        st.write("Deep Learning Powered Melanoma Detection System")

    st.divider()

    st.markdown("""
<div class="card">

### Project Overview

This artificial intelligence system analyzes dermoscopic images of skin lesions to detect **melanoma risk** using deep learning.

The model was trained using dermatology images from the **ISIC dataset**.

The goal of this project is to explore how artificial intelligence can assist doctors in **early detection of skin cancer**.

</div>
""", unsafe_allow_html=True)

# ------------------------------------------------
# ANALYZE IMAGE PAGE
# ------------------------------------------------

elif page == "Analyze Image":

    st.title("Analyze Skin Lesion")

    uploaded_file = st.file_uploader(
        "Upload a skin lesion image",
        type=["jpg","png","jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image,width=300,caption="Uploaded Image")

        # -------------------------------
        # IMAGE PREPROCESSING
        # -------------------------------

        img_resized = image.resize((IMG_SIZE, IMG_SIZE))

        img_array = np.array(img_resized).astype("float32")

        # Use MobileNetV2's official preprocessing
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        img_array_preprocessed = preprocess_input(img_array.copy())

        img_array_preprocessed = np.expand_dims(img_array_preprocessed, axis=0)

        # -------------------------------
        # MODEL PREDICTION
        # -------------------------------

        prediction = model.predict(img_array_preprocessed)[0][0]

        melanoma_prob = float(prediction)
        benign_prob = 1 - melanoma_prob

        st.write("Raw Model Output:", melanoma_prob)

        # -------------------------------
        # DIAGNOSIS
        # -------------------------------

        st.header("AI Diagnosis")

        # Lower threshold for medical safety
        if melanoma_prob >= 0.3:

            risk = "HIGH RISK (Possible Melanoma)"
            confidence = melanoma_prob

            st.markdown(
            f'<div class="high-risk">⚠️ {risk}<br>Confidence: {confidence*100:.2f}%</div>',
            unsafe_allow_html=True
            )

        else:

            risk = "LOW RISK (Likely Benign)"
            confidence = benign_prob

            st.markdown(
            f'<div class="low-risk">✅ {risk}<br>Confidence: {confidence*100:.2f}%</div>',
            unsafe_allow_html=True
            )

        st.warning("This AI tool is for research purposes only and not a medical diagnosis.")

        # -------------------------------
        # PROBABILITY METER
        # -------------------------------

        st.subheader("Melanoma Probability Meter")

        probability = melanoma_prob * 100

        st.progress(int(min(probability, 100)))

        st.write(f"Predicted Melanoma Probability: {probability:.2f}%")

        st.divider()

        # ------------------------------------------------
        # GRAD-CAM EXPLAINABILITY
        # ------------------------------------------------

        st.header("Explainable AI Visualization")

        try:
            # Find the last convolutional layer in the base model
            last_conv_layer = None
            
            # Look through all layers to find conv layers
            for layer in model.layers:
                # Check if it's a Sequential or Functional model (the base_model)
                if hasattr(layer, 'layers'):
                    for sublayer in reversed(layer.layers):
                        if 'conv' in sublayer.name.lower():
                            last_conv_layer = sublayer.name
                            break
                    if last_conv_layer:
                        break
                elif 'conv' in layer.name.lower():
                    last_conv_layer = layer.name

            if last_conv_layer:
                st.write(f"Using layer: {last_conv_layer}")
                
                # Prepare image for GradCAM (0-1 normalized)
                img_for_gradcam = img_array / 255.0
                img_for_gradcam = np.expand_dims(img_for_gradcam, axis=0)
                img_for_gradcam = preprocess_input(img_for_gradcam * 255.0)

                # Generate heatmap
                heatmap = make_gradcam_heatmap(
                    img_for_gradcam,
                    model,
                    last_conv_layer
                )

                # Overlay on original image
                original_img = np.array(image)
                original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

                heatmap_img = overlay_heatmap(original_img, heatmap)
                heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

                st.image(heatmap_img, width=350, caption="AI Attention Heatmap - Red areas show where the model is focusing")
                
            else:
                st.info("No convolutional layers found for GradCAM visualization.")
                
        except Exception as e:
            st.warning(f"Could not generate GradCAM visualization: {str(e)}")
            st.info("This may happen if the model architecture doesn't support GradCAM.")

# ------------------------------------------------
# MODEL INFO PAGE
# ------------------------------------------------

elif page == "Model Information":

    st.title("AI Model Information")

    st.markdown("""

### Model Architecture

Transfer Learning using **MobileNetV2**

Layers:

• Pretrained convolutional base (fine-tuned)
• Global Average Pooling  
• Dense hidden layer (128 units)
• Dropout layers for regularization
• Dense classification layer

### Training Details

• Optimizer: Adam (learning rate: 0.0001)
• Loss Function: Binary Crossentropy  
• Image Size: 224 × 224  
• Class Weights: Balanced for melanoma detection
• Data Augmentation: Flips, rotations, zoom

The model learns visual features such as:

• irregular borders  
• asymmetry  
• color variation  
• texture patterns

These patterns can help distinguish melanoma from benign lesions.
""")

# ------------------------------------------------
# DATASET PAGE
# ------------------------------------------------

elif page == "Dataset":

    st.title("Dataset")

    st.markdown("""

### ISIC Dermatology Dataset

The model was trained using images from the **International Skin Imaging Collaboration (ISIC)** dataset.

Dataset Characteristics:

• 25,331 dermoscopic images  
• Collected from dermatology clinics worldwide  
• Widely used in medical AI research

Classes used in this project:

• Benign Lesions (~20,809 images)
• Melanoma (~4,522 images)

Images were resized to **224 × 224 pixels** before training.

Class balancing techniques were applied to handle the imbalance.

""")

# ------------------------------------------------
# RESULTS PAGE
# ------------------------------------------------

elif page == "Results":

    st.title("Model Performance")

    col1,col2,col3,col4=st.columns(4)

    col1.metric("Accuracy","~75%")
    col2.metric("Precision","0.72")
    col3.metric("Recall","0.70")
    col4.metric("AUC","0.82")

    st.write("Performance metrics calculated using validation dataset.")
    st.info("Note: Recall (sensitivity) is prioritized for melanoma detection to minimize false negatives.")
    