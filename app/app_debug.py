import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import cv2

sys.path.append("../model")

from gradcam import make_gradcam_heatmap, overlay_heatmap

MODEL_PATH = "../model/melanoma_model.keras"
IMG_SIZE = 224

# ------------------------------------------------
# PAGE CONFIG WITH LOGO
# ------------------------------------------------

logo = Image.open("logo.png")

st.set_page_config(
    page_title="MelanoScan AI",
    page_icon=logo,
    layout="wide"
)

# ------------------------------------------------
# CUSTOM COLORS
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
# SIDEBAR
# ------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
"Go to",
[
"Home",
"Analyze Image"
]
)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ------------------------------------------------
# HOME PAGE
# ------------------------------------------------

if page == "Home":

    col1,col2 = st.columns([1,3])

    with col1:
        st.image(logo,width=120)

    with col2:
        st.title("MelanoScan AI")
        st.write("Deep Learning Powered Melanoma Detection")

    st.divider()

    st.markdown("""
<div class="card">

### Project Overview

This AI system analyzes dermoscopic skin images to detect melanoma risk using deep learning.

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

        img_resized = image.resize((IMG_SIZE,IMG_SIZE))
        img_array = np.array(img_resized)/255.0
        img_array = np.expand_dims(img_array,axis=0)

        prediction = model.predict(img_array)[0][0]

        # ------------------------------------------
        # DEBUG OUTPUT
        # ------------------------------------------

        st.subheader("Debug Information")

        st.write("Raw Model Output (0 = benign, 1 = melanoma):", prediction)

        st.write("Melanoma probability:", float(prediction)*100, "%")

        st.write("Benign probability:", float(1-prediction)*100, "%")

        # ------------------------------------------
        # NORMAL PREDICTION
        # ------------------------------------------

        st.header("AI Diagnosis")

        if prediction > 0.5:

            risk="HIGH RISK (Possible Melanoma)"
            confidence=prediction

            st.markdown(
            f'<div class="high-risk">⚠️ {risk}<br>Confidence: {confidence*100:.2f}%</div>',
            unsafe_allow_html=True)

        else:

            risk="LOW RISK (Likely Benign)"
            confidence=1-prediction

            st.markdown(
            f'<div class="low-risk">✅ {risk}<br>Confidence: {confidence*100:.2f}%</div>',
            unsafe_allow_html=True)

        st.warning("This tool is for research purposes only.")

        st.divider()

        # ------------------------------------------------
        # GRADCAM
        # ------------------------------------------------

        st.header("Explainable AI Visualization")

        base_model=model.get_layer("mobilenetv2_1.00_224")
        last_conv_layer_name="Conv_1"

        heatmap=make_gradcam_heatmap(
        img_array,
        base_model,
        last_conv_layer_name
        )

        original_img=np.array(image)
        original_img=cv2.cvtColor(original_img,cv2.COLOR_RGB2BGR)

        heatmap_img=overlay_heatmap(original_img,heatmap)
        heatmap_img=cv2.cvtColor(heatmap_img,cv2.COLOR_BGR2RGB)

        st.image(heatmap_img,width=350,caption="AI Attention Map")