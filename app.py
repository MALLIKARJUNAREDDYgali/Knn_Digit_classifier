import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import joblib

# Load the trained KNN model
model = joblib.load("knn_mnist_small.pkl")

# Set page settings
st.set_page_config(page_title="KNN Digit Classifier", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 Digit Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Draw a digit (0–9) below and click <b>Predict</b></p>", unsafe_allow_html=True)

# Custom CSS styling for buttons and selection
st.markdown("""
    <style>
        div.stButton > button:first-child {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.5em 1.2em;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            background-color: #388E3C;
            color: white;
        }
        .st-cp {
            background-color: #4CAF50 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Create a canvas component
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="black",
    background_color="white",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# When canvas has image data
if canvas_result.image_data is not None:
    img = canvas_result.image_data

    # Convert to grayscale PIL image
    pil_img = Image.fromarray((img[:, :, 0:3]).astype("uint8"))
    pil_img = pil_img.convert("L")  # Convert to grayscale
    pil_img = ImageOps.invert(pil_img)  # Invert to get white digit on black

    # Resize to 28x28 and normalize
    pil_img = pil_img.resize((28, 28), Image.LANCZOS)
    img_array = np.array(pil_img).astype("float32") / 255.0
    img_array = img_array.reshape(1, -1)

    # Predict when button is clicked
    if st.button("Predict"):
        prediction = model.predict(img_array)[0]
        confidence = model.predict_proba(img_array).max()
        
        st.success(f"🎯 Predicted Digit: {prediction}")
        st.info(f"📊 Confidence: {confidence:.2f}")
