import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import numpy as np
import joblib

# Load the trained KNN model
model = joblib.load("knn_mnist.pkl")

st.set_page_config(page_title="KNN Digit Classifier", layout="centered")
st.title("🧠 Digit Classifier")
st.markdown("Draw a digit (0–9) below and click **Predict**")

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
        
        st.success(f"Predicted Digit: {prediction}")
        st.info(f"Confidence: {confidence:.2f}")
