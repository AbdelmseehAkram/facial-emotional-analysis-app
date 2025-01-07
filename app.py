import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import gdown

# Download the model if it does not exist
if not os.path.exists("model.keras"):
    url = "https://drive.google.com/uc?id=1CszW4Xb8tvdv4DJM59Ob8mbitc4q1FG_"  # Direct download link
    output = "model.keras"
    gdown.download(url, output, quiet=False)

# Load the model
model = load_model("model.keras")

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Configure the Streamlit app
st.set_page_config(page_title="Emotion Classification App", layout="centered")
st.title("🎭 Emotion Classification App")
st.write("📷 **Upload an image to classify the emotion.**")

# Sidebar for additional settings
st.sidebar.title("⚙️ Settings")
bg_color = st.sidebar.color_picker("Choose Background Color", "#000000")  # Default color is black (#000000)

# Dynamically change the background color using CSS
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: {bg_color};
            color: white;  /* Text color to ensure visibility on black background */
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Upload an image
uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.subheader("📤 Uploaded Image")
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Process the image for prediction
    with st.spinner("🔄 Processing Image..."):
        img = img.convert("L")  # Convert image to grayscale
        img = img.resize((48, 48))  # Resize to 48x48
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict emotion
        prediction = model.predict(img_array)
        predicted_emotion = emotion_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100  # Confidence level

    # Display the prediction result
    st.subheader("🎯 Prediction")
    st.write(f"**Predicted Emotion:** `{predicted_emotion}`")
    st.write(f"**Confidence Level:** `{confidence:.2f}%`")

    # Display prediction probabilities as a bar chart
    st.subheader("📊 Prediction Probabilities")
    st.bar_chart(prediction[0])

# Display an info message if no image is uploaded
else:
    st.info("👆 Please upload an image to proceed.")

# Footer for additional information
st.markdown("---")
st.markdown(
    "Developed with ❤️ by [Abdelmseeh](https://www.linkedin.com/in/abdelmseeh-akram-347616262/)"
)
