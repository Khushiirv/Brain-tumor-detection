import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np
from PIL import Image

# Load models
cnn_model = load_model(r'C:\coding\final year project\ML_models\brain_tumor_detection_model_pnn.h5')
cnn_bayesian_model = load_model(r'C:\coding\final year project\ML_models\cnn_bayesian_hybrid_model.h5')

# Function to process uploaded image
def preprocess_image(img):
    img = img.resize((256, 256))  # Resize image
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Function to predict the class
def predict_class(model, img_array):
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions, axis=1)[0]
    class_labels = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'}
    return class_labels[class_idx], predictions[0][class_idx]

# Streamlit UI
st.title("Brain Tumor Detection")
st.subheader("Upload an MRI scan and get the tumor type prediction.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI scan image", type=["jpg", "png", "jpeg"])

# Model selection
model_choice = st.radio("Choose the model:", ("Standard CNN", "CNN-Bayesian Hybrid"))

# Submit button to trigger result
if uploaded_file is not None:
    # Open and resize the image for display
    img = Image.open(uploaded_file)
    img_resized = img.resize((400, 400))  # Resize image to 400x400 for display
    st.image(img_resized, caption='Uploaded Image', use_container_width=False)  # Display resized image

    # Add a submit button for prediction
    submit_button = st.button(label="Submit")

    if submit_button:
        # Preprocess image
        img_array = preprocess_image(img)

        # Predict with the selected model
        if model_choice == "Standard CNN":
            result, confidence = predict_class(cnn_model, img_array)
        else:
            result, confidence = predict_class(cnn_bayesian_model, img_array)

        # Display result
        st.write(f"Prediction: {result}")
        st.write(f"Confidence: {confidence*100:.2f}%")

        # Display message for certain predictions
        if result in ['Glioma', 'Meningioma', 'Pituitary']:
            st.warning(f"Consult a doctor. You may have a {result} tumor.")
