import streamlit as st
import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from dfcom_cv.predict import ImagePredictor
from dfcom_cv.train import CNNTrainer, TransferLearningTrainer

# Assuming you have initialized your models and label encoder elsewhere
cnn_trainer = CNNTrainer(input_shape=(100, 100, 3), num_classes=6)  # Adjust input shape and classes as necessary
transfer_learning_trainer = TransferLearningTrainer(input_shape=(100, 100, 3), num_classes=6)  # Same as above
label_encoder = LabelEncoder()  # Initialize your label encoder here if needed

# Create an image predictor instance
image_predictor = ImagePredictor(cnn_model=cnn_trainer.model, transfer_model=transfer_learning_trainer.model)

# Streamlit UI
st.title("Image Class Predictor")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
    # Resize and normalize the image for prediction
    resized_image = cv2.resize(image, (100, 100)) / 255.0  # Adjust size as necessary

    # Predict using the CNN model
    cnn_prediction = image_predictor.predict_image(resized_image, model='cnn')

    # Predict using the Transfer Learning model
    transfer_prediction = image_predictor.predict_image(resized_image, model='transfer_learning')

    # Display predictions
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"CNN Model Prediction: {cnn_prediction}")
    st.write(f"Transfer Learning Model Prediction: {transfer_prediction}")
