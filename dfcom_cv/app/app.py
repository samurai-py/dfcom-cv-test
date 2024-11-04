import streamlit as st
import cv2
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from dfcom_cv.predict import ImagePredictor
from tensorflow.keras.models import load_model

classes = ["forest", "street", "glacier", "mountain", "sea", "buildings"]
label_encoder = LabelEncoder()
label_encoder.fit(classes)

def load_model_from_path(model_path):
    return load_model(model_path)

st.title("Image Class Predictor")

model_choice = st.selectbox("Escolha o modelo para predição:", ["CNN", "Transfer Learning"])

if model_choice == "CNN":
    model_path = os.path.join('dfcom_cv', 'models', 'cnn', 'modelo_cnn.h5')
elif model_choice == "Transfer Learning":
    model_path = os.path.join('dfcom_cv', 'models', 'transfer', 'modelo_transfer.h5')

model = load_model_from_path(model_path)
image_predictor = ImagePredictor(model=model, label_encoder=label_encoder, target_size=(100, 100))

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    resized_image = cv2.resize(image, (100, 100)) / 255.0  # Normalizar para [0, 1]

    prediction = image_predictor.predict(resized_image)

    st.image(image, caption="Imagem Carregada", use_column_width=True)
    st.write(f"Predição do Modelo {model_choice}: {prediction}")
