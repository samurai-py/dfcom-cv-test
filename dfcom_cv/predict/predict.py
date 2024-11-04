import os
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder

class ImagePredictor:
    def __init__(self, model, label_encoder, target_size=(100, 100)):
        """
        Initialize the ImagePredictor with a model and label encoder.

        Args:
            model: The trained model to use for predictions.
            label_encoder: The label encoder for decoding predictions.
            target_size (tuple): Size to which images will be resized.
        """
        self.model = model
        self.label_encoder = label_encoder
        self.target_size = target_size

    def predict(self, image_path):
        """
        Predict the class of a new image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Predicted class label.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error loading image at path: {image_path}")

        # Resize and normalize the image
        image = cv2.resize(image, self.target_size) / 255.0
        image_array = np.expand_dims(image, axis=0)  # Add a new dimension

        # Make prediction
        prediction = self.model.predict(image_array)
        predicted_class_index = np.argmax(prediction)
        predicted_class = self.label_encoder.inverse_transform([predicted_class_index])

        return predicted_class[0]
