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

    def predict(self, image_array):
        """
        Predict the class of a new image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Predicted class label.
        """
        # Ensure the image is in the correct format
        if isinstance(image_array, np.ndarray):
            # Preprocess the image as needed by the model
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            predictions = self.model.predict(image_array)
            predicted_label = self.label_encoder.inverse_transform([np.argmax(predictions)])
            return predicted_label[0]
        else:
            raise ValueError("Expected image_array to be a numpy array")
