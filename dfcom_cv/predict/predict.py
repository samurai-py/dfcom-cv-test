import numpy as np

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

        if isinstance(image_array, np.ndarray):
            image_array = np.expand_dims(image_array, axis=0)
            predictions = self.model.predict(image_array)
            predicted_label = self.label_encoder.inverse_transform([np.argmax(predictions)])
            return predicted_label[0]
        else:
            raise ValueError("Expected image_array to be a numpy array")
