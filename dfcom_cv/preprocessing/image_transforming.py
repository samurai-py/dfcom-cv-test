import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

class ImageProcessor:
    def __init__(self, target_size=(100, 100)):
        """
        Initialize the ImageProcessor with the target size for resizing images.

        Args:
            target_size (tuple): Desired size for resizing images (width, height).
        """
        self.target_size = target_size
        self.label_encoder = LabelEncoder()

    def preprocess_images(self, images):
        """
        Resize and normalize a list of images.

        Args:
            images (list): List of images as numpy arrays.

        Returns:
            np.array: Array of preprocessed images.
        """
        processed_images = []

        for image in images:
            resized_image = cv2.resize(image, self.target_size)
            normalized_image = resized_image / 255.0
            processed_images.append(np.array(normalized_image, dtype=np.float32))

        return np.array(processed_images)

    def encode_labels(self, labels):
        """
        Encode class labels as categorical data.

        Args:
            labels (list): List of class labels.

        Returns:
            np.array: One-hot encoded labels.
        """
        encoded_labels = self.label_encoder.fit_transform(labels)
        return to_categorical(encoded_labels)

    def transform_labels(self, labels):
        """
        Transform labels using an already-fitted LabelEncoder.

        Args:
            labels (list): List of class labels.

        Returns:
            np.array: One-hot encoded labels.
        """
        transformed_labels = self.label_encoder.transform(labels)
        return to_categorical(transformed_labels)

    def prepare_data(self, train_images, train_labels, test_images, test_labels):
        """
        Prepare the training and test datasets.

        Args:
            train_images (list): List of training images.
            train_labels (list): List of training labels.
            test_images (list): List of test images.
            test_labels (list): List of test labels.

        Returns:
            tuple: Preprocessed and encoded (X_train, y_train, X_test, y_test) data.
        """
        # Preprocess images
        X_train = self.preprocess_images(train_images)
        X_test = self.preprocess_images(test_images)

        # Encode labels
        y_train = self.encode_labels(train_labels)
        y_test = self.transform_labels(test_labels)

        return X_train, y_train, X_test, y_test
