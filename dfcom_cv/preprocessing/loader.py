import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

class ImageLoader:
    def __init__(self, max_images=None):
        """
        Initialize the ImageLoader with an optional maximum number of images to load.
        
        Args:
            max_images (int, optional): Maximum number of images to load per category. Defaults to None, which loads all images.
        """
        self.max_images = max_images

    def load_images(self, folder_path):
        """
        Load images and their labels from a specified directory.
        
        Args:
            folder_path (str): Path to the main folder containing subfolders of images for each category.
        
        Returns:
            images (list): List of loaded images as numpy arrays.
            labels (list): List of labels corresponding to each image.
        """
        images = []
        labels = []

        for category in os.listdir(folder_path):
            category_path = os.path.join(folder_path, category)
            
            # Check if it's a directory
            if os.path.isdir(category_path):
                for i, image_name in enumerate(os.listdir(category_path)):
                    # Check if max_images limit has been reached
                    if self.max_images is not None and i >= self.max_images:
                        break

                    image_path = os.path.join(category_path, image_name)
                    image = cv2.imread(image_path)

                    if image is not None:
                        images.append(image)
                        labels.append(category)

        return images, labels

    def visualize_images(self, images, labels, max_images=None):
        """
        Visualize a given list of images with corresponding labels.
        
        Args:
            images (list): List of images to visualize.
            labels (list): List of labels corresponding to each image.
            max_images (int, optional): Maximum number of images to display. Defaults to None, which displays all images.
        """
        max_images = max_images if max_images is not None else len(images)

        for i in range(min(len(images), max_images)):
            image_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
            plt.title(f"Category: {labels[i]}")
            plt.axis("off")
            plt.show()
