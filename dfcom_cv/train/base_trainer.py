import os
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, input_shape, num_classes, learning_rate=0.001):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()

    @abstractmethod
    def _build_model(self):
        pass

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        return self.model.fit(X_train, y_train,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(X_val, y_val))

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")
        return loss, accuracy

    def save_model(self, save_dir, model_name):
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, model_name)
        self.model.save(model_path)
        print(f"Model saved at: {model_path}")
