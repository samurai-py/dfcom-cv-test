import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from .base_trainer import BaseTrainer

class TransferLearningTrainer(BaseTrainer):
    def _load_base_model(self):
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=self.input_shape)
        for layer in base_model.layers:
            layer.trainable = False
        return base_model

    def _build_model(self):
        base_model = self._load_base_model()
        top_model = Sequential([
            Flatten(input_shape=base_model.output_shape[1:]),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
