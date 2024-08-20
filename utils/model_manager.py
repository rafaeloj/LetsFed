import tensorflow as tf
import keras
from typing import List
import os

class ModelManager():
        def __init__(self, model_type: str, input_shape: List[int]):
            self.model_type = model_type
            self.input_shape = input_shape
            self.model_path = f"logs/{self.model_type}.weights.h5"  # Adiciona model_type ao nome do arquivo
            self.model = self.get_model(model_type)
            self.update_weights()

        def get_model(self, model_type) -> keras.Model:
            if model_type == 'dnn':
                return self.dnn()
            if model_type == 'cnn':
                return self.cnn()

        def dnn(self) -> keras.Model:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Input(shape=self.input_shape[1:]),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(64,  activation='relu'),
                tf.keras.layers.Dense(32,  activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax'),
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model
        
        def cnn(self)-> keras.Model:
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape[1:]),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(10, activation='softmax'),
            ])

            return model

        def update_weights(self):
            if os.path.exists(self.model_path):
                print(f"Loading weights from {self.model_path}")
                self.model.load_weights(self.model_path)
            else:
                print(f"Saving weights to {self.model_path}")
                self.model.save_weights(self.model_path)
