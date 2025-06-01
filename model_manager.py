import tensorflow as tf
import os

class ModelManager:
    def __init__(self, model_path, train_dir):
        self.model_path = model_path
        self.train_dir = train_dir
        self.model = self._load_model()
        self.clases = self._load_classes()

    def _load_model(self):
        print(f"[INFO] Cargando modelo desde: {self.model_path}")
        return tf.keras.models.load_model(self.model_path)

    def _load_classes(self):
        print(f"[INFO] Cargando clases desde: {self.train_dir}")
        return sorted([
            nombre for nombre in os.listdir(self.train_dir)
            if os.path.isdir(os.path.join(self.train_dir, nombre))
        ])
