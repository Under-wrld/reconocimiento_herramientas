import tensorflow as tf
import numpy as np
import cv2
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detectada:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No se detectó GPU")

modelo = tf.keras.models.load_model('modelos/modelo_final.h5')
clases = sorted([nombre for nombre in os.listdir('data/entrenar') if os.path.isdir(os.path.join('data/entrenar', nombre))])
print("Modelo cargado")

#Lee la imagen
imagen = cv2.imread('ejemplos/llave_inglesa.jpg')
imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
imagen_redimensionada = cv2.resize(imagen_rgb, (224, 224))
imagen_redimensionada = imagen_redimensionada.astype(np.float32) / 255.0
imagen_redimensionada = np.expand_dims(imagen_redimensionada, axis=0)

#las predicción
pred = modelo.predict(imagen_redimensionada, verbose=0)[0]
indice = np.argmax(pred)
clase = clases[indice]
confianza = pred[indice]

print(f"Predicción: {clase} ({confianza * 100:.2f}%)")
