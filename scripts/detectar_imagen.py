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
    print("No se detectó GPU. Asegúrate de haber instalado CUDA y cuDNN.")

def predecir_herramienta(imagen_path, top_k=3, modelo_path='modelos/modelo_final.h5', train_dir='data/entrenar/'):

    clases = sorted([nombre for nombre in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, nombre))]) #en base a las carpetas del directorio 

    #cargar el modelo
    if not os.path.exists(modelo_path):
        raise FileNotFoundError(f"No se encontró el modelo en {modelo_path}. Entrénalo primero.")
    
    #verificar imagen
    if not os.path.exists(imagen_path):
        raise FileNotFoundError(f"No se encontró la imagen en {imagen_path}.")
    
    modelo = tf.keras.models.load_model(modelo_path)#carga modelo

    img = cv2.imread(imagen_path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen. Asegúrate de que sea una imagen válida.")
    
    #normalizacion y adaptacion la dimensión para TensorFlow
    img = cv2.resize(img, (224, 224)) / 255.0
    img = img.reshape(1, 224, 224, 3)

    pred = modelo.predict(img)[0]  # obtener vector directamente
    indices = np.argsort(pred)[::-1][:top_k]


    resultados = [(clases[i], float(pred[i])) for i in indices]  # lista de (clase, confianza)
    
    print("Predicciones top-k:")
    for clase, conf in resultados:
        print(f"  {clase}: {conf:.2%}")

    return resultados  # una lista de tuplas (clase, confianza)
