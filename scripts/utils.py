import cv2
import numpy as np

def predecir_desde_modelo(modelo, clases, imagen_path, top_k=3):
    img = cv2.imread(imagen_path)
    if img is None:
        raise ValueError("No se pudo cargar la imagen.")
    
    img = cv2.resize(img, (224, 224)) / 255.0
    img = img.reshape(1, 224, 224, 3)

    pred = modelo.predict(img)[0]
    indices = np.argsort(pred)[::-1][:top_k]
    return [(clases[i], float(pred[i])) for i in indices]
