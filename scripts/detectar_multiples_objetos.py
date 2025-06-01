import cv2
import numpy as np
import json

def detectar_y_clasificar_multiples(imagen_path, modelo, clases, umbral_area=1000):
    img_original = cv2.imread(imagen_path)
    if img_original is None:
        raise ValueError("No se pudo cargar la imagen.")

    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    resultados = []

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)

        if area > umbral_area and w > 20 and h > 20 and 0.4 < aspect_ratio < 2.5:
            roi = img_original[y:y+h, x:x+w]
            if roi.size == 0:
                continue

            roi_resized = cv2.resize(roi, (224, 224)) / 255.0
            roi_resized = roi_resized.reshape(1, 224, 224, 3)

            pred = modelo.predict(roi_resized, verbose=0)
            top3_idx = pred[0].argsort()[-3:][::-1]
            top3 = [(clases[i], float(pred[0][i])) for i in top3_idx]

            clase_top1 = top3[0][0]
            confianza_top1 = top3[0][1]

            cv2.rectangle(img_original, (x, y), (x+w, y+h), (0, 255, 0), 2)
            texto = f"{clase_top1} ({confianza_top1*100:.1f}%)"
            cv2.putText(img_original, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            resultados.append({
                "top1": {"clase": clase_top1, "confianza": confianza_top1},
                "top3": top3,
                "bbox": {"x": x, "y": y, "w": w, "h": h}
            })

    return img_original, resultados
