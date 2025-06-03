import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from scripts.sort import Sort 
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detectada:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No se detectó GPU. Asegúrate de haber instalado CUDA y cuDNN.")

def procesar_video_con_tracking(video_path, modelo, clases, output_path='static/resultado_tracking.mp4', json_output='static/predicciones_tracking.json'):
    # Crear carpetas si no existen
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_output), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("No se pudo abrir el video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Asegúrate que coincide con la extensión del archivo
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = Sort()
    predicciones = []

    for _ in tqdm(range(total_frames), desc="Procesando video con tracking"):
        ret, frame = cap.read()
        if not ret:
            break

        detecciones = []
        frame_data = []

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contornos:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / float(h)

            if area > 1000 and w > 20 and h > 20 and 0.4 < aspect_ratio < 2.5:
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue

                roi_resized = cv2.resize(roi, (224, 224)) / 255.0
                roi_resized = roi_resized.reshape(1, 224, 224, 3)

                pred = modelo.predict(roi_resized, verbose=0)
                clase = clases[np.argmax(pred)]
                confianza = np.max(pred)

                detecciones.append([x, y, x+w, y+h, confianza])
                frame_data.append({
                    "clase": clase,
                    "confianza": round(float(confianza), 4),
                    "bounding_box": {"x": x, "y": y, "w": w, "h": h}
                })

        tracks = tracker.update(np.array(detecciones))

        for track, pred_data in zip(tracks, frame_data):
            x1, y1, x2, y2, track_id = track.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            texto = f"{pred_data['clase']} ID:{int(track_id)}"
            cv2.putText(frame, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            pred_data["id"] = int(track_id)

        predicciones.append(frame_data)
        out.write(frame)

    cap.release()
    out.release()

    with open(json_output, 'w') as f:
        json.dump(predicciones, f, indent=4)

    print(f"Video con tracking guardado en: {output_path}")
    print(f"Predicciones JSON en: {json_output}")
