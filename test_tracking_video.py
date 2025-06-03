from scripts.detectar_video import procesar_video_con_tracking
from model_manager import ModelManager
from config import Config
import tensorflow as tf

# Cargar el modelo y las clases desde la configuración
model_manager = ModelManager(Config.MODEL_PATH, Config.TRAIN_DIR)
modelo = model_manager.model
clases = model_manager.clases

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow configurado para usar GPU con crecimiento dinámico de memoria.")
    except RuntimeError as e:
        print("Error al configurar GPU:", e)

# Ejecutar el análisis de video con tracking
procesar_video_con_tracking(
    video_path='static/video_prueba.mp4',
    output_path='static/resultado_tracking.mp4',
    json_output='static/predicciones_tracking.json',
    modelo=modelo,
    clases=clases
)
