from scripts.detectar_video import procesar_video_con_tracking

# Ruta al video que deseas analizar
video_input_path = 'static/video_prueba.mp4'  # cámbialo si el nombre del video es distinto
video_output_path = 'static/resultado_tracking.mp4'
json_output_path = 'static/predicciones_tracking.json'

procesar_video_con_tracking(
    video_path=video_input_path,
    output_path=video_output_path,
    json_output=json_output_path
)
