from flask import Blueprint, render_template, request, flash, current_app, send_file
from scripts.detectar_video import procesar_video_con_tracking
import os
import uuid
import json

video_bp = Blueprint('video', __name__)

@video_bp.route('/deteccion-video-tracking', methods=['GET', 'POST'])
def deteccion_video_tracking():
    if request.method == 'POST':
        video = request.files['video']
        if not video.filename.lower().endswith('.mp4'):
            flash('Solo se permiten videos MP4.')
            return render_template('deteccion_video_tracking_formulario.html')

        filename = f"{uuid.uuid4().hex}_{video.filename}"
        ruta_video = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        video.save(ruta_video)

        modelo = current_app.model_manager.model
        clases = current_app.model_manager.clases

        procesar_video_con_tracking(
            video_path=ruta_video,
            output_path='static/resultado_tracking.mp4',
            json_output='static/predicciones_tracking.json',
            modelo=modelo,
            clases=clases
        )

        resumen = generar_resumen_desde_json()

        return render_template('deteccion_video_tracking_resultado.html',
                               video_resultado='resultado_tracking.mp4',
                               resumen=resumen)

    return render_template('deteccion_video_tracking_formulario.html')


@video_bp.route('/descargar-json-tracking')
def descargar_json_tracking():
    json_path = 'static/predicciones_tracking.json'
    return send_file(json_path, as_attachment=True)


def generar_resumen_desde_json(path='static/predicciones_tracking.json'):
    if not os.path.exists(path):
        return []

    with open(path, 'r') as f:
        data = json.load(f)

    resumen = {}
    for frame in data:
        for obj in frame:
            obj_id = obj.get('id', -1)
            clave = (obj_id, obj['clase'])
            if clave not in resumen:
                resumen[clave] = {'id': obj_id, 'clase': obj['clase'], 'confianzas': []}
            resumen[clave]['confianzas'].append(obj['confianza'])

    resultado = []
    for (id_val, clase), info in resumen.items():
        resultado.append({
            'id': id_val,
            'clase': clase,
            'veces': len(info['confianzas']),
            'promedio_confianza': sum(info['confianzas']) / len(info['confianzas'])
        })

    return resultado
