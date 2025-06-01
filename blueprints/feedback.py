from flask import Blueprint, request, redirect, url_for, flash, current_app, render_template
from scripts.reentrenar import integrar_feedback_y_reentrenar
import os
import shutil
import json
import hashlib

feedback_bp = Blueprint('feedback', __name__)

@feedback_bp.route('/feedback', methods=['POST'])
def feedback():
    etiqueta_correcta = request.form['etiqueta'].strip().lower()
    imagen_rel_path = request.form['imagen_path'].strip()  # ej: uploads/abc.jpg
    imagen_path = os.path.join(current_app.config['UPLOAD_FOLDER'], os.path.basename(imagen_rel_path))

    if not etiqueta_correcta:
        flash('No se envió ninguna etiqueta.')
        return redirect(url_for('imagen.index'))

    clases_validas = current_app.model_manager.clases
    if etiqueta_correcta not in clases_validas:
        flash(f'La clase "{etiqueta_correcta}" no existe.')
        return redirect(url_for('imagen.index'))

    log_path = current_app.config['LOG_PATH']
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            feedback_log = json.load(f)
    else:
        feedback_log = {}

    imagen_hash = calcular_hash_imagen(imagen_path)
    if imagen_hash in feedback_log:
        flash('Esta imagen ya fue corregida anteriormente.')
        return redirect(url_for('imagen.index'))

    destino = os.path.join(current_app.config['TRAIN_DIR'], etiqueta_correcta)
    os.makedirs(destino, exist_ok=True)
    shutil.copy(imagen_path, destino)

    feedback_log[imagen_hash] = etiqueta_correcta
    with open(log_path, 'w') as f:
        json.dump(feedback_log, f, indent=4)

    integrar_feedback_y_reentrenar()

    if os.path.exists(imagen_path):
        os.remove(imagen_path)

    # Redirigir a pantalla de éxito
    return render_template('feedback_exito.html', etiqueta=etiqueta_correcta)

def calcular_hash_imagen(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()
