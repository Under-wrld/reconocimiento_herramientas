from flask import Blueprint, request, render_template, redirect, flash, current_app
from scripts.utils import predecir_desde_modelo
import os
import uuid

imagen_bp = Blueprint('imagen', __name__)

@imagen_bp.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    ruta_relativa = None

    if request.method == 'POST':
        imagen = request.files['imagen']

        if not imagen.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            flash('Solo se permiten archivos JPG, JPEG o PNG.')
            return redirect(request.url)

        filename = f"{uuid.uuid4().hex}_{imagen.filename}"
        ruta = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        ruta_relativa = f'uploads/{filename}'
        imagen.save(ruta)

        modelo = current_app.model_manager.model
        clases = current_app.model_manager.clases
        top_predicciones = predecir_desde_modelo(modelo, clases, ruta, top_k=3)

        clase, confianza = top_predicciones[0]

        return render_template('feedback.html',
                               pred=clase,
                               conf=confianza,
                               img_rel=ruta_relativa,
                               clases=clases,
                               predicciones=top_predicciones)

    return render_template('index.html', resultado=resultado, imagen=ruta_relativa)
