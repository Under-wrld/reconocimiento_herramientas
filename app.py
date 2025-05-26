from flask import Flask, render_template, request, redirect, url_for, flash
from scripts.detectar_imagen import predecir_herramienta
from scripts.reentrenar import integrar_feedback_y_reentrenar
import uuid
import shutil
import os
import hashlib
import json

LOG_PATH = 'feedback_log.json'

#cargar registros previos
if os.path.exists(LOG_PATH):
    with open(LOG_PATH, 'r') as f:
        feedback_log = json.load(f)
else:
    feedback_log = {}

app = Flask(__name__)
app.secret_key = 'clave-secreta-para-flash' #mensajes de flash

#direcciones de las imagenes subidas y del feedback
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['FEEDBACK_FOLDER'] = 'feedback/'


#almacén temporal de última imagen analizada
last_image_path = None
last_prediction = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global last_image_path, last_prediction
    resultado = None
    ruta_relativa = None

    if request.method == 'POST':
        imagen = request.files['imagen']

        if not imagen.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            flash('Solo se permiten archivos JPG, JPEG o PNG.')
            return redirect(request.url)

        #asigna nombre unico 
        filename = f"{uuid.uuid4().hex}_{imagen.filename}"
        ruta = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        ruta_relativa = f'uploads/{filename}'
        imagen.save(ruta)

        top_predicciones = predecir_herramienta(ruta, top_k=3)
        clase, confianza = top_predicciones[0]
        last_image_path = ruta
        last_prediction = clase

        train_dir = 'data/entrenar/'
        clases = sorted([nombre for nombre in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, nombre))])
        return render_template('feedback.html', pred=clase, conf=confianza, img_rel=ruta_relativa, clases=clases, predicciones=top_predicciones)


    return render_template('index.html', resultado=resultado, imagen=ruta_relativa)

@app.route('/feedback', methods=['POST'])
def feedback():
    global last_image_path, last_prediction
    etiqueta_correcta = request.form['etiqueta'].strip().lower()

    if not etiqueta_correcta:
        flash('No se envió ninguna etiqueta.')
        return redirect(url_for('index'))

    #calcular hash de la imagen
    imagen_hash = calcular_hash_imagen(last_image_path)

    #verificar si ya se corrigió
    if imagen_hash in feedback_log:
        flash('Esta imagen ya fue corregida anteriormente. No se volverá a integrar.')
        return redirect(url_for('index'))

    #solo guardar si fue corregida correctamente
    if etiqueta_correcta != last_prediction:
        carpeta_destino = os.path.join(app.config['FEEDBACK_FOLDER'], etiqueta_correcta)
        os.makedirs(carpeta_destino, exist_ok=True)
        shutil.copy(last_image_path, carpeta_destino)

        feedback_log[imagen_hash] = etiqueta_correcta  #registrar corrección
        with open(LOG_PATH, 'w') as f:
            json.dump(feedback_log, f, indent=4)

        flash(f'Gracias por tu corrección. El modelo será actualizado con la clase "{etiqueta_correcta}".')
        integrar_feedback_y_reentrenar()
    else:
        flash('La predicción fue correcta. No se necesita reentrenamiento.')

    return redirect(url_for('index'))

@app.route('/stats')
def stats():
    ruta_img = 'stats/entrenamiento.png'
    if not os.path.exists(f'static/{ruta_img}'):
        flash('Aún no hay estadísticas disponibles. Entrena el modelo primero.')
        return redirect(url_for('index'))

    return render_template('stats.html', ruta_img=ruta_img)

#funcion para calcular hash de la imagen
def calcular_hash_imagen(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['FEEDBACK_FOLDER'], exist_ok=True)
    app.run(debug=True)
