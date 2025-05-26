from flask import Flask, render_template, request, redirect, url_for, flash
from scripts.detectar_imagen import predecir_herramienta
from scripts.reentrenar import integrar_feedback_y_reentrenar
import uuid
import shutil
import os

app = Flask(__name__)
app.secret_key = 'clave-secreta-para-flash' #mensajes de flash
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

        clase, confianza = predecir_herramienta(ruta)
        last_image_path = ruta
        last_prediction = clase

        train_dir = 'data/entrenar/'
        clases = sorted([nombre for nombre in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, nombre))])
        return render_template('feedback.html', pred=clase, conf=confianza, img_rel=ruta_relativa, clases=clases)


    return render_template('index.html', resultado=resultado, imagen=ruta_relativa)

@app.route('/feedback', methods=['POST'])
def feedback():
    global last_image_path, last_prediction
    etiqueta_correcta = request.form['etiqueta'].strip().lower()

    # Solo guardar si la etiqueta ha sido corregida
    if etiqueta_correcta and etiqueta_correcta != last_prediction:
        carpeta_destino = os.path.join(app.config['FEEDBACK_FOLDER'], etiqueta_correcta)
        os.makedirs(carpeta_destino, exist_ok=True)
        shutil.copy(last_image_path, carpeta_destino)

        flash(f'Gracias por tu corrección. El modelo será actualizado con la clase "{etiqueta_correcta}".')
    else:
        flash('No hubo necesidad de corrección.')

    integrar_feedback_y_reentrenar()

    return redirect(url_for('index'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['FEEDBACK_FOLDER'], exist_ok=True)
    app.run(debug=True)
