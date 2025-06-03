from flask import Blueprint, render_template, current_app, flash, redirect, url_for
import os
import random

stats_bp = Blueprint('stats', __name__)

@stats_bp.route('/stats')
def mostrar_stats():
    ruta_img = 'stats/entrenamiento.png'
    if not os.path.exists(os.path.join('static', ruta_img)):
        flash('Aún no hay estadísticas disponibles. Entrena el modelo primero.')
        return redirect(url_for('imagen.index'))

    return render_template('stats.html', config={'RANDOM': random.randint(0, 10000)})
