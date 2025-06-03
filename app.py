from flask import Flask
from config import Config
from model_manager import ModelManager

from blueprints.imagen import imagen_bp
from blueprints.video import video_bp
from blueprints.feedback import feedback_bp
from blueprints.stats import stats_bp

app = Flask(__name__)
app.config.from_object(Config)

model_manager = ModelManager(Config.MODEL_PATH, Config.TRAIN_DIR)
app.model_manager = model_manager

#registrar los blueprints
app.register_blueprint(imagen_bp)
app.register_blueprint(video_bp)
app.register_blueprint(feedback_bp)
app.register_blueprint(stats_bp)

if __name__ == '__main__':
    import os
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['FEEDBACK_FOLDER'], exist_ok=True)
    app.run(debug=True)
