import os

class Config:
    SECRET_KEY = 'clave-secreta-para-flash'
    UPLOAD_FOLDER = 'static/uploads/'
    FEEDBACK_FOLDER = 'feedback/'
    MODEL_PATH = 'modelos/modelo_final.h5'
    TRAIN_DIR = 'data/entrenar/'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}
    LOG_PATH = 'feedback_log.json'
