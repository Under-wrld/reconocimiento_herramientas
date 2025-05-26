import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

def construir_modelo(num_clases, input_shape=(224, 224, 3), metrics=['accuracy']):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Dropout(0.2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_clases, activation='softmax')  
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=metrics
    )

    return model

def graficar_historial(historial, ruta='static/stats/entrenamiento.png'):

    #detectamos todas las metricas que tienen version de validacion
    metricas = [k for k in historial.history.keys() if not k.startswith('val_') and k != 'loss']
    tiene_val = any(f'val_{m}' in historial.history for m in metricas)

    num_graficas = len(metricas) + 1  # +1 para perdida
    plt.figure(figsize=(5 * num_graficas, 4))
    
    #para la perdida
    plt.subplot(1, num_graficas, 1)
    plt.plot(historial.history['loss'], label='Entrenamiento')
    if 'val_loss' in historial.history:
        plt.plot(historial.history['val_loss'], label='Validación')
    plt.title('Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    #graficas de metricas dinamizado
    for i, metrica in enumerate(metricas):
        plt.subplot(1, num_graficas, i + 2)
        plt.plot(historial.history[metrica], label='Entrenamiento')
        val_key = f'val_{metrica}'
        if val_key in historial.history:
            plt.plot(historial.history[val_key], label='Validación')
        plt.title(metrica.capitalize())
        plt.xlabel('Época')
        plt.ylabel(metrica.capitalize())
        plt.legend()

    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    plt.tight_layout()
    plt.savefig(ruta)
    plt.close()

def generar_reporte_clasificacion(modelo, val_data, clases, salida='static/stats/reporte_clasificacion.csv'):
    #obtener predicciones
    val_data.reset()  # evitar errores
    pred = modelo.predict(val_data)
    y_pred = pred.argmax(axis=1)
    y_true = val_data.classes

    #reporte
    reporte = classification_report(y_true, y_pred, target_names=clases, output_dict=True)
    df_reporte = pd.DataFrame(reporte).transpose()

    os.makedirs(os.path.dirname(salida), exist_ok=True)
    df_reporte.to_csv(salida)
    print("Reporte de clasificación guardado en:", salida)

def generar_matriz_confusion(modelo, val_data, clases, ruta='static/stats/matriz_confusion.png'):
    val_data.reset()
    pred = modelo.predict(val_data)
    y_pred = pred.argmax(axis=1)
    y_true = val_data.classes

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases, yticklabels=clases)
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    plt.tight_layout()
    plt.savefig(ruta)
    plt.close()
    print("Matriz de confusión guardada en:", ruta)


def entrenar_modelo(data_dir='data/entrenar/', img_size=(224, 224), batch_size=4, epochs=15, metrics=None):
    #aumentación de datos y división en entrenamiento/validación
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.3,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    #datos de entrenamiento
    train_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    #datos de validación
    val_data = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    num_clases = len(train_data.class_indices)  #detecta automáticamente número de clases

    if metrics is None:
        metrics = ['accuracy']

    #construccion del modelo
    model = construir_modelo(num_clases=num_clases, input_shape=(img_size[0], img_size[1], 3))

    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True) 
    checkpoint = ModelCheckpoint('modelos/mejor_modelo.h5', monitor='val_loss', save_best_only=True)
  

    #entrenamiento
    historial = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=[early_stop, checkpoint]
    )

    #guardar el modelo ya entrenado
    os.makedirs('modelos', exist_ok=True)
    shutil.copy('modelos/mejor_modelo.h5', 'modelos/modelo_final.h5')

    graficar_historial(historial)
    clases = list(train_data.class_indices.keys())
    generar_reporte_clasificacion(model, val_data, clases)
    generar_matriz_confusion(model, val_data, clases)

    print("-------------Entrenamiento completo y modelo guardado.-------------")

if __name__ == "__main__":
    from tensorflow.keras.metrics import Precision, Recall
    entrenar_modelo(metrics=['accuracy', Precision(), Recall()])
