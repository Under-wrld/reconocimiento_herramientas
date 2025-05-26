import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def construir_modelo(num_clases, input_shape=(224, 224, 3)):
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
        metrics=['accuracy']
    )

    return model

def graficar_historial(historial, ruta='static/stats/entrenamiento.png'):
    plt.figure(figsize=(12,4))
    
    #para la precision
    plt.subplot(1, 2, 1)
    plt.plot(historial.history['accuracy'], label='Entrenamiento')
    plt.plot(historial.history['val_accuracy'], label='Validación')
    plt.title('Precisión')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.legend()
    
    #para la perdida
    plt.subplot(1, 2, 2)
    plt.plot(historial.history['loss'], label='Entrenamiento')
    plt.plot(historial.history['val_loss'], label='Validación')
    plt.title('Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.legend()
    
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    plt.tight_layout()
    plt.savefig(ruta)
    plt.close()

def entrenar_modelo(data_dir='data/entrenar/', img_size=(224, 224), batch_size=4, epochs=15):
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

    #construccion del modelo
    model = construir_modelo(num_clases=num_clases, input_shape=(img_size[0], img_size[1], 3))

    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) 
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

    print("-------------Entrenamiento completo y modelo guardado.-------------")

if __name__ == "__main__":
    entrenar_modelo()
