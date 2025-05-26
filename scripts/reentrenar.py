import os
import shutil
from scripts.entrenar_modelo import entrenar_modelo

def integrar_feedback_y_reentrenar(feedback_dir='feedback/', train_dir='data/entrenar/'):
    feedback_integrado = False

    for clase in os.listdir(feedback_dir):
        origen = os.path.join(feedback_dir, clase)
        destino = os.path.join(train_dir, clase)

        if os.path.isdir(origen):
            os.makedirs(destino, exist_ok=True)

            for archivo in os.listdir(origen):
                archivo_origen = os.path.join(origen, archivo)
                archivo_destino = os.path.join(destino, archivo)

                #solo si no existe una copia en el set de entrenamiento
                if not os.path.exists(archivo_destino):
                    shutil.move(archivo_origen, archivo_destino)
                    feedback_integrado = True

            #limpiamos la carpeta feedback una vez hayamos movido su contenido
            shutil.rmtree(origen)
        
    if feedback_integrado:
        print("Feedback nuevo detectado. Reentrenando modelo...")
        entrenar_modelo(epochs=5) #entrenamiento rápido
    else:
        print("No hay feedback nuevo para integrar.")

