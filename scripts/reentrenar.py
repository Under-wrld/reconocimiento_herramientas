import os
import shutil
from scripts.entrenar_modelo import entrenar_modelo

def integrar_feedback_y_reentrenar(feedback_dir='feedback/', train_dir='data/entrenar/', epochs=5):
    feedback_integrado = False

    clases_validas = sorted([nombre for nombre in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, nombre))])

    for clase in os.listdir(feedback_dir):
        if clase not in clases_validas:
            print(f"Clase inválida detectada en feedback: '{clase}'. Se ignora.")
            continue

        origen = os.path.join(feedback_dir, clase)
        destino = os.path.join(train_dir, clase)

        if os.path.isdir(origen):
            os.makedirs(destino, exist_ok=True)

            for archivo in os.listdir(origen):
                archivo_origen = os.path.join(origen, archivo)
                archivo_destino = os.path.join(destino, archivo)

                if not os.path.exists(archivo_destino):
                    shutil.move(archivo_origen, archivo_destino)
                    feedback_integrado = True

            shutil.rmtree(origen)

    if feedback_integrado:
        print("Feedback nuevo detectado. Reentrenando modelo...")
    else:
        print("No hay feedback nuevo, pero se procederá a reentrenar el modelo igual.")

    entrenar_modelo(epochs=epochs)  # Reentrena siempre
