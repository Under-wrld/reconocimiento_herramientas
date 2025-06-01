import os
import hashlib
from icrawler.builtin import GoogleImageCrawler
from entrenar_modelo import entrenar_modelo
from PIL import Image

LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)

def calcular_hash(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def limpiar_duplicados_y_renombrar(clase, carpeta):
    hashes = set()
    eliminadas = 0
    archivos = sorted(os.listdir(carpeta))
    i = 1
    log_file = os.path.join(LOG_DIR, f'urls_descargadas_{clase}.txt')

    for archivo in archivos:
        path = os.path.join(carpeta, archivo)

        if not os.path.isfile(path):
            continue

        try:
            h = calcular_hash(path)
            if h in hashes:
                os.remove(path)
                eliminadas += 1
                continue
            hashes.add(h)

            # Renombrar con formato clase_XXXX.jpg
            nuevo_nombre = f"{clase}_{i:04d}.jpg"
            nuevo_path = os.path.join(carpeta, nuevo_nombre)
            while os.path.exists(nuevo_path):
                i += 1
                nuevo_nombre = f"{clase}_{i:04d}.jpg"
                nuevo_path = os.path.join(carpeta, nuevo_nombre)
            os.rename(path, nuevo_path)
            i += 1
        except:
            os.remove(path)
            eliminadas += 1

    print(f"Imágenes duplicadas o inválidas eliminadas: {eliminadas}")

def descargar_imagenes(clase, n=100, destino_base='data/entrenar/'):
    destino = os.path.join(destino_base, clase)
    os.makedirs(destino, exist_ok=True)

    log_path = os.path.join(LOG_DIR, f'urls_descargadas_{clase}.txt')
    urls_guardadas = set()
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            urls_guardadas = set([line.strip() for line in f.readlines()])

    class MyCrawler(GoogleImageCrawler):
        def download(self, task, default_ext, timeout=5, max_retry=3, **kwargs):
            if task['file_url'] in urls_guardadas:
                return
            super().download(task, default_ext, timeout, max_retry, **kwargs)
            with open(log_path, 'a') as log:
                log.write(task['file_url'] + '\n')

    print(f"Buscando imágenes nuevas para la clase: {clase}")
    crawler = MyCrawler(storage={'root_dir': destino})
    crawler.crawl(keyword=clase + ' herramienta', max_num=n)

    limpiar_duplicados_y_renombrar(clase, destino)

def actualizar_y_entrenar(clases, imagenes_por_clase=100):
    for clase in clases:
        descargar_imagenes(clase, imagenes_por_clase)

    print("\nReentrenando modelo con nuevas imágenes...")
    entrenar_modelo()
    print("Proceso completo.")

# ----------- USO -------------
if __name__ == "__main__":
    clases_a_buscar = ['martillo', 'llave_inglesa', 'destornillador', 'gato_hidraulico', 
                       'alicate', 'chicharra_neumatica', 'copa_automotriz','lampara_de_prueba_automotriz',
                       'llave_allen', 'llave_de_impacto', 'multimetro']
    
    actualizar_y_entrenar(clases_a_buscar, imagenes_por_clase=100)