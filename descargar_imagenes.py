import os
import time
from icrawler.builtin import BingImageCrawler

# 1. RUTA ADAPTADA PARA CODESPACES
# os.getcwd() detecta automáticamente la ruta de tu proyecto en Codespaces
# (Ej. /workspaces/mi-tarea-redes/dataset)
ruta_base = os.getcwd() 
carpeta_principal = os.path.join(ruta_base, "dataset")

if not os.path.exists(carpeta_principal):
    os.makedirs(carpeta_principal)

# 2. Mis 3 categorías... (El resto del código se queda exactamente igual)
categorias = {
    "carros": "carro en la calle ciudad fotografia",
    "casas": "casa exterior diferentes paisajes",
    "motos": "moto manejando en carretera o calle"
}
print("Iniciando la construcción del Dataset...")

for nombre_carpeta, busqueda in categorias.items():
    print(f"\n📥 Descargando imágenes para la categoría: {nombre_carpeta}...")
    
    # Creamos la ruta exacta (ej. dataset/carros)
    ruta_guardado = os.path.join(carpeta_principal, nombre_carpeta)
    
    # Configuramos el Crawler
    crawler = BingImageCrawler(storage={"root_dir": ruta_guardado})
    
    # Iniciamos la búsqueda y descarga
    crawler.crawl(
        keyword=busqueda, 
        max_num=100,          # Pide 100 imágenes como exige la tarea
        overwrite=True,       # Sobrescribe si ya existen
        min_size=(200, 200),  # Evita descargar iconos pequeños o basura
        max_size=None
    )
    
    print(f"✅ {nombre_carpeta}: 100 imágenes descargadas en '{ruta_guardado}/'")
    time.sleep(2)  # Espera 2 segundos para no saturar el servidor de Bing

print("\n🎉 ¡Descarga completada! Tu carpeta 'dataset' está lista para entrenar la Red Neuronal.")