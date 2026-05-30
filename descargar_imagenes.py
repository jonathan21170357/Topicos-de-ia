import os
import time
from icrawler.builtin import BingImageCrawler
from PIL import Image

# 1. RUTA ADAPTADA PARA CODESPACES
ruta_base = os.getcwd() 
carpeta_principal = os.path.join(ruta_base, "dataset")

if not os.path.exists(carpeta_principal):
    os.makedirs(carpeta_principal)

# 2. Mis 3 categorías...
categorias = {
    "carros": "carro en la calle ciudad fotografia",
    "casas": "casa exterior diferentes paisajes",
    "motos": "moto manejando en carretera o calle"
}
print("Iniciando la construcción del Dataset...")

for nombre_carpeta, busqueda in categorias.items():
    print(f"\n📥 Descargando imágenes para la categoría: {nombre_carpeta}...")
    
    ruta_guardado = os.path.join(carpeta_principal, nombre_carpeta)
    
    crawler = BingImageCrawler(storage={"root_dir": ruta_guardado})
    
    crawler.crawl(
        keyword=busqueda, 
        max_num=130,          # <--- MODIFICADO: Pedimos 130 para tener un margen de error
        overwrite=True,
        min_size=(200, 200),
        max_size=None
    )
    
    print(f"✅ {nombre_carpeta}: Imágenes descargadas en '{ruta_guardado}/'")
    time.sleep(2)

print("\n🎉 ¡Descarga completada! Pasando a la fase de limpieza...")

# =====================================================================
# 3. LIMPIEZA DE IMÁGENES CORRUPTAS (EL CÓDIGO NUEVO)
# =====================================================================
print("\n🧹 Buscando y eliminando archivos corruptos o falsas imágenes...")

# Recorremos cada carpeta que acabamos de crear (carros, casas, motos)
for nombre_carpeta in categorias.keys():
    ruta_carpeta = os.path.join(carpeta_principal, nombre_carpeta)
    
    if os.path.exists(ruta_carpeta):
        # Revisamos cada archivo descargado dentro de la carpeta
        for archivo in os.listdir(ruta_carpeta):
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            
            try:
                # Intentamos abrir la imagen con PIL
                img = Image.open(ruta_archivo)
                img.verify()  # Verifica que los datos del archivo sean realmente de una imagen
            except Exception:
                # Si el código falla al abrirla (es un HTML, enlace roto o corrupta), la borramos
                print(f"   🗑️ Borrando imagen corrupta detectada: {archivo} en {nombre_carpeta}")
                os.remove(ruta_archivo)

print("\n✨ ¡Limpieza terminada! Tu dataset está 100% libre de errores y listo para la Red Neuronal.")