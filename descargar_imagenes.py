from icrawler.builtin import BingImageCrawler
import time

# Mis 3 categorías
categorias = {
    "carros": "carro",
    "casas": "casa",
    "motos": "moto"
}

for carpeta, busqueda in categorias.items():
    print(f"\n📥 Descargando {busqueda}...")
    
    # Asegurar 100 imágenes
    crawler = BingImageCrawler(storage={"root_dir": carpeta})
    crawler.crawl(
        keyword=busqueda, 
        max_num=100,  # ← Pide 100 imágenes
        overwrite=True,  # ← Sobrescribe si ya existen
        min_size=(200, 200),  # ← Solo imágenes de buen tamaño
        max_size=None
    )
    
    print(f"✅ {carpeta}: 100 imágenes descargadas")
    time.sleep(2)  # Espera 2 segundos entre categorías

print("\n🎉 ¡Descarga completada! Revisa las carpetas: carros/, casas/, motos/")