# limpiar_dataset.py
import os
import shutil

# Carpetas que NO quieres (las viejas)
carpetas_viejas = ["coches", "gatos", "perros"]

# Carpetas que QUIERES conservar
carpetas_nuevas = ["carros", "casas", "motos"]

print("🧹 LIMPIANDO DATASET...")
print("="*40)

# Eliminar carpetas viejas directamente de 'dataset/'
for carpeta in carpetas_viejas:
    ruta = f"dataset/{carpeta}"
    if os.path.exists(ruta):
        shutil.rmtree(ruta) # Borra la carpeta y todo su contenido
        print(f"❌ Eliminado: {ruta}")

print("\n✅ Dataset limpio. Ahora solo tienes:")
for carpeta in carpetas_nuevas:
    ruta_nueva = f"dataset/{carpeta}"
    # Contamos cuántas fotos hay en total en cada carpeta
    count = len(os.listdir(ruta_nueva)) if os.path.exists(ruta_nueva) else 0
    print(f"   📁 {carpeta}: {count} imágenes en total")

print("\n🎉 Ahora vuelve a ejecutar tu script de entrenamiento.")