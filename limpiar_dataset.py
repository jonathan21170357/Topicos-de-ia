# limpiar_dataset.py
import os
import shutil

# Carpetas que NO quieres (las viejas)
carpetas_viejas = ["coches", "gatos", "perros"]

# Carpetas que QUIERES conservar
carpetas_nuevas = ["carros", "casas", "motos"]

print("🧹 LIMPIANDO DATASET...")
print("="*40)

# Eliminar carpetas viejas de train
for carpeta in carpetas_viejas:
    ruta_train = f"dataset/train/{carpeta}"
    if os.path.exists(ruta_train):
        shutil.rmtree(ruta_train)
        print(f"❌ Eliminado: {ruta_train}")

# Eliminar carpetas viejas de test
for carpeta in carpetas_viejas:
    ruta_test = f"dataset/test/{carpeta}"
    if os.path.exists(ruta_test):
        shutil.rmtree(ruta_test)
        print(f"❌ Eliminado: {ruta_test}")

print("\n✅ Dataset limpio. Ahora solo tienes:")
for carpeta in carpetas_nuevas:
    train_count = len(os.listdir(f"dataset/train/{carpeta}")) if os.path.exists(f"dataset/train/{carpeta}") else 0
    test_count = len(os.listdir(f"dataset/test/{carpeta}")) if os.path.exists(f"dataset/test/{carpeta}") else 0
    print(f"   📁 {carpeta}: {train_count} train, {test_count} test")

print("\n🎉 Ahora vuelve a ejecutar: python clasificador_completo.py")