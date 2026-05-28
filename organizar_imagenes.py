import os
import random
import shutil

# MISMO NOMBRE que en código 1
categorias = ["carros", "casas", "motos"]

print("📂 ORGANIZANDO IMÁGENES...")
print("="*40)

# Crear estructura de carpetas
for categoria in categorias:
    os.makedirs(f"dataset/train/{categoria}", exist_ok=True)
    os.makedirs(f"dataset/test/{categoria}", exist_ok=True)

# Organizar imágenes
for categoria in categorias:
    if not os.path.exists(categoria):
        print(f"❌ Carpeta '{categoria}' no encontrada")
        print(f"   Ejecuta primero 'descargar_imagenes.py'")
        continue
    
    # Obtener lista de imágenes (varios formatos)
    imagenes = []
    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
        imagenes.extend([f for f in os.listdir(categoria) if f.lower().endswith(ext)])
    
    if len(imagenes) == 0:
        print(f"⚠️ No hay imágenes en {categoria}")
        continue
    
    print(f"\n📁 {categoria.upper()}: {len(imagenes)} imágenes encontradas")
    
    # Mezclar al azar
    random.shuffle(imagenes)
    
    # 80% entrenamiento, 20% prueba
    split_idx = int(len(imagenes) * 0.8)
    
    train_imgs = imagenes[:split_idx]
    test_imgs = imagenes[split_idx:]
    
    # Copiar a train
    for img in train_imgs:
        src = os.path.join(categoria, img)
        dst = f"dataset/train/{categoria}/{img}"
        shutil.copy2(src, dst)
    
    # Copiar a test
    for img in test_imgs:
        src = os.path.join(categoria, img)
        dst = f"dataset/test/{categoria}/{img}"
        shutil.copy2(src, dst)
    
    print(f"   ✅ Train: {len(train_imgs)} imágenes")
    print(f"   ✅ Test: {len(test_imgs)} imágenes")

print("\n" + "="*40)
print("🎉 ¡Dataset organizado correctamente!")
print("📁 Estructura creada en: dataset/")