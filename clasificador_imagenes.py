import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("="*50)
print("🚀 CLASIFICADOR DE IMÁGENES")
print("📋 Categorías: CARROS, CASAS, MOTOS")
print("="*50)

# Verificar que existan las carpetas
if not os.path.exists("dataset/train"):
    print("\n❌ Error: No se encuentra la carpeta 'dataset/train'")
    print("Ejecuta primero 'organizar_imagenes.py'")
    exit()

# 1. CARGAR IMÁGENES
print("\n📂 Cargando imágenes...")
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/train',
    image_size=(150, 150),
    batch_size=32,
    label_mode='categorical',
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset/test',
    image_size=(150, 150),
    batch_size=32,
    label_mode='categorical',
    shuffle=False
)

class_names = train_ds.class_names
print(f"\n📋 Clases encontradas: {class_names}")
print(f"✅ Total: {len(class_names)} clases")

# Optimizar
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 2. CREAR MODELO
print("\n🏗️ Creando modelo...")
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(150, 150, 3)),
    layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 3. ENTRENAR
print("\n🎯 Entrenando modelo...")
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=15,
    verbose=1
)

# 4. EVALUAR
print("\n📊 Evaluando modelo...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ PRECISIÓN DEL MODELO: {test_acc*100:.2f}%")
print(f"📉 Pérdida: {test_loss:.4f}")

# 5. MATRIZ DE CONFUSIÓN
print("\n📈 Generando matriz de confusión...")

y_true, y_pred = [], []
for images, labels in test_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)

# Guardar matriz
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={'size': 14})
plt.xlabel('Predicho', fontsize=12)
plt.ylabel('Real', fontsize=12)
plt.title('Matriz de Confusión', fontsize=14)
plt.tight_layout()
plt.savefig('matriz_confusion_tarea.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Guardado: matriz_confusion_tarea.png")

# 6. REPORTE
print("\n📋 REPORTE DE CLASIFICACIÓN:")
print("="*50)
print(classification_report(y_true, y_pred, target_names=class_names))

# 7. GRÁFICAS DE ENTRENAMIENTO
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('graficas_entrenamiento_tarea.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Guardado: graficas_entrenamiento_tarea.png")

# 8. EJEMPLOS DE IMÁGENES
print("\n🖼️ Generando ejemplos...")

for images, labels in test_ds.take(1):
    preds = model.predict(images, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.argmax(labels.numpy(), axis=1)
    
    plt.figure(figsize=(15, 10))
    for i in range(min(12, len(images))):
        plt.subplot(3, 4, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        color = "green" if true_classes[i] == pred_classes[i] else "red"
        true_label = class_names[true_classes[i]]
        pred_label = class_names[pred_classes[i]]
        confidence = np.max(preds[i]) * 100
        plt.title(f"Real: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%", 
                  color=color, fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Ejemplos de Clasificación (Verde=Acierto, Rojo=Error)', fontsize=14)
    plt.tight_layout()
    plt.savefig('ejemplos_clasificacion_tarea.png', dpi=300, bbox_inches='tight')
    plt.close()
    break

print("✅ Guardado: ejemplos_clasificacion_tarea.png")

# 9. ANÁLISIS POR CLASE
print("\n🔍 ANÁLISIS DETALLADO:")
print("="*50)

for i, clase in enumerate(class_names):
    VP = cm[i, i]
    FP = cm[:, i].sum() - VP
    FN = cm[i, :].sum() - VP
    total_real = cm[i, :].sum()
    
    precision = VP / (VP + FP) if (VP + FP) > 0 else 0
    recall = VP / (VP + FN) if (VP + FN) > 0 else 0
    
    print(f"\n📌 {clase.upper()}:")
    print(f"   ✅ Aciertos: {VP}/{total_real} ({VP/total_real*100:.1f}%)")
    print(f"   📊 Precisión: {precision*100:.1f}%")
    print(f"   📊 Sensibilidad: {recall*100:.1f}%")

# 10. CONCLUSIÓN
print("\n" + "="*50)
print("📝 CONCLUSIÓN:")
print("="*50)

if test_acc >= 0.85:
    print("✅ EXCELENTE: Clasificador muy apropiado")
elif test_acc >= 0.70:
    print("👍 BUENO: Clasificador apropiado para la tarea")
elif test_acc >= 0.55:
    print("⚠️ ACEPTABLE: Funciona pero puede mejorar")
else:
    print("❌ MEJORABLE: Aumentar imágenes o calidad")

print(f"\n🎯 Precisión final: {test_acc*100:.2f}%")
print("\n📁 Archivos generados:")
print("   • matriz_confusion_tarea.png")
print("   • graficas_entrenamiento_tarea.png")
print("   • ejemplos_clasificacion_tarea.png")