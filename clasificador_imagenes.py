import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Oculta advertencias de TensorFlow

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report

# 1. RUTAS ADAPTADAS A CODESPACES
ruta_base = os.getcwd() 
directorio_dataset = os.path.join(ruta_base, "dataset")

if not os.path.exists(directorio_dataset):
    print("❌ ERROR: No se encontró la carpeta 'dataset'. Ejecuta primero descargar_imagenes.py")
    exit()

# 2. CONFIGURACIÓN DEL DATASET
tamaño_lote = 32
tamaño_imagen = (128, 128) # TensorFlow redimensionará todas tus imágenes variadas a este tamaño

print("\nCargando imágenes de carros, casas y motos...")
# 80% Entrenamiento
dataset_entrenamiento = tf.keras.utils.image_dataset_from_directory(
    directorio_dataset,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=tamaño_imagen,
    batch_size=tamaño_lote
)

# 20% Pruebas / Validación
dataset_pruebas = tf.keras.utils.image_dataset_from_directory(
    directorio_dataset,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=tamaño_imagen,
    batch_size=tamaño_lote
)

nombres_clases = dataset_entrenamiento.class_names
print(f"\n✅ Clases detectadas: {nombres_clases}")

nombres_clases = dataset_entrenamiento.class_names
print(f"\n✅ Clases detectadas: {nombres_clases}")

# 3. CREACIÓN DE LA RED NEURONAL CONVOLUCIONAL (CNN)
modelo = models.Sequential([
    # Normalizamos los colores de los píxeles
    layers.Rescaling(1./255, input_shape=(tamaño_imagen[0], tamaño_imagen[1], 3)),
    
    # Capas de extracción visual (filtros fotográficos)
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # Capas de clasificación final
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Evita que la IA memorice y la obliga a generalizar
    layers.Dense(len(nombres_clases), activation='softmax') # 3 salidas
])

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. ENTRENAMIENTO
print("\n🧠 Iniciando entrenamiento visual...")
historial = modelo.fit(
    dataset_entrenamiento, 
    validation_data=dataset_pruebas, 
    epochs=15, 
    verbose=1
)

# Guardar Gráfica de Aprendizaje
plt.figure(figsize=(10, 5))
plt.plot(historial.history['accuracy'], label='Precisión Entrenamiento')
plt.plot(historial.history['val_accuracy'], label='Precisión Pruebas')
plt.title('Curva de Aprendizaje de la CNN')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.savefig('1_curva_aprendizaje_cnn.png')
plt.close()

# 5. MATRIZ DE CONFUSIÓN Y REPORTE
print("\nGenerando predicciones y Matriz de Confusión...")

clases_reales = []
clases_predichas = []

# Extraemos las predicciones lote por lote en el mismo orden exacto (A prueba de balas)
for imagenes, etiquetas in dataset_pruebas:
    predicciones_lote = modelo.predict(imagenes, verbose=0)
    clases_predichas.extend(np.argmax(predicciones_lote, axis=1))
    clases_reales.extend(etiquetas.numpy())

clases_reales = np.array(clases_reales)
clases_predichas = np.array(clases_predichas)

print("\n" + "="*40)
print("   REPORTE DE CLASIFICACIÓN")
print("="*40)
print(classification_report(clases_reales, clases_predichas, target_names=nombres_clases))

plt.figure(figsize=(8, 6))
matriz = confusion_matrix(clases_reales, clases_predichas)
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=nombres_clases, yticklabels=nombres_clases)
plt.title('Matriz de Confusión - Carros, Casas y Motos')
plt.xlabel('Predicción de la IA')
plt.ylabel('Imagen Real')
plt.savefig('2_matriz_confusion_imagenes.png')
plt.close()
# --------------------------------------------------------

# 6. ENTREGABLE VISUAL: REALES VS PREDICHOS
print("\n🖼️ Generando mosaico de ejemplos reales vs predichos...")
plt.figure(figsize=(12, 12))

# Tomamos un lote de imágenes del examen final
for imagenes, etiquetas in dataset_pruebas.take(1):
    predicciones_lote = modelo.predict(imagenes)
    etiquetas_predichas_lote = np.argmax(predicciones_lote, axis=1)
    
    # Mostramos 9 ejemplos
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        # Convertimos de tensores matemáticos a imagen visible
        plt.imshow(imagenes[i].numpy().astype("uint8"))
        
        nombre_real = nombres_clases[etiquetas[i]]
        nombre_predicho = nombres_clases[etiquetas_predichas_lote[i]]
        
        color = "green" if nombre_real == nombre_predicho else "red"
        plt.title(f"Real: {nombre_real}\nPred: {nombre_predicho}", color=color)
        plt.axis("off")

plt.tight_layout()
plt.savefig('3_ejemplos_reales_vs_predichos.png')
plt.close()

print("\n🎉 ¡Proceso terminado exitosamente! Tus 3 imágenes están listas en Codespaces.")