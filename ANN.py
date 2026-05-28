import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar los datos
print("Cargando datos...")
df = pd.read_csv('GeneroMusical.csv', header=0)

print(f"Dataset original: {df.shape}")
print("\nPrimeras 5 filas:")
print(df.head())

# 2. Análisis Exploratorio de Datos (EDA) básico
print("\n--- Análisis Exploratorio ---")
print("Distribución de clases:")
print(df['Class'].value_counts().sort_index())

print("\nValores nulos por columna:")
print(df.isnull().sum())

# 3. Preprocesamiento
print("\n--- Preprocesamiento ---")
# Eliminar columnas no útiles para la predicción
cols_to_drop = ['Artist Name', 'Track Name', 'duration_in min/ms']
df_clean = df.drop(columns=cols_to_drop)

# Eliminar filas con valores nulos
df_clean = df_clean.dropna()
print(f"Dataset después de limpiar nulos: {df_clean.shape}")

# Separar características (X) y etiquetas (y)
X = df_clean.drop('Class', axis=1).values
y = df_clean['Class'].values

# Dividir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Tamaño entrenamiento: {X_train.shape[0]}")
print(f"Tamaño prueba: {X_test.shape[0]}")

# Escalar las características (importante para redes neuronales)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Construir el modelo MLP (Red Neuronal con scikit-learn)
print("\n--- Construyendo la Red Neuronal MLP ---")
print(f"Número de características de entrada: {X_train_scaled.shape[1]}")
print(f"Número de clases a predecir: {len(np.unique(y))}")

# Capas ocultas: (128, 64) - similar a la versión de TensorFlow
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),  # Dos capas ocultas
    activation='relu',              # Función de activación ReLU
    solver='adam',                  # Optimizador Adam
    max_iter=200,                   # Número máximo de épocas
    random_state=42,
    verbose=True,                   # Para ver el progreso
    early_stopping=True,            # Detiene si no mejora
    validation_fraction=0.1        # Usa 10% para validación
)

# 5. Entrenar el modelo
print("\n--- Entrenando la Red Neuronal ---")
mlp.fit(X_train_scaled, y_train)

# 6. Evaluar el modelo
print("\n--- Evaluación del Modelo ---")
y_pred = mlp.predict(X_test_scaled)

# Calcular precisión
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión (Accuracy) global: {accuracy:.4f}")
print(f"Porcentaje de aciertos: {accuracy*100:.2f}%")
print(f"Porcentaje de fallos: {(1-accuracy)*100:.2f}%")

# Mostrar matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Matriz de Confusión para Clasificación de Género Musical')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()

# Reporte de clasificación detallado
print("\n--- Reporte de Clasificación ---")
print(classification_report(y_test, y_pred, 
                           target_names=[f'Género {i}' for i in range(len(np.unique(y)))]))

# Análisis de aciertos y fallos en las primeras 20 predicciones
print("\n--- Análisis de Aciertos/Fallos (Primeras 20 muestras de prueba) ---")
aciertos = 0
for i in range(min(20, len(y_test))):
    real = y_test[i]
    prediccion = y_pred[i]
    estado = "✓ ACIERTO" if real == prediccion else "✗ FALLO"
    if real == prediccion:
        aciertos += 1
    print(f"Muestra {i+1:2d}: Real = {real:2d}, Predicho = {prediccion:2d} -> {estado}")

print(f"\n📊 De las primeras 20 muestras, acertó {aciertos} veces ({(aciertos/20)*100:.1f}%)")
print(f"📊 En total: Acertó {int(accuracy * len(y_test))} de {len(y_test)} canciones")

# Análisis de error por clase
print("\n--- Análisis de Errores por Clase ---")
print("Clase | Aciertos | Total | Precisión")
print("-" * 45)
for clase in np.unique(y_test):
    indices_clase = np.where(y_test == clase)[0]
    aciertos_clase = np.sum(y_pred[indices_clase] == clase)
    total_clase = len(indices_clase)
    precision_clase = aciertos_clase / total_clase if total_clase > 0 else 0
    print(f"  {clase:2d}   |    {aciertos_clase:3d}   |  {total_clase:3d}  |   {precision_clase*100:.1f}%")

# Mostrar gráfica de pérdida (loss curve)
plt.figure(figsize=(10, 6))
plt.plot(mlp.loss_curve_, label='Pérdida durante entrenamiento')
plt.title('Curva de Pérdida del Modelo')
plt.xlabel('Iteración')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 7. Predicción para una nueva canción de ejemplo
print("\n--- Predicción para una nueva canción de ejemplo ---")
# Creamos un ejemplo con valores representativos (una canción de rock)
nueva_cancion = np.array([[
    75.0,   # popularity (popularidad alta)
    0.5,    # danceability (media)
    0.9,    # energy (alta, típica de rock)
    5.0,    # key
    -5.0,   # loudness (ruidoso)
    1,      # mode (mayor)
    0.05,   # speechiness (bajo, no hablado)
    0.1,    # acousticness (bajo, eléctrico)
    0.0,    # instrumentalness
    0.3,    # liveness
    0.6,    # valence (positivo)
    130.0,  # tempo (rápido)
    4       # time_signature (4/4)
]])

# Escalar la nueva canción
nueva_cancion_scaled = scaler.transform(nueva_cancion)

# Predecir el género
prediccion = mlp.predict(nueva_cancion_scaled)
probabilidades = mlp.predict_proba(nueva_cancion_scaled)

print(f"El género predicho para la canción de ejemplo es: Clase {prediccion[0]}")
print(f"Probabilidades por clase: {probabilidades[0]}")

print("\n--- Fin del Análisis ---")