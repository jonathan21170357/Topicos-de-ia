import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. CARGA DE DATOS
print("Cargando dataset...")
df = pd.read_csv('GeneroMusical.csv')

# 2. PREPROCESAMIENTO Y LIMPIEZA
df['Popularity'] = df['Popularity'].fillna(df['Popularity'].median())
df['key'] = df['key'].fillna(df['key'].mode()[0])
df['instrumentalness'] = df['instrumentalness'].fillna(0)

X = df.drop(columns=['Artist Name', 'Track Name', 'Class'])
y = df['Class']

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# 3. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
print("Generando y guardando gráficas EDA...")
plt.figure(figsize=(10, 6))
sns.countplot(x=y)
plt.title('Distribución de Clases de Género Musical')
plt.xlabel('Clase (Género)')
plt.ylabel('Cantidad')
plt.savefig('1_distribucion_clases_Genero musical.png') # Guarda la imagen
plt.close() # Cierra la figura para no mezclarla con la siguiente

plt.figure(figsize=(12, 10))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación de Características Musicales')
plt.savefig('2_matriz_correlacion_Genero musical.png')
plt.close()

# 4. DIVISIÓN DE DATOS (80% ENTRENAMIENTO, 20% PRUEBAS)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. CREACIÓN DEL MODELO RED NEURONAL (ANN/MLP)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 6. ENTRENAMIENTO CON VALIDACIÓN
print("Iniciando el entrenamiento de la Red Neuronal...")
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Guardar gráfica de Entrenamiento vs Validación
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión de la Red Neuronal')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento (Error)')
plt.plot(history.history['val_loss'], label='Validación (Error)')
plt.title('Pérdida (Error) del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.tight_layout()
plt.savefig('3_historial_entrenamiento_Genero musical.png')
plt.close()

# 7. PRUEBA FINAL EN EL CONJUNTO INTOCABLE (20%)
print("\nRealizando predicciones en el subconjunto de pruebas (20%)...")
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

y_test_orig = encoder.inverse_transform(y_test)
y_pred_orig = encoder.inverse_transform(y_pred)

# 8. IMPRESIÓN DE RESULTADOS
results_df = pd.DataFrame({'Clase_Real': y_test_orig, 'Clase_Predicha': y_pred_orig})
results_df['Acierto'] = results_df['Clase_Real'] == results_df['Clase_Predicha']

print("\n" + "="*40)
print("   MUESTRA: CLASE REAL VS PREDICHA")
print("="*40)
print(results_df.head(20).to_string(index=False))

print("\n" + "="*40)
print("   ANÁLISIS DE ACIERTOS Y FALLOS ")
print("="*40)
aciertos = results_df['Acierto'].sum()
fallos = len(results_df) - aciertos
precision = accuracy_score(y_test, y_pred) * 100
error = (1 - accuracy_score(y_test, y_pred)) * 100

print(f"Total de datos de prueba evaluados: {len(results_df)}")
print(f"✔️ Cantidad de veces que ACIERTA: {aciertos}")
print(f"❌ Cantidad de veces que FALLA: {fallos}")
print("-" * 40)
print(f"📈 Porcentaje de Precisión (Aciertos): {precision:.2f}%")
print(f"📉 Porcentaje de Error: {error:.2f}%")

# Guardar Matriz de confusión
print("\nGuardando matriz de confusión...")
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test_orig, y_pred_orig), annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Desempeño en Pruebas Reales')
plt.xlabel('Clase Predicha por la Red Neuronal')
plt.ylabel('Clase Real de la Canción')
plt.savefig('4_matriz_confusion_Genero musical.png')
plt.close()

print("\nProceso terminado")