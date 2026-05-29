import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Oculta advertencias de TensorFlow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# 1. CARGA DE DATOS
print("Cargando dataset de Carros...")
df = pd.read_csv('CarPrice_Assignment.csv')

# 2. PREPROCESAMIENTO Y LIMPIEZA
# Eliminamos car_ID (no sirve para predecir) y CarName (tiene demasiados valores únicos de texto que causan ruido)
df = df.drop(columns=['car_ID', 'CarName'])

# Separamos el objetivo (precio) de las características
X = df.drop(columns=['price'])
y = df['price']

# Convertimos las columnas de texto (gasolina, puertas, tipo de auto) a columnas binarias (0 y 1)
X = pd.get_dummies(X, drop_first=True) 

# 3. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
print("Generando gráficas EDA...")

# A) Distribución de los Precios
plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True, color='blue', bins=30)
plt.title('Distribución de Precios de los Carros')
plt.xlabel('Precio ($)')
plt.ylabel('Cantidad de Carros')
plt.savefig('1_distribucion_precios.png')
plt.close()

# B) Matriz de Correlación (Top 10 variables que más afectan el precio)
plt.figure(figsize=(10, 8))
# Unimos X y y temporalmente solo para ver qué variables numéricas suben el precio
datos_numericos = df.select_dtypes(include=['float64', 'int64']) 
correlaciones = datos_numericos.corr()
top_features = correlaciones.nlargest(10, 'price')['price'].index
sns.heatmap(datos_numericos[top_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Top 10 Características Numéricas que impactan el Precio')
plt.savefig('2_matriz_correlacion_precios.png')
plt.close()

# 4. DIVISIÓN DE DATOS (80% ENTRENAMIENTO, 20% PRUEBAS)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. CREACIÓN DEL MODELO RED NEURONAL (ANN/MLP PARA REGRESIÓN)
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    # CAPA DE SALIDA: 1 sola neurona con activación 'linear' para arrojar el precio sin límite
    Dense(1, activation='linear') 
])

# Para regresión el error se mide en MSE (Error Cuadrático Medio) y MAE (Error Absoluto Medio)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 6. ENTRENAMIENTO CON VALIDACIÓN EXPLÍCITA (20%)
print("Iniciando el entrenamiento de la Red Neuronal (Calculando Precios)...")
# Aquí incluimos el apartado específico para validación usando validation_split=0.2
history = model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=150, batch_size=16, verbose=1)

# Guardar gráfica de Historial de Aprendizaje
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Entrenamiento (Error Cuadrático)')
plt.plot(history.history['val_loss'], label='Validación (Error Cuadrático)')
plt.title('Disminución del Error del Modelo durante el Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida (MSE)')
plt.legend()
plt.savefig('3_historial_entrenamiento.png')
plt.close()

# 7. PRUEBA FINAL EN EL CONJUNTO INTOCABLE (20%)
print("\nRealizando predicciones en el subconjunto de pruebas (20%)...")
y_pred = model.predict(X_test_scaled).flatten() # Aplanamos el resultado para compararlo fácil

# 8. ANÁLISIS DE DISCREPANCIAS Y ERRORES
# Armamos la tabla de valores Reales vs Predichos
results_df = pd.DataFrame({
    'Precio_Real_($)': y_test.values, 
    'Precio_Predicho_($)': np.round(y_pred, 2)
})
# Calculamos por cuántos dólares se equivocó
results_df['Diferencia_($)'] = np.round(abs(results_df['Precio_Real_($)'] - results_df['Precio_Predicho_($)']), 2)

print("\n" + "="*50)
print("   MUESTRA: PRECIO REAL VS PREDICHO")
print("="*50)
print(results_df.head(20).to_string(index=False))

# Métricas Globales del Error
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred) * 100

print("\n" + "="*50)
print("   ANÁLISIS GENERAL DE ERROR EN LA PREDICCIÓN")
print("="*50)
print(f"Total de autos evaluados en prueba: {len(results_df)}")
print(f"📉 Error Absoluto Medio (MAE): La red se equivoca por un promedio de ${mae:.2f} dólares por carro.")
print(f"📏 Error Cuadrático Medio de la Raíz (RMSE): Castigo a errores muy grandes: ${rmse:.2f} dólares.")
print(f"🎯 Precisión R2: El modelo es capaz de explicar el {r2:.2f}% de la variación del precio.")

# Gráfica de Discrepancias (Real vs Predicho)
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.7, color='purple')
# Línea ideal perfecta
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3, label='Predicción Perfecta')
plt.title('Discrepancia General: Precios Reales vs Predicciones')
plt.xlabel('Precio Real ($)')
plt.ylabel('Precio Predicho por la Red ($)')
plt.legend()
plt.grid(True)
plt.savefig('4_grafica_discrepancias.png')
plt.close()

print("\n¡Proceso terminado! Revisa el panel de Codespaces para ver las 4 imágenes de análisis generadas.")