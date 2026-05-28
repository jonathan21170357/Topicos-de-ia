import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo para guardar gráficos
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
import warnings
import os

warnings.filterwarnings('ignore')

# Configuración de gráficos
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

print("=" * 90)
print(" " * 25 + "REGRESIÓN CON REDES NEURONALES")
print(" " * 20 + "Predicción de Precios de Automóviles")
print("=" * 90)

# ============================================================================
# 1. CARGA DE DATOS
# ============================================================================
print("\n" + "=" * 90)
print("1. CARGA DE DATOS")
print("=" * 90)

try:
    df = pd.read_csv('CarPrice_Assignment.csv')
    print(f"✓ Dataset cargado exitosamente")
    print(f"  - Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
except FileNotFoundError:
    print("✗ Error: No se encuentra el archivo 'CarPrice_Assignment.csv'")
    exit(1)

# ============================================================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS
# ============================================================================
print("\n" + "=" * 90)
print("2. ANÁLISIS EXPLORATORIO DE DATOS")
print("=" * 90)

print(f"\n▶ Estadísticas de la variable objetivo (PRECIO):")
print(f"  - Mínimo:     ${df['price'].min():>10,.2f}")
print(f"  - Máximo:     ${df['price'].max():>10,.2f}")
print(f"  - Media:      ${df['price'].mean():>10,.2f}")
print(f"  - Mediana:    ${df['price'].median():>10,.2f}")
print(f"  - Desv. Est.: ${df['price'].std():>10,.2f}")

# Correlaciones
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr()
corr_with_price = corr_matrix['price'].sort_values(ascending=False)
print(f"\n▶ Correlaciones con el precio (Top 10):")
for i, (col, corr) in enumerate(corr_with_price.items()):
    if col != 'price' and i < 11:
        print(f"      {i+1:2}. {col:<20} : {corr:>8.4f}")

# ============================================================================
# 3. PREPROCESAMIENTO DE DATOS
# ============================================================================
print("\n" + "=" * 90)
print("3. PREPROCESAMIENTO DE DATOS")
print("=" * 90)

# Selección de características
selected_features = [
    'enginesize', 'curbweight', 'horsepower', 'carwidth', 'carlength',
    'wheelbase', 'boreratio', 'compressionratio', 'citympg', 'highwaympg'
]

categorical_features = ['fueltype', 'aspiration', 'carbody', 'drivewheel']

print(f"\n▶ Características seleccionadas:")
print(f"  - Numéricas ({len(selected_features)}): {', '.join(selected_features)}")
print(f"  - Categóricas ({len(categorical_features)}): {', '.join(categorical_features)}")

# Codificar variables categóricas
df_processed = df.copy()
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le
    print(f"      {col:<15} : {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Características finales
final_features = selected_features + [col + '_encoded' for col in categorical_features]
X = df_processed[final_features].values
y = df_processed['price'].values

print(f"\n✓ Matriz de características X: {X.shape}")
print(f"✓ Vector objetivo y: {y.shape}")

# ============================================================================
# 4. DIVISIÓN TRAIN-TEST (80%-20%)
# ============================================================================
print("\n" + "=" * 90)
print("4. DIVISIÓN DE DATOS (ENTRENAMIENTO 80% - PRUEBA 20%)")
print("=" * 90)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n▶ Conjunto de ENTRENAMIENTO: {X_train.shape[0]} muestras (80%)")
print(f"▶ Conjunto de PRUEBA: {X_test.shape[0]} muestras (20%)")

# Escalado de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Escalado de características completado")

# ============================================================================
# 5. ENTRENAMIENTO DE REDES NEURONALES
# ============================================================================
print("\n" + "=" * 90)
print("5. ENTRENAMIENTO DE REDES NEURONALES")
print("=" * 90)

# Definición de arquitecturas
architectures = [
    {"name": "Arquitectura A", "hidden_layers": (50,), "alpha": 0.0001},
    {"name": "Arquitectura B", "hidden_layers": (100,), "alpha": 0.0001},
    {"name": "Arquitectura C", "hidden_layers": (100, 50), "alpha": 0.0001},
    {"name": "Arquitectura D", "hidden_layers": (100, 100, 50), "alpha": 0.0001},
    {"name": "Arquitectura E", "hidden_layers": (200, 100), "alpha": 0.0001},
    {"name": "Arquitectura F", "hidden_layers": (50,), "alpha": 0.001},
    {"name": "Arquitectura G", "hidden_layers": (100, 50), "alpha": 0.001},
    {"name": "Arquitectura H", "hidden_layers": (100, 80, 60, 40), "alpha": 0.0001},
]

results = []
best_test_r2 = -np.inf
best_model = None
best_model_info = None
best_test_mae = None

print("\n▶ Entrenando múltiples arquitecturas...\n")

for arch in architectures:
    print(f"  📊 {arch['name']}: Capas {arch['hidden_layers']}, Alpha={arch['alpha']}")
    
    mlp = MLPRegressor(
        hidden_layer_sizes=arch['hidden_layers'],
        activation='relu',
        solver='adam',
        alpha=arch['alpha'],
        batch_size=32,
        learning_rate='adaptive',
        max_iter=2000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False
    )
    
    mlp.fit(X_train_scaled, y_train)
    
    y_train_pred = mlp.predict(X_train_scaled)
    y_test_pred = mlp.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    results.append({
        'name': arch['name'],
        'hidden_layers': arch['hidden_layers'],
        'alpha': arch['alpha'],
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'n_iter': mlp.n_iter_
    })
    
    print(f"     Train → R²: {train_r2:.4f}, MAE: ${train_mae:,.0f}")
    print(f"     Test  → R²: {test_r2:.4f}, MAE: ${test_mae:,.0f}")
    print(f"     Iteraciones: {mlp.n_iter_}\n")
    
    if test_r2 > best_test_r2:
        best_test_r2 = test_r2
        best_model = mlp
        best_model_info = arch
        best_test_mae = test_mae
        best_y_test_pred = y_test_pred

# Mostrar resumen
print("▶ RESUMEN DE RESULTADOS:")
print("-" * 80)
print(f"{'Arquitectura':<15} {'Train R²':>10} {'Test R²':>10} {'Train MAE':>12} {'Test MAE':>12}")
print("-" * 80)
for r in results:
    print(f"{r['name']:<15} {r['train_r2']:>10.4f} {r['test_r2']:>10.4f} ${r['train_mae']:>10,.0f} ${r['test_mae']:>10,.0f}")
print("-" * 80)

print(f"\n⭐ MEJOR MODELO: {best_model_info['name']}")
print(f"   - Capas: {best_model_info['hidden_layers']}")
print(f"   - Alpha: {best_model_info['alpha']}")
print(f"   - Test R²: {best_test_r2:.4f}")
print(f"   - Test MAE: ${best_test_mae:,.2f}")

# ============================================================================
# 6. VALORES REALES VS PREDICHOS
# ============================================================================
print("\n" + "=" * 90)
print("6. VALORES REALES VS PREDICHOS (CONJUNTO DE PRUEBA - 20%)")
print("=" * 90)

y_test_pred_final = best_model.predict(X_test_scaled)

comparison_df = pd.DataFrame({
    'ID': range(1, len(y_test) + 1),
    'Precio_Real': y_test,
    'Precio_Predicho': y_test_pred_final,
    'Error_Absoluto': np.abs(y_test - y_test_pred_final),
    'Error_Relativo_%': (np.abs(y_test - y_test_pred_final) / y_test) * 100
})

print("\n" + "=" * 95)
print(f"{'ID':^6} {'Precio Real ($)':^18} {'Precio Predicho ($)':^20} {'Error Absoluto ($)':^18} {'Error Relativo (%)':^18}")
print("=" * 95)

# Mostrar todas las filas
for _, row in comparison_df.iterrows():
    print(f"{int(row['ID']):^6} ${row['Precio_Real']:>15,.2f} ${row['Precio_Predicho']:>17,.2f} ${row['Error_Absoluto']:>15,.2f} {row['Error_Relativo_%']:>15.2f}%")

print("=" * 95)

# ============================================================================
# 7. ANÁLISIS DEL ERROR GENERAL
# ============================================================================
print("\n" + "=" * 90)
print("7. ANÁLISIS DEL ERROR GENERAL")
print("=" * 90)

mae = comparison_df['Error_Absoluto'].mean()
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred_final))
mape = comparison_df['Error_Relativo_%'].mean()
r2 = r2_score(y_test, y_test_pred_final)

print(f"\n▶ MÉTRICAS DE RENDIMIENTO:")
print(f"  Error Medio Absoluto (MAE):     ${mae:>15,.2f}")
print(f"  Raíz Error Cuadrático Medio:    ${rmse:>15,.2f}")
print(f"  Error Medio Absoluto Porcentual: {mape:>15.2f}%")
print(f"  Coeficiente de Determinación:    {r2:>15.4f}")

print(f"\n▶ ESTADÍSTICAS DE ERROR:")
print(f"  Mediana del Error Absoluto:     ${comparison_df['Error_Absoluto'].median():>15,.2f}")
print(f"  Desviación Estándar del Error:  ${comparison_df['Error_Absoluto'].std():>15,.2f}")
print(f"  Máximo Error Absoluto:          ${comparison_df['Error_Absoluto'].max():>15,.2f}")
print(f"  Mínimo Error Absoluto:          ${comparison_df['Error_Absoluto'].min():>15,.2f}")

print(f"\n▶ PRECISIÓN DEL MODELO:")
exactitud_10 = (comparison_df['Error_Relativo_%'] < 10).sum()
exactitud_20 = (comparison_df['Error_Relativo_%'] < 20).sum()
exactitud_30 = (comparison_df['Error_Relativo_%'] < 30).sum()
print(f"  Predicciones con error < 10%:   {exactitud_10}/{len(comparison_df)} ({exactitud_10/len(comparison_df)*100:.1f}%)")
print(f"  Predicciones con error < 20%:   {exactitud_20}/{len(comparison_df)} ({exactitud_20/len(comparison_df)*100:.1f}%)")
print(f"  Predicciones con error < 30%:   {exactitud_30}/{len(comparison_df)} ({exactitud_30/len(comparison_df)*100:.1f}%)")

# ============================================================================
# 8. GENERAR GRÁFICOS (GUARDAR COMO PNG)
# ============================================================================
print("\n" + "=" * 90)
print("8. GENERANDO GRÁFICOS DE DISCREPANCIAS")
print("=" * 90)

# Crear directorio para gráficos si no existe
os.makedirs('graficos', exist_ok=True)

# Gráfico 1: Dispersión Real vs Predicho
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_test_pred_final, alpha=0.6, edgecolors='black', linewidth=0.5, c='steelblue', s=80)
min_val = min(y_test.min(), y_test_pred_final.min())
max_val = max(y_test.max(), y_test_pred_final.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
plt.xlabel('Precio Real ($)', fontsize=12)
plt.ylabel('Precio Predicho ($)', fontsize=12)
plt.title(f'Valores Reales vs Predichos (R² = {r2:.4f})', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graficos/1_real_vs_predicho.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Gráfico 1 guardado: 'graficos/1_real_vs_predicho.png'")

# Gráfico 2: Histograma de errores absolutos
plt.figure(figsize=(10, 8))
plt.hist(comparison_df['Error_Absoluto'], bins=20, edgecolor='black', alpha=0.7, color='coral')
plt.axvline(mae, color='red', linestyle='--', linewidth=2, label=f'MAE: ${mae:,.0f}')
plt.axvline(comparison_df['Error_Absoluto'].median(), color='green', linestyle='--', linewidth=2, 
            label=f'Mediana: ${comparison_df["Error_Absoluto"].median():,.0f}')
plt.xlabel('Error Absoluto ($)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.title('Distribución de Errores Absolutos', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graficos/2_histograma_errores_absolutos.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Gráfico 2 guardado: 'graficos/2_histograma_errores_absolutos.png'")

# Gráfico 3: Histograma de errores relativos
plt.figure(figsize=(10, 8))
plt.hist(comparison_df['Error_Relativo_%'], bins=20, edgecolor='black', alpha=0.7, color='lightgreen')
plt.axvline(mape, color='red', linestyle='--', linewidth=2, label=f'MAPE: {mape:.1f}%')
plt.xlabel('Error Relativo (%)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.title('Distribución de Errores Relativos', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graficos/3_histograma_errores_relativos.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Gráfico 3 guardado: 'graficos/3_histograma_errores_relativos.png'")

# Gráfico 4: Residuales
plt.figure(figsize=(10, 8))
residuals = y_test - y_test_pred_final
plt.scatter(y_test_pred_final, residuals, alpha=0.6, edgecolors='black', linewidth=0.5, s=80, c='purple')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Precio Predicho ($)', fontsize=12)
plt.ylabel('Residual (Real - Predicho)', fontsize=12)
plt.title('Análisis de Residuales', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graficos/4_residuales.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Gráfico 4 guardado: 'graficos/4_residuales.png'")

# Gráfico 5: Comparación ordenada
plt.figure(figsize=(12, 6))
sorted_idx = np.argsort(y_test)
plt.plot(range(len(y_test)), y_test[sorted_idx], 'b-', label='Precio Real', linewidth=2, alpha=0.8)
plt.plot(range(len(y_test)), y_test_pred_final[sorted_idx], 'r--', label='Precio Predicho', linewidth=2, alpha=0.8)
plt.fill_between(range(len(y_test)), y_test[sorted_idx], y_test_pred_final[sorted_idx], alpha=0.2, color='gray')
plt.xlabel('Muestra (ordenada por precio)', fontsize=12)
plt.ylabel('Precio ($)', fontsize=12)
plt.title('Comparación de Predicciones (Ordenadas por Precio)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('graficos/5_comparacion_ordenada.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Gráfico 5 guardado: 'graficos/5_comparacion_ordenada.png'")

# Gráfico 6: Boxplot por rango de precio (CORREGIDO)
plt.figure(figsize=(12, 6))
# Crear los rangos de precio como categorías
price_ranges = pd.cut(y_test, bins=[0, 8000, 12000, 18000, 25000, 50000], 
                      labels=['< $8k', '$8k-12k', '$12k-18k', '$18k-25k', '> $25k'])

# Preparar los datos para el boxplot
data_to_plot = []
for label in ['< $8k', '$8k-12k', '$12k-18k', '$18k-25k', '> $25k']:
    mask = price_ranges == label
    data_to_plot.append(comparison_df['Error_Relativo_%'][mask].values)

# Crear el boxplot
bp = plt.boxplot(data_to_plot, labels=['< $8k', '$8k-12k', '$12k-18k', '$18k-25k', '> $25k'], 
                 patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_alpha(0.7)
for median in bp['medians']:
    median.set_color('red')
    median.set_linewidth(2)

plt.xlabel('Rango de Precio', fontsize=12)
plt.ylabel('Error Relativo (%)', fontsize=12)
plt.title('Discrepancias por Rango de Precio', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('graficos/6_boxplot_por_rango.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Gráfico 6 guardado: 'graficos/6_boxplot_por_rango.png'")

# Gráfico 7: Comparación de arquitecturas - R²
plt.figure(figsize=(10, 6))
names = [r['name'] for r in results]
test_r2_values = [r['test_r2'] for r in results]
colors = ['green' if r['test_r2'] == best_test_r2 else 'steelblue' for r in results]
plt.barh(names, test_r2_values, color=colors, alpha=0.7)
plt.axvline(best_test_r2, color='red', linestyle='--', linewidth=2, label=f'Mejor R²: {best_test_r2:.4f}')
plt.xlabel('R² Score (Conjunto de Prueba)', fontsize=12)
plt.title('Comparación de Arquitecturas - R²', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('graficos/7_comparacion_r2.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Gráfico 7 guardado: 'graficos/7_comparacion_r2.png'")

# Gráfico 8: Comparación de arquitecturas - MAE
plt.figure(figsize=(10, 6))
test_mae_values = [r['test_mae'] for r in results]
best_mae = min(test_mae_values)
colors = ['green' if r['test_mae'] == best_mae else 'coral' for r in results]
plt.barh(names, test_mae_values, color=colors, alpha=0.7)
plt.axvline(best_mae, color='red', linestyle='--', linewidth=2, label=f'Mejor MAE: ${best_mae:,.0f}')
plt.xlabel('MAE (Conjunto de Prueba) - Dólares', fontsize=12)
plt.title('Comparación de Arquitecturas - MAE', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('graficos/8_comparacion_mae.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Gráfico 8 guardado: 'graficos/8_comparacion_mae.png'")

# Gráfico 9: Gráfico combinado grande (para resumen)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('ANÁLISIS COMPLETO DE PREDICCIÓN DE PRECIOS', fontsize=16, fontweight='bold')

# Subplot 1: Real vs Predicho
axes[0, 0].scatter(y_test, y_test_pred_final, alpha=0.6, edgecolors='black', c='steelblue', s=60)
axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
axes[0, 0].set_xlabel('Precio Real ($)')
axes[0, 0].set_ylabel('Precio Predicho ($)')
axes[0, 0].set_title(f'Real vs Predicho (R² = {r2:.4f})')
axes[0, 0].grid(True, alpha=0.3)

# Subplot 2: Histograma errores absolutos
axes[0, 1].hist(comparison_df['Error_Absoluto'], bins=20, edgecolor='black', alpha=0.7, color='coral')
axes[0, 1].axvline(mae, color='red', linestyle='--', lw=2, label=f'MAE: ${mae:,.0f}')
axes[0, 1].set_xlabel('Error Absoluto ($)')
axes[0, 1].set_ylabel('Frecuencia')
axes[0, 1].set_title('Distribución de Errores')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Subplot 3: Residuales
axes[1, 0].scatter(y_test_pred_final, residuals, alpha=0.6, edgecolors='black', c='purple', s=60)
axes[1, 0].axhline(y=0, color='red', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Precio Predicho ($)')
axes[1, 0].set_ylabel('Residual')
axes[1, 0].set_title('Análisis de Residuales')
axes[1, 0].grid(True, alpha=0.3)

# Subplot 4: Comparación arquitecturas
axes[1, 1].barh(names, test_r2_values, color=colors, alpha=0.7)
axes[1, 1].axvline(best_test_r2, color='red', linestyle='--', lw=2, label=f'Mejor: {best_test_r2:.4f}')
axes[1, 1].set_xlabel('R² Score')
axes[1, 1].set_title('Rendimiento por Arquitectura')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('graficos/9_resumen_completo.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Gráfico 9 guardado: 'graficos/9_resumen_completo.png'")

# ============================================================================
# 9. GUARDAR RESULTADOS EN CSV
# ============================================================================
print("\n" + "=" * 90)
print("9. GUARDANDO RESULTADOS")
print("=" * 90)

comparison_df.to_csv('predicciones_precios_carros.csv', index=False)
print("✓ Archivo CSV guardado: 'predicciones_precios_carros.csv'")

# Guardar métricas en un archivo de texto
with open('resultados_metricas.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("RESULTADOS DE PREDICCIÓN DE PRECIOS DE AUTOMÓVILES\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"MEJOR MODELO: {best_model_info['name']}\n")
    f.write(f"Capas ocultas: {best_model_info['hidden_layers']}\n")
    f.write(f"Alpha: {best_model_info['alpha']}\n\n")
    f.write("MÉTRICAS DE RENDIMIENTO:\n")
    f.write(f"  R² Score: {r2:.4f}\n")
    f.write(f"  MAE: ${mae:,.2f}\n")
    f.write(f"  RMSE: ${rmse:,.2f}\n")
    f.write(f"  MAPE: {mape:.2f}%\n\n")
    f.write("ESTADÍSTICAS DE ERROR:\n")
    f.write(f"  Mediana Error Absoluto: ${comparison_df['Error_Absoluto'].median():,.2f}\n")
    f.write(f"  Máximo Error Absoluto: ${comparison_df['Error_Absoluto'].max():,.2f}\n")
    f.write(f"  Mínimo Error Absoluto: ${comparison_df['Error_Absoluto'].min():,.2f}\n\n")
    f.write("PRECISIÓN DEL MODELO:\n")
    f.write(f"  Predicciones con error < 10%: {exactitud_10}/{len(comparison_df)} ({exactitud_10/len(comparison_df)*100:.1f}%)\n")
    f.write(f"  Predicciones con error < 20%: {exactitud_20}/{len(comparison_df)} ({exactitud_20/len(comparison_df)*100:.1f}%)\n")
    f.write(f"  Predicciones con error < 30%: {exactitud_30}/{len(comparison_df)} ({exactitud_30/len(comparison_df)*100:.1f}%)\n")

print("✓ Archivo de métricas guardado: 'resultados_metricas.txt'")

# ============================================================================
# 10. CONCLUSIONES FINALES
# ============================================================================
print("\n" + "=" * 90)
print("10. CONCLUSIONES DEL ANÁLISIS")
print("=" * 90)

print(f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RESUMEN DEL MODELO                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Mejor Arquitectura   : {best_model_info['name']}                                              │
│  Capas Ocultas        : {str(best_model_info['hidden_layers']):<50} │
│  Regularización (α)   : {best_model_info['alpha']}                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                           MÉTRICAS DE RENDIMIENTO                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  R² (Prueba)          : {best_test_r2:.4f} ({best_test_r2*100:.1f}%)                                    │
│  MAE (Prueba)         : ${best_test_mae:,.2f}                                    │
│  MAPE (Prueba)        : {mape:.2f}%                                             │
│  RMSE (Prueba)        : ${rmse:,.2f}                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                           ARCHIVOS GENERADOS                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  📁 Carpeta 'graficos/': 9 gráficos PNG                                     │
│  📄 predicciones_precios_carros.csv                                         │
│  📄 resultados_metricas.txt                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
""")

print("\n✅ PROGRAMA COMPLETADO EXITOSAMENTE")
print(f"   - Revisa la carpeta 'graficos/' para ver los gráficos generados")
print("   - Los resultados completos están en 'predicciones_precios_carros.csv'")
print("=" * 90)