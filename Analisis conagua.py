import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Generación del Dataset Simulado (Parámetros CONAGUA)
np.random.seed(42)
n_samples = 200

data = {
    'ID_Muestra': [f'M-{i}' for i in range(1, n_samples + 1)],
    'pH': np.random.normal(7.2, 0.5, n_samples),
    'Turbidez_NTU': np.random.uniform(0.5, 10.0, n_samples),
    'Oxigeno_Disuelto_mgL': np.random.normal(6.5, 1.2, n_samples),
    'Conductividad_uS': np.random.normal(500, 150, n_samples),
    'Dureza_mgL': np.random.normal(120, 30, n_samples),
    'Nitratos_mgL': np.random.exponential(5, n_samples),
    'Sulfatos_mgL': np.random.normal(40, 10, n_samples),
    'Cloruros_mgL': np.random.uniform(10, 100, n_samples),
    'Temperatura_Agua_C': np.random.normal(22, 3, n_samples),
    'Categoria_Calidad': np.random.choice(['Buena', 'Regular', 'Mala'], n_samples, p=[0.6, 0.3, 0.1])
}
df = pd.DataFrame(data)

# Inyección de Outliers (Picos de contaminación o fallos de sensor)
df.loc[10:15, 'pH'] = 14.0 
df.loc[45:50, 'Conductividad_uS'] = 3500.0

# Inyección de Datos Faltantes (NaN - Simulando pérdida de telemetría)
cols_con_nan = ['Turbidez_NTU', 'Oxigeno_Disuelto_mgL', 'Nitratos_mgL', 'Categoria_Calidad']
for col in cols_con_nan:
    indices_nan = np.random.choice(df.index, size=15, replace=False)
    df.loc[indices_nan, col] = np.nan

# 2. Análisis Estadístico Inicial
print("--- ESTADÍSTICAS DESCRIPTIVAS INICIALES ---")
variables_numericas = df.select_dtypes(include=[np.number]).columns

for col in variables_numericas:
    media = df[col].mean()
    mediana = df[col].median()
    varianza = df[col].var()
    std = df[col].std()
    print(f"\n[{col}]")
    print(f"Media: {media:.2f} | Mediana: {mediana:.2f} | Varianza: {varianza:.2f} | Desv. Est: {std:.2f}")

# Moda para variables categóricas
moda_categoria = df['Categoria_Calidad'].mode()[0]
print(f"\n[Categoria_Calidad]")
print(f"Moda: {moda_categoria}")

# 3. Tratamiento de Datos (Imputación y Outliers)
df_tratado = df.copy()

# Imputación de variables numéricas usando la Mediana (robusta ante outliers)
for col in ['Turbidez_NTU', 'Oxigeno_Disuelto_mgL', 'Nitratos_mgL']:
    mediana_col = df_tratado[col].median()
    df_tratado[col] = df_tratado[col].fillna(mediana_col)

# Imputación de variable categórica usando la Moda
df_tratado['Categoria_Calidad'] = df_tratado['Categoria_Calidad'].fillna(moda_categoria)

# Acotamiento de Outliers (Clipping) usando el Rango Intercuartílico (IQR) en pH y Conductividad
for col in ['pH', 'Conductividad_uS']:
    Q1 = df_tratado[col].quantile(0.25)
    Q3 = df_tratado[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    df_tratado[col] = np.clip(df_tratado[col], limite_inferior, limite_superior)

# 4. Escalamiento de Variables
scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()

# MinMax (0 a 1) para Turbidez y Cloruros (distribuciones uniformes o no normales)
cols_minmax = ['Turbidez_NTU', 'Cloruros_mgL']
df_tratado[cols_minmax] = scaler_minmax.fit_transform(df_tratado[cols_minmax])

# StandardScaler (Z-score) para variables con distribución normal
cols_standard = ['pH', 'Oxigeno_Disuelto_mgL', 'Conductividad_uS', 'Dureza_mgL', 'Temperatura_Agua_C']
df_tratado[cols_standard] = scaler_standard.fit_transform(df_tratado[cols_standard])

# 5. Visualización de Resultados
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('Impacto del Tratamiento de Datos en Parámetros Hídricos', fontsize=18, weight='bold')

# Histograma con KDE: Oxígeno Disuelto ANTES y DESPUÉS de imputar
sns.histplot(df['Oxigeno_Disuelto_mgL'].dropna(), ax=axes[0,0], color='red', kde=True)
axes[0,0].set_title('ANTES: Oxígeno Disuelto (Con valores faltantes)')
sns.histplot(df_tratado['Oxigeno_Disuelto_mgL'], ax=axes[0,1], color='green', kde=True)
axes[0,1].set_title('DESPUÉS: Oxígeno Disuelto (Imputado y Estandarizado)')

# Boxplot: Conductividad ANTES y DESPUÉS de outliers/escalamiento
sns.boxplot(x=df['Conductividad_uS'], ax=axes[1,0], color='lightcoral')
axes[1,0].set_title('ANTES: Conductividad con Outliers Extremos')
sns.boxplot(x=df_tratado['Conductividad_uS'], ax=axes[1,1], color='lightgreen')
axes[1,1].set_title('DESPUÉS: Conductividad (Outliers acotados y Escalamiento Z)')

# Scatterplot: Comparación de escalas
axes[2,0].scatter(df['Turbidez_NTU'], df['Temperatura_Agua_C'], color='purple', alpha=0.6)
axes[2,0].set_title('ANTES: Turbidez vs Temperatura (Escalas Incompatibles)')
axes[2,0].set_xlabel('Turbidez (NTU)')
axes[2,0].set_ylabel('Temperatura (°C)')

axes[2,1].scatter(df_tratado['Turbidez_NTU'], df_tratado['Temperatura_Agua_C'], color='teal', alpha=0.6)
axes[2,1].set_title('DESPUÉS: Variables Escaladas (Listas para Modelos Predictivos)')
axes[2,1].set_xlabel('Turbidez (MinMax: 0 a 1)')
axes[2,1].set_ylabel('Temperatura (Z-Score)')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()
