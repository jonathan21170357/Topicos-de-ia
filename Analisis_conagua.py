import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os

# ==========================================
# 1. CARGA DE DATOS Y VALIDACIÓN
# ==========================================
archivo_csv = 'Calidad_Agua_subterranea_2012_2024.csv'

if not os.path.exists(archivo_csv):
    print(f"❌ ERROR: No se encontró el archivo '{archivo_csv}'.")
    print(f"📁 Directorio actual: {os.getcwd()}")
    exit()

# Cargar el Dataset Real (utf-8 previene errores con los acentos de la palabra SEMÁFORO)
df = pd.read_csv(archivo_csv, encoding='utf-8', low_memory=False)

# ==========================================
# 2. CONFIGURACIÓN DE LAS 10 VARIABLES CONAGUA
# ==========================================
# Variable categórica
col_categoria = 'SEMÁFORO' 

# Las 9 variables numéricas seleccionadas que tienen datos faltantes
cols_numericas = [
    'ALC_mg/L',             # Alcalinidad
    'CONDUCT_mS/cm',        # Conductividad
    'SDT_mg/L',             # Sólidos Disueltos Totales
    'FLUORUROS_mg/L',       # Fluoruros
    'DUR_mg/L',             # Dureza
    'COLI_FEC_NMP/100_mL',  # Coliformes Fecales
    'N_NO3_mg/L',           # Nitratos
    'AS_TOT_mg/L',          # Arsénico
    'FE_TOT_mg/L'           # Hierro
]

# ==========================================
# LIMPIEZA VITAL (Versión Forzada)
# ==========================================
# Los datos de CONAGUA tienen símbolos como '<' y comas en los números.
# Obligamos a todas las variables numéricas a limpiarse y convertirse en float.
for col in cols_numericas:
    # 1. Forzamos la columna a texto para poder usar .str.replace
    df[col] = df[col].astype(str)
    # 2. Reemplazamos símbolos problemáticos
    df[col] = df[col].str.replace('<', '').str.replace('>', '').str.replace(',', '.')
    # 3. Forzamos la conversión a numérico. 'coerce' convierte cualquier texto no válido (como "ND") en NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
# Definir el tratamiento que se le dará a cada variable
cols_imputar_mediana = cols_numericas # A todas se les aplica mediana
cols_clipping = ['ALC_mg/L', 'CONDUCT_mS/cm']
cols_minmax = ['COLI_FEC_NMP/100_mL', 'FLUORUROS_mg/L']
cols_standard = ['ALC_mg/L', 'CONDUCT_mS/cm', 'SDT_mg/L', 'DUR_mg/L', 'N_NO3_mg/L', 'AS_TOT_mg/L', 'FE_TOT_mg/L']

# Variables para las gráficas visuales
col_grafica_1 = 'ALC_mg/L'
col_grafica_2 = 'CONDUCT_mS/cm'
col_scatter_x = 'SDT_mg/L'
col_scatter_y = 'CONDUCT_mS/cm'

# ==========================================
# 3. ANÁLISIS ESTADÍSTICO INICIAL
# ==========================================
print("--- ESTADÍSTICAS DESCRIPTIVAS INICIALES (Variables con NaN) ---")
for col in cols_numericas:
    media = df[col].mean()
    mediana = df[col].median()
    varianza = df[col].var()
    std = df[col].std()
    nulos = df[col].isna().sum()
    print(f"\n[{col}] -> (Datos Faltantes: {nulos})")
    print(f"Media: {media:.4f} | Mediana: {mediana:.4f} | Varianza: {varianza:.4f} | Desv. Est: {std:.4f}")

# Moda segura para variable categórica
modas = df[col_categoria].dropna().mode()
moda_categoria = modas[0] if not modas.empty else "Desconocida"
print(f"\n[{col_categoria}]")
print(f"Moda: {moda_categoria}")

# ==========================================
# 4. TRATAMIENTO DE DATOS (Imputación y Outliers)
# ==========================================
df_tratado = df.copy()

# Imputación de numéricas por Mediana
for col in cols_imputar_mediana:
    mediana_col = df_tratado[col].median()
    df_tratado[col] = df_tratado[col].fillna(mediana_col)

# Imputación de categórica por Moda (Si existieran faltantes)
if moda_categoria != "Desconocida":
    df_tratado[col_categoria] = df_tratado[col_categoria].fillna(moda_categoria)

# Acotamiento de Outliers (Clipping IQR) para proteger la integridad matemática
for col in cols_clipping:
    Q1 = df_tratado[col].quantile(0.25)
    Q3 = df_tratado[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    df_tratado[col] = np.clip(df_tratado[col], limite_inferior, limite_superior)

# ==========================================
# 5. ESCALAMIENTO DE VARIABLES
# ==========================================
scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()

# MinMax (0 a 1) para coliformes y fluoruros
df_tratado[cols_minmax] = scaler_minmax.fit_transform(df_tratado[cols_minmax])

# StandardScaler (Z-score) para el resto de químicos
df_tratado[cols_standard] = scaler_standard.fit_transform(df_tratado[cols_standard])

# ==========================================
# 6. VISUALIZACIÓN DE RESULTADOS
# ==========================================
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('Impacto del Tratamiento de Datos en Parámetros de CONAGUA', fontsize=18, weight='bold')

# Fila 1: Histograma ANTES y DESPUÉS
sns.histplot(df[col_grafica_1].dropna(), ax=axes[0,0], color='red', kde=True)
axes[0,0].set_title(f'ANTES: {col_grafica_1} (Con Valores Faltantes)')

sns.histplot(df_tratado[col_grafica_1], ax=axes[0,1], color='green', kde=True)
axes[0,1].set_title(f'DESPUÉS: {col_grafica_1} (Outliers Acotados y Escalamiento Z)')

# Fila 2: Boxplot ANTES y DESPUÉS
sns.boxplot(x=df[col_grafica_2].dropna(), ax=axes[1,0], color='lightcoral')
axes[1,0].set_title(f'ANTES: {col_grafica_2} con Outliers Extremos')

sns.boxplot(x=df_tratado[col_grafica_2], ax=axes[1,1], color='lightgreen')
axes[1,1].set_title(f'DESPUÉS: {col_grafica_2} (Outliers acotados y Escalamiento Z)')

# Fila 3: Scatterplot ANTES y DESPUÉS
axes[2,0].scatter(df[col_scatter_x], df[col_scatter_y], color='purple', alpha=0.6)
axes[2,0].set_title('ANTES: Dispersión (Escalas Incompatibles)')
axes[2,0].set_xlabel(f'{col_scatter_x} Crudo')
axes[2,0].set_ylabel(f'{col_scatter_y} Crudo')

axes[2,1].scatter(df_tratado[col_scatter_x], df_tratado[col_scatter_y], color='teal', alpha=0.6)
axes[2,1].set_title('DESPUÉS: Variables Normalizadas (Z-Score)')
axes[2,1].set_xlabel(f'{col_scatter_x} (Z-Score)')
axes[2,1].set_ylabel(f'{col_scatter_y} (Z-Score)')

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()