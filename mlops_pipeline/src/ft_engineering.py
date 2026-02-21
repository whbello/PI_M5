# Feature Engineering - Avance 2

## Objetivo
# Transformaciones a Aplicar (del EDA)
#1. **Imputación de nulos**
#2. **Creación de features derivados**
#3. **Encoding de variables categóricas**
#4. **Scaling de variables numéricas**
#5. **División train/test estratificada**
#6. **Balanceo de clases con SMOTE**

# ============================================
# 1. CONFIGURACIÓN E IMPORTS
# ============================================

# Manipulación de datos
import pandas as pd
import numpy as np
from pathlib import Path
import sys

print(sys.executable)

# Preprocesamiento
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Balanceo de clases
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Feature engineering avanzado
from feature_engine.imputation import MeanMedianImputer, CategoricalImputer
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder as FEOneHotEncoder
from feature_engine.outliers import OutlierTrimmer

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Guardar objetos
import joblib

# Configuración
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')

print("✓ Librerías importadas correctamente")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")

# ============================================
# 2. DEFINIR RUTAS
# ============================================

# src/ft_engineering.py → subir 2 niveles → mlops_pipeline
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Crear carpetas necesarias
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Rutas del proyecto:")
print(f"  PROJECT_ROOT:   {PROJECT_ROOT}")
print(f"  DATA_INTERIM:   {DATA_INTERIM}")
print(f"  DATA_PROCESSED: {DATA_PROCESSED}")
print(f"  MODELS_DIR:     {MODELS_DIR}")

# Validar carpeta crítica
if not DATA_INTERIM.exists():
    raise FileNotFoundError(
        f"No existe la carpeta data/interim en: {DATA_INTERIM}"
    )

# ============================================
# 3. CARGAR DATOS LIMPIOS
# ============================================

# Cargar desde interim
pkl_path = DATA_INTERIM / "creditos_limpio.pkl"
csv_path = DATA_INTERIM / "creditos_limpio.csv"

if pkl_path.exists():
    df = pd.read_pickle(pkl_path)
    print(f"✓ Datos cargados desde: {pkl_path.name}")

elif csv_path.exists():
    df = pd.read_csv(csv_path)
    print(f"✓ Datos cargados desde: {csv_path.name}")

else:
    raise FileNotFoundError(
        "No se encontró 'creditos_limpio.pkl' ni 'creditos_limpio.csv' "
        f"en {DATA_INTERIM}"
    )

print(f"\nDimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"Período: {df['fecha_prestamo'].min()} a {df['fecha_prestamo'].max()}")

# Mostrar primeras filas
df.head()

# ============================================
# 4. PREPARACIÓN: SEPARAR FEATURES Y TARGET
# ============================================

# Separar target
TARGET = 'Pago_atiempo'

if TARGET not in df.columns:
    raise ValueError(f"Target '{TARGET}' no encontrado en el dataset")

y = df[TARGET].copy()
X = df.drop(columns=[TARGET])

print(f"Features (X): {X.shape}")
print(f"Target (y):   {y.shape}")

# Verificar desbalance
print(f"\nDistribución del target:")
print(y.value_counts())
print(f"\nPorcentajes:")
print(y.value_counts(normalize=True) * 100)

# ============================================
# 5. IDENTIFICAR TIPOS DE VARIABLES
# ============================================

# Variables a eliminar (identificadas en EDA)
VARS_TO_DROP = [
    'fecha_prestamo',  # Temporal, no usamos directamente
    'cuota_pactada',   # Multicolinealidad con capital_prestado y plazo_meses
]

# Eliminar variables
X = X.drop(columns=[col for col in VARS_TO_DROP if col in X.columns])

# Variables numéricas
NUMERIC_FEATURES = [
    'capital_prestado',
    'plazo_meses',
    'edad_cliente',
    'salario_cliente',
    'total_otros_prestamos',
    'puntaje',
    'puntaje_datacredito',
    'cant_creditosvigentes',
    'huella_consulta',
    'saldo_mora',
    'saldo_total',
    'saldo_principal',
    'saldo_mora_codeudor',
    'creditos_sectorFinanciero',
    'creditos_sectorCooperativo',
    'creditos_sectorReal',
    'promedio_ingresos_datacredito'
]

# Variables categóricas nominales
CATEGORICAL_NOMINAL = [
    'tipo_laboral',
    'tipo_credito'
]

# Variables categóricas ordinales
CATEGORICAL_ORDINAL = [
    'tendencia_ingresos'  # Orden: Decreciente < Estable < Creciente
]

# Filtrar solo las que existen en X
NUMERIC_FEATURES = [col for col in NUMERIC_FEATURES if col in X.columns]
CATEGORICAL_NOMINAL = [col for col in CATEGORICAL_NOMINAL if col in X.columns]
CATEGORICAL_ORDINAL = [col for col in CATEGORICAL_ORDINAL if col in X.columns]

print(f"Variables numéricas: {len(NUMERIC_FEATURES)}")
print(f"Variables categóricas nominales: {len(CATEGORICAL_NOMINAL)}")
print(f"Variables categóricas ordinales: {len(CATEGORICAL_ORDINAL)}")
print(f"\nTotal features: {len(NUMERIC_FEATURES) + len(CATEGORICAL_NOMINAL) + len(CATEGORICAL_ORDINAL)}")

# ============================================
# 6. CREAR FEATURES DERIVADOS
# ============================================

print("Creando features derivados...")

# Ratio cuota/salario
if 'cuota_pactada' in df.columns and 'salario_cliente' in X.columns:
    # Usar cuota_pactada de df original antes de eliminarla
    X['ratio_cuota_salario'] = df['cuota_pactada'] / (X['salario_cliente'] + 1)
    NUMERIC_FEATURES.append('ratio_cuota_salario')
    print("✓ ratio_cuota_salario creado")

# Ratio crédito/ingreso
if 'capital_prestado' in X.columns and 'salario_cliente' in X.columns:
    X['ratio_credito_ingreso'] = X['capital_prestado'] / (X['salario_cliente'] + 1)
    NUMERIC_FEATURES.append('ratio_credito_ingreso')
    print("✓ ratio_credito_ingreso creado")

# Total créditos en todos los sectores
sector_cols = ['creditos_sectorFinanciero', 'creditos_sectorCooperativo', 'creditos_sectorReal']
if all(col in X.columns for col in sector_cols):
    X['total_creditos_sectores'] = X[sector_cols].sum(axis=1)
    NUMERIC_FEATURES.append('total_creditos_sectores')
    print("✓ total_creditos_sectores creado")

# Indicador de si tiene codeudor
if 'saldo_mora_codeudor' in X.columns:
    X['tiene_codeudor'] = (X['saldo_mora_codeudor'].notna()).astype(int)
    NUMERIC_FEATURES.append('tiene_codeudor')
    print("✓ tiene_codeudor creado")

print(f"\nTotal features después de derivados: {X.shape[1]}")

# ============================================
# 7. DIVISIÓN TRAIN/TEST ESTRATIFICADA
# ============================================

# División 80/20 con estratificación (mantiene proporción de clases)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Mantiene proporción 95/5 en train y test
)

print("División de datos:")
print(f"  Train: {X_train.shape[0]:,} registros ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Test:  {X_test.shape[0]:,} registros ({X_test.shape[0]/len(X)*100:.1f}%)")

print(f"\nDistribución del target en train:")
print(y_train.value_counts())
print(f"\nDistribución del target en test:")
print(y_test.value_counts())

# ============================================
# 8. PIPELINE DE TRANSFORMACIÓN
# ============================================

# Pipeline para variables numéricas
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para variables categóricas nominales
categorical_nominal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Pipeline para variables categóricas ordinales
# Orden: Decreciente < Estable < Creciente
ordinal_categories = [['Decreciente', 'Estable', 'Creciente']]

categorical_ordinal_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(
        categories=ordinal_categories,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ))
])

# Combinar todos los pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat_nom', categorical_nominal_transformer, CATEGORICAL_NOMINAL),
        ('cat_ord', categorical_ordinal_transformer, CATEGORICAL_ORDINAL)
    ],
    remainder='drop'  # Eliminar columnas no especificadas
)

print("✓ Pipeline de transformación creado")
print(f"\nTransformaciones:")
print(f"  Numéricas ({len(NUMERIC_FEATURES)}): Imputación (mediana) + StandardScaler")
print(f"  Categóricas nominales ({len(CATEGORICAL_NOMINAL)}): Imputación + One-Hot Encoding")
print(f"  Categóricas ordinales ({len(CATEGORICAL_ORDINAL)}): Imputación + Ordinal Encoding")

# ============================================
# 9. APLICAR TRANSFORMACIONES
# ============================================

print("Aplicando transformaciones a los datos de entrenamiento...")

# Fit y transform en train
X_train_transformed = preprocessor.fit_transform(X_train)

# Solo transform en test (usar parámetros aprendidos de train)
X_test_transformed = preprocessor.transform(X_test)

print(f"\n✓ Transformaciones aplicadas")
print(f"  X_train_transformed: {X_train_transformed.shape}")
print(f"  X_test_transformed:  {X_test_transformed.shape}")

# Obtener nombres de features después de transformación
feature_names = []

# Numéricas
feature_names.extend(NUMERIC_FEATURES)

# Categóricas nominales (One-Hot genera múltiples columnas)
if CATEGORICAL_NOMINAL:
    onehot_encoder = preprocessor.named_transformers_['cat_nom'].named_steps['onehot']
    cat_features = onehot_encoder.get_feature_names_out(CATEGORICAL_NOMINAL)
    feature_names.extend(cat_features)

# Categóricas ordinales (mantienen su nombre)
feature_names.extend(CATEGORICAL_ORDINAL)

print(f"\nTotal features después de transformación: {len(feature_names)}")

# Convertir a DataFrame
X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

print("\n✓ DataFrames creados con nombres de features")

# ============================================
# 10. BALANCEO DE CLASES CON SMOTE
# ============================================

print("="*80)
print("BALANCEO DE CLASES CON SMOTE")
print("="*80)

print(f"\nANTES de SMOTE:")
print(f"  Clase 0 (No pagó): {(y_train==0).sum():,}")
print(f"  Clase 1 (Sí pagó): {(y_train==1).sum():,}")
print(f"  Ratio: {(y_train==1).sum()/(y_train==0).sum():.2f}:1")

# Aplicar SMOTE solo en conjunto de entrenamiento
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_df, y_train)

print(f"\nDESPUÉS de SMOTE:")
print(f"  Clase 0 (No pagó): {(y_train_balanced==0).sum():,}")
print(f"  Clase 1 (Sí pagó): {(y_train_balanced==1).sum():,}")
print(f"  Ratio: {(y_train_balanced==1).sum()/(y_train_balanced==0).sum():.2f}:1")

print(f"\nRegistros agregados sintéticamente: {len(y_train_balanced) - len(y_train):,}")
print(f"\n✓ Clases balanceadas exitosamente")
print("="*80)

# Visualizar el balanceo
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Antes de SMOTE
y_train.value_counts().plot(kind='bar', ax=axes[0], color=['#e74c3c', '#2ecc71'], 
                            edgecolor='black', alpha=0.8)
axes[0].set_title('ANTES de SMOTE', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Pago a Tiempo')
axes[0].set_ylabel('Frecuencia')
axes[0].set_xticklabels(['No (0)', 'Sí (1)'], rotation=0)
axes[0].grid(axis='y', alpha=0.3)

# Después de SMOTE
pd.Series(y_train_balanced).value_counts().plot(kind='bar', ax=axes[1], 
                                                color=['#e74c3c', '#2ecc71'],
                                                edgecolor='black', alpha=0.8)
axes[1].set_title('DESPUÉS de SMOTE', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Pago a Tiempo')
axes[1].set_ylabel('Frecuencia')
axes[1].set_xticklabels(['No (0)', 'Sí (1)'], rotation=0)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# 11. GUARDAR DATOS PROCESADOS
# ============================================

print("Guardando datos procesados...")

# Guardar datos de entrenamiento balanceados
X_train_balanced.to_csv(DATA_PROCESSED / 'X_train.csv', index=False)
pd.Series(y_train_balanced, name=TARGET).to_csv(DATA_PROCESSED / 'y_train.csv', index=False)

# Guardar datos de test (SIN balancear - mantener distribución real)
X_test_df.to_csv(DATA_PROCESSED / 'X_test.csv', index=False)
y_test.to_csv(DATA_PROCESSED / 'y_test.csv', index=False, header=True)

# Guardar pipeline de transformación
joblib.dump(preprocessor, MODELS_DIR / 'pipeline_transformacion.pkl')

print(f"\n✓ Datos guardados en: {DATA_PROCESSED}")
print(f"  - X_train.csv: {X_train_balanced.shape}")
print(f"  - y_train.csv: {len(y_train_balanced)}")
print(f"  - X_test.csv: {X_test_df.shape}")
print(f"  - y_test.csv: {len(y_test)}")

print(f"\n✓ Pipeline guardado en: {MODELS_DIR}")
print(f"  - pipeline_transformacion.pkl")

# ============================================
# 12. RESUMEN Y VERIFICACIÓN
# ============================================

print("="*80)
print("RESUMEN DE FEATURE ENGINEERING")
print("="*80)

print(f"\n DATOS ORIGINALES:")
print(f"  Registros totales: {len(df):,}")
print(f"  Features originales: {df.shape[1]}")

print(f"\n TRANSFORMACIONES APLICADAS:")
print(f"  1. Variables eliminadas: {len(VARS_TO_DROP)}")
print(f"  2. Features derivados creados: {X.shape[1] - (df.shape[1] - 1 - len(VARS_TO_DROP))}")
print(f"  3. Features después de encoding: {len(feature_names)}")

print(f"\n DIVISIÓN DE DATOS:")
print(f"  Train: {len(X_train):,} ({len(X_train)/len(df)*100:.1f}%)")
print(f"  Test:  {len(X_test):,} ({len(X_test)/len(df)*100:.1f}%)")

print(f"\n  BALANCEO DE CLASES (Solo Train):")
print(f"  Antes de SMOTE: {len(y_train):,} registros")
print(f"  Después de SMOTE: {len(y_train_balanced):,} registros")
print(f"  Registros sintéticos añadidos: {len(y_train_balanced) - len(y_train):,}")

print(f"\n ARCHIVOS GENERADOS:")
print(f"  1. X_train.csv ({X_train_balanced.shape[0]:,} × {X_train_balanced.shape[1]})")
print(f"  2. y_train.csv ({len(y_train_balanced):,} registros)")
print(f"  3. X_test.csv ({X_test_df.shape[0]:,} × {X_test_df.shape[1]})")
print(f"  4. y_test.csv ({len(y_test):,} registros)")
print(f"  5. pipeline_transformacion.pkl")

print(f"\n FEATURE ENGINEERING COMPLETADO")
print(f" Datos listos para modelamiento en: {DATA_PROCESSED}")
print("="*80)



