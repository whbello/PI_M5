""" Modelamiento y Evaluación - Avance 2

Objetivo
Entrenar y evaluar múltiples modelos de Machine Learning para predecir el pago a tiempo de créditos.

Contexto del Problema
- **Desbalance crítico:** 95.25% pagó vs 4.75% no pagó (identificado en EDA)
- **Datos balanceados:** Train ya procesado con SMOTE en feature engineering
- **Métrica principal:** **Recall** (detectar clientes que NO pagarán)
- **Métricas secundarias:** F1-Score, Precision, AUC-ROC

¿Por qué Recall es la métrica principal?
- **Costo de Falsos Negativos es ALTO:** No detectar un cliente que no pagará = pérdida de dinero
- **Costo de Falsos Positivos es MENOR:** Rechazar un cliente bueno = pérdida de oportunidad
- **Objetivo:** Maximizar la detección de clientes con riesgo de impago

Inputs
- `data/processed/X_train.csv` (8,610+ registros, balanceado con SMOTE)
- `data/processed/X_test.csv` (2,153 registros, distribución real)
- `data/processed/y_train.csv` (balanceado)
- `data/processed/y_test.csv` (distribución real)

Outputs
- `models/modelo_final.pkl`
- `models/metricas_comparacion.csv`
- Visualizaciones y análisis de performance

Modelos a Evaluar
1. Baseline (Predicción mayoritaria)
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. XGBoost"
"""

# ============================================
# 1. CONFIGURACIÓN E IMPORTS
# ============================================

# Manipulación de datos
import pandas as pd
import numpy as np
from pathlib import Path

# Modelos de clasificación
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Evaluación de modelos
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)

# Validación cruzada
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Persistencia de modelos
import joblib

# Tiempo
import time

# Configuración
import warnings
warnings.filterwarnings('ignore')

# Configuración de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', '{:.4f}'.format)

# Configuración de matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("CONFIGURACIÓN COMPLETADA")
print("="*80)
print(f"✓ Librerías importadas correctamente")
print(f"  - Pandas: {pd.__version__}")
print(f"  - NumPy: {np.__version__}")
print(f"  - Scikit-learn disponible")
print(f"  - XGBoost disponible")
print("="*80)

# ============================================
# 2. DEFINIR RUTAS Y CARGAR DATOS
# ============================================

# Definir rutas del proyecto
# Toda la estructura está dentro de src/
PROJECT_ROOT = Path.cwd()  # Estamos en mlops_pipeline/src/

# Definir todas las rutas dentro de src/
DATA_DIR = PROJECT_ROOT / 'data'
DATA_RAW = DATA_DIR / 'raw'
DATA_INTERIM = DATA_DIR / 'interim'
DATA_PROCESSED = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

# Crear carpetas si no existen
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("CONFIGURACIÓN DE RUTAS")
print("="*80)
print(f"\nDirectorio de trabajo: {PROJECT_ROOT}")
print(f"\nRutas configuradas:")
print(f"  📁 DATA_RAW:       {DATA_RAW.relative_to(PROJECT_ROOT)}")
print(f"  📁 DATA_INTERIM:   {DATA_INTERIM.relative_to(PROJECT_ROOT)}")
print(f"  📁 DATA_PROCESSED: {DATA_PROCESSED.relative_to(PROJECT_ROOT)}")
print(f"  📁 MODELS_DIR:     {MODELS_DIR.relative_to(PROJECT_ROOT)}")
print(f"  📁 REPORTS_DIR:    {REPORTS_DIR.relative_to(PROJECT_ROOT)}")

# Verificar que la carpeta processed existe
if not DATA_PROCESSED.exists():
    print(f"\nX ERROR: La carpeta {DATA_PROCESSED} no existe")
    print(f"\n SOLUCIÓN:")
    print(f"  1. Ejecuta primero: 03_ft_engineering.ipynb")
    print(f"  2. O crea la carpeta manualmente en: {DATA_PROCESSED}")
    raise FileNotFoundError(f"Carpeta no encontrada: {DATA_PROCESSED}")

print(f"\n✓ Carpeta de datos procesados encontrada")

print("\n" + "="*80)
print("CARGA DE DATOS PROCESADOS")
print("="*80)

# Verificar archivos requeridos
archivos_requeridos = {
    'X_train.csv': 'Features de entrenamiento (balanceado)',
    'X_test.csv': 'Features de prueba (distribución real)',
    'y_train.csv': 'Target de entrenamiento (balanceado)',
    'y_test.csv': 'Target de prueba (distribución real)'
}

print(f"\nBuscando archivos en: {DATA_PROCESSED}")
archivos_faltantes = []

for archivo, descripcion in archivos_requeridos.items():
    archivo_path = DATA_PROCESSED / archivo
    existe = archivo_path.exists()
    
    if existe:
        size_mb = archivo_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {archivo:<15} ({size_mb:.2f} MB) - {descripcion}")
    else:
        print(f"  ✗ {archivo:<15} - FALTANTE")
        archivos_faltantes.append(archivo)

if archivos_faltantes:
    print(f"\nX ERROR: Archivos faltantes en {DATA_PROCESSED}:")
    for archivo in archivos_faltantes:
        print(f"  - {archivo}")
    print(f"\n  SOLUCIÓN:")
    print(f"  1. Abre y ejecuta: 03_ft_engineering.ipynb")
    print(f"  2. Verifica que se generan los 4 archivos CSV")
    print(f"  3. Vuelve a ejecutar este notebook")
    raise FileNotFoundError("Datos procesados incompletos")

# Cargar datos
print(f"\n Cargando datos en memoria...")

try:
    X_train = pd.read_csv(DATA_PROCESSED / 'X_train.csv')
    X_test = pd.read_csv(DATA_PROCESSED / 'X_test.csv')
    y_train = pd.read_csv(DATA_PROCESSED / 'y_train.csv').squeeze()
    y_test = pd.read_csv(DATA_PROCESSED / 'y_test.csv').squeeze()
    
    print(f"✓ Datos cargados exitosamente")
    
except Exception as e:
    print(f"\nX ERROR al cargar datos: {e}")
    raise

# Información de los datos cargados
print(f"\n INFORMACIÓN DE LOS DATOS:")
print(f"\n  Conjunto de Entrenamiento:")
print(f"    X_train: {X_train.shape[0]:>6,} filas × {X_train.shape[1]:>3} features")
print(f"    y_train: {len(y_train):>6,} registros")
print(f"    Memoria:  {X_train.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\n  Conjunto de Prueba:")
print(f"    X_test:  {X_test.shape[0]:>6,} filas × {X_test.shape[1]:>3} features")
print(f"    y_test:  {len(y_test):>6,} registros")
print(f"    Memoria:  {X_test.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Distribución del target
print(f"\n DISTRIBUCIÓN DEL TARGET:")

print(f"\n  Training Set (Balanceado con SMOTE):")
train_dist = y_train.value_counts().sort_index()
print(f"    Clase 0 (No pagó): {train_dist[0]:>6,} ({train_dist[0]/len(y_train)*100:>5.2f}%)")
print(f"    Clase 1 (Sí pagó): {train_dist[1]:>6,} ({train_dist[1]/len(y_train)*100:>5.2f}%)")
print(f"    Ratio: {train_dist[1]/train_dist[0]:.2f}:1  ✓ Balanceado")

print(f"\n  Test Set (Distribución Real):")
test_dist = y_test.value_counts().sort_index()
print(f"    Clase 0 (No pagó): {test_dist[0]:>6,} ({test_dist[0]/len(y_test)*100:>5.2f}%)")
print(f"    Clase 1 (Sí pagó): {test_dist[1]:>6,} ({test_dist[1]/len(y_test)*100:>5.2f}%)")
print(f"    Ratio: {test_dist[1]/test_dist[0]:.2f}:1   Desbalanceado (real)")

print("\n" + "="*80)
print(" DATOS LISTOS PARA MODELAMIENTO")
print("="*80)

# ============================================
# 3. FUNCIÓN DE EVALUACIÓN DE MODELOS
# ============================================

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Entrena y evalúa un modelo de clasificación.
    
    Parámetros:
    -----------
    model : estimator
        Modelo de sklearn a entrenar
    X_train, y_train : arrays
        Datos de entrenamiento
    X_test, y_test : arrays
        Datos de prueba
    model_name : str
        Nombre del modelo para reportes
    
    Retorna:
    --------
    metrics : dict
        Diccionario con todas las métricas
    trained_model : estimator
        Modelo entrenado
    """
    
    print(f"\n{'='*80}")
    print(f"ENTRENANDO: {model_name}")
    print(f"{'='*80}")
    
    # Medir tiempo de entrenamiento
    start_time = time.time()
    
    # Entrenar modelo
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilidades (para AUC-ROC)
    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Para modelos sin predict_proba (ej: SVM sin probability=True)
        y_train_proba = y_train_pred
        y_test_proba = y_test_pred
    
    # Calcular todas las métricas
    metrics = {
        'model': model_name,
        'training_time': training_time,
        
        # Métricas en Train
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
        'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
        'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
        'train_auc_roc': roc_auc_score(y_train, y_train_proba) if hasattr(model, 'predict_proba') else None,
        
        # Métricas en Test
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, zero_division=0),
        'test_auc_roc': roc_auc_score(y_test, y_test_proba) if hasattr(model, 'predict_proba') else None,
    }
    
    # Matriz de confusión (solo test)
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['test_tn'] = tn
    metrics['test_fp'] = fp
    metrics['test_fn'] = fn
    metrics['test_tp'] = tp
    
    # Calcular especificidad (True Negative Rate)
    metrics['test_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Mostrar resultados
    print(f"\n  Tiempo de entrenamiento: {training_time:.2f} segundos")
    
    print(f"\n MÉTRICAS EN TEST (Distribución Real):")
    print(f"  {'Métrica':<15} {'Valor':>10}")
    print(f"  {'-'*25}")
    print(f"  {'Accuracy':<15} {metrics['test_accuracy']:>10.4f}")
    print(f"  {'Precision':<15} {metrics['test_precision']:>10.4f}")
    print(f"  {'Recall':<15} {metrics['test_recall']:>10.4f}  ⭐ PRINCIPAL")
    print(f"  {'F1-Score':<15} {metrics['test_f1']:>10.4f}")
    print(f"  {'AUC-ROC':<15} {metrics['test_auc_roc']:>10.4f}" if metrics['test_auc_roc'] else "")
    print(f"  {'Specificity':<15} {metrics['test_specificity']:>10.4f}")
    
    print(f"\n MATRIZ DE CONFUSIÓN (Test):")
    print(f"                    Predicción")
    print(f"                 No Pagó  |  Sí Pagó")
    print(f"  Real  No Pagó    {tn:>5}  |  {fp:>5}  ← FP")
    print(f"        Sí Pagó    {fn:>5}  |  {tp:>5}  ← FN")
    print(f"                     ↑         ↑")
    print(f"                    TN        TP")
    
    # Interpretación de negocio
    total_no_pago = tn + fn
    total_si_pago = fp + tp
    
    print(f"\n INTERPRETACIÓN DE NEGOCIO:")
    print(f"  • De {total_no_pago} clientes que NO pagaron:")
    print(f"    - Detectados correctamente (TN): {tn} ({tn/total_no_pago*100:.1f}%)")
    print(f"    - NO detectados - PÉRDIDA (FN): {fn} ({fn/total_no_pago*100:.1f}%)")
    
    print(f"\n  • De {total_si_pago} clientes que SÍ pagaron:")
    print(f"    - Aprobados correctamente (TP): {tp} ({tp/total_si_pago*100:.1f}%)")
    print(f"    - Rechazados incorrectamente (FP): {fp} ({fp/total_si_pago*100:.1f}%)")
    
    # Detectar overfitting
    if metrics['train_f1'] and metrics['test_f1']:
        overfit = metrics['train_f1'] - metrics['test_f1']
        if overfit > 0.15:
            print(f"\n  OVERFITTING DETECTADO")
            print(f"  Diferencia Train-Test F1: {overfit:.4f}")
            print(f"  El modelo se ajustó demasiado a los datos de entrenamiento")
        elif overfit > 0.10:
            print(f"\n  Posible ligero overfitting")
            print(f"  Diferencia Train-Test F1: {overfit:.4f}")
    
    print(f"\n{'='*80}")
    
    return metrics, model

print("✓ Función evaluate_model() creada")
print("  Esta función entrenará, evaluará y reportará métricas de cada modelo")

# ============================================
# 4. MODELO BASELINE (Predicción Mayoritaria)
# ============================================

print("\n" + "="*80)
print("MODELO 1/6: BASELINE (Predicción Mayoritaria)")
print("="*80)
print("\n Este modelo siempre predice la clase más frecuente.")
print("   Sirve como punto de referencia mínimo.")
print("   Cualquier modelo real debe superarlo significativamente.")

# Crear modelo baseline
baseline = DummyClassifier(strategy='most_frequent', random_state=42)

# Evaluar
metrics_baseline, model_baseline = evaluate_model(
    baseline, X_train, y_train, X_test, y_test,
    "Baseline (Most Frequent)"
)

print(f"\n  NOTA: Este modelo tiene Recall = 0 para la clase 0")
print(f"   NO detecta NINGÚN cliente que no pagará")
print(f"   Esto es exactamente lo que queremos EVITAR")

# ============================================
# 5.1 LOGISTIC REGRESSION
# ============================================

print("\n" + "="*80)
print("MODELO 2/6: LOGISTIC REGRESSION")
print("="*80)
print("\n Modelo lineal simple y rápido.")
print("   Bueno para interpretabilidad y como baseline avanzado.")

lr = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced',  # Ajuste adicional de pesos
    solver='lbfgs'
)

metrics_lr, model_lr = evaluate_model(
    lr, X_train, y_train, X_test, y_test,
    "Logistic Regression"
)

# ============================================
# 5.2 DECISION TREE
# ============================================

print("\n" + "="*80)
print("MODELO 3/6: DECISION TREE")
print("="*80)
print("\n Árbol de decisión con limitaciones para evitar overfitting.")

dt = DecisionTreeClassifier(
    random_state=42,
    max_depth=10,
    min_samples_split=100,
    min_samples_leaf=50,
    class_weight='balanced',
    criterion='gini'
)

metrics_dt, model_dt = evaluate_model(
    dt, X_train, y_train, X_test, y_test,
    "Decision Tree"
)

# ============================================
# 5.3 RANDOM FOREST
# ============================================

print("\n" + "="*80)
print("MODELO 4/6: RANDOM FOREST")
print("="*80)
print("\n Ensamble de árboles de decisión.")
print("   Más robusto que un árbol individual.")

rf = RandomForestClassifier(
    random_state=42,
    n_estimators=100,
    max_depth=15,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',
    n_jobs=-1,  # Usar todos los cores disponibles
    max_features='sqrt'
)

metrics_rf, model_rf = evaluate_model(
    rf, X_train, y_train, X_test, y_test,
    "Random Forest"
)

# ============================================
# 5.4 GRADIENT BOOSTING
# ============================================

print("\n" + "="*80)
print("MODELO 5/6: GRADIENT BOOSTING")
print("="*80)
print("\n Boosting: construye árboles secuencialmente.")
print("   Cada árbol corrige errores del anterior.")

gb = GradientBoostingClassifier(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=50,
    min_samples_leaf=20,
    subsample=0.8
)

metrics_gb, model_gb = evaluate_model(
    gb, X_train, y_train, X_test, y_test,
    "Gradient Boosting"
)

# ============================================
# 5.5 XGBOOST
# ============================================

print("\n" + "="*80)
print("MODELO 6/6: XGBOOST")
print("="*80)
print("\n Versión optimizada de Gradient Boosting.")
print("   Generalmente el mejor performance en competencias.")

# Calcular scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"\n   scale_pos_weight calculado: {scale_pos_weight:.2f}")

xgb = XGBClassifier(
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    use_label_encoder=False
)

metrics_xgb, model_xgb = evaluate_model(
    xgb, X_train, y_train, X_test, y_test,
    "XGBoost"
)

print("\n TODOS LOS MODELOS ENTRENADOS")

# ============================================
# 6. COMPARACIÓN DE MODELOS
# ============================================

# Crear DataFrame con todas las métricas
results_df = pd.DataFrame([
    metrics_baseline,
    metrics_lr,
    metrics_dt,
    metrics_rf,
    metrics_gb,
    metrics_xgb
])

print("\n" + "="*80)
print("COMPARACIÓN DE MODELOS - MÉTRICAS EN TEST")
print("="*80)

# Seleccionar métricas de test para mostrar
test_metrics = results_df[[
    'model', 'training_time',
    'test_accuracy', 'test_precision', 'test_recall', 
    'test_f1', 'test_auc_roc'
]].copy()

# Renombrar columnas para mejor visualización
test_metrics.columns = [
    'Modelo', 'Tiempo (s)',
    'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'
]

print("\n Tabla Comparativa:")
print(test_metrics.to_string(index=False))

# Guardar resultados completos
results_path = MODELS_DIR / 'metricas_comparacion.csv'
results_df.to_csv(results_path, index=False)
print(f"\n✓ Métricas completas guardadas en:")
print(f"  {results_path.relative_to(PROJECT_ROOT)}")

# Identificar mejor modelo por cada métrica
print(f"\n MEJORES MODELOS POR MÉTRICA:")
print(f"  Recall (Principal):  {test_metrics.loc[test_metrics['Recall'].idxmax(), 'Modelo']:<25} ({test_metrics['Recall'].max():.4f})")
print(f"  F1-Score:            {test_metrics.loc[test_metrics['F1-Score'].idxmax(), 'Modelo']:<25} ({test_metrics['F1-Score'].max():.4f})")
print(f"  Precision:           {test_metrics.loc[test_metrics['Precision'].idxmax(), 'Modelo']:<25} ({test_metrics['Precision'].max():.4f})")
print(f"  AUC-ROC:             {test_metrics.loc[test_metrics['AUC-ROC'].idxmax(), 'Modelo']:<25} ({test_metrics['AUC-ROC'].max():.4f})")
print(f"  Más Rápido:          {test_metrics.loc[test_metrics['Tiempo (s)'].idxmin(), 'Modelo']:<25} ({test_metrics['Tiempo (s)'].min():.2f}s)")

print("\n" + "="*80)

# ============================================
# 7. VISUALIZACIÓN COMPARATIVA DE MODELOS
# ============================================

# Preparar datos para visualización
models_list = test_metrics['Modelo'].values
metrics_to_plot = ['Recall', 'Precision', 'F1-Score', 'AUC-ROC']

# Paleta de colores (suficientes para todos los modelos)
colors_palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#95a5a6']

# Crear figura con 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Comparación de Modelos - Métricas en Test', fontsize=18, fontweight='bold', y=0.995)

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    
    values = test_metrics[metric].values
    
    # Asignar colores de forma segura
    colors = []
    for i, model in enumerate(models_list):
        if 'Baseline' in model:
            colors.append(colors_palette[0])  # Rojo para baseline
        else:
            # Usar índice seguro dentro del rango
            color_idx = (i % (len(colors_palette) - 1)) + 1
            colors.append(colors_palette[color_idx])
    
    # Crear barras horizontales
    bars = ax.barh(range(len(models_list)), values, color=colors, 
                   edgecolor='black', alpha=0.85, linewidth=1.5)
    
    ax.set_yticks(range(len(models_list)))
    ax.set_yticklabels(models_list, fontsize=10)
    ax.set_xlabel(metric, fontsize=12, fontweight='bold')
    ax.set_title(f'{metric} por Modelo', fontsize=13, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1.0)
    
    # Añadir valores en las barras
    for i, (bar, value) in enumerate(zip(bars, values)):
        width = bar.get_width()
        label_position = width + 0.02 if width < 0.85 else width - 0.02
        ha = 'left' if width < 0.85 else 'right'
        color = 'black' if width < 0.85 else 'white'
        
        ax.text(label_position, bar.get_y() + bar.get_height()/2,
                f'{value:.3f}', va='center', ha=ha,
                fontsize=10, fontweight='bold', color=color)
    
    # Marcar el mejor modelo
    best_idx = values.argmax()
    ax.get_yticklabels()[best_idx].set_weight('bold')
    ax.get_yticklabels()[best_idx].set_color(colors[best_idx])
    
    # Añadir línea de referencia (solo para métricas donde aplique)
    if metric in ['Recall', 'F1-Score']:
        ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
        ax.text(0.5, len(models_list) - 0.5, 'Mínimo aceptable', 
                fontsize=8, ha='center', va='bottom', color='gray',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()

# Guardar figura
try:
    plt.savefig(FIGURES_DIR / 'comparacion_modelos.png', dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado en: {FIGURES_DIR / 'comparacion_modelos.png'}")
except Exception as e:
    print(f"⚠️ No se pudo guardar el gráfico: {e}")

plt.show()
# ============================================
# 8. SELECCIÓN DEL MEJOR MODELO
# ============================================

# Seleccionar mejor modelo basado en Recall (métrica principal)
best_idx = test_metrics['Recall'].idxmax()
best_model_name = test_metrics.loc[best_idx, 'Modelo']
best_recall = test_metrics.loc[best_idx, 'Recall']

# Diccionario de modelos entrenados
models_dict = {
    'Baseline (Most Frequent)': model_baseline,
    'Logistic Regression': model_lr,
    'Decision Tree': model_dt,
    'Random Forest': model_rf,
    'Gradient Boosting': model_gb,
    'XGBoost': model_xgb
}

# Obtener el modelo seleccionado
best_model = models_dict[best_model_name]

print("\n" + "="*80)
print(f"🏆 MODELO SELECCIONADO")
print("="*80)
print(f"\nModelo: {best_model_name}")
print(f"Criterio: Mayor Recall en Test")
print(f"Recall: {best_recall:.4f}")

# Métricas completas del mejor modelo
best_metrics = results_df[results_df['model'] == best_model_name].iloc[0]

print(f"\n MÉTRICAS COMPLETAS DEL MEJOR MODELO:")
print(f"  {'Métrica':<20} {'Train':>10} {'Test':>10} {'Diferencia':>12}")
print(f"  {'-'*54}")
print(f"  {'Accuracy':<20} {best_metrics['train_accuracy']:>10.4f} {best_metrics['test_accuracy']:>10.4f} {best_metrics['train_accuracy']-best_metrics['test_accuracy']:>12.4f}")
print(f"  {'Precision':<20} {best_metrics['train_precision']:>10.4f} {best_metrics['test_precision']:>10.4f} {best_metrics['train_precision']-best_metrics['test_precision']:>12.4f}")
print(f"  {'Recall':<20} {best_metrics['train_recall']:>10.4f} {best_metrics['test_recall']:>10.4f} {best_metrics['train_recall']-best_metrics['test_recall']:>12.4f}")
print(f"  {'F1-Score':<20} {best_metrics['train_f1']:>10.4f} {best_metrics['test_f1']:>10.4f} {best_metrics['train_f1']-best_metrics['test_f1']:>12.4f}")

# Classification Report completo
print(f"\n CLASSIFICATION REPORT DETALLADO:")
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best, 
                        target_names=['No pagó (0)', 'Pagó (1)'],
                        digits=4))

# Matriz de confusión visual
cm = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm.ravel()

fig, ax = plt.subplots(figsize=(10, 8))

# Crear heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicción: No pagó', 'Predicción: Sí pagó'],
            yticklabels=['Real: No pagó', 'Real: Sí pagó'],
            ax=ax, cbar_kws={'label': 'Cantidad de casos'},
            annot_kws={'size': 16, 'weight': 'bold'})

ax.set_title(f'Matriz de Confusión - {best_model_name}', 
            fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Valor Real', fontsize=13, fontweight='bold')
ax.set_xlabel('Valor Predicho', fontsize=13, fontweight='bold')

# Añadir porcentajes y etiquetas
total = cm.sum()
for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / total * 100
        text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        
        # Añadir porcentaje
        ax.text(j + 0.5, i + 0.75, f'({percentage:.1f}%)', 
                ha='center', va='center', fontsize=11, 
                color=text_color, style='italic')
        
        # Añadir etiquetas TN, FP, FN, TP
        labels = [['TN', 'FP'], ['FN', 'TP']]
        ax.text(j + 0.1, i + 0.1, labels[i][j],
                ha='left', va='top', fontsize=10,
                color='red', weight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig(FIGURES_DIR / f'confusion_matrix_{best_model_name.lower().replace(" ", "_")}.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# Interpretación de negocio
total_no_pago = tn + fn
total_si_pago = fp + tp

print(f"\n INTERPRETACIÓN DE NEGOCIO:")
print(f"\n  Clientes que NO pagaron ({total_no_pago} en test):")
print(f"    ✓ Detectados (TN):          {tn:>5} ({tn/total_no_pago*100:>5.1f}%)  ← Pérdida evitada")
print(f"    ✗ NO detectados (FN):       {fn:>5} ({fn/total_no_pago*100:>5.1f}%)  ← Pérdida real")
print(f"\n  Clientes que SÍ pagaron ({total_si_pago} en test):")
print(f"    ✓ Aprobados (TP):           {tp:>5} ({tp/total_si_pago*100:>5.1f}%)  ← Ganancia")
print(f"    ✗ Rechazados (FP):          {fp:>5} ({fp/total_si_pago*100:>5.1f}%)  ← Oportunidad perdida")

print(f"\n✓ MEJORA vs BASELINE:")
baseline_recall = metrics_baseline['test_recall']
improvement = best_recall - baseline_recall
print(f"  Baseline Recall: {baseline_recall:.4f}")
print(f"  Mejor Modelo:    {best_recall:.4f}")
print(f"  Mejora:          +{improvement:.4f} ({improvement*100:.1f}% más detección)")

print("\n" + "="*80)

# ============================================
# 9. ANÁLISIS DE IMPORTANCIA DE FEATURES
# ============================================

print("\n" + "="*80)
print("ANÁLISIS DE IMPORTANCIA DE FEATURES")
print("="*80)

if hasattr(best_model, 'feature_importances_'):
    print(f"\n✓ {best_model_name} tiene feature_importances_")
    
    # Obtener importancias
    importances = best_model.feature_importances_
    feature_names = X_train.columns
    
    # Crear DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Calcular importancia acumulada
    feature_importance_df['cumulative_importance'] = feature_importance_df['importance'].cumsum()
    
    # Features que explican el 80% de la importancia
    features_80 = feature_importance_df[feature_importance_df['cumulative_importance'] <= 0.8]
    
    print(f"\n Estadísticas de Importancia:")
    print(f"  Total de features: {len(feature_importance_df)}")
    print(f"  Features con importancia > 0: {(importances > 0).sum()}")
    print(f"  Features que explican 80% de importancia: {len(features_80)}")
    
    # Top 20 features
    top_n = 20
    top_features = feature_importance_df.head(top_n)
    
    print(f"\n Top {top_n} Features Más Importantes:")
    print(f"  {'#':<3} {'Feature':<40} {'Importancia':>12} {'% Acumulado':>15}")
    print(f"  {'-'*72}")
    for idx, row in top_features.iterrows():
        print(f"  {idx+1:<3} {row['feature']:<40} {row['importance']:>12.6f} {row['cumulative_importance']*100:>14.2f}%")
    
    # Guardar importancias completas
    importance_path = MODELS_DIR / f'feature_importance_{best_model_name.lower().replace(" ", "_")}.csv'
    feature_importance_df.to_csv(importance_path, index=False)
    print(f"\n✓ Importancias guardadas en: {importance_path.relative_to(PROJECT_ROOT)}")
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f'Importancia de Features - {best_model_name}', 
                fontsize=16, fontweight='bold')
    
    # Gráfico 1: Top 20 features
    ax1 = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    bars = ax1.barh(range(len(top_features)), top_features['importance'], 
                    color=colors, edgecolor='black', alpha=0.8, linewidth=1)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'], fontsize=9)
    ax1.set_xlabel('Importancia', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top {top_n} Features Más Importantes', 
                fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Añadir valores
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        width = bar.get_width()
        ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.4f}', va='center', fontsize=8)
    
    # Gráfico 2: Importancia acumulada
    ax2 = axes[1]
    x_vals = range(1, len(feature_importance_df) + 1)
    ax2.plot(x_vals, feature_importance_df['cumulative_importance']*100,
            linewidth=2.5, color='#3498db', marker='o', markersize=3,
            markevery=max(1, len(x_vals)//20))
    
    # Línea del 80%
    ax2.axhline(80, color='red', linestyle='--', linewidth=2, 
                alpha=0.7, label='80% de importancia')
    ax2.axvline(len(features_80), color='red', linestyle='--', 
                linewidth=2, alpha=0.7)
    
    # Área sombreada
    ax2.fill_between(x_vals, 0, 
                    feature_importance_df['cumulative_importance']*100,
                    alpha=0.3, color='#3498db')
    
    ax2.set_xlabel('Número de Features', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Importancia Acumulada (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Importancia Acumulada de Features', 
                fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='lower right', fontsize=10)
    
    # Anotación
    ax2.annotate(f'{len(features_80)} features\nexplican 80%',
                xy=(len(features_80), 80), xytext=(len(features_80)+10, 60),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'feature_importance_{best_model_name.lower().replace(" ", "_")}.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n INSIGHTS:")
    print(f"  • Solo {len(features_80)} features ({len(features_80)/len(feature_importance_df)*100:.1f}%) explican el 80% de las predicciones")
    print(f"  • Los 3 features más importantes son:")
    for i in range(min(3, len(top_features))):
        print(f"    {i+1}. {top_features.iloc[i]['feature']}: {top_features.iloc[i]['importance']:.4f}")

elif hasattr(best_model, 'coef_'):
    print(f"\n✓ {best_model_name} tiene coeficientes (modelo lineal)")
    
    # Para modelos lineales (Logistic Regression)
    coefficients = best_model.coef_[0]
    feature_names = X_train.columns
    
    # Crear DataFrame
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    # Top 20
    top_n = 20
    top_coefs = coef_df.head(top_n)
    
    print(f"\n Top {top_n} Features por Magnitud de Coeficiente:")
    print(f"  {'#':<3} {'Feature':<40} {'Coeficiente':>15} {'Magnitud':>12}")
    print(f"  {'-'*72}")
    for idx, row in enumerate(top_coefs.iterrows(), 1):
        _, data = row
        print(f"  {idx:<3} {data['feature']:<40} {data['coefficient']:>15.6f} {data['abs_coefficient']:>12.6f}")
    
    # Visualización
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in top_coefs['coefficient']]
    bars = ax.barh(range(len(top_coefs)), top_coefs['coefficient'],
                color=colors, edgecolor='black', alpha=0.8, linewidth=1)
    
    ax.set_yticks(range(len(top_coefs)))
    ax.set_yticklabels(top_coefs['feature'], fontsize=10)
    ax.set_xlabel('Coeficiente', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Coeficientes - {best_model_name}',
                fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'coefficients_{best_model_name.lower().replace(" ", "_")}.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n Coeficientes:")
    print(f"  • Positivos (verde): Aumentan probabilidad de pago a tiempo")
    print(f"  • Negativos (rojo): Disminuyen probabilidad de pago a tiempo")

else:
    print(f"\n  {best_model_name} no tiene feature_importances_ ni coef_")
    print(f"   Este modelo no permite analizar importancia de features directamente")

print("\n" + "="*80)

# ============================================
# 10. CURVAS ROC Y PRECISION-RECALL
# ============================================

print("\n" + "="*80)
print("CURVAS DE EVALUACIÓN")
print("="*80)

# Modelos a graficar (excluir baseline)
models_for_curves = {
    'Logistic Regression': model_lr,
    'Decision Tree': model_dt,
    'Random Forest': model_rf,
    'Gradient Boosting': model_gb,
    'XGBoost': model_xgb
}

# Crear figura con 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Curvas de Evaluación - Comparación de Modelos', 
            fontsize=16, fontweight='bold')

# ============================================
# CURVA ROC (Receiver Operating Characteristic)
# ============================================
ax1 = axes[0]

for name, model in models_for_curves.items():
    # Obtener probabilidades
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Estilo de línea
    if name == best_model_name:
        linestyle = '-'
        linewidth = 3
        alpha = 1.0
        label = f'{name} (AUC = {roc_auc:.3f}) ⭐'
    else:
        linestyle = '--'
        linewidth = 2
        alpha = 0.7
        label = f'{name} (AUC = {roc_auc:.3f})'
    
    # Plotear
    ax1.plot(fpr, tpr, label=label, linestyle=linestyle, 
            linewidth=linewidth, alpha=alpha)

# Línea diagonal (random classifier)
ax1.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', 
        linewidth=1.5, alpha=0.5)

ax1.set_xlabel('False Positive Rate (FPR)', fontsize=12, fontweight='bold')
ax1.set_ylabel('True Positive Rate (TPR = Recall)', fontsize=12, fontweight='bold')
ax1.set_title('Curva ROC', fontsize=14, fontweight='bold', pad=15)
ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax1.grid(alpha=0.3, linestyle='--')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])

# Área sombreada para el mejor modelo
if best_model_name in models_for_curves:
    y_proba_best = best_model.predict_proba(X_test)[:, 1]
    fpr_best, tpr_best, _ = roc_curve(y_test, y_proba_best)
    ax1.fill_between(fpr_best, 0, tpr_best, alpha=0.2, label='_nolegend_')

# ============================================
# CURVA PRECISION-RECALL
# ============================================
ax2 = axes[1]

for name, model in models_for_curves.items():
    # Obtener probabilidades
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calcular curva Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    # Estilo de línea
    if name == best_model_name:
        linestyle = '-'
        linewidth = 3
        alpha = 1.0
        label = f'{name} (AUC = {pr_auc:.3f}) ⭐'
    else:
        linestyle = '--'
        linewidth = 2
        alpha = 0.7
        label = f'{name} (AUC = {pr_auc:.3f})'
    
    # Plotear
    ax2.plot(recall, precision, label=label, linestyle=linestyle,
            linewidth=linewidth, alpha=alpha)

# Baseline (proporción de la clase positiva)
baseline_precision = (y_test == 1).sum() / len(y_test)
ax2.axhline(baseline_precision, color='gray', linestyle='--', 
            label=f'Baseline (P = {baseline_precision:.3f})',
            linewidth=1.5, alpha=0.5)

ax2.set_xlabel('Recall (Sensibilidad)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax2.set_title('Curva Precision-Recall', fontsize=14, fontweight='bold', pad=15)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax2.grid(alpha=0.3, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])

# Área sombreada para el mejor modelo
if best_model_name in models_for_curves:
    y_proba_best = best_model.predict_proba(X_test)[:, 1]
    precision_best, recall_best, _ = precision_recall_curve(y_test, y_proba_best)
    ax2.fill_between(recall_best, 0, precision_best, alpha=0.2, label='_nolegend_')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'curvas_evaluacion.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✓ Curvas guardadas en: {FIGURES_DIR / 'curvas_evaluacion.png'}")

print(f"\n INTERPRETACIÓN DE LAS CURVAS:")
print(f"\n  Curva ROC:")
print(f"    • Mide la capacidad del modelo de distinguir entre clases")
print(f"    • AUC = 1.0: Clasificación perfecta")
print(f"    • AUC = 0.5: Clasificación aleatoria")
print(f"    • Mejor modelo: {best_model_name}")

print(f"\n  Curva Precision-Recall:")
print(f"    • Más informativa para clases desbalanceadas")
print(f"    • Muestra trade-off entre Precision y Recall")
print(f"    • Importante cuando los FP y FN tienen costos diferentes")

print("\n" + "="*80)

# ============================================
# 11. GUARDAR MODELO FINAL Y CONCLUSIONES
# ============================================

print("\n" + "="*80)
print("GUARDANDO MODELO FINAL")
print("="*80)

# Nombre del archivo del modelo
model_filename = f'modelo_final_{best_model_name.lower().replace(" ", "_")}.pkl'
model_path = MODELS_DIR / model_filename

# Guardar modelo
joblib.dump(best_model, model_path)
print(f"\n✓ Modelo guardado exitosamente:")
print(f"  Ubicación: {model_path.relative_to(PROJECT_ROOT)}")
print(f"  Tamaño: {model_path.stat().st_size / 1024:.2f} KB")

# Verificar que el pipeline de transformación existe
pipeline_path = MODELS_DIR / 'pipeline_transformacion.pkl'
if pipeline_path.exists():
    print(f"\n✓ Pipeline de transformación disponible:")
    print(f"  Ubicación: {pipeline_path.relative_to(PROJECT_ROOT)}")
    print(f"  Tamaño: {pipeline_path.stat().st_size / 1024:.2f} KB")
else:
    print(f"\n  Pipeline de transformación NO encontrado")
    print(f"   Ejecuta 03_ft_engineering.ipynb para generarlo")

# Crear resumen del proyecto
summary = {
    'modelo_seleccionado': best_model_name,
    'criterio_seleccion': 'Máximo Recall en Test',
    'fecha_entrenamiento': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'metricas_test': {
        'recall': float(best_metrics['test_recall']),
        'precision': float(best_metrics['test_precision']),
        'f1_score': float(best_metrics['test_f1']),
        'auc_roc': float(best_metrics['test_auc_roc']),
        'accuracy': float(best_metrics['test_accuracy'])
    },
    'matriz_confusion': {
        'true_negatives': int(best_metrics['test_tn']),
        'false_positives': int(best_metrics['test_fp']),
        'false_negatives': int(best_metrics['test_fn']),
        'true_positives': int(best_metrics['test_tp'])
    },
    'dataset': {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'num_features': X_train.shape[1],
        'balanceado': 'SMOTE aplicado en train'
    }
}

# Guardar resumen
import json
summary_path = MODELS_DIR / 'resumen_modelo.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Resumen guardado en: {summary_path.relative_to(PROJECT_ROOT)}")

print("\n" + "="*80)
print("CONCLUSIONES Y RECOMENDACIONES FINALES")
print("="*80)

# Métricas finales
print(f"\n MODELO FINAL: {best_model_name}")
print(f"\n MÉTRICAS EN TEST (Distribución Real):")
print(f"  {'Métrica':<15} {'Valor':>10} {'Interpretación'}")
print(f"  {'-'*60}")
print(f"  {'Recall':<15} {best_metrics['test_recall']:>10.4f}  ⭐ Detecta {best_metrics['test_recall']*100:.1f}% de impagos")
print(f"  {'Precision':<15} {best_metrics['test_precision']:>10.4f}     {best_metrics['test_precision']*100:.1f}% de alertas son correctas")
print(f"  {'F1-Score':<15} {best_metrics['test_f1']:>10.4f}     Balance Precision-Recall")
print(f"  {'AUC-ROC':<15} {best_metrics['test_auc_roc']:>10.4f}     Capacidad discriminativa")
print(f"  {'Accuracy':<15} {best_metrics['test_accuracy']:>10.4f}     Exactitud global")

# Interpretación de negocio
total_no_pago = int(best_metrics['test_tn'] + best_metrics['test_fn'])
detectados = int(best_metrics['test_tn'])
no_detectados = int(best_metrics['test_fn'])

print(f"\n IMPACTO DE NEGOCIO:")
print(f"\n  De {total_no_pago} clientes que NO pagaron en test:")
print(f"    ✓ Detectados y rechazados:  {detectados:>4} ({detectados/total_no_pago*100:>5.1f}%)  ← PÉRDIDA EVITADA")
print(f"    ✗ NO detectados (aprobados): {no_detectados:>4} ({no_detectados/total_no_pago*100:>5.1f}%)  ← PÉRDIDA REAL")

# Comparación con baseline
baseline_recall = metrics_baseline['test_recall']
improvement = (best_metrics['test_recall'] - baseline_recall)
print(f"\n MEJORA vs BASELINE:")
print(f"  Baseline (siempre predice 'Pagó'):  Recall = {baseline_recall:.4f} (0%)")
print(f"  {best_model_name:<30} Recall = {best_metrics['test_recall']:.4f} ({best_metrics['test_recall']*100:.1f}%)")
print(f"  Mejora absoluta: +{improvement:.4f}")
print(f"  Clientes adicionales detectados: {int(improvement * total_no_pago)}")

# Recomendaciones
print(f"\n RECOMENDACIONES PARA PRODUCCIÓN:")
print(f"\n  1. IMPLEMENTACIÓN:")
print(f"     • Cargar pipeline: joblib.load('{pipeline_path.name}')")
print(f"     • Cargar modelo: joblib.load('{model_filename}')")
print(f"     • Aplicar pipeline a datos nuevos")
print(f"     • Predecir con el modelo")

print(f"\n  2. AJUSTE DE THRESHOLD:")
print(f"     • Threshold actual: 0.5 (default)")
print(f"     • Considerar reducir a 0.3-0.4 para mayor Recall")
print(f"     • Trade-off: Más detección, pero más rechazos")

print(f"\n  3. MONITOREO:")
print(f"     • Evaluar performance cada mes")
print(f"     • Detectar data drift")
print(f"     • Re-entrenar si Recall cae < {best_metrics['test_recall']*0.9:.3f}")

print(f"\n  4. PRÓXIMAS MEJORAS:")
print(f"     • Optimización de hiperparámetros (GridSearch)")
print(f"     • Feature engineering adicional")
print(f"     • Modelos ensamblados (Stacking/Voting)")
print(f"     • Validación cruzada estratificada")

print(f"\n  5. CONSIDERACIONES DE NEGOCIO:")
print(f"     • Definir costo real de FN vs FP")
print(f"     • Ajustar threshold según costo")
print(f"     • Implementar sistema de alertas tempranas")
print(f"     • Estratificar por monto de crédito")

print("\n" + "="*80)
print(" MODELAMIENTO COMPLETADO EXITOSAMENTE")
print("="*80)

print(f"\n📁 ARCHIVOS GENERADOS:")
print(f"  • Modelo:              {model_path.relative_to(PROJECT_ROOT)}")
print(f"  • Métricas:            {MODELS_DIR / 'metricas_comparacion.csv'}")
print(f"  • Resumen:             {summary_path.relative_to(PROJECT_ROOT)}")
if hasattr(best_model, 'feature_importances_'):
    print(f"  • Feature Importance:  models/feature_importance_*.csv")
print(f"  • Visualizaciones:     {FIGURES_DIR.relative_to(PROJECT_ROOT)}/*.png")

print(f"\n✓ Modelo listo para evaluación y deployment!")
print("\n" + "="*80)





#from fastapi import FastAPI
#import uvicorn

#app = FastAPI()

#@app.get("/")
#def read_root():
#    return {"message": "Model Deployment API"}

#if __name__ == "__main__":
#    uvicorn.run(app, host="0.0.0.0", port=8000)
