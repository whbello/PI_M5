"""
Script de Entrenamiento y Evaluación de Modelos

Fecha: 2026-Febrero
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from datetime import datetime
import logging
import time

# Modelos
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# Evaluación
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================
# CONFIGURACIÓN DE RUTAS
# ============================================

SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_PROCESSED = SCRIPT_DIR / 'data' / 'processed'
MODELS_DIR = SCRIPT_DIR / 'models'
REPORTS_DIR = SCRIPT_DIR / 'reports'

# Crear carpetas
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# FUNCIONES
# ============================================

def load_data():
    """Cargar datos procesados"""
    logger.info("\n" + "="*80)
    logger.info("CARGA DE DATOS PROCESADOS")
    logger.info("="*80)
    
    # Verificar archivos
    archivos_requeridos = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']
    archivos_faltantes = [f for f in archivos_requeridos if not (DATA_PROCESSED / f).exists()]
    
    if archivos_faltantes:
        logger.error(f"Archivos faltantes: {archivos_faltantes}")
        logger.error(f"Ejecuta primero: 03_ft_engineering.ipynb")
        raise FileNotFoundError("Datos procesados no encontrados")
    
    # Cargar
    X_train = pd.read_csv(DATA_PROCESSED / 'X_train.csv')
    X_test = pd.read_csv(DATA_PROCESSED / 'X_test.csv')
    y_train = pd.read_csv(DATA_PROCESSED / 'y_train.csv').squeeze()
    y_test = pd.read_csv(DATA_PROCESSED / 'y_test.csv').squeeze()
    
    logger.info(f"\n✓ Datos cargados:")
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_test:  {X_test.shape}")
    logger.info(f"  y_train: {len(y_train)} ({y_train.value_counts()[1]/len(y_train)*100:.1f}% clase 1)")
    logger.info(f"  y_test:  {len(y_test)} ({y_test.value_counts()[1]/len(y_test)*100:.1f}% clase 1)")
    
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Entrenar y evaluar un modelo"""
    logger.info(f"\n{'='*80}")
    logger.info(f"ENTRENANDO: {model_name}")
    logger.info(f"{'='*80}")
    
    # Entrenar
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Predecir
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Probabilidades
    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_train_proba = y_train_pred
        y_test_proba = y_test_pred
    
    # Métricas
    metrics = {
        'model': model_name,
        'training_time': training_time,
        
        # Train
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred, zero_division=0),
        'train_recall': recall_score(y_train, y_train_pred, zero_division=0),
        'train_f1': f1_score(y_train, y_train_pred, zero_division=0),
        'train_auc_roc': roc_auc_score(y_train, y_train_proba) if hasattr(model, 'predict_proba') else None,
        
        # Test
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
        'test_recall': recall_score(y_test, y_test_pred, zero_division=0),
        'test_f1': f1_score(y_test, y_test_pred, zero_division=0),
        'test_auc_roc': roc_auc_score(y_test, y_test_proba) if hasattr(model, 'predict_proba') else None,
    }
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics['test_tn'] = int(tn)
    metrics['test_fp'] = int(fp)
    metrics['test_fn'] = int(fn)
    metrics['test_tp'] = int(tp)
    
    # Mostrar resultados
    logger.info(f"\n  Tiempo: {training_time:.2f}s")
    logger.info(f"\n MÉTRICAS EN TEST:")
    logger.info(f"  Accuracy:  {metrics['test_accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['test_precision']:.4f}")
    logger.info(f"  Recall:    {metrics['test_recall']:.4f}  ⭐")
    logger.info(f"  F1-Score:  {metrics['test_f1']:.4f}")
    logger.info(f"  AUC-ROC:   {metrics['test_auc_roc']:.4f}" if metrics['test_auc_roc'] else "")
    
    logger.info(f"\n Matriz de Confusión:")
    logger.info(f"  TN: {tn:>5}  |  FP: {fp:>5}")
    logger.info(f"  FN: {fn:>5}  |  TP: {tp:>5}")
    
    return model, metrics


def main():
    """Función principal"""
    logger.info("="*80)
    logger.info("ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
    logger.info("="*80)
    
    # Cargar datos
    X_train, X_test, y_train, y_test = load_data()
    
    # Definir modelos
    models = {
        'Baseline': DummyClassifier(strategy='most_frequent', random_state=42),
        
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        ),
        
        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_split=100,
            min_samples_leaf=50,
            class_weight='balanced'
        ),
        
        'Random Forest': RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=15,
            min_samples_split=50,
            min_samples_leaf=20,
            class_weight='balanced',
            n_jobs=-1
        ),
        
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=50,
            min_samples_leaf=20
        ),
        
        'XGBoost': XGBClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            eval_metric='logloss',
            use_label_encoder=False
        )
    }
    
    # Entrenar todos
    results = []
    trained_models = {}
    
    for name, model in models.items():
        trained_model, metrics = evaluate_model(
            model, X_train, y_train, X_test, y_test, name
        )
        results.append(metrics)
        trained_models[name] = trained_model
    
    # DataFrame de resultados
    results_df = pd.DataFrame(results)
    
    # Mejor modelo por Recall
    best_idx = results_df['test_recall'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model']
    best_model = trained_models[best_model_name]
    
    logger.info("\n" + "="*80)
    logger.info("COMPARACIÓN DE MODELOS")
    logger.info("="*80)
    logger.info(f"\n{results_df[['model', 'test_recall', 'test_precision', 'test_f1', 'test_auc_roc']].to_string(index=False)}")
    
    logger.info("\n" + "="*80)
    logger.info(f" MEJOR MODELO: {best_model_name}")
    logger.info("="*80)
    logger.info(f"  Recall:    {results_df.loc[best_idx, 'test_recall']:.4f}")
    logger.info(f"  Precision: {results_df.loc[best_idx, 'test_precision']:.4f}")
    logger.info(f"  F1-Score:  {results_df.loc[best_idx, 'test_f1']:.4f}")
    logger.info(f"  AUC-ROC:   {results_df.loc[best_idx, 'test_auc_roc']:.4f}")
    
    # Guardar modelo
    model_filename = f"modelo_final_{best_model_name.lower().replace(' ', '_')}.pkl"
    model_path = MODELS_DIR / model_filename
    joblib.dump(best_model, model_path)
    logger.info(f"\n✓ Modelo guardado: {model_path}")
    
    # Guardar métricas
    metrics_path = MODELS_DIR / 'metricas_comparacion.csv'
    results_df.to_csv(metrics_path, index=False)
    logger.info(f"✓ Métricas guardadas: {metrics_path}")
    
    # Guardar resumen
    summary = {
        'timestamp': datetime.now().isoformat(),
        'best_model': best_model_name,
        'test_recall': float(results_df.loc[best_idx, 'test_recall']),
        'test_precision': float(results_df.loc[best_idx, 'test_precision']),
        'test_f1': float(results_df.loc[best_idx, 'test_f1']),
        'test_auc_roc': float(results_df.loc[best_idx, 'test_auc_roc']),
        'confusion_matrix': {
            'tn': int(results_df.loc[best_idx, 'test_tn']),
            'fp': int(results_df.loc[best_idx, 'test_fp']),
            'fn': int(results_df.loc[best_idx, 'test_fn']),
            'tp': int(results_df.loc[best_idx, 'test_tp'])
        }
    }
    
    summary_path = MODELS_DIR / 'resumen_modelo.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"✓ Resumen guardado: {summary_path}")
    
    logger.info("\n" + "="*80)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("="*80)


if __name__ == "__main__":
    main()