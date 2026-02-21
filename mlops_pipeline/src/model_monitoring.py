"""
Objetivo del Avance-3

Implementar un sistema de monitoreo para detectar **Data Drift** (cambios en la distribución de datos)
que puedan afectar el desempeño del modelo en producción.

Este módulo implementa:
- Cálculo de métricas de data drift (KS, PSI, JS Divergence)
- Monitoreo de distribución de features
- Detección de cambios en la población
- Generación de reportes y alertas
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Estadística
from scipy import stats
from scipy.spatial.distance import jensenshannon

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN GLOBAL
# ============================================

class MonitoringConfig:
    """Configuración del sistema de monitoreo"""
    
    # Rutas
    PROJECT_ROOT = Path.cwd()
    DATA_DIR = PROJECT_ROOT / 'data'
    MONITORING_DIR = DATA_DIR / 'monitoring'
    DRIFT_REPORTS_DIR = MONITORING_DIR / 'drift_reports'
    MODELS_DIR = PROJECT_ROOT / 'models'
    
    # Umbrales de alerta
    THRESHOLD_KS = 0.2        # Kolmogorov-Smirnov
    THRESHOLD_PSI = 0.2       # Population Stability Index
    THRESHOLD_JS = 0.15       # Jensen-Shannon Divergence
    THRESHOLD_CHI2 = 0.05     # P-value para Chi-cuadrado
    
    # Categorías de drift
    DRIFT_LOW = "🟢 Bajo (Sin riesgo)"
    DRIFT_MEDIUM = "🟡 Moderado (Revisar)"
    DRIFT_HIGH = "🔴 Alto (Reentrenar)"
    
    @classmethod
    def setup_directories(cls):
        """Crear directorios necesarios"""
        cls.MONITORING_DIR.mkdir(parents=True, exist_ok=True)
        cls.DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# CLASE: CALCULADORA DE DRIFT
# ============================================
class DriftCalculator:
    """Calcula métricas de data drift para variables numéricas y categóricas"""
    
    @staticmethod
    def kolmogorov_smirnov(baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
        """
        Test de Kolmogorov-Smirnov para variables numéricas.
        
        Mide la máxima diferencia entre las distribuciones acumuladas.
        
        Returns:
            (statistic, p_value): estadístico KS y p-value
        """
        statistic, pvalue = stats.ks_2samp(baseline, current)
        return statistic, pvalue
    
    @staticmethod
    def population_stability_index(baseline: np.ndarray, current: np.ndarray, 
                                bins: int = 10) -> float:
        """
        Population Stability Index (PSI) para variables numéricas.
        
        PSI mide el cambio en la distribución usando bins.
        
        Interpretación:
        - PSI < 0.1: Sin cambio significativo
        - 0.1 ≤ PSI < 0.2: Cambio moderado
        - PSI ≥ 0.2: Cambio significativo (requiere investigación)
        
        Returns:
            psi_value: valor del PSI
        """
        # Crear bins basados en la baseline
        try:
            breakpoints = np.percentile(baseline, np.linspace(0, 100, bins + 1))
            breakpoints = np.unique(breakpoints)  # Eliminar duplicados
            
            if len(breakpoints) <= 2:
                # Si hay muy pocos breakpoints únicos, retornar 0
                return 0.0
            
            # Calcular histogramas
            baseline_hist, _ = np.histogram(baseline, bins=breakpoints)
            current_hist, _ = np.histogram(current, bins=breakpoints)
            
            # Evitar división por cero
            baseline_hist = baseline_hist + 1e-10
            current_hist = current_hist + 1e-10
            
            # Normalizar (convertir a proporciones)
            baseline_pct = baseline_hist / len(baseline)
            current_pct = current_hist / len(current)
            
            # Calcular PSI
            psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
            
            return psi
        
        except Exception as e:
            print(f"Error calculando PSI: {e}")
            return 0.0
    
    @staticmethod
    def jensen_shannon_divergence(baseline: np.ndarray, current: np.ndarray, 
                                bins: int = 10) -> float:
        """
        Jensen-Shannon Divergence para variables numéricas.
        
        Mide la similitud entre dos distribuciones de probabilidad.
        
        Returns:
            js_divergence: valor entre 0 (idénticas) y 1 (completamente diferentes)
        """
        try:
            # Crear bins
            all_data = np.concatenate([baseline, current])
            breakpoints = np.linspace(all_data.min(), all_data.max(), bins + 1)
            
            # Calcular histogramas
            baseline_hist, _ = np.histogram(baseline, bins=breakpoints)
            current_hist, _ = np.histogram(current, bins=breakpoints)
            
            # Normalizar
            baseline_prob = (baseline_hist + 1e-10) / (baseline_hist.sum() + 1e-10 * bins)
            current_prob = (current_hist + 1e-10) / (current_hist.sum() + 1e-10 * bins)
            
            # Calcular JS divergence
            js_div = jensenshannon(baseline_prob, current_prob)
            
            return js_div
        
        except Exception as e:
            print(f"Error calculando JS Divergence: {e}")
            return 0.0
    
    @staticmethod
    def chi_square_test(baseline: pd.Series, current: pd.Series) -> Tuple[float, float]:
        """
        Test Chi-cuadrado para variables categóricas.
        
        Mide si hay diferencia significativa en las frecuencias de categorías.
        
        Returns:
            (statistic, p_value): estadístico chi2 y p-value
        """
        try:
            # Obtener todas las categorías únicas
            all_categories = pd.concat([baseline, current]).unique()
            
            # Contar frecuencias
            baseline_counts = baseline.value_counts()
            current_counts = current.value_counts()
            
            # Asegurar que ambos tienen todas las categorías
            baseline_freq = [baseline_counts.get(cat, 0) for cat in all_categories]
            current_freq = [current_counts.get(cat, 0) for cat in all_categories]
            
            # Crear tabla de contingencia
            contingency_table = np.array([baseline_freq, current_freq])
            
            # Test chi-cuadrado
            statistic, pvalue, dof, expected = stats.chi2_contingency(contingency_table)
            
            return statistic, pvalue
        
        except Exception as e:
            print(f"Error calculando Chi-cuadrado: {e}")
            return 0.0, 1.0

# ============================================
# CLASE: MONITOR DE DRIFT
# ============================================

class DataDriftMonitor:
    """Sistema de monitoreo de data drift"""
    
    def __init__(self, baseline_data: pd.DataFrame, model_path: Optional[Path] = None):
        """
        Inicializa el monitor con datos baseline.
        
        Args:
            baseline_data: DataFrame con datos de entrenamiento (baseline)
            model_path: Ruta al modelo entrenado (opcional)
        """
        self.baseline_data = baseline_data
        self.model_path = model_path
        self.model = None
        
        # Cargar modelo si se proporciona
        if model_path and model_path.exists():
            self.model = joblib.load(model_path)
        
        # Identificar tipos de variables
        self.numeric_features = baseline_data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = baseline_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        print(f"✓ Monitor inicializado")
        print(f"  Features numéricas: {len(self.numeric_features)}")
        print(f"  Features categóricas: {len(self.categorical_features)}")
    
    def calculate_drift_numeric(self, feature: str, current_data: pd.DataFrame) -> Dict:
        """
        Calcula drift para una variable numérica.
        
        Returns:
            dict con métricas de drift
        """
        baseline_values = self.baseline_data[feature].dropna().values
        current_values = current_data[feature].dropna().values
        
        # Calcular métricas
        ks_stat, ks_pval = DriftCalculator.kolmogorov_smirnov(baseline_values, current_values)
        psi = DriftCalculator.population_stability_index(baseline_values, current_values)
        js_div = DriftCalculator.jensen_shannon_divergence(baseline_values, current_values)
        
        # Determinar nivel de drift (basado en PSI principalmente)
        if psi < MonitoringConfig.THRESHOLD_PSI * 0.5:
            drift_level = MonitoringConfig.DRIFT_LOW
        elif psi < MonitoringConfig.THRESHOLD_PSI:
            drift_level = MonitoringConfig.DRIFT_MEDIUM
        else:
            drift_level = MonitoringConfig.DRIFT_HIGH
        
        return {
            'feature': feature,
            'type': 'numeric',
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'psi': psi,
            'js_divergence': js_div,
            'drift_level': drift_level,
            'baseline_mean': baseline_values.mean(),
            'current_mean': current_values.mean(),
            'baseline_std': baseline_values.std(),
            'current_std': current_values.std()
        }
    
    def calculate_drift_categorical(self, feature: str, current_data: pd.DataFrame) -> Dict:
        """
        Calcula drift para una variable categórica.
        
        Returns:
            dict con métricas de drift
        """
        baseline_values = self.baseline_data[feature].dropna()
        current_values = current_data[feature].dropna()
        
        # Chi-cuadrado
        chi2_stat, chi2_pval = DriftCalculator.chi_square_test(baseline_values, current_values)
        
        # Determinar nivel de drift (basado en p-value)
        if chi2_pval > 0.1:
            drift_level = MonitoringConfig.DRIFT_LOW
        elif chi2_pval > MonitoringConfig.THRESHOLD_CHI2:
            drift_level = MonitoringConfig.DRIFT_MEDIUM
        else:
            drift_level = MonitoringConfig.DRIFT_HIGH
        
        # Calcular cambios en distribución
        baseline_dist = baseline_values.value_counts(normalize=True).to_dict()
        current_dist = current_values.value_counts(normalize=True).to_dict()
        
        return {
            'feature': feature,
            'type': 'categorical',
            'chi2_statistic': chi2_stat,
            'chi2_pvalue': chi2_pval,
            'drift_level': drift_level,
            'baseline_distribution': baseline_dist,
            'current_distribution': current_dist,
            'n_categories_baseline': len(baseline_dist),
            'n_categories_current': len(current_dist)
        }
    
    def monitor_dataset(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """
        Monitorea drift en todo el dataset.
        
        Returns:
            DataFrame con métricas de drift por feature
        """
        print("\n" + "="*80)
        print("ANÁLISIS DE DATA DRIFT")
        print("="*80)
        
        results = []
        
        # Monitorear variables numéricas
        print(f"\n Analizando {len(self.numeric_features)} features numéricas...")
        for feature in self.numeric_features:
            if feature in current_data.columns:
                try:
                    drift_metrics = self.calculate_drift_numeric(feature, current_data)
                    results.append(drift_metrics)
                    
                    # Mostrar si hay drift significativo
                    if MonitoringConfig.DRIFT_HIGH in drift_metrics['drift_level']:
                        print(f"  🔴 {feature}: PSI = {drift_metrics['psi']:.4f}")
                    elif MonitoringConfig.DRIFT_MEDIUM in drift_metrics['drift_level']:
                        print(f"  🟡  {feature}: PSI = {drift_metrics['psi']:.4f}")
                
                except Exception as e:
                    print(f"  🟡  Error en {feature}: {e}")
        
        # Monitorear variables categóricas
        if self.categorical_features:
            print(f"\n Analizando {len(self.categorical_features)} features categóricas...")
            for feature in self.categorical_features:
                if feature in current_data.columns:
                    try:
                        drift_metrics = self.calculate_drift_categorical(feature, current_data)
                        results.append(drift_metrics)
                        
                        # Mostrar si hay drift significativo
                        if MonitoringConfig.DRIFT_HIGH in drift_metrics['drift_level']:
                            print(f"  🔴 {feature}: p-value = {drift_metrics['chi2_pvalue']:.4f}")
                        elif MonitoringConfig.DRIFT_MEDIUM in drift_metrics['drift_level']:
                            print(f"  🟡  {feature}: p-value = {drift_metrics['chi2_pvalue']:.4f}")
                    
                    except Exception as e:
                        print(f"  🟡  Error en {feature}: {e}")
        
        # Crear DataFrame de resultados
        drift_df = pd.DataFrame(results)
        
        # Resumen
        print(f"\n RESUMEN:")
        if len(drift_df) > 0:
            high_drift = drift_df['drift_level'].str.contains('Alto').sum()
            medium_drift = drift_df['drift_level'].str.contains('Moderado').sum()
            low_drift = drift_df['drift_level'].str.contains('Bajo').sum()
            
            print(f"  🔴 Drift Alto:     {high_drift:>3} features")
            print(f"  🟡 Drift Moderado: {medium_drift:>3} features")
            print(f"  🟢 Drift Bajo:     {low_drift:>3} features")
            
            if high_drift > 0:
                print(f"\n  🟡  ALERTA: {high_drift} features con drift significativo")
                print(f"     Recomendación: Considerar reentrenamiento del modelo")
        
        print("="*80)
        
        return drift_df
    
    def generate_report(self, drift_df: pd.DataFrame, report_name: str = None) -> Path:
        """
        Genera reporte de drift en formato JSON.
        
        Returns:
            Path al archivo del reporte
        """
        if report_name is None:
            report_name = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_path = MonitoringConfig.DRIFT_REPORTS_DIR / report_name
        
        # Preparar datos para JSON
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_features': len(drift_df),
                'high_drift': int(drift_df['drift_level'].str.contains('Alto').sum()),
                'medium_drift': int(drift_df['drift_level'].str.contains('Moderado').sum()),
                'low_drift': int(drift_df['drift_level'].str.contains('Bajo').sum())
            },
            'features': drift_df.to_dict('records')
        }
        
        # Guardar reporte
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"\n✓ Reporte guardado: {report_path}")
        
        return report_path
    
    def visualize_drift(self, feature: str, current_data: pd.DataFrame, 
                    save_path: Optional[Path] = None):
        """
        Visualiza drift para una feature específica.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Data Drift: {feature}', fontsize=14, fontweight='bold')
        
        baseline_values = self.baseline_data[feature].dropna()
        current_values = current_data[feature].dropna()
        
        if feature in self.numeric_features:
            # Histogramas superpuestos
            axes[0].hist(baseline_values, bins=30, alpha=0.6, label='Baseline', 
                        color='#3498db', edgecolor='black')
            axes[0].hist(current_values, bins=30, alpha=0.6, label='Current',
                        color='#e74c3c', edgecolor='black')
            axes[0].set_xlabel(feature)
            axes[0].set_ylabel('Frecuencia')
            axes[0].set_title('Distribución')
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            
            # CDFs (distribuciones acumuladas)
            axes[1].hist(baseline_values, bins=50, cumulative=True, density=True,
                        histtype='step', linewidth=2, label='Baseline', color='#3498db')
            axes[1].hist(current_values, bins=50, cumulative=True, density=True,
                        histtype='step', linewidth=2, label='Current', color='#e74c3c')
            axes[1].set_xlabel(feature)
            axes[1].set_ylabel('Probabilidad Acumulada')
            axes[1].set_title('CDF (para KS test)')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
        
        else:  # Categórica
            # Distribuciones de frecuencias
            baseline_dist = baseline_values.value_counts()
            current_dist = current_values.value_counts()
            
            # Combinar índices
            all_cats = sorted(set(baseline_dist.index) | set(current_dist.index))
            
            x = np.arange(len(all_cats))
            width = 0.35
            
            baseline_vals = [baseline_dist.get(cat, 0) for cat in all_cats]
            current_vals = [current_dist.get(cat, 0) for cat in all_cats]
            
            axes[0].bar(x - width/2, baseline_vals, width, label='Baseline',
                    color='#3498db', edgecolor='black', alpha=0.8)
            axes[0].bar(x + width/2, current_vals, width, label='Current',
                    color='#e74c3c', edgecolor='black', alpha=0.8)
            axes[0].set_xlabel('Categorías')
            axes[0].set_ylabel('Frecuencia')
            axes[0].set_title('Frecuencias')
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(all_cats, rotation=45, ha='right')
            axes[0].legend()
            axes[0].grid(axis='y', alpha=0.3)
            
            # Proporciones
            baseline_pct = (baseline_dist / len(baseline_values) * 100)
            current_pct = (current_dist / len(current_values) * 100)
            
            baseline_vals_pct = [baseline_pct.get(cat, 0) for cat in all_cats]
            current_vals_pct = [current_pct.get(cat, 0) for cat in all_cats]
            
            axes[1].bar(x - width/2, baseline_vals_pct, width, label='Baseline',
                    color='#3498db', edgecolor='black', alpha=0.8)
            axes[1].bar(x + width/2, current_vals_pct, width, label='Current',
                    color='#e74c3c', edgecolor='black', alpha=0.8)
            axes[1].set_xlabel('Categorías')
            axes[1].set_ylabel('Porcentaje (%)')
            axes[1].set_title('Proporciones')
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(all_cats, rotation=45, ha='right')
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

"""
Niveles de Alerta

- 🟢 **Bajo:** Sin riesgo, no requiere acción
- 🟡 **Moderado:** Revisar, monitorear de cerca
- 🔴 **Alto:** Reentrenamiento necesario
"""
# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def main():
    """Función principal de monitoreo"""
    
    # Configurar directorios
    MonitoringConfig.setup_directories()
    
    print("="*80)
    print("SISTEMA DE MONITOREO DE DATA DRIFT")
    print("="*80)
    
    # Cargar datos baseline (datos de entrenamiento)
    baseline_path = MonitoringConfig.DATA_DIR / 'processed' / 'X_train.csv'
    
    if not baseline_path.exists():
        print(f"\nX ERROR: No se encuentra el archivo baseline")
        print(f"   Esperado en: {baseline_path}")
        return
    
    print(f"\n Cargando datos baseline...")
    baseline_data = pd.read_csv(baseline_path)
    print(f"✓ Baseline cargado: {baseline_data.shape}")
    
    # Para este ejemplo, simularemos datos "current" con una muestra del test
    # En producción, estos serían datos nuevos de producción
    current_path = MonitoringConfig.DATA_DIR / 'processed' / 'X_test.csv'
    
    if current_path.exists():
        print(f"\n Cargando datos actuales (simulación con test)...")
        current_data = pd.read_csv(current_path)
        print(f"✓ Datos actuales cargados: {current_data.shape}")
    else:
        print(f"\n🟡  No hay datos actuales. Usando muestra de baseline.")
        current_data = baseline_data.sample(frac=0.2, random_state=42)
    
    # Inicializar monitor
    monitor = DataDriftMonitor(baseline_data)
    
    # Analizar drift
    drift_df = monitor.monitor_dataset(current_data)
    
    # Generar reporte
    if len(drift_df) > 0:
        report_path = monitor.generate_report(drift_df)
        
        # Guardar también como CSV para fácil acceso
        csv_path = report_path.with_suffix('.csv')
        drift_df.to_csv(csv_path, index=False)
        print(f"✓ CSV guardado: {csv_path}")
    
    print("\n Monitoreo completado")


if __name__ == "__main__":
    main()




