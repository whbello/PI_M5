#  APLICACIÓN STREAMLIT - DASHBOARD DE MONITOREO

"""
Dashboard de Monitoreo de Data Drift
Aplicación Streamlit para visualización interactiva
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Importar el módulo de monitoreo
import sys
sys.path.append(str(Path(__file__).parent))
from model_monitoring import DataDriftMonitor, MonitoringConfig, DriftCalculator

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================

st.set_page_config(
    page_title="Monitor de Data Drift - Créditos",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        padding: 15px;
        border-left: 5px solid #f44336;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-medium {
        background-color: #fff3e0;
        padding: 15px;
        border-left: 5px solid #ff9800;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-low {
        background-color: #e8f5e9;
        padding: 15px;
        border-left: 5px solid #4caf50;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# FUNCIONES AUXILIARES
# ============================================

@st.cache_data
def load_baseline_data():
    """Carga datos baseline (entrenamiento)"""
    baseline_path = MonitoringConfig.DATA_DIR / 'processed' / 'X_train.csv'
    if baseline_path.exists():
        return pd.read_csv(baseline_path)
    return None

@st.cache_data
def load_current_data():
    """Carga datos actuales (test o producción)"""
    current_path = MonitoringConfig.DATA_DIR / 'processed' / 'X_test.csv'
    if current_path.exists():
        return pd.read_csv(current_path)
    return None

@st.cache_resource
def initialize_monitor(baseline_data):
    """Inicializa el monitor de drift"""
    return DataDriftMonitor(baseline_data)

def load_latest_report():
    """Carga el reporte de drift más reciente"""
    reports_dir = MonitoringConfig.DRIFT_REPORTS_DIR
    if reports_dir.exists():
        json_files = list(reports_dir.glob('drift_report_*.json'))
        if json_files:
            latest_report = max(json_files, key=lambda p: p.stat().st_mtime)
            with open(latest_report, 'r') as f:
                return json.load(f)
    return None

def get_drift_color(drift_level):
    """Retorna color según nivel de drift"""
    if '🔴' in drift_level or 'Alto' in drift_level:
        return '#f44336'
    elif '🟡' in drift_level or 'Moderado' in drift_level:
        return '#ff9800'
    else:
        return '#4caf50'

def create_gauge_chart(value, title, threshold_low, threshold_high):
    """Crea un gráfico de gauge (velocímetro)"""
    
    # Determinar color
    if value < threshold_low:
        color = '#4caf50'
    elif value < threshold_high:
        color = '#ff9800'
    else:
        color = '#f44336'
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'size': 20}},
        delta={'reference': threshold_low},
        gauge={
            'axis': {'range': [0, max(1.0, threshold_high * 1.5)]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, threshold_low], 'color': '#e8f5e9'},
                {'range': [threshold_low, threshold_high], 'color': '#fff3e0'},
                {'range': [threshold_high, max(1.0, threshold_high * 1.5)], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_high
            }
        }
    ))
    
    fig.update_layout(height=250)
    return fig

# ============================================
# HEADER
# ============================================

st.markdown('<div class="main-header"> Dashboard de Monitoreo de Data Drift</div>', 
            unsafe_allow_html=True)

st.markdown("""
Este dashboard monitorea cambios en la distribución de datos que podrían afectar 
el desempeño del modelo de predicción de pagos de créditos.
""")

# ============================================
# SIDEBAR - CONFIGURACIÓN
# ============================================

st.sidebar.header("Configuración")

# Cargar datos
baseline_data = load_baseline_data()
current_data = load_current_data()

if baseline_data is None:
    st.error("X No se encontraron datos baseline. Ejecuta primero ft_engineering.ipynb")
    st.stop()

if current_data is None:
    st.warning("🟡 No hay datos actuales. Usando muestra de baseline para demostración.")
    current_data = baseline_data.sample(frac=0.3, random_state=42)

# Información de datos
st.sidebar.metric("Registros Baseline", f"{len(baseline_data):,}")
st.sidebar.metric("Registros Actuales", f"{len(current_data):,}")

# Actualizar análisis
if st.sidebar.button("🔵 Actualizar Análisis", type="primary"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# Filtros
st.sidebar.header("Filtros")
drift_filter = st.sidebar.multiselect(
    "Nivel de Drift",
    ["Alto", "Moderado", "Bajo"],
    default=["Alto", "Moderado", "Bajo"]
)

feature_type_filter = st.sidebar.multiselect(
    "Tipo de Variable",
    ["numeric", "categorical"],
    default=["numeric", "categorical"]
)

# ============================================
# ANÁLISIS DE DRIFT
# ============================================

# Inicializar monitor
monitor = initialize_monitor(baseline_data)

# Calcular drift
with st.spinner("Analizando data drift..."):
    drift_df = monitor.monitor_dataset(current_data)

# Aplicar filtros
filtered_df = drift_df.copy()
if drift_filter:
    filter_pattern = '|'.join(drift_filter)
    filtered_df = filtered_df[filtered_df['drift_level'].str.contains(filter_pattern)]
if feature_type_filter:
    filtered_df = filtered_df[filtered_df['type'].isin(feature_type_filter)]

# ============================================
# MÉTRICAS PRINCIPALES
# ============================================

st.header("Resumen Ejecutivo")

col1, col2, col3, col4 = st.columns(4)

total_features = len(drift_df)
high_drift = drift_df['drift_level'].str.contains('Alto').sum()
medium_drift = drift_df['drift_level'].str.contains('Moderado').sum()
low_drift = drift_df['drift_level'].str.contains('Bajo').sum()

with col1:
    st.metric(
        label="Total Features",
        value=total_features,
        help="Número total de variables monitoreadas"
    )

with col2:
    st.metric(
        label="🔴 Drift Alto",
        value=high_drift,
        delta=f"{high_drift/total_features*100:.1f}%",
        delta_color="inverse"
    )

with col3:
    st.metric(
        label="🟡 Drift Moderado",
        value=medium_drift,
        delta=f"{medium_drift/total_features*100:.1f}%",
        delta_color="off"
    )

with col4:
    st.metric(
        label="🟢 Drift Bajo",
        value=low_drift,
        delta=f"{low_drift/total_features*100:.1f}%",
        delta_color="normal"
    )

# ============================================
# ALERTAS
# ============================================

if high_drift > 0:
    st.markdown('<div class="alert-high">', unsafe_allow_html=True)
    st.error(f"""
    🔴 **ALERTA CRÍTICA:** Se detectaron {high_drift} variables con drift alto.
    
    **Recomendación:** Considerar reentrenamiento del modelo inmediatamente.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

elif medium_drift > 0:
    st.markdown('<div class="alert-medium">', unsafe_allow_html=True)
    st.warning(f"""
    🟡 **ATENCIÓN:** {medium_drift} variables presentan drift moderado.
    
    **Recomendación:** Monitorear de cerca y preparar pipeline de reentrenamiento.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="alert-low">', unsafe_allow_html=True)
    st.success("""
    🟢 **SISTEMA ESTABLE:** No se detectaron cambios significativos en los datos.
    
    El modelo continúa operando dentro de parámetros normales.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================
# VISUALIZACIÓN: DISTRIBUCIÓN DE DRIFT
# ============================================

st.header("Distribución de Drift por Feature")

tab1, tab2, tab3 = st.tabs(["Vista General", "Por Métrica", "Detalle por Feature"])

with tab1:
    # Gráfico de barras con niveles de drift
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución por nivel
        drift_counts = pd.DataFrame({
            'Nivel': ['Alto', 'Moderado', 'Bajo'],
            'Cantidad': [high_drift, medium_drift, low_drift],
            'Color': ['#f44336', '#ff9800', '#4caf50']
        })
        
        fig = px.bar(
            drift_counts,
            x='Nivel',
            y='Cantidad',
            color='Nivel',
            color_discrete_map={'Alto': '#f44336', 'Moderado': '#ff9800', 'Bajo': '#4caf50'},
            title='Distribución de Niveles de Drift',
            text='Cantidad'
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribución por tipo de variable
        type_counts = drift_df['type'].value_counts().reset_index()
        type_counts.columns = ['Tipo', 'Cantidad']
        
        fig = px.pie(
            type_counts,
            names='Tipo',
            values='Cantidad',
            title='Distribución por Tipo de Variable',
            color_discrete_sequence=['#1f77b4', '#ff7f0e']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Visualización de métricas numéricas
    numeric_drift = drift_df[drift_df['type'] == 'numeric'].copy()
    
    if len(numeric_drift) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # PSI promedio
            avg_psi = numeric_drift['psi'].mean()
            fig = create_gauge_chart(
                avg_psi,
                "PSI Promedio",
                MonitoringConfig.THRESHOLD_PSI * 0.5,
                MonitoringConfig.THRESHOLD_PSI
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # KS promedio
            avg_ks = numeric_drift['ks_statistic'].mean()
            fig = create_gauge_chart(
                avg_ks,
                "KS Statistic Promedio",
                MonitoringConfig.THRESHOLD_KS * 0.5,
                MonitoringConfig.THRESHOLD_KS
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # JS promedio
            avg_js = numeric_drift['js_divergence'].mean()
            fig = create_gauge_chart(
                avg_js,
                "JS Divergence Promedio",
                MonitoringConfig.THRESHOLD_JS * 0.5,
                MonitoringConfig.THRESHOLD_JS
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Top features por PSI
        st.subheader("🔵 Top 10 Features con Mayor PSI")
        top_psi = numeric_drift.nlargest(10, 'psi')[['feature', 'psi', 'drift_level']]
        
        fig = px.bar(
            top_psi,
            x='psi',
            y='feature',
            orientation='h',
            color='drift_level',
            color_discrete_map={
                MonitoringConfig.DRIFT_HIGH: '#f44336',
                MonitoringConfig.DRIFT_MEDIUM: '#ff9800',
                MonitoringConfig.DRIFT_LOW: '#4caf50'
            },
            title='Population Stability Index (PSI) por Feature',
            text='psi'
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Detalle por feature individual
    st.subheader(" Análisis Detallado por Feature")
    
    # Selector de feature
    selected_feature = st.selectbox(
        "Selecciona una feature para análisis detallado:",
        filtered_df['feature'].tolist()
    )
    
    if selected_feature:
        feature_data = filtered_df[filtered_df['feature'] == selected_feature].iloc[0]
        
        # Información de la feature
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Información General")
            st.markdown(f"**Feature:** `{feature_data['feature']}`")
            st.markdown(f"**Tipo:** {feature_data['type']}")
            st.markdown(f"**Nivel de Drift:** {feature_data['drift_level']}")
            
            if feature_data['type'] == 'numeric':
                st.markdown("### Métricas")
                st.metric("PSI", f"{feature_data['psi']:.4f}")
                st.metric("KS Statistic", f"{feature_data['ks_statistic']:.4f}")
                st.metric("JS Divergence", f"{feature_data['js_divergence']:.4f}")
                
                st.markdown("### Estadísticas")
                st.metric("Mean Baseline", f"{feature_data['baseline_mean']:.2f}")
                st.metric("Mean Current", f"{feature_data['current_mean']:.2f}")
                st.metric("Cambio", f"{((feature_data['current_mean']/feature_data['baseline_mean'])-1)*100:.1f}%")
        
        with col2:
            st.markdown("### Visualización de Distribución")
            
            # Crear visualización de distribución
            baseline_values = baseline_data[selected_feature].dropna()
            current_values = current_data[selected_feature].dropna()
            
            if feature_data['type'] == 'numeric':
                # Histogramas superpuestos
                fig = go.Figure()
                
                fig.add_trace(go.Histogram(
                    x=baseline_values,
                    name='Baseline',
                    opacity=0.6,
                    marker_color='#3498db',
                    nbinsx=30
                ))
                
                fig.add_trace(go.Histogram(
                    x=current_values,
                    name='Current',
                    opacity=0.6,
                    marker_color='#e74c3c',
                    nbinsx=30
                ))
                
                fig.update_layout(
                    title=f'Distribución: {selected_feature}',
                    barmode='overlay',
                    xaxis_title=selected_feature,
                    yaxis_title='Frecuencia',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Box plots comparativos
                fig = go.Figure()
                fig.add_trace(go.Box(y=baseline_values, name='Baseline', marker_color='#3498db'))
                fig.add_trace(go.Box(y=current_values, name='Current', marker_color='#e74c3c'))
                fig.update_layout(
                    title=f'Box Plot: {selected_feature}',
                    yaxis_title=selected_feature,
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                # Gráfico de barras para categóricas
                baseline_counts = baseline_values.value_counts()
                current_counts = current_values.value_counts()
                
                # Combinar categorías
                all_cats = sorted(set(baseline_counts.index) | set(current_counts.index))
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=all_cats,
                    y=[baseline_counts.get(cat, 0) for cat in all_cats],
                    name='Baseline',
                    marker_color='#3498db'
                ))
                fig.add_trace(go.Bar(
                    x=all_cats,
                    y=[current_counts.get(cat, 0) for cat in all_cats],
                    name='Current',
                    marker_color='#e74c3c'
                ))
                
                fig.update_layout(
                    title=f'Distribución: {selected_feature}',
                    barmode='group',
                    xaxis_title='Categoría',
                    yaxis_title='Frecuencia',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ============================================
# TABLA DE RESULTADOS
# ============================================

st.header(" Tabla de Resultados Detallados")

# Preparar tabla para mostrar
display_df = filtered_df.copy()

# Formatear columnas según tipo
if 'psi' in display_df.columns:
    display_df['psi'] = display_df['psi'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
if 'ks_statistic' in display_df.columns:
    display_df['ks_statistic'] = display_df['ks_statistic'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
if 'js_divergence' in display_df.columns:
    display_df['js_divergence'] = display_df['js_divergence'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

# Seleccionar columnas relevantes
if len(display_df[display_df['type'] == 'numeric']) > 0:
    cols_to_show = ['feature', 'type', 'drift_level', 'psi', 'ks_statistic', 'js_divergence']
else:
    cols_to_show = ['feature', 'type', 'drift_level', 'chi2_statistic', 'chi2_pvalue']

available_cols = [col for col in cols_to_show if col in display_df.columns]
st.dataframe(
    display_df[available_cols],
    use_container_width=True,
    height=400
)

# Descargar reporte
csv = drift_df.to_csv(index=False)
st.download_button(
    label="Descargar Reporte Completo (CSV)",
    data=csv,
    file_name=f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

# ============================================
# RECOMENDACIONES
# ============================================

st.header(" Recomendaciones")

if high_drift > 3:
    st.error(f"""
    ### 🔴 Acción Inmediata Requerida
    
    Se detectaron **{high_drift} variables** con drift significativo.
    
    **Pasos a seguir:**
    1. ✓ Reentrenar el modelo con datos recientes
    2. ✓ Evaluar performance en datos actuales
    3. ✓ Implementar monitoreo continuo
    4. ✓ Revisar proceso de ingesta de datos
    """)

elif medium_drift > 5 or high_drift > 0:
    st.warning(f"""
    ### 🟡 Monitoreo Cercano Necesario
    
    Cambios moderados detectados en {medium_drift + high_drift} variables.
    
    **Pasos sugeridos:**
    1. 📊 Continuar monitoreando semanalmente
    2. 🔍 Investigar causas de los cambios
    3. 📝 Documentar cambios en el proceso de datos
    4. 🔄 Preparar pipeline de reentrenamiento
    """)

else:
    st.success(f"""
    ### 🟢 Sistema Operando Normalmente
    
    No se detectaron cambios significativos que requieran acción inmediata.
    
    **Próximos pasos:**
    1.  Continuar monitoreo mensual
    2.  Revisar métricas de performance del modelo
    3.  Mantener documentación actualizada
    """)

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Dashboard de Monitoreo de Data Drift v1.0.0</p>
    <p>Última actualización: {}</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)


"""
## ASI SE EJECUTAR LA APLICACIÓN

Estando en la carpeta src/
cd mlops_pipeline/src

La app se ejecuta así:
streamlit run app_streamlit.py

Abrir en el navegador

La aplicación se abrirá automáticamente en: `http://localhost:8501`
"""

