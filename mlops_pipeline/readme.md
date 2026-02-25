### Proyecto Integrador M5 - Data Science
#  Sistema de Predicción y Monitoreo de Pago de Créditos


## Tabla de Contenidos

- [Problema de Negocio](#-problema-de-negocio)
- [Solución Implementada](#-solución-implementada)
- [Dataset](#-dataset)
- [Hallazgos Principales](#-hallazgos-principales)
- [Pipeline de ML](#-pipeline-de-ml)
- [Resultados del Modelo](#-resultados-del-modelo)
- [Sistema de Monitoreo](#-sistema-de-monitoreo)
- [Instalación](#-instalación)
- [Uso del Sistema](#-uso-del-sistema)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Tecnologías Utilizadas](#-tecnologías-utilizadas)
- [Versionamiento](#-versionamiento)
- [Autores](#-autores)
- [Conclusiones](#-conclusiones)

---
## Problema de Negocio

### Contexto

Las instituciones financieras enfrentan un desafío crítico: **predecir qué clientes pagarán sus créditos a tiempo**. Un sistema de predicción inexacto puede resultar en:

- **Pérdidas financieras** por créditos no pagados
- **Oportunidades perdidas** por rechazar clientes buenos
- **Deterioro del portafolio** de créditos

### Objetivo

Desarrollar un **sistema de ML end-to-end** que:

1. Prediga con alta precisión qué clientes NO pagarán a tiempo
2. Monitoree continuamente la calidad de las predicciones
3. Alerte cuando los datos cambien y el modelo necesite reentrenamiento
4. Proporcione insights accionables para el negocio

### Métrica de Éxito

**Recall (Sensibilidad)** en la clase minoritaria (No pagó):
-  Objetivo: **>60%** de detección de impagos
-  Trade-off: Balance entre detectar impagos y no rechazar clientes buenos

---

##  Solución Implementada

### Componentes del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                   SISTEMA COMPLETO                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. CARGA Y LIMPIEZA DE DATOS                              │
│     └─→ Cargar_datos.ipynb                                 │
│                                                             │
│  2. ANÁLISIS EXPLORATORIO (EDA)                            │
│     └─→ comprension_eda.ipynb                              │
│         ├─ Identificación de desbalance crítico           │
│         ├─ Análisis bivariado con tests estadísticos      │
│         └─ Detección de multicolinealidad                 │
│                                                             │
│  3. FEATURE ENGINEERING                                     │
│     └─→ ft_engineering.ipynb                            │
│         ├─ Creación de features derivados                  │
│         ├─ Pipelines de transformación                     │
│         └─ Balanceo con SMOTE                              │
│                                                             │
│  4. MODELAMIENTO                                            │
│     └─→ model_training_evaluation.ipynb                 │
│         ├─ Entrenamiento de 6 modelos                      │
│         ├─ Evaluación con métricas apropiadas             │
│         └─ Selección del mejor modelo                      │
│                                                             │
│  5. MONITOREO Y DRIFT DETECTION                            │
│     ├─→ model_monitoring.py                                │
│     │   ├─ Cálculo de métricas de drift                   │
│     │   ├─ KS, PSI, JS Divergence, Chi²                   │
│     │   └─ Generación de reportes                         │
│     │                                                       │
│     └─→ app_streamlit.py                                   │
│         ├─ Dashboard interactivo                           │
│         ├─ Visualización de drift                          │
│         ├─ Sistema de alertas                              │
│         └─ Recomendaciones automáticas                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---
## Dataset

### Descripción

- **Nombre:** Base de Créditos
- **Período:** Diciembre 2024 - Enero 2026
- **Registros:** 10,763 préstamos
- **Variables:** 23 columnas

### Variables Principales

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `Pago_atiempo` | Target | 1=Pagó a tiempo, 0=No pagó (4.75% de impagos) |
| `capital_prestado` | Numérica | Monto del préstamo en pesos |
| `puntaje_datacredito` | Numérica | Score de Datacrédito (0-1000) |
| `edad_cliente` | Numérica | Edad del titular |
| `salario_cliente` | Numérica | Ingreso mensual declarado |
| `tipo_laboral` | Categórica | Empleado / Independiente |
| `tendencia_ingresos` | Categórica | Creciente / Estable / Decreciente |
| `cant_creditosvigentes` | Numérica | Número de créditos activos |
| `saldo_mora` | Numérica | Saldo en mora del cliente |

### Calidad de Datos

- **Sin duplicados:** 0 registros duplicados
- **Nulos significativos:**
  - `tendencia_ingresos`: 27.24% (2,931 registros)
  - `promedio_ingresos_datacredito`: 27.22%
- **Limpieza aplicada:**
  - Conversión de tipos de datos
  - Eliminación de valores inválidos en `tendencia_ingresos`
  - Validación de estructura

---

## Hallazgos Principales

### 1. Desbalance de Clases CRÍTICO

**Problema más importante del dataset:**

```
Clase 1 (Pagó):    10,252 clientes (95.25%) ████████████████████
Clase 0 (No pagó):    511 clientes ( 4.75%) █
                                             ↑
                                    Ratio: 20:1
```

**Implicaciones:**
- ❌ Modelo "tonto" logra 95% accuracy sin aprender nada
- ❌ Sin técnicas de balanceo, el modelo ignora la clase minoritaria
- **Solución:** SMOTE aplicado en entrenamiento

### 2. Variables Más Predictivas

Identificadas mediante análisis bivariado con tests estadísticos:

| Variable | Test | P-value | Significancia |
|----------|------|---------|---------------|
| `puntaje_datacredito` | t-test | < 0.001 | ⭐⭐⭐ Alta |
| `saldo_mora` | t-test | < 0.001 | ⭐⭐⭐ Alta |
| `tipo_laboral` | Chi² | < 0.001 | ⭐⭐⭐ Alta |
| `capital_prestado` | t-test | < 0.01 | ⭐⭐ Media |
| `edad_cliente` | t-test | < 0.05 | ⭐ Baja |

### 3. Multicolinealidad Detectada

Variables altamente correlacionadas (|r| > 0.8):
- `capital_prestado` ↔ `cuota_pactada` (r = 0.92)
- `capital_prestado` ↔ `plazo_meses` (r = 0.85)

**Acción tomada:** Eliminación de `cuota_pactada` en feature engineering.

### 4. Insights de Negocio

**Perfil de Cliente con Mayor Riesgo:**
-  Puntaje Datacrédito bajo (< 500)
-  Trabajador independiente
-  Tendencia de ingresos decreciente
-  Saldo en mora existente
-  Múltiples créditos vigentes (> 3)

---

## Pipeline de ML

### Fase 1: Feature Engineering

```python
# Features Derivados Creados
ratio_cuota_salario = cuota_pactada / salario_cliente
ratio_credito_ingreso = capital_prestado / salario_cliente
total_creditos_sectores = sum(creditos_sector*)
tiene_codeudor = (saldo_mora_codeudor > 0)
```

**Transformaciones Aplicadas:**

| Tipo | Variables | Transformación |
|------|-----------|----------------|
| Numéricas (17) | capital_prestado, puntaje, etc. | Mediana + StandardScaler |
| Categóricas Nominales (2) | tipo_laboral, tipo_credito | Imputación + One-Hot |
| Categóricas Ordinales (1) | tendencia_ingresos | Imputación + Ordinal |

**Balanceo de Clases:**
- Train ANTES: 511 vs 10,252 (desbalanceado)
- Train DESPUÉS: ~8,610 vs ~8,610  (balanceado con SMOTE)
- Test: Mantiene distribución real (95.25% vs 4.75%)

### Fase 2: División de Datos

```
Total: 10,763 registros
├─ Train: 8,610 (80%) → Balanceado con SMOTE
└─ Test:  2,153 (20%) → Distribución real
```

---

##  Resultados del Modelo

### Modelos Entrenados

| # | Modelo | Recall | Precision | F1-Score | AUC-ROC | Tiempo (s) |
|---|--------|--------|-----------|----------|---------|--------|
| 1 | Baseline | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.0074 |
| 2 | Logistic Regression | 0.9980 | 1.0000 | 0.9990 | 0.9999 | 0.2656 |
| 3 | **Decision Tree** ⭐ | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.1597** |
| 4 | Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.0357 |
| 5 | Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4.2260 |
| 6 | XGBoost  | 1.0000 | 0.9985 | 0.9993 | 1.0000 | 0.6641 |

### Modelo Seleccionado

**[Decision Tree]** seleccionado por ejecutarse en menor tiempo a pesar de haber varios con resultados iguales pero con mayor tiempo de ejecución en los cálculos.

**Métricas en Test:**
-  **Recall:** 1.0000 (detecta 100% de impagos)
-  **Precision:** 1.0000 (100% de alertas son correctas)
-  **F1-Score:** 1.000 (balance Precision-Recall)
-  **AUC-ROC:** 1.000 (capacidad discriminativa)

**Matriz de Confusión:**
```
                Predicción
              No Pagó  Sí Pagó
Real  No Pagó    TN       FP
      Sí Pagó    FN       TP
```

### Interpretación de Negocio

De **0 clientes que NO pagaron** en el conjunto de prueba:
- 🟢 **2051 detectados** (TN): Pérdida evitada
- 🔴 **0 no detectados** (FN): Pérdida real

**Mejora vs Baseline:**
- Baseline detecta: 0% de impagos
- Nuestro modelo: **0%** de impagos
- **Mejora:** +1.0000 (100%) puntos porcentuales

### Top 5 Features Más Importantes

1.  `puntaje_datacredito` (importancia: 1.000)
2.  `saldo_mora` (importancia: 0.000)
3.  `ratio_cuota_salario` (importancia: 0.000)
4.  `edad_cliente` (importancia: 0.000)
5.  `cant_creditosvigentes` (importancia: 0.000)

---

##  Sistema de Monitoreo

### Métricas de Data Drift Implementadas

| Métrica | Tipo | Umbral | Interpretación |
|---------|------|--------|----------------|
| **KS Test** | Numérica | < 0.2 | Diferencia entre CDFs |
| **PSI** | Numérica | < 0.2 | < 0.1: OK, 0.1-0.2: Moderado, >0.2: Alto |
| **JS Divergence** | Numérica | < 0.15 | 0=idénticas, 1=diferentes |
| **Chi²** | Categórica | p>0.05 | Diferencia en frecuencias |

### Sistema de Alertas

```
🟢 Drift Bajo (0-0.1):       Sin acción requerida
🟡 Drift Moderado (0.1-0.2): Monitoreo cercano
🔴 Drift Alto (>0.2):        Reentrenamiento necesario
```

### Dashboard Interactivo

Acceso vía **Streamlit** con:
-  Visualización de métricas de drift
-  Comparación distribuciones (baseline vs actual)
-  Evolución temporal del drift
-  Alertas automáticas
-  Recomendaciones de acción

---

##  Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes)
- Git

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/whbello/PI_M5.git
cd PI_M5/mlops_pipeline
```

### Paso 2: Crear Entorno Virtual 

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

*Contenido de requirements.txt:**
```txt
pandas
numpy
scikit-learn
feature-engine
seaborn
matplotlib
fastapi
uvicorn
streamlit
pypdf
python-pptx
openpyxl
python-dotenv
db-dtypes
jupyter
notebook
ipykernel
ipywidgets
xgboost
pydantic
uvicorn[standard]
imbalanced-learn
streamlit plotly
```

---

##  Uso del Sistema

### 1. Ejecutar Pipeline Completo

#### Paso 1: Carga de Datos

```bash
jupyter notebook
# Abrir y ejecutar: src/Cargar_datos.ipynb
```

**Output:** `data/interim/creditos_limpio.csv`

#### Paso 2: Análisis Exploratorio

```bash
# Ejecutar: src/comprension_eda.ipynb
```

**Outputs:** 
- Análisis de desbalance
- Tests estadísticos
- Visualizaciones

#### Paso 3: Feature Engineering

```bash
# Ejecutar: src/ft_engineering.ipynb
```

**Outputs:**
- `data/processed/X_train.csv`
- `data/processed/X_test.csv`
- `data/processed/y_train.csv`
- `data/processed/y_test.csv`
- `models/pipeline_transformacion.pkl`

#### Paso 4: Entrenamiento de Modelos

```bash
# Ejecutar: src/model_training_evaluation.ipynb
```

**Outputs:**
- `models/modelo_final_*.pkl`
- `models/metricas_comparacion.csv`
- `reports/figures/*.png`

### 2. Sistema de Monitoreo

#### Ejecutar Análisis de Drift

```bash
cd src
python model_monitoring.py
```

**Output:** 
- `data/monitoring/drift_reports/drift_report_*.json`
- `data/monitoring/drift_reports/drift_report_*.csv`

#### Visualizar Dashboard

```bash
cd src
streamlit run app_streamlit.py
```

Abre automáticamente: `http://localhost:8501`

**Funcionalidades del Dashboard:**
-  Métricas de drift en tiempo real
-  Comparación de distribuciones
-  Gráficos interactivos con Plotly
-  Análisis detallado por feature
-  Exportación de reportes

---
##  Estructura del Proyecto

```
PI_M5/mlops_pipeline/
├── src/
│   ├── data/
│   │   ├── raw/                          # Datos originales
│   │   │   └── Base_de_datos.xlsx
│   │   ├── interim/                      # Datos limpios
│   │   │   ├── creditos_limpio.csv
│   │   │   └── creditos_limpio.pkl
│   │   ├── processed/                    # Datos procesados
│   │   │   ├── X_train.csv
│   │   │   ├── X_test.csv
│   │   │   ├── y_train.csv
│   │   │   └── y_test.csv
│   │   └── monitoring/                   # Monitoreo
│   │       ├── drift_reports/
│   │       └── baseline_stats.pkl
│   │
│   ├── models/                           # Modelos entrenados
│   │   ├── modelo_final_decision_tree.pkl
│   │   ├── pipeline_transformacion.pkl
│   │   ├── metricas_comparacion.csv
│   │   └── resumen_modelo.json
│   │
│   ├── reports/                          # Reportes y figuras
│   │   └── figures/
│   │       ├── comparacion_modelos.png
│   │       ├── confusion_matrix_decision_tree.png
│   │       ├── feature_importance_decision_tree.png
│   │       └── curvas_evaluacion.png
│   │
│   ├── Cargar_datos.ipynb               # Notebook 1: Carga
│   ├── comprension_eda.ipynb            # Notebook 2: EDA
│   ├── ft_engineering.ipynb             # Notebook 3: Features
│   ├── model_training_evaluation.ipynb  # Notebook 4: Modelos
│   ├── model_monitoring.py              # Script: Monitoreo
│   └── app_streamlit.py                 # App: Dashboard
│
├── .gitignore                           # Archivos ignorados
├── requirements.txt                     # Dependencias
└── README.md                            # Este archivo
```

---

##  Tecnologías Utilizadas

### Análisis y Modelamiento
- ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) **Python 3.8+**
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) **Pandas** - Manipulación de datos
- ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) **NumPy** - Operaciones numéricas
- ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) **Scikit-learn** - Machine Learning
- ![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat) **XGBoost** - Gradient Boosting

### Balanceo y Feature Engineering
- **imbalanced-learn** - SMOTE para balanceo
- **feature-engine** - Transformaciones avanzadas

### Visualización
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat) **Matplotlib** - Gráficos estáticos
- ![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat) **Seaborn** - Visualizaciones estadísticas
- ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat&logo=plotly&logoColor=white) **Plotly** - Gráficos interactivos

### Dashboard y Monitoreo
- ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white) **Streamlit** - Aplicación web
- **SciPy** - Tests estadísticos (KS, Chi²)

### Entorno y Versionamiento
- ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) **Jupyter** - Notebooks interactivos
- ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white) **Git** - Control de versiones
- ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white) **GitHub** - Repositorio remoto

---

##  Versionamiento

El proyecto sigue **Git Flow** con las siguientes versiones:

### Releases

| Versión | Fecha | Descripción | Branch |
|---------|-------|-------------|--------|
| **v1.0.0** | 2026-02-10 | Estructura inicial del proyecto | main |
| **v1.0.1** | 2026-02-12 | Carga de datos y EDA completado | main |
| **v1.1.0** | 2026-02-14 | Feature Engineering pipeline | main |
| **v1.1.1** | 2026-02-14 | Entrenamiento y evaluación de modelos | main |
| **v1.2.0** | 2026-02-15 | Sistema de monitoreo y dashboard | main |

---

##  Autor

**[Tu Nombre]**
- Email: whbello@yahoo.es
- LinkedIn: en construcción
- 🐙 GitHub: https://github.com/whbello/PI_M5.git

---

##  Conclusiones

### Logros Principales

1.  **Sistema ML End-to-End Implementado**
   - Pipeline completo desde datos crudos hasta predicciones
   - Automatizado y reproducible

2.  **Desbalance de Clases Resuelto**
   - SMOTE aplicado exitosamente
   - Mejora significativa en detección de impagos

3.  **Modelo con Performance Competitiva**
   - Recall: 100% (vs 0% del baseline)
   - Balance apropiado entre Precision y Recall

4.  **Sistema de Monitoreo Robusto**
   - 4 métricas de drift implementadas
   - Dashboard interactivo funcional
   - Alertas automáticas configuradas

### Impacto de Negocio

**Estimación de Impacto Financiero:**

Asumiendo:
- 511 clientes que no pagaron en el test
- Pérdida promedio por impago: $1,000,000 COP
- Modelo detecta 100% de impagos

```
Pérdida SIN modelo:  511 × $1,000,000 = $511,000,000
Pérdida CON modelo:  [FN] × $1,000,000 = $[000,000,000]
────────────────────────────────────────────────────────
AHORRO ESTIMADO:                       $[511,000,000]
```

### Limitaciones Identificadas

1.  **Desbalance Extremo del Dataset**
   - Ratio 20:1 limita el aprendizaje
   - Recolectar más ejemplos de clase minoritaria mejoraría resultados

2.  **Valores Nulos Significativos**
   - 27% de nulos en `tendencia_ingresos`
   - Mejorar calidad de captura de datos

3.  **Datos Históricos Limitados**
   - Solo 14 meses de datos
   - Más historia permitiría capturar estacionalidad

### Recomendaciones Futuras

#### Corto Plazo (1-3 meses)
1.  **Monitoreo Semanal**
   - Ejecutar `model_monitoring.py` cada semana
   - Revisar dashboard para detectar drift temprano

2.  **Optimización de Hiperparámetros**
   - GridSearchCV en el modelo seleccionado
   - Potencial mejora de 2-5% en métricas

3.  **Threshold Tuning**
   - Ajustar umbral de decisión (actualmente 0.5)
   - Optimizar según costo de FP vs FN

#### Mediano Plazo (3-6 meses)
1.  **Modelos Ensamblados**
   - Stacking de mejores modelos
   - Voting Classifier

2.  **Features Adicionales**
   - Comportamiento de pago histórico
   - Variables macroeconómicas
   - Indicadores de red social

3.  **Pipeline de Reentrenamiento Automático**
   - Trigger cuando drift > umbral
   - CI/CD para deployment

#### Largo Plazo (6-12 meses)
1.  **Deployment en Producción**
   - API REST con FastAPI
   - Containerización con Docker
   - Orquestación con Kubernetes

2.  **Interfaz para Analistas de Crédito**
   - App web para scoring en tiempo real
   - Explicabilidad de predicciones (SHAP/LIME)

3.  **Segmentación de Clientes**
   - Modelos específicos por segmento
   - Personalización de umbrales

---
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=whbello_PI_M5&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=whbello_PI_M5)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=whbello_PI_M5&metric=coverage)](https://sonarcloud.io/summary/new_code?id=whbello_PI_M5)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=whbello_PI_M5&metric=bugs)](https://sonarcloud.io/summary/new_code?id=whbello_PI_M5)
---