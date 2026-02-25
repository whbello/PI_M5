"""
API de Deployment para Modelo de Predicción de Pagos
FastAPI REST API para servir predicciones del modelo de ML

Versión: 1.0.0
Autor: [Tu Nombre]
Fecha: 2026-02-22

Endpoints:
- GET  /                  : Health check
- POST /predict           : Predicción individual
- POST /predict_batch     : Predicción por lotes
- GET  /model_info        : Información del modelo
- GET  /docs              : Documentación Swagger
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURACIÓN
# ============================================

# Rutas
SCRIPT_DIR = Path(__file__).parent.resolve()
MODELS_DIR = SCRIPT_DIR / 'models'
MODEL_PATH = MODELS_DIR / 'modelo_final_decision_tree.pkl'  #
PIPELINE_PATH = MODELS_DIR / 'pipeline_transformacion.pkl'

# Metadata
API_TITLE = "API de Predicción de Pagos de Créditos"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
## API de Machine Learning para Predicción de Pagos

Esta API permite predecir si un cliente pagará su crédito a tiempo.

Características:
-  Predicción individual y por lotes
-  Documentación automática (Swagger/ReDoc)
-  Validación de datos de entrada
-  Respuestas estructuradas en JSON
-  Manejo de errores robusto

Modelo:
- Entrenado con 10,763 registros
- Balanceado con SMOTE
- Métrica principal: Recall
- Features: 30+ después de encoding
"""

# ============================================
# INICIALIZAR FASTAPI
# ============================================

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================
# CARGAR MODELO Y PIPELINE
# ============================================

try:
    # Buscar el archivo del modelo (puede tener diferentes nombres)
    model_files = list(MODELS_DIR.glob('modelo_final_*.pkl'))
    
    if not model_files:
        raise FileNotFoundError(f"No se encontró modelo en {MODELS_DIR}")
    
    MODEL_PATH = model_files[0]  # Usar el primero encontrado
    
    logger.info(f"Cargando modelo desde: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info("✓ Modelo cargado exitosamente")
    
    logger.info(f"Cargando pipeline desde: {PIPELINE_PATH}")
    pipeline = joblib.load(PIPELINE_PATH)
    logger.info("✓ Pipeline cargado exitosamente")
    
    MODEL_LOADED = True
    MODEL_NAME = MODEL_PATH.stem
    
except Exception as e:
    logger.error(f"Error cargando modelo: {e}")
    MODEL_LOADED = False
    MODEL_NAME = "No cargado"
    model = None
    pipeline = None

# ============================================
# MODELOS DE DATOS (PYDANTIC)
# ============================================

class CreditoInput(BaseModel):
    """Schema para entrada de datos de un crédito individual"""
    
    # Variables numéricas principales
    capital_prestado: float = Field(..., gt=0, description="Monto del préstamo en pesos")
    plazo_meses: int = Field(..., ge=1, le=360, description="Plazo en meses")
    edad_cliente: int = Field(..., ge=18, le=100, description="Edad del cliente")
    salario_cliente: float = Field(..., gt=0, description="Salario mensual")
    puntaje: float = Field(..., ge=0, le=1000, description="Puntaje interno")
    puntaje_datacredito: float = Field(..., ge=0, le=1000, description="Puntaje Datacrédito")
    
    # Variables de comportamiento crediticio
    cant_creditosvigentes: int = Field(default=0, ge=0, description="Créditos vigentes")
    huella_consulta: int = Field(default=0, ge=0, description="Consultas en Datacrédito")
    total_otros_prestamos: float = Field(default=0, ge=0, description="Total otros préstamos")
    
    # Variables de mora
    saldo_mora: float = Field(default=0, ge=0, description="Saldo en mora")
    saldo_total: float = Field(default=0, ge=0, description="Saldo total")
    saldo_principal: float = Field(default=0, ge=0, description="Saldo principal")
    saldo_mora_codeudor: Optional[float] = Field(default=None, ge=0, description="Mora codeudor")
    
    # Créditos por sector
    creditos_sectorFinanciero: int = Field(default=0, ge=0, description="Créditos sector financiero")
    creditos_sectorCooperativo: int = Field(default=0, ge=0, description="Créditos sector cooperativo")
    creditos_sectorReal: int = Field(default=0, ge=0, description="Créditos sector real")
    
    # Variables de ingresos
    promedio_ingresos_datacredito: Optional[float] = Field(default=None, ge=0, description="Promedio ingresos")
    
    # Variables categóricas
    tipo_laboral: str = Field(..., description="Empleado o Independiente")
    tipo_credito: str = Field(..., description="Tipo de crédito")
    tendencia_ingresos: str = Field(..., description="Creciente, Estable o Decreciente")
    
    @validator('tipo_laboral')
    def validate_tipo_laboral(cls, v):
        allowed = ['Empleado', 'Independiente']
        if v not in allowed:
            raise ValueError(f'tipo_laboral debe ser uno de: {allowed}')
        return v
    
    @validator('tendencia_ingresos')
    def validate_tendencia_ingresos(cls, v):
        allowed = ['Creciente', 'Estable', 'Decreciente']
        if v not in allowed:
            raise ValueError(f'tendencia_ingresos debe ser uno de: {allowed}')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "capital_prestado": 5000000,
                "plazo_meses": 36,
                "edad_cliente": 35,
                "salario_cliente": 3000000,
                "puntaje": 700,
                "puntaje_datacredito": 650,
                "cant_creditosvigentes": 2,
                "huella_consulta": 5,
                "total_otros_prestamos": 2000000,
                "saldo_mora": 0,
                "saldo_total": 1500000,
                "saldo_principal": 1500000,
                "saldo_mora_codeudor": None,
                "creditos_sectorFinanciero": 1,
                "creditos_sectorCooperativo": 0,
                "creditos_sectorReal": 1,
                "promedio_ingresos_datacredito": 2800000,
                "tipo_laboral": "Empleado",
                "tipo_credito": "Consumo",
                "tendencia_ingresos": "Estable"
            }
        }


class PredictionResponse(BaseModel):
    """Schema para respuesta de predicción"""
    prediction: int = Field(..., description="0 = No pagará, 1 = Pagará")
    probability: float = Field(..., description="Probabilidad de pagar (0-1)")
    risk_level: str = Field(..., description="Nivel de riesgo: Bajo, Medio, Alto")
    recommendation: str = Field(..., description="Recomendación de acción")
    timestamp: str = Field(..., description="Timestamp de la predicción")


class BatchPredictionResponse(BaseModel):
    """Schema para respuesta de predicción por lotes"""
    predictions: List[Dict[str, Any]]
    total_predictions: int
    timestamp: str


# ============================================
# FUNCIONES AUXILIARES
# ============================================

def prepare_input(data: CreditoInput) -> pd.DataFrame:
    """Convierte input de Pydantic a DataFrame para el modelo"""
    
    # Convertir a diccionario
    data_dict = data.dict()
    
    # Crear DataFrame
    df = pd.DataFrame([data_dict])
    
    # Asegurar que las columnas estén en el orden correcto
    # (el mismo orden que en el entrenamiento)
    
    return df


def get_risk_level(probability: float) -> str:
    """Determina el nivel de riesgo basado en la probabilidad"""
    if probability >= 0.7:
        return "Bajo"
    elif probability >= 0.4:
        return "Medio"
    else:
        return "Alto"


def get_recommendation(prediction: int, probability: float) -> str:
    """Genera recomendación basada en predicción y probabilidad"""
    if prediction == 1 and probability >= 0.7:
        return "🟢 Aprobar crédito - Bajo riesgo de impago"
    elif prediction == 1 and probability >= 0.4:
        return "🟡 Revisar caso - Riesgo moderado, considerar garantías adicionales"
    else:
        return "🔴 Rechazar crédito - Alto riesgo de impago"


# ============================================
# ENDPOINTS
# ============================================

@app.get("/", tags=["Health"])
async def root():
    """
    Health check endpoint
    
    Retorna el estado del servicio y del modelo
    """
    return {
        "status": "online",
        "service": API_TITLE,
        "version": API_VERSION,
        "model_loaded": MODEL_LOADED,
        "model_name": MODEL_NAME,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check detallado"""
    
    health_status = {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "model_path": str(MODEL_PATH) if MODEL_LOADED else None,
        "pipeline_loaded": pipeline is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(data: CreditoInput):
    """
    Realizar predicción individual
    
    Recibe los datos de un crédito y retorna:
    - Predicción (0 = No pagará, 1 = Pagará)
    - Probabilidad de pago
    - Nivel de riesgo
    - Recomendación de acción
    """
    
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Verifica que el archivo del modelo exista."
        )
    
    try:
        # Preparar datos
        df = prepare_input(data)
        
        # Aplicar pipeline de transformación
        X_transformed = pipeline.transform(df)
        
        # Hacer predicción
        prediction = int(model.predict(X_transformed)[0])
        probability = float(model.predict_proba(X_transformed)[0][1])
        
        # Determinar riesgo y recomendación
        risk_level = get_risk_level(probability)
        recommendation = get_recommendation(prediction, probability)
        
        # Crear respuesta
        response = PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            risk_level=risk_level,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Predicción exitosa: {prediction} (prob: {probability:.4f})")
        
        return response
    
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(data: List[CreditoInput]):
    """
    Realizar predicción por lotes
    
    Recibe una lista de créditos y retorna predicciones para todos
    """
    
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible"
        )
    
    if len(data) == 0:
        raise HTTPException(
            status_code=400,
            detail="La lista de datos está vacía"
        )
    
    if len(data) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Máximo 1000 registros por lote"
        )
    
    try:
        predictions_list = []
        
        for idx, item in enumerate(data):
            # Preparar datos
            df = prepare_input(item)
            
            # Aplicar pipeline
            X_transformed = pipeline.transform(df)
            
            # Predicción
            prediction = int(model.predict(X_transformed)[0])
            probability = float(model.predict_proba(X_transformed)[0][1])
            
            # Crear respuesta individual
            pred_result = {
                "index": idx,
                "prediction": prediction,
                "probability": round(probability, 4),
                "risk_level": get_risk_level(probability),
                "recommendation": get_recommendation(prediction, probability)
            }
            
            predictions_list.append(pred_result)
        
        response = BatchPredictionResponse(
            predictions=predictions_list,
            total_predictions=len(predictions_list),
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Predicción batch exitosa: {len(predictions_list)} registros")
        
        return response
    
    except Exception as e:
        logger.error(f"Error en predicción batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/model_info", tags=["Model"])
async def model_info():
    """
    Información del modelo
    
    Retorna metadatos del modelo cargado
    """
    
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible"
        )
    
    try:
        # Cargar resumen del modelo si existe
        summary_path = MODELS_DIR / 'resumen_modelo.json'
        
        if summary_path.exists():
            import json
            with open(summary_path, 'r') as f:
                model_summary = json.load(f)
        else:
            model_summary = {}
        
        info = {
            "model_name": MODEL_NAME,
            "model_path": str(MODEL_PATH),
            "model_type": str(type(model).__name__),
            "pipeline_available": pipeline is not None,
            "summary": model_summary,
            "loaded_at": datetime.now().isoformat()
        }
        
        return info
    
    except Exception as e:
        logger.error(f"Error obteniendo info del modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# STARTUP Y SHUTDOWN EVENTS
# ============================================

@app.on_event("startup")
async def startup_event():
    """Evento al iniciar la aplicación"""
    logger.info("="*80)
    logger.info("API DE PREDICCIÓN DE PAGOS INICIADA")
    logger.info("="*80)
    logger.info(f"Versión: {API_VERSION}")
    logger.info(f"Modelo cargado: {MODEL_LOADED}")
    if MODEL_LOADED:
        logger.info(f"Modelo: {MODEL_NAME}")
    logger.info("="*80)


@app.on_event("shutdown")
async def shutdown_event():
    """Evento al cerrar la aplicación"""
    logger.info("API cerrándose...")


# ============================================
# EJECUTAR SERVIDOR
# ============================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "model_deploy:app",
        host="localhost",
        port=8000,
        reload=True,  # Solo en desarrollo
        log_level="info"
    )

