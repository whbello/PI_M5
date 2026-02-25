"""
Tests para la API de Predicción
Usa pytest y httpx para testing asíncrono

Ejecución:
    pytest test_api.py -v
    pytest test_api.py --cov=api_deploy
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent))

# Importar la app
from api_deploy import app

# Cliente de testing
client = TestClient(app)

# ============================================
# TESTS DE ENDPOINTS BÁSICOS
# ============================================

def test_root_endpoint():
    """Test del endpoint raíz /"""
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert data["status"] == "online"
    assert "service" in data
    assert "version" in data
    assert "model_loaded" in data


def test_health_endpoint():
    """Test del endpoint de salud /health"""
    response = client.get("/health")
    
    # Puede ser 200 (healthy) o 503 (unhealthy)
    assert response.status_code in [200, 503]
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "timestamp" in data


def test_docs_endpoint():
    """Test de que la documentación está disponible"""
    response = client.get("/docs")
    assert response.status_code == 200


def test_redoc_endpoint():
    """Test de ReDoc"""
    response = client.get("/redoc")
    assert response.status_code == 200


# ============================================
# TESTS DE PREDICCIÓN INDIVIDUAL
# ============================================

def test_predict_endpoint_valid_data():
    """Test de predicción con datos válidos"""
    
    # Datos de ejemplo válidos
    payload = {
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
    
    response = client.post("/predict", json=payload)
    
    # Verificar respuesta
    if response.status_code == 200:
        data = response.json()
        
        # Verificar estructura
        assert "prediction" in data
        assert "probability" in data
        assert "risk_level" in data
        assert "recommendation" in data
        assert "timestamp" in data
        
        # Verificar tipos
        assert isinstance(data["prediction"], int)
        assert data["prediction"] in [0, 1]
        assert isinstance(data["probability"], float)
        assert 0 <= data["probability"] <= 1
        assert isinstance(data["risk_level"], str)
        assert data["risk_level"] in ["Bajo", "Medio", "Alto"]
    
    elif response.status_code == 503:
        # Modelo no cargado (esperado en CI/CD sin modelo)
        data = response.json()
        assert "detail" in data


def test_predict_endpoint_invalid_data():
    """Test de predicción con datos inválidos"""
    
    # Datos inválidos (edad negativa)
    payload = {
        "capital_prestado": 5000000,
        "plazo_meses": 36,
        "edad_cliente": -5,  # Inválido
        "salario_cliente": 3000000,
        "puntaje": 700,
        "puntaje_datacredito": 650,
        "cant_creditosvigentes": 2,
        "huella_consulta": 5,
        "total_otros_prestamos": 2000000,
        "saldo_mora": 0,
        "saldo_total": 1500000,
        "saldo_principal": 1500000,
        "creditos_sectorFinanciero": 1,
        "creditos_sectorCooperativo": 0,
        "creditos_sectorReal": 1,
        "tipo_laboral": "Empleado",
        "tipo_credito": "Consumo",
        "tendencia_ingresos": "Estable"
    }
    
    response = client.post("/predict", json=payload)
    
    # Debe retornar error de validación
    assert response.status_code == 422


def test_predict_endpoint_missing_fields():
    """Test con campos faltantes"""
    
    payload = {
        "capital_prestado": 5000000,
        # Faltan muchos campos requeridos
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_invalid_tipo_laboral():
    """Test con valor inválido en tipo_laboral"""
    
    payload = {
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
        "creditos_sectorFinanciero": 1,
        "creditos_sectorCooperativo": 0,
        "creditos_sectorReal": 1,
        "tipo_laboral": "Desempleado",  # No permitido
        "tipo_credito": "Consumo",
        "tendencia_ingresos": "Estable"
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ============================================
# TESTS DE PREDICCIÓN POR LOTES
# ============================================

def test_predict_batch_endpoint_valid_data():
    """Test de predicción por lotes con datos válidos"""
    
    payload = [
        {
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
            "creditos_sectorFinanciero": 1,
            "creditos_sectorCooperativo": 0,
            "creditos_sectorReal": 1,
            "tipo_laboral": "Empleado",
            "tipo_credito": "Consumo",
            "tendencia_ingresos": "Estable"
        },
        {
            "capital_prestado": 3000000,
            "plazo_meses": 24,
            "edad_cliente": 28,
            "salario_cliente": 2000000,
            "puntaje": 600,
            "puntaje_datacredito": 550,
            "cant_creditosvigentes": 1,
            "huella_consulta": 3,
            "total_otros_prestamos": 1000000,
            "saldo_mora": 0,
            "saldo_total": 800000,
            "saldo_principal": 800000,
            "creditos_sectorFinanciero": 1,
            "creditos_sectorCooperativo": 0,
            "creditos_sectorReal": 0,
            "tipo_laboral": "Independiente",
            "tipo_credito": "Consumo",
            "tendencia_ingresos": "Creciente"
        }
    ]
    
    response = client.post("/predict_batch", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        
        assert "predictions" in data
        assert "total_predictions" in data
        assert "timestamp" in data
        
        assert data["total_predictions"] == 2
        assert len(data["predictions"]) == 2
        
        # Verificar estructura de predicciones
        for pred in data["predictions"]:
            assert "index" in pred
            assert "prediction" in pred
            assert "probability" in pred
            assert "risk_level" in pred
            assert "recommendation" in pred
    
    elif response.status_code == 503:
        # Modelo no disponible
        pass


def test_predict_batch_empty_list():
    """Test con lista vacía"""
    payload = []
    
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 400


def test_predict_batch_too_many():
    """Test con demasiados registros"""
    
    # Crear 1001 registros (límite es 1000)
    payload = [
        {
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
            "creditos_sectorFinanciero": 1,
            "creditos_sectorCooperativo": 0,
            "creditos_sectorReal": 1,
            "tipo_laboral": "Empleado",
            "tipo_credito": "Consumo",
            "tendencia_ingresos": "Estable"
        }
    ] * 1001
    
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 400


# ============================================
# TESTS DE MODEL INFO
# ============================================

def test_model_info_endpoint():
    """Test del endpoint de información del modelo"""
    response = client.get("/model_info")
    
    if response.status_code == 200:
        data = response.json()
        
        assert "model_name" in data
        assert "model_path" in data
        assert "model_type" in data
    
    elif response.status_code == 503:
        # Modelo no disponible
        data = response.json()
        assert "detail" in data


# ============================================
# TESTS DE RENDIMIENTO
# ============================================

def test_response_time():
    """Test de tiempo de respuesta del health check"""
    import time
    
    start = time.time()
    response = client.get("/health")
    elapsed = time.time() - start
    
    # Debe responder en menos de 2 segundos
    assert elapsed < 2.0


# ============================================
# FIXTURE PARA LIMPIEZA
# ============================================

@pytest.fixture(autouse=True)
def reset_state():
    """Fixture que se ejecuta antes y después de cada test"""
    yield
    # Limpieza después del test si es necesario


# ============================================
# CONFIGURACIÓN DE PYTEST
# ============================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])