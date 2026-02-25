# ============================================
# DOCKERFILE PARA API DE PREDICCIÓN
# ============================================

# Imagen base: Python 3.10 slim (liviana)
FROM python:3.10-slim

# Metadata del contenedor
LABEL maintainer="whbello@gmail.com"
LABEL version="1.0.0"
LABEL description="API REST para predicción de pagos de créditos"

# Configurar directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar archivo de dependencias
COPY requirements_api.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements_api.txt

# Copiar el código de la API
COPY api_deploy.py .

# Crear carpeta para modelos
RUN mkdir -p models

# Copiar modelos entrenados (solo los necesarios)
COPY models/modelo_final*.pkl ./models/
COPY models/pipeline_transformacion.pkl ./models/

# Exponer el puerto 8000
EXPOSE 8000

# Variable de entorno para Python
ENV PYTHONUNBUFFERED=1

# Comando para ejecutar la API cuando inicie el contenedor
CMD ["uvicorn", "api_deploy:app", "--host", "0.0.0.0", "--port", "8000"]