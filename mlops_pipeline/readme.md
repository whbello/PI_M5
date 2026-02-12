# Proyecto Integrador M5 - Data Science

## Descripción del Proyecto
Este proyecto forma parte del Bootcamp de Data Science de SoyHenry. El objetivo es desarrollar un modelo de riesgo crediticio para una institución financiera, desplegarlo mediante una API y crear una aplicación web para su visualización.

## Estructura del Proyecto
El proyecto sigue una estructura de carpetas estrictamente definida para asegurar la compatibilidad con los pipelines de despliegue automatizados.

```
/
├── data/                   # Datos crudos y procesados
├── notebooks/              # Jupyter notebooks para EDA y entrenamiento
│   ├── cargar_datos.ipynb
│   └── comprension_eda.ipynb
├── src/                    # Código fuente de la aplicación y API
│   └── model_deploy.py
├── models/                 # Modelos serializados (pickle, joblib)
├── .gitignore              # Archivos ignorados por Git
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Documentación del proyecto
└── Dockerfile              # Configuración de Docker
```

## Ramas de Git
- **master**: Rama principal de producción.
- **certification**: Rama de pruebas y certificación.
- **developer**: Rama de desarrollo activo.

## Instalación y Uso
1. Clonar el repositorio.
2. Crear un entorno virtual: `python -m venv venv`
3. Activar el entorno virtual.
4. Instalar dependencias: `pip install -r requirements.txt`

## Tecnologías
- Python
- Pandas, Scikit-learn, Seaborn
- FastAPI
- Streamlit
- Docker
