# Parte 1 y 2: Aplicación Streamlit y Dockerización

Este directorio contiene la aplicación Streamlit y los archivos para su dockerización.

## Configuración y Ejecución de la Aplicación Streamlit

1. Navega al directorio `app`:
   `cd app`
2. Instala las dependencias usando Poetry:
   `poetry install`
3. Ejecuta la aplicación Streamlit:
   `poetry run streamlit run app.py`

## Construcción y Ejecución de la Imagen Docker

1. Asegúrate de estar en el directorio `app`:
   `cd app`
2. Construye la imagen Docker:
   `docker build -t titanic-streamlit-app .`
3. Ejecuta el contenedor Docker, mapeando el puerto 8501:
   `docker run -p 8501:8501 -d titanic-streamlit-app`

4. Abre tu navegador en:
   `http://localhost:8501`

