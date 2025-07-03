# Parte 0: Entrenamiento y Serialización del Modelo

Este directorio contiene el código para entrenar y serializar el modelo de Keras.

## Configuración y Ejecución

1. Navega al directorio `parte0`:
   `cd parte0`
2. Instala las dependencias usando Poetry:
   `poetry install`

4. Instala manualmente TensorFlow 2.15.0 dentro del entorno Poetry (para evitar problemas con dependencias de wheels):
   `poetry run pip install tensorflow==2.15.0`
   
4. Ejecuta el script de entrenamiento:
   `poetry run python train_model.py`