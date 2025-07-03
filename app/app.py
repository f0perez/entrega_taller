import streamlit as st
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib # Si guardaste el preprocesador

# Función para cargar el modelo (cacheada)
@st.cache_resource
def load_keras_model():
    model = tf.keras.models.load_model('model.keras')
    return model

# # Función para cargar el preprocesador (si fue guardado)
# @st.cache_resource
# def load_preprocessor():
#     preprocessor = joblib.load('preprocessor.joblib')
#     return preprocessor

# Cargar el modelo
model = load_keras_model()
# preprocessor = load_preprocessor() # Cargar el preprocesador si lo necesitas

# --- Recrear el preprocesador para el ejemplo, si no lo guardaste ---
# Esto es crucial para que las predicciones sean correctas
@st.cache_resource # Caching el preprocesador si se recrea en la app
def create_preprocessor():
    numerical_features = ['age', 'sibsp', 'parch', 'fare']
    categorical_features = ['pclass', 'embarked']

    # Entrenar un preprocesador dummy para obtener la estructura necesaria
    # En una aplicación real, se entrenaría con un subconjunto de datos o se cargaría
    # el preprocesador ya entrenado. Para este ejemplo, solo creamos la estructura.
    # Asumiendo que 'sex' y 'embarked' ya se manejan como numéricos en la entrada del usuario
    # y que 'pclass' es categórica para OHE.
    # Para que funcione correctamente, el preprocesador debe ser entrenado
    # con datos reales o cargar uno entrenado.
    # Aquí, para simplicidad, asumiremos que las entradas se mapean directamente
    # a los valores que el modelo espera después del preprocesamiento.
    # O: cargar un dataset pequeño y entrenar el preprocesador de nuevo.
    # df_dummy = pd.DataFrame({
    #     'pclass': [1, 2, 3], 'sex': [0, 1, 0], 'age': [22, 38, 26],
    #     'sibsp': [1, 0, 0], 'parch': [0, 0, 0], 'fare': [7.25, 71.28, 7.92],
    #     'embarked': [0, 1, 2] # Mapeados a numéricos
    # })
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', StandardScaler(), numerical_features),
    #         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    #     ])
    # preprocessor.fit(df_dummy[numerical_features + categorical_features])
    # return preprocessor

    # Para simplificar, si no se guardó el preprocesador, debemos asegurarnos de
    # que las entradas del usuario se transformen de la misma manera que los datos de entrenamiento.
    # Un enfoque más robusto sería cargar el preprocesador entrenado.
    return None # Placeholder, si no se usa un preprocesador de sklearn aquí
preprocessor = create_preprocessor()


# Título y descripción
st.title("Predicción de Supervivencia en el Titanic")
st.write("Ingrese los datos del pasajero para predecir si sobrevivió al hundimiento del Titanic.")

# Widgets de entrada
st.sidebar.header("Características del Pasajero")

pclass = st.sidebar.selectbox("Clase de Pasajero (Pclass)", [1, 2, 3])
sex = st.sidebar.radio("Sexo", ["Hombre", "Mujer"])
age = st.sidebar.slider("Edad", 1, 100, 30)
sibsp = st.sidebar.number_input("Número de hermanos/cónyuges a bordo (SibSp)", min_value=0, max_value=8, value=0)
parch = st.sidebar.number_input("Número de padres/hijos a bordo (Parch)", min_value=0, max_value=6, value=0)
fare = st.sidebar.number_input("Tarifa (Fare)", min_value=0.0, max_value=500.0, value=30.0, format="%.2f")
embarked = st.sidebar.selectbox("Puerto de Embarque", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

# Mapeo de entradas a valores numéricos esperados por el modelo
sex_map = {"Hombre": 0, "Mujer": 1} # Asumiendo que 0=male, 1=female
embarked_map = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2} # Asumiendo S=0, C=1, Q=2

input_data = {
    'pclass': pclass,
    'sex': sex_map[sex],
    'age': age,
    'sibsp': sibsp,
    'parch': parch,
    'fare': fare,
    'embarked': embarked_map[embarked]
}

# Convertir a DataFrame y preprocesar (CRUCIAL: debe ser consistente con el entrenamiento)
input_df = pd.DataFrame([input_data])

# Si usaste un ColumnTransformer en tu modelo de Keras, DEBES usarlo aquí.
# Si no, asegúrate de que las columnas estén en el orden correcto y escaladas.
# Para el ejemplo, asumiremos un preprocesamiento manual si no se guardó el preprocesador.
# Esto es una simplificación; lo ideal es usar el mismo preprocesador entrenado.

# Ejemplo de preprocesamiento manual (solo si no se usa ColumnTransformer cargado)
# y si tus features son exactamente los que el modelo espera, ya escalados/codificados
# numerical_features = ['age', 'sibsp', 'parch', 'fare']
# categorical_features_one_hot = ['pclass_1', 'pclass_2', 'pclass_3', 'embarked_0', 'embarked_1', 'embarked_2'] # Ejemplo OHE
# Asegúrate de que las columnas del DataFrame de entrada coincidan con las del entrenamiento.

# Crear una fila de entrada que coincida con las características de entrenamiento
# Esto es lo más delicado: el modelo espera una entrada específica (orden y tipo de datos)
# Si el preprocesador no se guardó, tendrías que replicar la lógica de preprocesamiento aquí.
# Por ejemplo:
# 1. Escalar 'age', 'sibsp', 'parch', 'fare' con los mismos escaladores usados en el entrenamiento.
# 2. Aplicar One-Hot Encoding a 'pclass' y 'embarked' (si no los mapeaste directamente a numéricos).

# Para este esqueleto, asumiremos que el modelo espera las entradas en el orden y formato
# en que las pasamos, lo cual es MUY simplificado.
# Una solución robusta implica:
# A) Guardar el `ColumnTransformer` junto con el modelo.
# B) Usar `preprocessor.transform(input_df)`

# Ejemplo de cómo se vería si el preprocesador fue guardado:
# processed_input = preprocessor.transform(input_df)

# Si no guardamos el preprocesador, una solución simple (menos robusta):
# Asegúrate de que el orden de las características sea el mismo que en el entrenamiento
# Y que los valores numéricos estén escalados y categóricos codificados de la misma manera.
# Aquí estamos asumiendo que el modelo puede manejar las entradas numéricas directamente
# después de mapear 'sex' y 'embarked', y que 'pclass' ya es numérico.
# Esto es un placeholder para que el script funcione, pero la lógica de preprocesamiento
# debe ser la correcta y coherente.

# Simulación de un input_array listo para el modelo:
# Esto DEBE coincidir con el formato de entrada del modelo después del preprocesamiento
# Si tu modelo espera 10 características (4 numéricas escaladas, 3 categóricas OHE resultando en 6, 1 sex),
# el array debe tener esa forma.
# Un ejemplo MUY básico si el modelo espera todo en un array plano sin más preprocesamiento:

# Convertir input_df a un array de numpy
# Asegúrate de que las columnas estén en el mismo orden que las características que el modelo fue entrenado.
# Por ejemplo, si el modelo fue entrenado con ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
# y luego se preprocesaron. Aquí, para que funcione, tendremos que simular ese resultado.

# Esto es un punto crítico: la preparación de `input_for_prediction` debe ser idéntica al entrenamiento.
# Para la demo, lo haremos lo más simple posible, asumiendo que el modelo puede manejarlo así,
# lo cual es poco probable para un modelo complejo.

# Mejor enfoque: Re-crear el ColumnTransformer (o cargarlo si lo guardaste)
# y usarlo en el input_df

# Asumiendo que preprocessor es el mismo ColumnTransformer que en train_model.py
# Y que fue guardado y cargado.
# Si no, tendrías que replicar la lógica de transformaciones (StandardScaler, OneHotEncoder) aquí manualmente
# sobre input_df para obtener processed_input.

# Para el ejemplo más simple:
# Asumimos que `input_df` es un DataFrame de una fila con las columnas correctas
# y que el modelo de Keras puede procesar esto directamente (lo cual no es común sin preprocesamiento).
# **La forma correcta es usar el `preprocessor` entrenado en `parte0`.**
# Si guardaste el preprocesador:
# processed_input = preprocessor.transform(input_df)
# Si NO lo guardaste, debes aplicar las transformaciones aquí manualmente.

# Ejemplo muy simplificado de la creación del array de entrada si NO hay un preprocesador cargado:
# Esto es SOLO para que el código sea ejecutable, pero no es la mejor práctica.
# Necesitas que esto coincida con lo que tu modelo espera.
# Si tu modelo usa un StandardScaler y OneHotEncoder, DEBES aplicarlos aquí.

# Aquí, asumimos que el modelo recibe un array numérico con todas las características.
# Lo más probable es que necesites escalar 'age', 'sibsp', 'parch', 'fare'
# y aplicar one-hot encoding a 'pclass' y 'embarked'.
# Si no guardaste el preprocesador, tendrías que instanciar y aplicar StandardScaler y OneHotEncoder
# con los parámetros de entrenamiento.

# Para un modelo Keras simple que acepta entradas numéricas directas (después de mapear categóricas):
features_order = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
input_for_prediction = input_df[features_order].values

if st.button("Predecir Supervivencia"):
    if model:
        try:
            # Realizar la predicción
            prediction = model.predict(input_for_prediction)[0][0] # Ajusta según la salida de tu modelo

            st.subheader("Resultado de la Predicción:")
            if prediction > 0.5:
                st.success(f"¡El pasajero **probablemente sobrevivió**! (Probabilidad: {prediction:.2f})")
                st.image("https://media.giphy.com/media/oR8Sx09kG14JgQY9Pj/giphy.gif", caption="Sobrevivió!")
            else:
                st.error(f"El pasajero **probablemente no sobrevivió**. (Probabilidad: {prediction:.2f})")
                st.image("https://media.giphy.com/media/vQG7Gg8x7kX5K/giphy.gif", caption="No sobrevivió.")

            st.info(f"Nota: Una probabilidad mayor a 0.5 indica supervivencia.")

        except Exception as e:
            st.error(f"Ocurrió un error al realizar la predicción: {e}")
            st.write("Asegúrate de que las entradas y el modelo sean compatibles.")
    else:
        st.warning("El modelo no ha sido cargado correctamente.")

st.markdown("---")
st.write("Creado como parte de un taller de Machine Learning y Streamlit.")