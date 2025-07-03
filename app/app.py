import streamlit as st
import tensorflow as tf
import pandas as pd
import joblib

# --- Funciones de Carga Cacheada ---
@st.cache_resource
def load_keras_model():
    try:
        model = tf.keras.models.load_model('model.keras')
        return model
    except Exception as e:
        import traceback
        st.error("Error al cargar el modelo:")
        st.text(traceback.format_exc())
        return None

@st.cache_resource
def load_preprocessor():
    try:
        preprocessor = joblib.load('preprocessor.joblib')
        return preprocessor
    except Exception as e:
        import traceback
        st.error("Error al cargar el preprocesador:")
        st.text(traceback.format_exc())
        return None


# Cargar el modelo y el preprocesador
model = load_keras_model()
preprocessor = load_preprocessor()

# --- Interfaz de Usuario de Streamlit ---

st.title("Predicción de Supervivencia en el Titanic")
st.write("Ingrese los datos del pasajero para predecir si sobrevivió al hundimiento del Titanic.")

st.sidebar.header("Características del Pasajero")

# Widgets de entrada para el usuario
pclass = st.sidebar.selectbox("Clase de Pasajero (Pclass)", [1, 2, 3])
sex = st.sidebar.radio("Sexo", ["Hombre", "Mujer"])
age = st.sidebar.slider("Edad", 1, 100, 30)
sibsp = st.sidebar.number_input("Número de hermanos/cónyuges a bordo (SibSp)", min_value=0, max_value=8, value=0)
parch = st.sidebar.number_input("Número de padres/hijos a bordo (Parch)", min_value=0, max_value=6, value=0)
fare = st.sidebar.number_input("Tarifa (Fare)", min_value=0.0, max_value=500.0, value=30.0, format="%.2f")
embarked = st.sidebar.selectbox("Puerto de Embarque", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

# Mapeo de entradas de texto a valores numéricos esperados por el preprocesador/modelo
sex_map = {"Hombre": 0, "Mujer": 1}
# En el preprocesador, 'Embarked' se maneja como categórica, no como numérica mapeada.
# Necesitamos pasar el string original para que OneHotEncoder lo maneje.
embarked_map = {"Southampton (S)": "S", "Cherbourg (C)": "C", "Queenstown (Q)": "Q"}

# Crear DataFrame con los datos de entrada
input_data = {
    'Pclass': pclass,
    'Sex': sex_map[sex], # Sex ya mapeado a 0/1
    'Age': float(age), # Aseguramos float
    'SibSp': float(sibsp),
    'Parch': float(parch),
    'Fare': float(fare),
    'Embarked': embarked_map[embarked] # Pasamos el string original para OneHotEncoder
}
input_df = pd.DataFrame([input_data])

# Orden de las columnas debe ser consistente con el entrenamiento original
# (importante si el preprocesador depende del orden)
# Las columnas utilizadas en el ColumnTransformer eran:
# numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
# categorical_features = ['Pclass', 'Embarked']
# 'Sex' se incluía con remainder='passthrough'
expected_features_order = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
input_df = input_df[expected_features_order]


if st.button("Predecir Supervivencia"):
    if model and preprocessor:
        try:
            # Preprocesar los datos de entrada usando el preprocesador cargado
            processed_input = preprocessor.transform(input_df)

            # Realizar la predicción
            prediction = model.predict(processed_input)[0][0] # Ajusta según la salida de tu modelo

            st.subheader("Resultado de la Predicción:")
            if prediction > 0.5:
                st.success(f"¡El pasajero **probablemente sobrevivió**! (Probabilidad: {prediction:.2f})")
                st.image("https://media.giphy.com/media/oR8Sx09kG14JgQY9Pj/giphy.gif", caption="Sobrevivió!")
            else:
                st.error(f"El pasajero **probablemente no sobrevivió**. (Probabilidad: {prediction:.2f})")
                st.image("https://media.giphy.com/media/vQG7Gg8x7kX5K/giphy.gif", caption="No sobrevivió.")

            st.info(f"Nota: Una probabilidad mayor a 0.5 indica supervivencia. Los valores se basan en un modelo entrenado con datos del Titanic.")

        except Exception as e:
            st.error(f"Ocurrió un error al realizar la predicción: {e}")
            st.write("Asegúrate de que las entradas y el modelo sean compatibles. Verifica los mensajes de error anteriores para más detalles.")
    else:
        st.warning("El modelo o el preprocesador no han sido cargados correctamente. Asegúrate de que `model.keras` y `preprocessor.joblib` estén en el directorio `app/`.")

st.markdown("---")
st.write("Creado como parte de un taller de Machine Learning y Streamlit.")