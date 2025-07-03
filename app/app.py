import streamlit as st
import tensorflow as tf
import pandas as pd
import joblib
import traceback

# --- Funciones de carga cacheada ---
@st.cache_resource
def load_keras_model():
    try:
        return tf.keras.models.load_model('model.keras')
    except Exception as e:
        st.error("Error al cargar el modelo:")
        st.text(traceback.format_exc())
        return None

@st.cache_resource
def load_preprocessor():
    try:
        return joblib.load('preprocessor.joblib')
    except Exception as e:
        st.error("Error al cargar el preprocesador:")
        st.text(traceback.format_exc())
        return None

# --- Cargar recursos ---
model = load_keras_model()
preprocessor = load_preprocessor()

# --- Interfaz de usuario ---
st.title("Predicción de Supervivencia en el Titanic")
st.write("Ingrese los datos del pasajero para predecir si sobrevivió al hundimiento del Titanic.")

st.sidebar.header("Características del Pasajero")
pclass = st.sidebar.selectbox("Clase de Pasajero (Pclass)", [1, 2, 3])
sex = st.sidebar.radio("Sexo", ["Hombre", "Mujer"])
age = st.sidebar.slider("Edad", 1, 100, 30)
sibsp = st.sidebar.number_input("Hermanos/Cónyuge a bordo (SibSp)", 0, 8, 0)
parch = st.sidebar.number_input("Padres/Hijos a bordo (Parch)", 0, 6, 0)
fare = st.sidebar.number_input("Tarifa (Fare)", 0.0, 500.0, 30.0, format="%.2f")
embarked = st.sidebar.selectbox("Puerto de Embarque", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

# --- Preparar datos ---
sex_map = {"Hombre": 0, "Mujer": 1}
embarked_map = {"Southampton (S)": "S", "Cherbourg (C)": "C", "Queenstown (Q)": "Q"}

input_df = pd.DataFrame([{
    'Pclass': pclass,
    'Sex': sex_map[sex],
    'Age': float(age),
    'SibSp': float(sibsp),
    'Parch': float(parch),
    'Fare': float(fare),
    'Embarked': embarked_map[embarked]
}])

expected_order = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
input_df = input_df[expected_order]

# --- Predicción ---
if st.button("Predecir Supervivencia"):
    if model and preprocessor:
        try:
            processed_input = preprocessor.transform(input_df)
            prediction = model.predict(processed_input)[0][0]

            st.subheader("Resultado de la Predicción:")
            if prediction > 0.5:
                st.success(f"¡Probablemente sobrevivió! (Probabilidad: {prediction:.2f})")
                st.image("https://media.giphy.com/media/oR8Sx09kG14JgQY9Pj/giphy.gif", caption="Sobrevivió")
            else:
                st.error(f"No sobrevivió. (Probabilidad: {prediction:.2f})")
                st.image("https://media.giphy.com/media/vQG7Gg8x7kX5K/giphy.gif", caption="No sobrevivió")

            st.info("Probabilidad > 0.5 indica supervivencia.")

        except Exception as e:
            st.error("Ocurrió un error al realizar la predicción.")
            st.text(traceback.format_exc())
    else:
        st.warning("Modelo o preprocesador no cargados correctamente.")

st.markdown("---")
st.caption("Creado como parte de un taller de Machine Learning y Streamlit.")
