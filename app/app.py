import streamlit as st
import tensorflow as tf
import pandas as pd
import joblib
import traceback

# --- Carga cacheada del modelo y preprocesador ---
@st.cache_resource
def load_keras_model():
    try:
        return tf.keras.models.load_model("model.keras")
    except Exception:
        st.error("Error al cargar el modelo:")
        st.text(traceback.format_exc())
        return None

@st.cache_resource
def load_preprocessor():
    try:
        return joblib.load("preprocessor.joblib")
    except Exception:
        st.error("Error al cargar el preprocesador:")
        st.text(traceback.format_exc())
        return None

# --- Recursos cargados ---
model = load_keras_model()
preprocessor = load_preprocessor()

# --- Interfaz principal ---
st.set_page_config(page_title="Predicción Titanic", layout="centered")
st.title("🎯 Predicción de Supervivencia - Titanic")
st.markdown("Completa el formulario para predecir si un pasajero habría sobrevivido.")

with st.form("form_prediccion"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Clase del pasajero", [1, 2, 3])
        sex = st.radio("Sexo", ["Hombre", "Mujer"])
        age = st.slider("Edad", 1, 100, 30)

    with col2:
        sibsp = st.number_input("Hermanos/Cónyuge a bordo", 0, 8, 0)
        parch = st.number_input("Padres/Hijos a bordo", 0, 6, 0)
        fare = st.number_input("Tarifa (USD)", 0.0, 500.0, 30.0, format="%.2f")
    
    embarked = st.selectbox("Puerto de embarque", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])
    
    submit = st.form_submit_button("Predecir")

# --- Procesamiento y predicción ---
if submit:
    if model and preprocessor:
        sex_map = {"Hombre": 0, "Mujer": 1}
        embarked_map = {
            "Southampton (S)": "S",
            "Cherbourg (C)": "C",
            "Queenstown (Q)": "Q"
        }

        input_data = pd.DataFrame([{
            'Pclass': pclass,
            'Sex': sex_map[sex],
            'Age': float(age),
            'SibSp': float(sibsp),
            'Parch': float(parch),
            'Fare': float(fare),
            'Embarked': embarked_map[embarked]
        }])

        input_df = input_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

        try:
            processed = preprocessor.transform(input_df)
            prob = model.predict(processed)[0][0]

            st.subheader("🔎 Resultado")
            if prob > 0.5:
                st.success(f"✅ Probablemente sobrevivió. (Probabilidad: {prob:.2f})")
            else:
                st.error(f"❌ Probablemente no sobrevivió. (Probabilidad: {prob:.2f})")

        except Exception:
            st.error("Ocurrió un error durante la predicción.")
            st.text(traceback.format_exc())
    else:
        st.warning("Modelo o preprocesador no cargados correctamente.")

st.markdown("---")
st.caption("🚢 Aplicación desarrollada por Fernando Canales para el curso de Plataformas de Machine Learning.")
