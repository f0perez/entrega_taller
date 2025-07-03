import pandas as pd
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# ---------------------------
# Cargar y limpiar los datos
# ---------------------------
df = pd.read_csv('titanic.csv')

# Eliminar filas con valores nulos en columnas críticas
df.dropna(subset=['Survived', 'Embarked', 'Age'], inplace=True)

# Simplificar variables categóricas
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Imputar 'Fare' si tiene NaN
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Selección de características relevantes
selected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

X = df[selected_features]
y = df[target]

# ---------------------------
# Preprocesamiento
# ---------------------------
numerical_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Embarked']  # 'Sex' ya está en 0/1

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'  # incluye 'Sex' tal como está
)

# ---------------------------
# Dividir y transformar datos
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ---------------------------
# Modelo Keras
# ---------------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_processed.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ---------------------------
# Entrenamiento
# ---------------------------
model.fit(X_train_processed, y_train, epochs=10, batch_size=32, validation_data=(X_test_processed, y_test))

# ---------------------------
# Guardar modelo y preprocesador
# ---------------------------
output_dir = os.path.join(os.path.dirname(os.getcwd()), 'app')
os.makedirs(output_dir, exist_ok=True)

model_save_path = os.path.join(output_dir, 'model.keras')
model.save(model_save_path)
print(f"✅ Modelo guardado en: {model_save_path}")

preprocessor_save_path = os.path.join(output_dir, 'preprocessor.joblib')
joblib.dump(preprocessor, preprocessor_save_path)
print(f"✅ Preprocesador guardado en: {preprocessor_save_path}")
