import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd

# Título de la aplicación
st.title('Aplicación para Predicción con Random Forest')

# Carga de datos Excel
uploaded_file = st.file_uploader("Cargar un archivo Excel para análisis", type=["xlsx", "xls"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, engine='openpyxl')  # Asegúrate de especificar el motor adecuado
    # Muestra las primeras filas del conjunto de datos
    st.write("Vista previa de los datos:", df.head())

    # Si el usuario carga el archivo, proceder con el procesamiento
    if st.button('Entrenar Modelo'):
        features = ['NoDesembolso', 'RecenciaDesembolso', 'Años', 'Sector', 'SubSector', 'TipodePrestamo', 'Pais', 'IDOperacion']
        X = df[features]
        y = df['Porcentaje Acumulado']
        
        # Preprocesamiento y modelo
        categorical_features = ['Sector', 'SubSector', 'TipodePrestamo', 'Pais', 'IDOperacion']
        numerical_features = ['NoDesembolso', 'RecenciaDesembolso', 'Años']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', SimpleImputer(strategy='mean'), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))])

        # División de datos y entrenamiento
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        # Predicción y evaluación
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        # Mostrar métricas
        st.write(f"RMSE: {rmse}")
        st.write(f"R^2: {r2}")

        # Crea secciones en tu aplicación para las entradas del usuario
        st.header('Entradas para Predicción de Porcentaje Acumulado')

        # Para demostración, aquí definimos manualmente las opciones
        # En una aplicación real, considera extraer estos valores de tu conjunto de datos o definirlos según tu conocimiento
        categorical_options = {
            'Sector': ['SOC', 'INF', 'PRO'],
            'SubSector': ['AYS', 'TYL', 'SYE','PRO','GOB','VDU','AMB','FIN','ENE'],
            'TipodePrestamo': ['EMERGENCIA', 'PRÉSTAMO', 'PROGRAMA GLOBAL DE OBRAS MULTIPLES', 'PROYECTOS ESPECIFICOS DE INVERSION','PROGRAMA EN FUNCION DE RESULTADOS', 'PROGRAMA PARA EL FINANCIAMIENTO PROPORCIONAL DE INVERSIONES'],
            'Pais': ['ARGENTINA', 'BOLIVIA', 'BRASIL', 'PARAGUAY', 'URUGUA'],
            'IDOperacion' :['AR019_1' ,'AR020_1', 'AR021_1', 'AR022_1', 'AR024_1', 'AR025_1', 'AR026_1', 'AR027_1', 'AR028_1', 'AR030_1', 'AR031_1', 'AR031_2', 'AR033_1', 'AR036_1', 'AR037_1', 'AR038_1', 'AR040_1', 'AR043_1', 'AR043_2', 'AR044_1', 'AR044_2', 'AR046_1', 'AR048_1', 'AR050_1', 'BO020_1', 'BO021_1', 'BO022_1', 'BO023_1', 'BO024_1', 'BO025_1', 'BO028_1', 'BO029_1', 'BO030_1', 'BO032_1', 'BO032_2', 'BR016_1', 'BR025_1', 'PY020_1', 'PY020_2', 'PY021_1', 'PY026_1', 'UR014_1', 'UR016_1', 'UR017_1', 'UR018_1', 'UR019_1', 'UR020_1', 'UR021_1', 'UR022_1', 'UR023_1', 'UR024_1']
        }

        # Características categóricas: para cada una, creamos un selectbox con las opciones
        inputs_categorical = {
            feature: st.selectbox(f"{feature}", options=options)
            for feature, options in categorical_options.items()
        }

        # Características numéricas: para cada una, creamos un input numérico
        numerical_features = ['NoDesembolso', 'RecenciaDesembolso', 'Años']
        inputs_numerical = {
            feature: st.number_input(f"{feature}", step=1.0)
            for feature in numerical_features
        }

        if st.button('Predecir'):
            # Combina inputs numéricos y categóricos en un único DataFrame
            input_data = {**inputs_numerical, **inputs_categorical}
            input_df = pd.DataFrame([input_data])

            # Asegúrate de que las columnas sigan el orden correcto
            input_df = input_df[features]

            # Realiza la predicción
            prediction = model.predict(input_df)

            # Muestra la predicción
            st.write(f"Predicción de Porcentaje Acumulado: {prediction[0]}")
