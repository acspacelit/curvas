import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import polyfit, poly1d
from sklearn.metrics import r2_score
import io

def dataframe_to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Sheet1')
    excel_bytes = output.getvalue()
    return excel_bytes

# Función para calcular y graficar R^2 y las líneas de tendencia
def calculate_and_plot_r2_for_country(df, country):
    fig, ax = plt.subplots(figsize=(5, 4))
    country_data = df[df['Pais'] == country]
    if not country_data.empty:
        x = country_data['Años']
        y = country_data['Porcentaje Acumulado']
        z = polyfit(x, y, 3)
        p = poly1d(z)
        y_pred = p(x)
        r2 = r2_score(y, y_pred)

        # Graficar
        ax.scatter(x, y, label=f'{country} Datos')
        ax.plot(np.sort(x), p(np.sort(x)), color='red', label=f'Línea de tendencia (R²={r2:.3f})')
        ax.set_title(f'{country}')
        ax.set_xlabel('Años')
        ax.set_ylabel('Porcentaje Acumulado')
        ax.legend()

    plt.tight_layout()
    st.pyplot(fig)

def app():
    st.title('Análisis de Desembolsos')

    uploaded_file = st.file_uploader("Elige un archivo Excel (.xlsx)", type=['xlsx'])
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

        # Filtrar para excluir valores negativos de "Años"
        df_filtered = df[df['Años'] >= 0]

        # Permitir al usuario elegir un país específico para el análisis
        all_countries = df_filtered['Pais'].unique()
        selected_country = st.selectbox('Elige un país para el análisis', all_countries)

        # Input para IDOperacion
        specified_id_default = "AR019_1 ,AR020_1, AR021_1, AR022_1, AR024_1, AR025_1, AR026_1, AR027_1, AR028_1, AR030_1, AR031_1, AR031_2, AR033_1, AR036_1, AR037_1, AR038_1, AR040_1, AR043_1, AR043_2, AR044_1, AR044_2, AR046_1, AR048_1, AR050_1, BO020_1, BO021_1, BO022_1, BO023_1, BO024_1, BO025_1, BO028_1, BO029_1, BO030_1, BO032_1, BO032_2, BR016_1, BR025_1, PY020_1, PY020_2, PY021_1, PY026_1, UR014_1, UR016_1, UR017_1, UR018_1, UR019_1, UR020_1, UR021_1, UR022_1, UR023_1, UR024_1"
        specified_id = st.text_area("Introduce los IDOperacion separados por comas", value=specified_id_default)
        specified_id_list = [id.strip() for id in specified_id.split(",")]

        # Filtrar el DataFrame según los IDOperacion especificados
        df_filtered_specified = df_filtered[df_filtered['IDOperacion'].isin(specified_id_list)]
        st.write(df_filtered_specified)
        # Convertir el DataFrame a bytes y agregar botón de descarga para ambas tablas
        excel_bytes_monto = dataframe_to_excel_bytes(df_filtered_specified)
        st.download_button(
            label="Descargar DataFrame en Excel",
            data=excel_bytes_monto,
            file_name="IDOperaciones.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        if st.button('Calcular R^2 y mostrar gráfico'):
            calculate_and_plot_r2_for_country(df_filtered_specified[df_filtered_specified['Pais'] == selected_country], selected_country)

if __name__ == "__main__":
    app()
