import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import threading
import io
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Configuración inicial
LOGGER = st.logger.get_logger(__name__)
_lock = threading.Lock()

# URLs de las hojas de Google Sheets
sheet_url_operaciones = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSeeSag2FV6X2E2aS7PIXfZmNOW7RQfjAfN9L9R_EaW_q0Z91DZYwK1eLtQago7LFy8qya-ltrJkosb/pub?gid=1582263470&single=true&output=csv"

# Inicializar la aplicación de Streamlit
st.title("Análisis de Desembolsos por País")

# Función para cargar los datos desde las hojas de Google Sheets
def load_data(url):
    with _lock:
        return pd.read_csv(url)

# Función para convertir el monto a un número flotante
def convert_to_float(monto_str):
    try:
        monto_str = monto_str.replace('.', '').replace(',', '.')
        return float(monto_str)
    except ValueError:
        return np.nan

# Función para procesar los datos
def process_data(df_operaciones):
    # Preparar los DataFrames seleccionando las columnas requeridas

    # Convertir 'Monto' a numérico
    df_operaciones['Monto'] = df_operaciones['Monto'].apply(convert_to_float)
    df_operaciones = df_operaciones.iloc[:, 1:].drop_duplicates()

    # Fusionar DataFrames
    merged_df = (df_operaciones)
    
    # Convertir fechas y calcular años
    merged_df['FechaEfectiva'] = pd.to_datetime(merged_df['FechaEfectiva'], dayfirst=True, errors='coerce')
    merged_df['FechaVigencia'] = pd.to_datetime(merged_df['FechaVigencia'], dayfirst=True, errors='coerce')
    merged_df['Ano'] = ((merged_df['FechaEfectiva'] - merged_df['FechaVigencia']).dt.days / 365).fillna(-1)
    merged_df['Ano_FechaEfectiva'] = pd.to_datetime(merged_df['FechaEfectiva']).dt.year
    # Convierte las columnas 'Monto' y 'AporteFONPLATAVigente' a numéricas
    merged_df['Monto'] = pd.to_numeric(merged_df['Monto'], errors='coerce')
    merged_df['AporteFONPLATAVigente'] = pd.to_numeric(merged_df['AporteFONPLATAVigente'], errors='coerce')
    merged_df['Porcentaje'] = ((merged_df['Monto'] / merged_df['AporteFONPLATAVigente']) * 100).round(2)
    filtered_df = merged_df[merged_df['Ano'] >= 0]
    filtered_df['Ano'] = filtered_df['Ano'].astype(int)
    st.write(filtered_df)

    # Obtener valores únicos y ordenarlos alfabéticamente por país
    unique_paises = sorted(filtered_df['Pais'].unique())

    # Selectbox para filtrar por País
    selected_pais = st.selectbox('Seleccione un país para filtrar', unique_paises)
    filtered_result_df = filtered_df[filtered_df['Pais'] == selected_pais]

    # Realizar cálculos
    result_df = filtered_result_df.groupby(['Pais', 'Ano'])['Monto'].sum().reset_index()
    result_df['Monto'] = result_df['Monto'] / 1_000_000  # Ajustar Monto a millones
    result_df['Monto Acumulado'] = result_df.groupby('Pais')['Monto'].cumsum().round(2).reset_index(drop=True)

    # Calcular el Monto Total por País y ajustar a millones
    total_monto_por_pais = df_operaciones.groupby('Pais')['Monto'].sum().reset_index()
    total_monto_por_pais['Monto'] = total_monto_por_pais['Monto'] / 1_000_000  # Ajustar MontoTotal a millones
    total_monto_por_pais.rename(columns={'Monto': 'MontoTotal'}, inplace=True)

    # Unir el Monto Total por País a result_df
    result_df = pd.merge(result_df, total_monto_por_pais, on='Pais', how='left')

    # Calcular el nuevo Porcentaje como la división del Monto por Año entre el Monto Total del País
    result_df['Porcentaje'] = (result_df['Monto'] / result_df['MontoTotal'] * 100).round(2)

    # Calcular el Porcentaje Acumulado por País
    result_df['Porcentaje Acumulado'] = result_df.groupby('Pais')['Porcentaje'].cumsum().round(2)

    # Realizar cálculos para el Año de Fecha Efectiva
    result_df_ano_efectiva = filtered_result_df.groupby(['Pais', 'Ano_FechaEfectiva'])['Monto'].sum().reset_index()
    result_df_ano_efectiva['Monto'] = result_df_ano_efectiva['Monto'] / 1_000_000  # Ajustar Monto a millones
    result_df_ano_efectiva['Monto Acumulado'] = result_df_ano_efectiva.groupby('Pais')['Monto'].cumsum().round(2).reset_index(drop=True)

    # Unir el Monto Total por País a result_df_ano_efectiva
    result_df_ano_efectiva = pd.merge(result_df_ano_efectiva, total_monto_por_pais, on='Pais', how='left')

    # Calcular el nuevo Porcentaje como la división del Monto por Año entre el Monto Total del País
    result_df_ano_efectiva['Porcentaje'] = (result_df_ano_efectiva['Monto'] / result_df_ano_efectiva['MontoTotal'] * 100).round(2)

    # Calcular el Porcentaje Acumulado por País
    result_df_ano_efectiva['Porcentaje Acumulado'] = result_df_ano_efectiva.groupby('Pais')['Porcentaje'].cumsum().round(2)

    return result_df, result_df_ano_efectiva


# Función para convertir DataFrame a Excel
def dataframe_to_excel_bytes(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Resultados', index=False)
    output.seek(0)
    return output

# Función para crear una gráfica de líneas con etiquetas
def line_chart_with_labels(data, x_col, y_col, title, color):
    chart = alt.Chart(data).mark_line(point=True, color=color).encode(
        x=alt.X(f'{x_col}:O', axis=alt.Axis(title='Año', labelAngle=0)),
        y=alt.Y(f'{y_col}:Q', axis=alt.Axis(title=y_col)),
        tooltip=[x_col, y_col]
    ).properties(
        title=title,
        width=600,
        height=400
    )
    text = chart.mark_text(
        align='left',
        baseline='middle',
        dx=18,
        dy=-18
    ).encode(
        text=alt.Text(f'{y_col}:Q', format='.2f')
    )
    return chart + text

#Funcion
def run():
    # Cargar y procesar los datos
    df_operaciones = load_data(sheet_url_operaciones)
    result_df, result_df_ano_efectiva = process_data(df_operaciones)

    # Define los colores para cada gráfico
    color_monto = 'steelblue'
    color_acumulado = 'goldenrod'
    color_porcentaje = 'salmon'

    # Crear y mostrar gráficos para result_df
    chart_monto = line_chart_with_labels(result_df, 'Ano', 'Monto', 'Monto por Año en Millones', color_monto)
    chart_monto_acumulado = line_chart_with_labels(result_df, 'Ano', 'Monto Acumulado', 'Monto Acumulado por Año en Millones', color_acumulado)
    chart_porcentaje_acumulado = line_chart_with_labels(result_df, 'Ano', 'Porcentaje Acumulado', 'Porcentaje Acumulado del Monto por Año', color_porcentaje)

    st.altair_chart(chart_monto, use_container_width=True)
    st.altair_chart(chart_monto_acumulado, use_container_width=True)
    st.altair_chart(chart_porcentaje_acumulado, use_container_width=True)
    
    # Mostrar la tabla "Tabla por Año"
    st.write("Tabla por Año:", result_df)
  
    # Crear y mostrar gráficos para result_df_ano_efectiva
    chart_monto_efectiva = line_chart_with_labels(result_df_ano_efectiva, 'Ano_FechaEfectiva', 'Monto', 'Monto por Año de Fecha Efectiva en Millones', color_monto)
    chart_monto_acumulado_efectiva = line_chart_with_labels(result_df_ano_efectiva, 'Ano_FechaEfectiva', 'Monto Acumulado', 'Monto Acumulado por Año de Fecha Efectiva en Millones', color_acumulado)
    chart_porcentaje_acumulado_efectiva = line_chart_with_labels(result_df_ano_efectiva, 'Ano_FechaEfectiva', 'Porcentaje Acumulado', 'Porcentaje Acumulado del Monto por Año de Fecha Efectiva', color_porcentaje)

    st.altair_chart(chart_monto_efectiva, use_container_width=True)
    st.altair_chart(chart_monto_acumulado_efectiva, use_container_width=True)
    st.altair_chart(chart_porcentaje_acumulado_efectiva, use_container_width=True)

    # Mostrar la tabla "Tabla por Año de Fecha Efectiva"
    st.write("Tabla por Año de Fecha Efectiva:", result_df_ano_efectiva)

if __name__ == "__main__":
    run()