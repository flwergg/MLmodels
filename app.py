import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

# Configuración de la página
st.set_page_config(
    page_title="EDA Deportes - Análisis Exploratorio",
    page_icon="⚽",
    layout="wide"
)

# Función para generar datos sintéticos de deportes
@st.cache_data
def generar_datos_sinteticos(n_muestras, columnas_seleccionadas):
    np.random.seed(42)  # Para reproducibilidad
    
    # Definir todas las posibles columnas con sus generadores
    generadores_columnas = {
        'edad': lambda n: np.random.randint(16, 45, n),
        'altura_cm': lambda n: np.random.normal(175, 15, n).astype(int),
        'peso_kg': lambda n: np.random.normal(70, 12, n).round(1),
        'tiempo_entrenamiento_horas': lambda n: np.random.exponential(2, n).round(1),
        'salario_miles': lambda n: np.random.lognormal(3, 1, n).round(1),
        'rendimiento_score': lambda n: np.random.beta(2, 2, n).round(3) * 100,
        'deporte': lambda n: np.random.choice(['Fútbol', 'Baloncesto', 'Tenis', 'Natación', 'Atletismo'], n),
        'posicion': lambda n: np.random.choice(['Delantero', 'Defensa', 'Mediocampo', 'Portero'], n),
        'nivel': lambda n: np.random.choice(['Principiante', 'Intermedio', 'Avanzado', 'Profesional'], n, 
                                          p=[0.3, 0.3, 0.25, 0.15]),
        'lesiones_anuales': lambda n: np.random.poisson(1.5, n),
        'victorias': lambda n: np.random.randint(0, 25, n),
        'derrotas': lambda n: np.random.randint(0, 20, n),
        'categoria': lambda n: np.random.choice(['Amateur', 'Semi-profesional', 'Profesional'], n),
        'experiencia_años': lambda n: np.random.randint(1, 20, n),
        'pais': lambda n: np.random.choice(['España', 'Brasil', 'Argentina', 'Francia', 'Alemania', 
                                          'Italia', 'Inglaterra', 'México', 'Colombia', 'Holanda'], n),
        'imc': lambda n: (np.random.normal(70, 12, n) / (np.random.normal(1.75, 0.15, n)**2)).round(1)
    }
    
    # Generar datos solo para las columnas seleccionadas
    data = {}
    for columna in columnas_seleccionadas:
        if columna in generadores_columnas:
            data[columna] = generadores_columnas[columna](n_muestras)
    
    return pd.DataFrame(data)

# Función para mostrar estadísticas descriptivas
def mostrar_estadisticas(df):
    st.subheader("📊 Estadísticas Descriptivas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Variables Numéricas:**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe())
        else:
            st.write("No hay variables numéricas")
    
    with col2:
        st.write("**Variables Categóricas:**")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            st.write(f"**{col}:**")
            value_counts = df[col].value_counts()
            st.write(value_counts)

# Función para crear gráficos
def crear_graficos(df, tipo_grafico, columnas_graf):
    if tipo_grafico == "Histograma":
        if len(columnas_graf) >= 1:
            col = columnas_graf[0]
            if df[col].dtype in ['int64', 'float64']:
                fig = px.histogram(df, x=col, title=f'Distribución de {col}')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Por favor selecciona una variable numérica para el histograma")
    
    elif tipo_grafico == "Gráfico de Barras":
        if len(columnas_graf) >= 1:
            col = columnas_graf[0]
            if df[col].dtype == 'object':
                value_counts = df[col].value_counts()
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=f'Frecuencia de {col}')
                fig.update_xaxes(title=col)
                fig.update_yaxes(title='Frecuencia')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Por favor selecciona una variable categórica para el gráfico de barras")
    
    elif tipo_grafico == "Gráfico de Dispersión":
        if len(columnas_graf) >= 2:
            x_col, y_col = columnas_graf[0], columnas_graf[1]
            if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
                fig = px.scatter(df, x=x_col, y=y_col, title=f'{x_col} vs {y_col}')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Por favor selecciona dos variables numéricas para el gráfico de dispersión")
        else:
            st.warning("Selecciona al menos 2 columnas para el gráfico de dispersión")
    
    elif tipo_grafico == "Gráfico de Pastel":
        if len(columnas_graf) >= 1:
            col = columnas_graf[0]
            if df[col].dtype == 'object':
                value_counts = df[col].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index,
                           title=f'Distribución de {col}')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Por favor selecciona una variable categórica para el gráfico de pastel")
    
    elif tipo_grafico == "Boxplot":
        if len(columnas_graf) >= 1:
            col = columnas_graf[0]
            if df[col].dtype in ['int64', 'float64']:
                fig = px.box(df, y=col, title=f'Boxplot de {col}')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Por favor selecciona una variable numérica para el boxplot")
    
    elif tipo_grafico == "Línea de Tendencia":
        if len(columnas_graf) >= 2:
            x_col, y_col = columnas_graf[0], columnas_graf[1]
            if df[x_col].dtype in ['int64', 'float64'] and df[y_col].dtype in ['int64', 'float64']:
                # Ordenar por la variable x para la línea de tendencia
                df_sorted = df.sort_values(x_col)
                fig = px.line(df_sorted, x=x_col, y=y_col, title=f'Tendencia: {x_col} vs {y_col}')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Por favor selecciona dos variables numéricas para la línea de tendencia")
        else:
            st.warning("Selecciona al menos 2 columnas para la línea de tendencia")

# Función principal
def main():
    st.title("⚽ EDA Deportes - Análisis Exploratorio de Datos")
    st.markdown("---")
    
    # Sidebar para configuración
    st.sidebar.header("🔧 Configuración de Datos")
    
    # Control deslizante para número de muestras
    n_muestras = st.sidebar.slider("Número de muestras", 10, 500, 100)
    
    # Selección de columnas disponibles
    columnas_disponibles = [
        'edad', 'altura_cm', 'peso_kg', 'tiempo_entrenamiento_horas',
        'salario_miles', 'rendimiento_score', 'deporte', 'posicion',
        'nivel', 'lesiones_anuales', 'victorias', 'derrotas',
        'categoria', 'experiencia_años', 'pais', 'imc'
    ]
    
    # Multiselect para elegir columnas (máximo 6)
    columnas_seleccionadas = st.sidebar.multiselect(
        "Selecciona columnas (máximo 6)",
        columnas_disponibles,
        default=['edad', 'altura_cm', 'peso_kg', 'deporte'],
        max_selections=6
    )
    
    if len(columnas_seleccionadas) == 0:
        st.warning("Por favor selecciona al menos una columna")
        return
    
    # Generar datos
    if st.sidebar.button("🔄 Generar Nuevos Datos", key="generar"):
        st.cache_data.clear()
    
    # Generar dataset
    df = generar_datos_sinteticos(n_muestras, columnas_seleccionadas)
    
    # Tabs para organizar el contenido
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Datos", "📊 Estadísticas", "📈 Visualizaciones", "🔍 Análisis"])
    
    with tab1:
        st.subheader("📋 Conjunto de Datos Generado")
        st.write(f"**Dimensiones:** {df.shape[0]} filas × {df.shape[1]} columnas")
        
        # Mostrar información del dataset
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de registros", df.shape[0])
        with col2:
            st.metric("Total de columnas", df.shape[1])
        with col3:
            st.metric("Valores faltantes", df.isnull().sum().sum())
        
        st.dataframe(df, use_container_width=True)
        
        # Botón para descargar datos
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Descargar datos como CSV",
            data=csv,
            file_name='datos_deportes.csv',
            mime='text/csv'
        )
    
    with tab2:
        mostrar_estadisticas(df)
        
        # Matriz de correlación si hay variables numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.subheader("🔗 Matriz de Correlación")
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                          title="Matriz de Correlación")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Visualizaciones Interactivas")
        
        # Selección de tipo de gráfico
        tipos_graficos = [
            "Histograma", "Gráfico de Barras", "Gráfico de Dispersión", 
            "Gráfico de Pastel", "Boxplot", "Línea de Tendencia"
        ]
        
        tipo_grafico = st.selectbox("Selecciona el tipo de gráfico:", tipos_graficos)
        
        # Selección de columnas para el gráfico
        if tipo_grafico in ["Gráfico de Dispersión", "Línea de Tendencia"]:
            columnas_graf = st.multiselect(
                "Selecciona columnas para el gráfico (mínimo 2):",
                df.columns.tolist(),
                default=df.columns.tolist()[:2] if len(df.columns) >= 2 else df.columns.tolist()
            )
        else:
            columnas_graf = st.multiselect(
                "Selecciona columna para el gráfico:",
                df.columns.tolist(),
                default=[df.columns.tolist()[0]] if len(df.columns) > 0 else []
            )
        
        if st.button("📊 Generar Gráfico"):
            crear_graficos(df, tipo_grafico, columnas_graf)
    
    with tab4:
        st.subheader("🔍 Análisis Exploratorio Avanzado")
        
        # Análisis por categorías si existe una columna categórica
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_col = st.selectbox("Selecciona variable categórica para análisis:", categorical_cols)
            
            st.write(f"**Análisis por {cat_col}:**")
            
            # Estadísticas por categoría
            if len(df.select_dtypes(include=[np.number]).columns) > 0:
                numeric_col = st.selectbox("Selecciona variable numérica:", 
                                         df.select_dtypes(include=[np.number]).columns)
                
                # Gráfico de caja por categoría
                fig = px.box(df, x=cat_col, y=numeric_col, 
                           title=f'{numeric_col} por {cat_col}')
                st.plotly_chart(fig, use_container_width=True)
                
                # Estadísticas por grupo
                group_stats = df.groupby(cat_col)[numeric_col].describe()
                st.dataframe(group_stats)
        
        # Información adicional
        st.subheader("ℹ️ Información del Dataset")
        buffer = df.dtypes.to_frame('Tipo de Dato')
        buffer['Valores Únicos'] = df.nunique()
        buffer['Valores Faltantes'] = df.isnull().sum()
        buffer['% Faltantes'] = (df.isnull().sum() / len(df) * 100).round(2)
        st.dataframe(buffer)

if __name__ == "__main__":
    main()
