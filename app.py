import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# Configuración de la página
st.set_page_config(
    page_title="EDA Deportivo",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título de la aplicación
st.title("🏆 Análisis Exploratorio de Datos Deportivos")
st.markdown("""
Esta aplicación te permite generar datos sintéticos relacionados con deportes y realizar un análisis exploratorio interactivo.
Puedes elegir el número de muestras (hasta 500) y hasta 6 columnas con diferentes tipos de variables.
""")

# Sidebar para controles
with st.sidebar:
    st.header("⚙️ Configuración de Datos")
    
    # Selector de número de muestras
    n_muestras = st.slider("Número de muestras", min_value=50, max_value=500, value=200, step=50)
    
    # Selector de número de columnas
    n_columnas = st.slider("Número de columnas", min_value=2, max_value=6, value=4, step=1)
    
    # Selector de tipos de datos
    tipos_disponibles = ["Edad", "Altura", "Peso", "Deporte", "Nivel", "Puntuación", 
                         "Género", "Equipo", "Lesiones", "Experiencia", "Victorias"]
    
    columnas_seleccionadas = st.multiselect(
        "Selecciona las columnas para tu dataset:",
        tipos_disponibles,
        default=["Edad", "Deporte", "Puntuación", "Nivel"]
    )
    
    # Asegurarse de que no se seleccionen más columnas de las permitidas
    if len(columnas_seleccionadas) > n_columnas:
        st.warning(f"Solo puedes seleccionar hasta {n_columnas} columnas. Se tomarán las primeras {n_columnas}.")
        columnas_seleccionadas = columnas_seleccionadas[:n_columnas]
    
    # Botón para generar datos
    generar_datos = st.button("Generar Datos", type="primary")

# Función para generar datos sintéticos
def generar_datos_deportivos(n_muestras, columnas_seleccionadas):
    datos = {}
    np.random.seed(42)  # Para reproducibilidad
    
    if "Edad" in columnas_seleccionadas:
        # Edad entre 15 y 40 años, distribución normal
        edad = np.random.normal(25, 5, n_muestras)
        edad = np.clip(edad, 15, 40).astype(int)
        datos["Edad"] = edad
    
    if "Altura" in columnas_seleccionadas:
        # Altura en cm, distribución normal diferente por género
        altura = np.random.normal(175, 10, n_muestras)
        datos["Altura"] = np.clip(altura, 150, 210).astype(int)
    
    if "Peso" in columnas_seleccionadas:
        # Peso en kg, correlacionado con altura
        if "Altura" in datos:
            peso = datos["Altura"] * 0.4 + np.random.normal(0, 5, n_muestras)
        else:
            peso = np.random.normal(70, 10, n_muestras)
        datos["Peso"] = np.clip(peso, 50, 120).astype(int)
    
    if "Deporte" in columnas_seleccionadas:
        # Deportes con probabilidades diferentes
        deportes = ["Fútbol", "Baloncesto", "Tenis", "Natación", "Atletismo", "Ciclismo", "Voleibol"]
        probabilidades = [0.3, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05]
        datos["Deporte"] = np.random.choice(deportes, n_muestras, p=probabilidades)
    
    if "Nivel" in columnas_seleccionadas:
        # Nivel de habilidad (Principiante, Intermedio, Avanzado, Élite)
        niveles = ["Principiante", "Intermedio", "Avanzado", "Élite"]
        datos["Nivel"] = np.random.choice(niveles, n_muestras, p=[0.3, 0.4, 0.2, 0.1])
    
    if "Puntuación" in columnas_seleccionadas:
        # Puntuación de rendimiento (0-100)
        if "Nivel" in datos:
            # La puntuación depende del nivel
            puntuacion = np.zeros(n_muestras)
            for i, nivel in enumerate(datos["Nivel"]):
                if nivel == "Principiante":
                    puntuacion[i] = np.random.normal(50, 10)
                elif nivel == "Intermedio":
                    puntuacion[i] = np.random.normal(70, 8)
                elif nivel == "Avanzado":
                    puntuacion[i] = np.random.normal(85, 6)
                else:  # Élite
                    puntuacion[i] = np.random.normal(95, 3)
        else:
            puntuacion = np.random.normal(70, 15, n_muestras)
        datos["Puntuación"] = np.clip(puntuacion, 0, 100).astype(int)
    
    if "Género" in columnas_seleccionadas:
        # Género binario para simplificar
        datos["Género"] = np.random.choice(["Masculino", "Femenino"], n_muestras, p=[0.6, 0.4])
    
    if "Equipo" in columnas_seleccionadas:
        # Nombres de equipos
        equipos = ["Águilas", "Leones", "Tiburones", "Halcones", "Panteras", "Osos", "Dragones"]
        datos["Equipo"] = np.random.choice(equipos, n_muestras)
    
    if "Lesiones" in columnas_seleccionadas:
        # Número de lesiones en el último año (distribución de Poisson)
        lesiones = np.random.poisson(0.7, n_muestras)
        datos["Lesiones"] = lesiones
    
    if "Experiencia" in columnas_seleccionadas:
        # Años de experiencia (correlacionado con la edad)
        if "Edad" in datos:
            experiencia = datos["Edad"] - 15 + np.random.randint(-3, 4, n_muestras)
            experiencia = np.clip(experiencia, 0, 25)
        else:
            experiencia = np.random.randint(0, 20, n_muestras)
        datos["Experiencia"] = experiencia
    
    if "Victorias" in columnas_seleccionadas:
        # Porcentaje de victorias en competiciones
        if "Nivel" in datos:
            victorias = np.zeros(n_muestras)
            for i, nivel in enumerate(datos["Nivel"]):
                if nivel == "Principiante":
                    victorias[i] = np.random.normal(30, 10)
                elif nivel == "Intermedio":
                    victorias[i] = np.random.normal(50, 12)
                elif nivel == "Avanzado":
                    victorias[i] = np.random.normal(70, 8)
                else:  # Élite
                    victorias[i] = np.random.normal(85, 5)
        else:
            victorias = np.random.normal(50, 20, n_muestras)
        datos["% Victorias"] = np.clip(victorias, 0, 100).astype(int)
    
    return pd.DataFrame(datos)

# Generar datos si se ha hecho clic en el botón o si no hay datos aún
if generar_datos or 'df' not in st.session_state:
    with st.spinner('Generando datos...'):
        df = generar_datos_deportivos(n_muestras, columnas_seleccionadas)
        st.session_state.df = df
        st.session_state.columnas_seleccionadas = columnas_seleccionadas
        st.success('¡Datos generados exitosamente!')

# Mostrar datos si están disponibles
if 'df' in st.session_state:
    df = st.session_state.df
    columnas_seleccionadas = st.session_state.columnas_seleccionadas
    
    # Mostrar información básica del dataset
    st.header("📊 Dataset Generado")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Mostrar estadísticas descriptivas
    st.subheader("📈 Estadísticas Descriptivas")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Resumen de variables numéricas:**")
        st.write(df.describe())
    
    with col2:
        st.write("**Resumen de variables categóricas:**")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            st.write(f"**{col}**: {df[col].nunique()} categorías")
            st.write(df[col].value_counts())
    
    # Análisis Exploratorio de Datos
    st.header("🔍 Análisis Exploratorio de Datos")
    
    # Selección de tipo de gráfico
    tipo_grafico = st.selectbox(
        "Selecciona el tipo de gráfico:",
        ["Histograma", "Gráfico de Barras", "Gráfico de Dispersión", 
         "Gráfico de Pastel", "Boxplot", "Heatmap de Correlación"]
    )
    
    # Generar el gráfico seleccionado
    if tipo_grafico == "Histograma":
        st.subheader("📊 Histograma")
        col_num = st.selectbox("Selecciona una columna numérica:", 
                              df.select_dtypes(include=[np.number]).columns.tolist())
        
        fig = px.histogram(df, x=col_num, nbins=20, 
                          title=f"Distribución de {col_num}",
                          color_discrete_sequence=['#1f77b4'])
        st.plotly_chart(fig, use_container_width=True)
        
        # Prueba de normalidad
        if len(df[col_num]) > 2:
            stat, p_value = stats.normaltest(df[col_num].dropna())
            st.write(f"**Prueba de normalidad (D'Agostino-Pearson):**")
            st.write(f"Estadístico = {stat:.3f}, p-valor = {p_value:.3f}")
            if p_value > 0.05:
                st.write("Los datos parecen seguir una distribución normal (p > 0.05)")
            else:
                st.write("Los datos no siguen una distribución normal (p ≤ 0.05)")
    
    elif tipo_grafico == "Gráfico de Barras":
        st.subheader("📊 Gráfico de Barras")
        col_cat = st.selectbox("Selecciona una columna categórica:", 
                              df.select_dtypes(include=['object']).columns.tolist())
        
        if st.checkbox("Mostrar por porcentajes"):
            counts = df[col_cat].value_counts(normalize=True) * 100
            fig = px.bar(x=counts.index, y=counts.values, 
                        title=f"Distribución de {col_cat} (%)",
                        labels={'x': col_cat, 'y': 'Porcentaje'})
        else:
            counts = df[col_cat].value_counts()
            fig = px.bar(x=counts.index, y=counts.values, 
                        title=f"Distribución de {col_cat}",
                        labels={'x': col_cat, 'y': 'Frecuencia'})
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif tipo_grafico == "Gráfico de Dispersión":
        st.subheader("📈 Gráfico de Dispersión")
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("Variable X:", 
                                df.select_dtypes(include=[np.number]).columns.tolist())
        with col2:
            y_col = st.selectbox("Variable Y:", 
                                df.select_dtypes(include=[np.number]).columns.tolist())
        
        color_col = st.selectbox("Color por (opcional):", 
                                [None] + df.columns.tolist())
        
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        title=f"{y_col} vs {x_col}",
                        trendline="ols" if st.checkbox("Mostrar línea de tendencia") else None)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calcular correlación si ambas variables son numéricas
        if df[x_col].dtype in [np.number] and df[y_col].dtype in [np.number]:
            corr = df[x_col].corr(df[y_col])
            st.write(f"**Coeficiente de correlación de Pearson:** {corr:.3f}")
    
    elif tipo_grafico == "Gráfico de Pastel":
        st.subheader("🥧 Gráfico de Pastel")
        col_cat = st.selectbox("Selecciona una columna categórica para el gráfico de pastel:", 
                              df.select_dtypes(include=['object']).columns.tolist())
        
        counts = df[col_cat].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, 
                    title=f"Distribución de {col_cat}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif tipo_grafico == "Boxplot":
        st.subheader("📦 Boxplot")
        col_num = st.selectbox("Selecciona una columna numérica:", 
                              df.select_dtypes(include=[np.number]).columns.tolist())
        col_cat = st.selectbox("Selecciona una columna categórica (opcional):", 
                              [None] + df.select_dtypes(include=['object']).columns.tolist())
        
        if col_cat:
            fig = px.box(df, x=col_cat, y=col_num, 
                        title=f"Distribución de {col_num} por {col_cat}")
        else:
            fig = px.box(df, y=col_num, title=f"Distribución de {col_num}")
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif tipo_grafico == "Heatmap de Correlación":
        st.subheader("🔥 Heatmap de Correlación")
        
        # Seleccionar solo columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                           text_auto=True, 
                           aspect="auto",
                           title="Matriz de Correlación",
                           color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
            
            # Mostrar pares de variables con mayor correlación
            st.write("**Pares de variables con mayor correlación (absoluta):**")
            corr_pairs = corr_matrix.unstack().sort_values(key=abs, ascending=False)
            # Eliminar autocorrelaciones (valor 1.0)
            corr_pairs = corr_pairs[corr_pairs != 1.0]
            # Mostrar los 5 pares principales
            for idx, value in corr_pairs.head(5).items():
                st.write(f"{idx[0]} - {idx[1]}: {value:.3f}")
        else:
            st.warning("Se necesitan al menos 2 variables numéricas para el heatmap de correlación.")
    
    # Análisis adicional por deporte si está disponible
    if "Deporte" in df.columns:
        st.header("🏅 Análisis por Deporte")
        
        deporte_seleccionado = st.selectbox("Selecciona un deporte para analizar:", 
                                           df["Deporte"].unique())
        
        df_deporte = df[df["Deporte"] == deporte_seleccionado]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Estadísticas para {deporte_seleccionado}:**")
            st.write(df_deporte.describe())
        
        with col2:
            if "Nivel" in df.columns:
                st.write(f"**Distribución por nivel en {deporte_seleccionado}:**")
                nivel_counts = df_deporte["Nivel"].value_counts()
                fig_nivel = px.pie(values=nivel_counts.values, names=nivel_counts.index,
                                  title=f"Niveles en {deporte_seleccionado}")
                st.plotly_chart(fig_nivel, use_container_width=True)
    
    # Opción para descargar los datos
    st.header("💾 Descargar Datos")
    csv = df.to_csv(index=False)
    st.download_button(
        label="Descargar datos como CSV",
        data=csv,
        file_name="datos_deportivos.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Configura los parámetros en la barra lateral y haz clic en 'Generar Datos' para comenzar.")

# Footer
st.markdown("---")
st.markdown("### 🏆 Aplicación de Análisis Exploratorio de Datos Deportivos")
st.markdown("Genera datos sintéticos y explora visualizaciones interactivas.")
