import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# Generación de dataset sintético
# ==============================
def generar_dataset(n_muestras, n_columnas):
    np.random.seed(42)  # reproducibilidad

    deportes = ["Fútbol", "Baloncesto", "Tenis", "Natación", "Ciclismo", "Atletismo"]
    equipos = ["Equipo A", "Equipo B", "Equipo C", "Equipo D"]
    paises = ["Colombia", "Brasil", "España", "EEUU", "Argentina"]

    data = {
        "Deporte": np.random.choice(deportes, n_muestras),
        "Equipo": np.random.choice(equipos, n_muestras),
        "País": np.random.choice(paises, n_muestras),
        "Edad": np.random.randint(16, 40, n_muestras),
        "Puntaje": np.random.randint(0, 100, n_muestras),
        "Altura_cm": np.random.normal(175, 10, n_muestras).astype(int),
    }

    df = pd.DataFrame(data)

    # limitar el número de columnas
    columnas_seleccionadas = list(df.columns[:n_columnas])
    return df[columnas_seleccionadas]


# ==============================
# Streamlit APP
# ==============================
st.set_page_config(page_title="EDA en Deportes", layout="wide")

st.title("🏅 Análisis Exploratorio de Datos (EDA) - Deportes")

# Sidebar opciones
st.sidebar.header("Opciones de configuración")

n_muestras = st.sidebar.slider("Número de muestras", min_value=50, max_value=500, value=200, step=10)
n_columnas = st.sidebar.slider("Número de columnas", min_value=2, max_value=6, value=4)

df = generar_dataset(n_muestras, n_columnas)

# Mostrar dataset
if st.checkbox("Mostrar tabla de datos"):
    st.write(df.head(20))

# Selección de columnas para graficar
st.sidebar.subheader("Opciones de visualización")
columna_x = st.sidebar.selectbox("Columna en eje X", df.columns)
columna_y = st.sidebar.selectbox("Columna en eje Y (si aplica)", [None] + list(df.columns))
tipo_grafica = st.sidebar.selectbox(
    "Tipo de gráfica",
    ["Histograma", "Gráfico de barras", "Gráfico de dispersión", "Gráfico de pastel", "Tendencia (line plot)"]
)

# ==============================
# Generar visualizaciones
# ==============================
st.subheader("📊 Visualización de datos")

fig, ax = plt.subplots(figsize=(8, 5))

if tipo_grafica == "Histograma":
    sns.histplot(df[columna_x], bins=15, kde=True, ax=ax)
    ax.set_title(f"Histograma de {columna_x}")

elif tipo_grafica == "Gráfico de barras":
    conteo = df[columna_x].value_counts()
    conteo.plot(kind="bar", ax=ax)
    ax.set_title(f"Gráfico de barras de {columna_x}")
    ax.set_ylabel("Frecuencia")

elif tipo_grafica == "Gráfico de dispersión":
    if columna_y:
        sns.scatterplot(data=df, x=columna_x, y=columna_y, ax=ax)
        ax.set_title(f"Dispersión entre {columna_x} y {columna_y}")
    else:
        st.warning("Selecciona una columna para el eje Y.")

elif tipo_grafica == "Gráfico de pastel":
    conteo = df[columna_x].value_counts()
    ax.pie(conteo, labels=conteo.index, autopct="%1.1f%%")
    ax.set_title(f"Gráfico de pastel de {columna_x}")

elif tipo_grafica == "Tendencia (line plot)":
    if columna_y:
        sns.lineplot(data=df, x=columna_x, y=columna_y, ax=ax)
        ax.set_title(f"Tendencia de {columna_y} respecto a {columna_x}")
    else:
        st.warning("Selecciona una columna para el eje Y.")

st.pyplot(fig)

# ==============================
# Estadísticas descriptivas
# ==============================
if st.checkbox("Mostrar estadísticas descriptivas"):
    st.write(df.describe(include="all"))
