import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_absolute_error

from io import BytesIO

st.set_page_config(page_title="COVID-19 Viz – Pregunta 2", layout="wide")

GITHUB_BASE = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports"

@st.cache_data(show_spinner=False)
def load_daily_report(yyyy_mm_dd: str):
    yyyy, mm, dd = yyyy_mm_dd.split("-")
    url = f"{GITHUB_BASE}/{mm}-{dd}-{yyyy}.csv"
    df = pd.read_csv(url)
    # normalizar nombres por si varían
    lower = {c.lower(): c for c in df.columns}
    cols = {
        "country": lower.get("country_region", "Country_Region"),
        "province": lower.get("province_state", "Province_State"),
        "confirmed": lower.get("confirmed", "Confirmed"),
        "deaths": lower.get("deaths", "Deaths"),
        "recovered": lower.get("recovered", "Recovered") if "recovered" in lower else None,
        "active": lower.get("active", "Active") if "active" in lower else None,
    }
    return df, url, cols

st.sidebar.title("Opciones")
fecha = st.sidebar.date_input("Fecha del reporte (JHU CSSE)", value=pd.to_datetime("2022-09-09"))
fecha_str = pd.to_datetime(fecha).strftime("%Y-%m-%d")
df, source_url, cols = load_daily_report(fecha_str)
st.sidebar.caption(f"Fuente: {source_url}")

# === NUEVO: selector de país para modelado ===
pais_sel = st.sidebar.selectbox("Selecciona un país para modelado", sorted(df[cols["country"]].unique()), index=0)

st.title("Exploración COVID-19 – Versión Streamlit (Preg2)")
st.caption("Adaptación fiel del script original: mostrar/ocultar filas/columnas y varios gráficos (líneas, barras, sectores, histograma y boxplot).")

# ———————————————————————————————————————————————
# a) Mostrar todas las filas del dataset
# ———————————————————————————————————————————————
st.header("a) Mostrar filas")
mostrar_todas = st.checkbox("Mostrar todas las filas", value=False)
if mostrar_todas:
    st.dataframe(df, use_container_width=True)
else:
    st.dataframe(df.head(25), use_container_width=True)

# ———————————————————————————————————————————————
# b) Mostrar todas las columnas del dataset
# ———————————————————————————————————————————————
st.header("b) Mostrar columnas")
with st.expander("Vista de columnas"):
    st.write(list(df.columns))

st.caption("Usa el scroll horizontal de la tabla para ver todas las columnas en pantalla.")

# ———————————————————————————————————————————————
# c) Línea del total de fallecidos (>2500) vs Confirmed/Recovered/Active por país
# ———————————————————————————————————————————————
st.header("c) Gráfica de líneas por país (muertes > 2500)")
C, D = cols["confirmed"], cols["deaths"]
R, A = cols["recovered"], cols["active"]

metrics = [m for m in [C, D, R, A] if m and m in df.columns]
base = df[[cols["country"]] + metrics].copy()
base = base.rename(columns={cols["country"]: "Country_Region"})

filtrado = base.loc[base[D] > 2500]
agr = filtrado.groupby("Country_Region").sum(numeric_only=True)
orden = agr.sort_values(D)

if not orden.empty:
    st.line_chart(orden[[c for c in [C, R, A] if c in orden.columns]])

# ———————————————————————————————————————————————
# d) Barras de fallecidos de estados de Estados Unidos
# ———————————————————————————————————————————————
st.header("d) Barras: fallecidos por estado de EE.UU.")
country_col = cols["country"]
prov_col = cols["province"]

dfu = df[df[country_col] == "US"]
if len(dfu) == 0:
    st.info("Para esta fecha no hay registros con Country_Region='US'.")
else:
    agg_us = dfu.groupby(prov_col)[D].sum(numeric_only=True).sort_values(ascending=False)
    top_n = st.slider("Top estados por fallecidos", 5, 50, 20)
    st.bar_chart(agg_us.head(top_n))

# ———————————————————————————————————————————————
# e) Gráfica de sectores (simulada)
# ———————————————————————————————————————————————
st.header("e) Gráfica de sectores (simulada)")
lista_paises = ["Colombia", "Chile", "Peru", "Argentina", "Mexico"]
sel = st.multiselect("Países", sorted(df[country_col].unique().tolist()), default=lista_paises)
agg_latam = df[df[country_col].isin(sel)].groupby(country_col)[D].sum(numeric_only=True)
if agg_latam.sum() > 0:
    st.write("Participación de fallecidos")
    st.dataframe(agg_latam)
    normalized = agg_latam / agg_latam.sum()
    st.bar_chart(normalized)
else:
    st.warning("Sin datos para los países seleccionados")

# ———————————————————————————————————————————————
# f) Histograma del total de fallecidos por país
# ———————————————————————————————————————————————
st.header("f) Histograma de fallecidos por país")
muertes_pais = df.groupby(country_col)[D].sum(numeric_only=True)
st.bar_chart(muertes_pais)

# ———————————————————————————————————————————————
# g) Boxplot simulado
# ———————————————————————————————————————————————
st.header("g) Boxplot (simulado)")
cols_box = [c for c in [C, D, R, A] if c and c in df.columns]
subset = df[cols_box].fillna(0)
subset_plot = subset.head(25)
st.write("Resumen estadístico (simulación de boxplot):")
st.dataframe(subset_plot.describe().T)


 =========================
# Cargar datos
# =========================
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/04-18-2022.csv"
    df = pd.read_csv(url)
    df["Last_Update"] = pd.to_datetime(df["Last_Update"])
    return df

df = load_data()

# =========================
# Sidebar con filtros (5.1)
# =========================
st.sidebar.header("Filtros")

paises = st.sidebar.multiselect("Países", df["Country_Region"].unique(), default=["Peru","Brazil","Mexico"])
fecha_min, fecha_max = df["Last_Update"].min(), df["Last_Update"].max()
rango_fechas = st.sidebar.date_input("Rango de fechas", [fecha_min, fecha_max])
umbral_confirmados = st.sidebar.slider("Umbral de confirmados", 0, int(df["Confirmed"].max()), 1000)

# =========================
# KPIs principales (5.2)
# =========================
st.title("📊 Dashboard COVID-19")

df_filtrado = df[df["Country_Region"].isin(paises)]
total_confirmados = df_filtrado["Confirmed"].sum()
total_fallecidos = df_filtrado["Deaths"].sum()
cfr = (total_fallecidos / total_confirmados) * 100 if total_confirmados > 0 else 0

col1, col2, col3 = st.columns(3)
col1.metric("Confirmados", f"{total_confirmados:,}")
col2.metric("Fallecidos", f"{total_fallecidos:,}")
col3.metric("CFR (%)", f"{cfr:.2f}")

# =========================
# Tabs (5.3)
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Visión general", "Estadística avanzada", 
    "Modelado temporal", "Clustering y PCA", "Calidad de datos"
])

# -------- Tab 1: Visión general
with tab1:
    st.subheader("Top-N países por confirmados")
    top_paises = df.groupby("Country_Region")["Confirmed"].sum().sort_values(ascending=False).head(10)
    fig1 = px.bar(top_paises, x=top_paises.index, y=top_paises.values, title="Top 10 países confirmados")
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Mapa interactivo")
    fig2 = px.scatter_geo(df, locations="Country_Region", locationmode="country names",
                          size="Confirmed", color="Deaths",
                          hover_name="Country_Region", title="Mapa global COVID-19")
    st.plotly_chart(fig2, use_container_width=True)

# -------- Tab 2: Estadística avanzada
with tab2:
    st.subheader("Boxplot Confirmados/Deaths")
    fig3 = px.box(df_filtrado, y=["Confirmed","Deaths"])
    st.plotly_chart(fig3, use_container_width=True)

    st.info("Aquí irían los test de hipótesis y los intervalos de confianza.")

# -------- Tab 3: Modelado temporal
with tab3:
    st.subheader("Series de tiempo (ejemplo simple)")
    ts = df_filtrado.groupby("Last_Update")["Confirmed"].sum().reset_index()
    fig4 = px.line(ts, x="Last_Update", y="Confirmed", title="Confirmados en el tiempo")
    st.plotly_chart(fig4, use_container_width=True)
    st.warning("Aquí se agregaría forecast y validación.")

# -------- Tab 4: Clustering y PCA
with tab4:
    st.subheader("Clustering/PCA")
    st.info("Pendiente: aplicar K-means y PCA para agrupar países.")

# -------- Tab 5: Calidad de datos
with tab5:
    st.subheader("Valores nulos")
    st.write(df.isna().sum())
    st.subheader("Gráfico de control (muertes diarias)")
    st.warning("Pendiente: control chart 3σ")

# =========================
# Exportación de datos (5.4)
# =========================
st.sidebar.header("Exportar resultados")

def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

csv = convert_df(df_filtrado)

st.sidebar.download_button(
    label="📥 Descargar CSV",
    data=csv,
    file_name="covid_export.csv",
    mime="text/csv"
)

# =========================
# Narrativa automática (5.5)
# =========================
st.subheader("Narrativa automática")
st.write(f"En los países seleccionados ({', '.join(paises)}), se registran {total_confirmados:,} casos confirmados y {total_fallecidos:,} fallecidos, con una CFR estimada en {cfr:.2f}%.")

