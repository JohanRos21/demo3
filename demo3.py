import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_absolute_error

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


# =======================
# PARTE 3: Modelado y proyecciones
# =======================
st.header("3) Modelado temporal y proyecciones (14 días)")

# === NUEVO: construir serie temporal df_ts ===
df_ts = df[df[cols["country"]] == pais_sel].copy()
if "Date" not in df_ts.columns:
    df_ts["Date"] = pd.to_datetime(fecha)

df_ts = df_ts.groupby("Date")[[cols["confirmed"], cols["deaths"]]].sum().reset_index()
df_ts = df_ts.rename(columns={cols["confirmed"]: "Confirmed", cols["deaths"]: "Deaths"})

df_ts["NewConfirmed"] = df_ts["Confirmed"].diff().fillna(0).clip(lower=0)
df_ts["NewDeaths"]    = df_ts["Deaths"].diff().fillna(0).clip(lower=0)


# --- 3.1 Serie de tiempo con suavizado 7 días ---
st.subheader("3.1 Serie con suavizado de 7 días (nuevos casos y muertes)")

df_ts["NewConfirmed_7d"] = df_ts["NewConfirmed"].rolling(7, min_periods=1).mean()
df_ts["NewDeaths_7d"]    = df_ts["NewDeaths"].rolling(7, min_periods=1).mean()

fig_smooth = go.Figure()
fig_smooth.add_trace(go.Scatter(x=df_ts["Date"], y=df_ts["NewConfirmed_7d"], name="Nuevos Confirmados (7d)"))
fig_smooth.add_trace(go.Scatter(x=df_ts["Date"], y=df_ts["NewDeaths_7d"],    name="Nuevas Muertes (7d)"))
fig_smooth.update_layout(title=f"Suavizado (7 días) – {pais_sel}", xaxis_title="Fecha", yaxis_title="Casos diarios (media móvil)")
st.plotly_chart(fig_smooth, use_container_width=True)

st.divider()

# --- Controles para el modelo ---
st.subheader("3.2 Pronóstico con SARIMAX (14 días)")
target_var = st.selectbox("Variable a pronosticar", ["NewConfirmed", "NewDeaths"], index=1)
horizon = st.slider("Horizonte de pronóstico (días)", 7, 21, 14)
use_log = st.checkbox("Usar transformación log1p (estabiliza varianza)", value=True)
seasonal = st.checkbox("Estacionalidad semanal (7 días)", value=True)

# Parámetros
order_p = st.number_input("AR (p)", 0, 3, 1)
order_d = st.number_input("Diferencias (d)", 0, 2, 1)
order_q = st.number_input("MA (q)", 0, 3, 1)
if seasonal:
    sp, sd, sq, s = st.columns([1,1,1,1])
    with sp: P = st.number_input("SAR (P)", 0, 3, 1)
    with sd: D = st.number_input("SDiff (D)", 0, 2, 1)
    with sq: Q = st.number_input("SMA (Q)", 0, 3, 1)
    with s:  S = st.number_input("Periodo estacional (S)", 2, 14, 7)
else:
    P=D=Q=0
    S=0

# Prepara serie
ts = df_ts[["Date", target_var]].dropna().copy()
ts = ts.set_index("Date").asfreq("D")
ts[target_var] = ts[target_var].fillna(0)

y = np.log1p(ts[target_var]) if use_log else ts[target_var]

valid_len = min(28, max(14, int(len(y)*0.2)))
train = y.iloc[:-valid_len]
test  = y.iloc[-valid_len:]

def fit_sarimax(endog):
    if seasonal and S > 0:
        model = SARIMAX(endog, order=(order_p, order_d, order_q),
                        seasonal_order=(P, D, Q, S), enforce_stationarity=False, enforce_invertibility=False)
    else:
        model = SARIMAX(endog, order=(order_p, order_d, order_q),
                        enforce_stationarity=False, enforce_invertibility=False)
    return model.fit(disp=False)

# --- 3.3 Backtesting ---
st.subheader("3.3 Validación con backtesting (MAE / MAPE)")
def walk_forward_mae_mape(train_series, test_series):
    history = train_series.copy()
    preds = []
    for t in range(len(test_series)):
        res = fit_sarimax(history)
        fc = res.get_forecast(steps=1).predicted_mean.iloc[0]
        preds.append(fc)
        history = history.append(test_series.iloc[t:t+1])
    preds = pd.Series(preds, index=test_series.index)

    if use_log:
        preds_inv = np.expm1(preds.clip(max=20))
        test_inv  = np.expm1(test_series)
    else:
        preds_inv = preds.clip(min=0)
        test_inv  = test_series

    mae  = float(mean_absolute_error(test_inv, preds_inv))
    mape = float((np.abs((test_inv - preds_inv) / np.maximum(test_inv, 1e-9))).mean() * 100)
    return mae, mape, preds_inv, test_inv

mae, mape, preds_bt, test_bt = walk_forward_mae_mape(train, test)
c1, c2 = st.columns(2)
with c1: st.metric("MAE (validación)", f"{mae:,.2f}")
with c2: st.metric("MAPE (validación)", f"{mape:,.2f}%")

fig_bt = go.Figure()
fig_bt.add_trace(go.Scatter(x=test_bt.index, y=test_bt, name="Real (validación)"))
fig_bt.add_trace(go.Scatter(x=preds_bt.index, y=preds_bt, name="Pronóstico (walk-forward)"))
fig_bt.update_layout(title="Backtesting en ventana de validación", xaxis_title="Fecha", yaxis_title=target_var)
st.plotly_chart(fig_bt, use_container_width=True)

st.divider()

# --- 3.4 Pronóstico final ---
st.subheader("3.4 Pronóstico final y bandas de confianza")

res_final = fit_sarimax(y)
fc_res = res_final.get_forecast(steps=horizon)
fc_mean = fc_res.predicted_mean
fc_ci   = fc_res.conf_int(alpha=0.05)

if use_log:
    fc_mean_plot = np.expm1(fc_mean.clip(max=20))
    lower = np.expm1(fc_ci.iloc[:, 0].clip(max=20))
    upper = np.expm1(fc_ci.iloc[:, 1].clip(max=20))
    hist_line = np.expm1(y)
else:
    fc_mean_plot = fc_mean.clip(min=0)
    lower = fc_ci.iloc[:, 0].clip(lower=0)
    upper = fc_ci.iloc[:, 1].clip(lower=0)
    hist_line = y

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(x=hist_line.index, y=hist_line, name="Histórico", mode="lines"))
fig_fc.add_trace(go.Scatter(x=fc_mean_plot.index, y=fc_mean_plot, name="Forecast", mode="lines"))
fig_fc.add_trace(go.Scatter(
    x=list(fc_mean_plot.index)+list(fc_mean_plot.index[::-1]),
    y=list(upper)+list(lower[::-1]),
    fill="toself", name="IC 95%"
))
fig_fc.update_layout(
    title=f"Pronóstico {target_var} a {horizon} días – {pais_sel}",
    xaxis_title="Fecha", yaxis_title=target_var
)
st.plotly_chart(fig_fc, use_container_width=True)


