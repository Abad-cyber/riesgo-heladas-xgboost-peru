import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta
import pickle
import time

import streamlit as st
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="Pronóstico de heladas (1 día)", page_icon="❄️", layout="wide")
st.title("❄️ Pronóstico de heladas (1 día)")
st.caption("XGBoost + Open-Meteo + OpenStreetMap (Nominatim).")
st.divider()


# ============================================================
# PATHS + MODEL
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "modelo_heladas.pkl"

@st.cache_resource
def cargar_modelo():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = cargar_modelo()


# ============================================================
# SESSION STATE
# ============================================================

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_error" not in st.session_state:
    st.session_state.last_error = None


# ============================================================
# HELPERS
# ============================================================

def dia_juliano(fecha: date) -> int:
    return fecha.timetuple().tm_yday

def request_json_with_retries(url, params=None, headers=None, timeout=20, retries=3, sleep_s=1.0):
    """Requests robusto con reintentos para evitar parpadeo por fallas momentáneas."""
    last_exc = None
    for _ in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(sleep_s)
    raise last_exc

def obtener_ubicacion(lat: float, lon: float):
    """Devuelve (departamento, distrito/localidad)."""
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"format": "json", "lat": lat, "lon": lon, "zoom": 12, "addressdetails": 1}
    headers = {
        "User-Agent": "modelo-heladas-dashboard/1.0 (contacto@ejemplo.com)",
        "Accept-Language": "es",
    }
    data = request_json_with_retries(url, params=params, headers=headers, timeout=20, retries=3, sleep_s=1.0)
    address = data.get("address", {})

    dept = address.get("state") or address.get("region") or address.get("state_district")
    dist = (
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("municipality")
        or address.get("hamlet")
        or address.get("locality")
        or address.get("suburb")
    )
    return dept, dist

def obtener_meteo_rango(lat: float, lon: float, fecha_inicio: date, fecha_fin: date):
    """
    Open-Meteo: devuelve (daily, altitud) para un rango [fecha_inicio, fecha_fin].
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_mean",
            "dewpoint_2m_mean",
            "relative_humidity_2m_mean",
            "wind_speed_10m_mean",
            "wind_speed_10m_min",
            "wind_speed_10m_max",
            "precipitation_sum",
        ]),
        "start_date": fecha_inicio.isoformat(),
        "end_date": fecha_fin.isoformat(),
        "timezone": "auto",
    }
    data = request_json_with_retries(url, params=params, headers=None, timeout=20, retries=3, sleep_s=1.0)

    daily = data.get("daily", {})
    times = daily.get("time", [])
    if not times:
        raise RuntimeError("Open-Meteo no devolvió datos diarios para el rango solicitado.")

    altitud = float(data.get("elevation", 3800.0))
    return daily, altitud

def construir_X(lat, lon, altitud, fecha_str, meteo):
    """
    Construye el vector de entrada (orden fijo).
    IMPORTANTE: estos nombres deben coincidir con los usados al entrenar tu modelo.
    """
    fecha_dt = datetime.strptime(fecha_str, "%Y-%m-%d").date()

    X = pd.DataFrame([{
        "año": float(fecha_dt.year),
        "dia": float(dia_juliano(fecha_dt)),
        "temp_max_2m": float(meteo["temp_max_2m"]),
        "temp_2m": float(meteo["temp_2m"]),
        "punto_rocio_2m": float(meteo["punto_rocio_2m"]),
        "humedad_rel_2m": float(meteo["humedad_rel_2m"]),
        "viento_10m": float(meteo["viento_10m"]),
        "viento_min_10m": float(meteo["viento_min_10m"]),
        "viento_max_10m": float(meteo["viento_max_10m"]),
        "precipitacion": float(meteo["precipitacion"]),
        "latitud": float(lat),
        "longitud": float(lon),
        "altitud": float(altitud),
    }])

    cols = [
        "año", "dia",
        "temp_max_2m", "temp_2m", "punto_rocio_2m", "humedad_rel_2m",
        "viento_10m", "viento_min_10m", "viento_max_10m",
        "precipitacion",
        "latitud", "longitud", "altitud"
    ]
    return X[cols]

def nivel_riesgo(prob_pct: float) -> str:
    if prob_pct >= 70:
        return "ALTO"
    if prob_pct >= 40:
        return "MODERADO"
    return "BAJO"

def probabilidades_5_dias(model, lat: float, lon: float, fecha_objetivo: date):
    """
    Devuelve:
    - df_hist: 5 días EXACTOS (fecha_objetivo-4 ... fecha_objetivo)
      columnas: fecha (YYYY-MM-DD), prob_pct (0-100)
    - altitud: altitud de Open-Meteo
    - meteo_ultimo: meteo del día consultado (para tabla)
    """
    fecha_inicio = fecha_objetivo - timedelta(days=4)
    fecha_fin = fecha_objetivo

    daily, altitud = obtener_meteo_rango(lat, lon, fecha_inicio, fecha_fin)
    fechas = daily["time"]

    probs = []
    meteo_ultimo = None

    for i, fstr in enumerate(fechas):
        met = {
            "temp_max_2m": float(daily["temperature_2m_max"][i]),
            "temp_2m": float(daily["temperature_2m_mean"][i]),
            "punto_rocio_2m": float(daily["dewpoint_2m_mean"][i]),
            "humedad_rel_2m": float(daily["relative_humidity_2m_mean"][i]),
            "viento_10m": float(daily["wind_speed_10m_mean"][i]),
            "viento_min_10m": float(daily["wind_speed_10m_min"][i]),
            "viento_max_10m": float(daily["wind_speed_10m_max"][i]),
            "precipitacion": float(daily["precipitation_sum"][i]),
        }

        X = construir_X(lat, lon, altitud, fstr, met)
        prob_pct = float(model.predict_proba(X)[0, 1]) * 100.0
        probs.append({"fecha": fstr, "prob_pct": prob_pct})

        if i == len(fechas) - 1:
            meteo_ultimo = met

    df_hist = pd.DataFrame(probs)
    return df_hist, altitud, meteo_ultimo


# ============================================================
# SIDEBAR INPUTS
# ============================================================

with st.sidebar:
    st.header("Parámetros de consulta")

    lat = st.number_input("Latitud", value=-15.85, format="%.6f", key="lat_input")
    lon = st.number_input("Longitud", value=-70.02, format="%.6f", key="lon_input")
    fecha = st.date_input("Fecha a predecir", value=date.today(), key="fecha_input")

    with st.form("form_pronostico"):
        submitted = st.form_submit_button("Calcular pronóstico")

    if st.button("Limpiar resultado"):
        st.session_state.last_result = None
        st.session_state.last_error = None
        st.rerun()


# ============================================================
# RUN (solo cuando submit)
# ============================================================

if submitted:
    try:
        st.session_state.last_error = None

        with st.spinner("Consultando APIs y ejecutando el modelo..."):
            dep, dist = obtener_ubicacion(lat, lon)

            df_hist, altitud, meteo_dia = probabilidades_5_dias(model, lat, lon, fecha)

            prob_pct_dia = float(df_hist["prob_pct"].iloc[-1])
            pred_text = "Helada" if prob_pct_dia >= 50 else "No helada"
            riesgo = nivel_riesgo(prob_pct_dia)

            st.session_state.last_result = {
                "params": (float(lat), float(lon), str(fecha)),
                "dep": dep,
                "dist": dist,
                "altitud": altitud,
                "df_hist": df_hist,
                "prob_pct_dia": prob_pct_dia,
                "pred_text": pred_text,
                "riesgo": riesgo,
                "meteo_dia": meteo_dia,
            }

    except Exception as e:
        # No borra el último resultado, solo avisa
        st.session_state.last_error = str(e)


# ============================================================
# RENDER
# ============================================================

if st.session_state.last_error:
    st.warning(f"No se pudo actualizar por un fallo temporal (API/timeout). Detalle: {st.session_state.last_error}")

res = st.session_state.last_result

if res is None:
    st.info("Configura los parámetros y pulsa **Calcular pronóstico**.")
else:
    colA, colB = st.columns([1.25, 1])

    # ---- IZQUIERDA: MÉTRICAS + HISTOGRAMA + TABLA ----
    with colA:
        st.subheader("Resultado del día consultado")
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilidad helada", f"{res['prob_pct_dia']:.1f} %")
        c2.metric("Riesgo", res["riesgo"])
        c3.metric("Predicción", res["pred_text"])

        st.subheader("Ubicación aproximada")
        u1, u2, u3 = st.columns(3)
        u1.metric("Departamento", res["dep"] or "No disponible")
        u2.metric("Distrito/Localidad", res["dist"] or "No disponible")
        u3.metric("Altitud (m.s.n.m.)", f"{res['altitud']:.0f}")

        # ---------------- HISTOGRAMA 5 DÍAS (MEJORADO) ----------------
        st.subheader("Probabilidad de helada (últimos 5 días)")

        df_hist = res["df_hist"].copy()
        df_hist["fecha_dt"] = pd.to_datetime(df_hist["fecha"])
        df_hist = df_hist.sort_values("fecha_dt")

        # Eje X más corto: dd/mm
        df_hist["fecha_fmt"] = df_hist["fecha_dt"].dt.strftime("%d/%m")

        # Asegurar numérico
        df_hist["prob_pct"] = pd.to_numeric(df_hist["prob_pct"], errors="coerce").fillna(0.0)

        # Escala Y dinámica (evita gráfico “vacío” si todo está bajo)
        max_prob = float(df_hist["prob_pct"].max())
        ymax = max(5.0, min(100.0, max_prob * 1.35))  # mínimo 5%, máximo 100%

        fig, ax = plt.subplots(figsize=(7.5, 3.6))

        bars = ax.bar(df_hist["fecha_fmt"], df_hist["prob_pct"])
        ax.plot(df_hist["fecha_fmt"], df_hist["prob_pct"], marker="o", linewidth=1.2)

        # Etiquetas encima de cada barra
        for b in bars:
            h = float(b.get_height())
            ax.text(
                b.get_x() + b.get_width() / 2,
                h + (ymax * 0.02),
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9
            )

        ax.set_xlabel("Fecha (dd/mm)")
        ax.set_ylabel("Probabilidad (%)")
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

        # ---------------- TABLA METEO DEL DÍA ----------------
        st.subheader("Variables meteorológicas del día consultado")

        met = res["meteo_dia"] or {}
        df_vars = pd.DataFrame({
            "Variable": [
                "Temp. máxima (°C)",
                "Temp. media (°C)",
                "Punto de rocío (°C)",
                "Humedad relativa (%)",
                "Viento medio (m/s)",
                "Viento mínimo (m/s)",
                "Viento máximo (m/s)",
                "Precipitación (mm)",
            ],
            "Valor": [
                met.get("temp_max_2m", None),
                met.get("temp_2m", None),
                met.get("punto_rocio_2m", None),
                met.get("humedad_rel_2m", None),
                met.get("viento_10m", None),
                met.get("viento_min_10m", None),
                met.get("viento_max_10m", None),
                met.get("precipitacion", None),
            ],
        })
        st.dataframe(df_vars, use_container_width=True)

    # ---- DERECHA: MAPA ----
    with colB:
        st.subheader("Mapa de riesgo")

        color = "red" if res["riesgo"] == "ALTO" else ("orange" if res["riesgo"] == "MODERADO" else "green")
        m = folium.Map(location=[st.session_state.lat_input, st.session_state.lon_input], zoom_start=7)

        popup = (
            f"{res['dep'] or ''} - {res['dist'] or ''}<br>"
            f"Prob: {res['prob_pct_dia']:.1f}%<br>"
            f"Riesgo: {res['riesgo']}<br>"
            f"Pred: {res['pred_text']}"
        )

        folium.CircleMarker(
            location=[st.session_state.lat_input, st.session_state.lon_input],
            radius=9,
            color=color,
            fill=True,
            fill_opacity=0.85,
            popup=folium.Popup(popup, max_width=300),
        ).add_to(m)

        st_folium(m, height=500, width=None)
