import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import pickle
import sys

# ---------------------------------------------------
# 1. Rutas y carga del modelo
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "modelo_heladas.pkl"

print(f"Cargando modelo desde: {MODEL_PATH}")
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("No se encontró el archivo del modelo 'modelo_heladas.pkl'.")
    sys.exit(1)

print("Modelo cargado correctamente.\n")

# Columnas que espera el modelo (mismo orden que X_train)
FEATURE_COLUMNS = [
    "año",
    "dia",
    "temp_max_2m",
    "temp_2m",
    "punto_rocio_2m",
    "humedad_rel_2m",
    "viento_10m",
    "viento_min_10m",
    "viento_max_10m",
    "precipitacion",
    "latitud",
    "longitud",
    "altitud",
]

# ---------------------------------------------------
# 2. Funciones auxiliares
# ---------------------------------------------------

def leer_float(mensaje: str) -> float:
    texto = input(mensaje)
    texto = (
        texto.strip()
        .replace("´", "")
        .replace("’", "")
        .replace(",", ".")
    )
    return float(texto)

def obtener_ubicacion(lat: float, lon: float):

    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "format": "json",
        "lat": lat,
        "lon": lon,
        "zoom": 12,
        "addressdetails": 1,
    }
    # IMPORTANTE: Nominatim exige un User-Agent identificable
    headers = {
        "User-Agent": "modelo-heladas-peru/1.0 (tu_correo@ejemplo.com)"
        # Reemplaza con tu correo si lo deseas
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException:
        return None, None, None

    address = data.get("address", {})

    # Mapeo aproximado para Perú
    departamento = address.get("state")            # ej. "Puno"
    provincia = (address.get("county") or
                 address.get("province"))         # ej. "Puno"
    distrito = (address.get("city") or
                address.get("town") or
                address.get("village") or
                address.get("municipality") or
                address.get("suburb"))

    return departamento, provincia, distrito

# ---------------------------------------------------
# 3. Entradas del usuario
# ---------------------------------------------------

print("Introduce los datos para obtener el pronóstico y la predicción de helada:\n")

latitud = leer_float("Latitud (grados decimales, ej. -15.84): ")
longitud = leer_float("Longitud (grados decimales, ej. -70.02): ")

fecha_str = input("Fecha a predecir (formato YYYY-MM-DD, ej. 2025-08-15): ")

# Validar fecha y extraer año + día juliano
try:
    fecha_dt = datetime.strptime(fecha_str, "%Y-%m-%d")
except ValueError:
    print("❌ Formato de fecha inválido. Usa YYYY-MM-DD.")
    sys.exit(1)

anio = fecha_dt.year
dia_juliano = fecha_dt.timetuple().tm_yday

print(f"\nFecha ingresada: {fecha_str} (año={anio}, día_juliano={dia_juliano})")

# ---------------------------------------------------
# 4. Obtener ubicación (departamento, provincia, distrito)
# ---------------------------------------------------

print("\nConsultando ubicación aproximada (Nominatim / OpenStreetMap)...")
dep, prov, dist = obtener_ubicacion(latitud, longitud)

if dep or prov or dist:
    print("Ubicación aproximada detectada:")
    if dep:
        print(f"  Departamento: {dep}")
    if prov:
        print(f"  Provincia   : {prov}")
    if dist:
        print(f"  Distrito/Localidad: {dist}")
else:
    print("⚠ No se pudo obtener la ubicación detallada a partir de las coordenadas.")

print()

# ---------------------------------------------------
# 5. Llamar a la API de Open-Meteo (meteo + altitud)
# ---------------------------------------------------

base_url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": latitud,
    "longitude": longitud,
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
    "timezone": "auto",
    "start_date": fecha_str,
    "end_date": fecha_str,
}

print("Llamando a la API de Open-Meteo...\n")
try:
    resp = requests.get(base_url, params=params, timeout=20)
    resp.raise_for_status()
except requests.RequestException as e:
    print(f"Error al llamar a la API: {e}")
    sys.exit(1)

data = resp.json()

# ALTITUD (elevation) en metros
altitud = data.get("elevation", None)
if altitud is None:
    print("⚠ La API no devolvió 'elevation'. Se usará altitud = 3800 m por defecto.")
    altitud = 3800.0

# ---------------------------------------------------
# 6. Extraer variables diarias
# ---------------------------------------------------

try:
    daily = data["daily"]
    temp_max_2m = daily["temperature_2m_max"][0]
    temp_2m = daily["temperature_2m_mean"][0]
    punto_rocio_2m = daily["dewpoint_2m_mean"][0]
    humedad_rel_2m = daily["relative_humidity_2m_mean"][0]
    viento_10m = daily["wind_speed_10m_mean"][0]
    viento_min_10m = daily["wind_speed_10m_min"][0]
    viento_max_10m = daily["wind_speed_10m_max"][0]
    precipitacion = daily["precipitation_sum"][0]
except KeyError as e:
    print(f"La respuesta de la API no contiene la clave esperada: {e}")
    sys.exit(1)
except IndexError:
    print("La API no devolvió datos diarios para la fecha indicada.")
    sys.exit(1)

print("Variables diarias obtenidas de la API:")
print(f"  temp_max_2m        = {temp_max_2m} °C")
print(f"  temp_2m (media)    = {temp_2m} °C")
print(f"  punto_rocio_2m     = {punto_rocio_2m} °C")
print(f"  humedad_rel_2m     = {humedad_rel_2m} %")
print(f"  viento_10m (media) = {viento_10m} m/s")
print(f"  viento_min_10m     = {viento_min_10m} m/s")
print(f"  viento_max_10m     = {viento_max_10m} m/s")
print(f"  precipitacion      = {precipitacion} mm")
print(f"  altitud (API)      = {altitud} m\n")

# ---------------------------------------------------
# 7. Construir vector de entrada al modelo
# ---------------------------------------------------

fila = {
    "año": anio,
    "dia": dia_juliano,
    "temp_max_2m": temp_max_2m,
    "temp_2m": temp_2m,
    "punto_rocio_2m": punto_rocio_2m,
    "humedad_rel_2m": humedad_rel_2m,
    "viento_10m": viento_10m,
    "viento_min_10m": viento_min_10m,
    "viento_max_10m": viento_max_10m,
    "precipitacion": precipitacion,
    "latitud": latitud,
    "longitud": longitud,
    "altitud": altitud,
}

X_nuevo = pd.DataFrame([fila], columns=FEATURE_COLUMNS)

print("Vector de entrada al modelo:")
print(X_nuevo, "\n")

# ---------------------------------------------------
# 8. Predicción con el modelo
# ---------------------------------------------------

pred_clase = model.predict(X_nuevo)[0]           # 0 = no helada, 1 = helada
pred_proba = model.predict_proba(X_nuevo)[0, 1]  # probabilidad de helada

print("RESULTADO DE LA PREDICCIÓN")
print("---------------------------")
if dep or prov or dist:
    print("Ubicación aproximada de la predicción:")
    if dep:
        print(f"  Departamento: {dep}")
    if prov:
        print(f"  Provincia   : {prov}")
    if dist:
        print(f"  Distrito/Localidad: {dist}")
    print()

print(f"Probabilidad de helada (clase 1): {pred_proba*100:.2f} %")

if pred_clase == 1:
    print("✅ El modelo PREDICE: SÍ HABRÁ HELADA en la fecha indicada.")
else:
    print("✅ El modelo PREDICE: NO habrá helada en la fecha indicada.")



