import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# ------------ 1. Rutas básicas del proyecto ------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "DATASET.csv"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# ------------ 2. Cargar dataset original ------------

print(f"Cargando dataset desde: {DATA_RAW}")
df = pd.read_csv(DATA_RAW)
print("Shape original:", df.shape)  # (filas, columnas)


# ------------ 3. Crear variable objetivo 'helada' ------------

# Regla de etiquetado histórico: helada = 1 si temp_min_2m <= 0 °C
df["helada"] = (df["temp_min_2m"] <= 0).astype(int)

print("\nDistribución de 'helada':")
print(df["helada"].value_counts(), "\n")

# ------------ 4. Seleccionar SOLO variables que usará el modelo ------------


columnas_modelo = [
    "año",
    "dia",
    "temp_max_2m",
    "temp_2m",          # temperatura media del día
    "punto_rocio_2m",
    "humedad_rel_2m",
    "viento_10m",
    "viento_min_10m",
    "viento_max_10m",
    "precipitacion",
    "latitud",
    "longitud",
    "altitud",
    "helada"
]

df_modelo = df[columnas_modelo].copy()
print("Shape df_modelo:", df_modelo.shape)
print("Columnas usadas por el modelo:")
print(df_modelo.columns.tolist(), "\n")

# Guardar dataset filtrado para referencia / análisis
df_modelo.to_csv(DATA_PROCESSED / "DATASET_FILTRADO.csv", index=False)

# ------------ 5. Separar X (features) e y (target) ------------

X = df_modelo.drop(columns=["helada"])
y = df_modelo["helada"]

# ------------ 6. División train/test con estratificación ------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,      # 80% train, 20% test
    random_state=42,
    stratify=y,          # mantiene proporción helada/no helada
    shuffle=True
)

print("Tamaños de los conjuntos:")
print("  X_train:", X_train.shape)
print("  X_test :", X_test.shape)
print("  y_train:", y_train.shape)
print("  y_test :", y_test.shape, "\n")

# ------------ 7. Guardar datasets procesados ------------

X_train.to_csv(DATA_PROCESSED / "X_train.csv", index=False)
X_test.to_csv(DATA_PROCESSED / "X_test.csv", index=False)
y_train.to_csv(DATA_PROCESSED / "y_train.csv", index=False)
y_test.to_csv(DATA_PROCESSED / "y_test.csv", index=False)

print("Archivos guardados en:", DATA_PROCESSED)
print(" - DATASET_FILTRADO.csv")
print(" - X_train.csv")
print(" - X_test.csv")
print(" - y_train.csv")
print(" - y_test.csv")
print("\n✅ P1 FINAL completado correctamente.")

