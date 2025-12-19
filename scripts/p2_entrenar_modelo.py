import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
import pickle

# ------------ 1. Rutas del proyecto ------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------ 2. Cargar conjuntos de entrenamiento ------------

print("Cargando conjuntos de entrenamiento...\n")

X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv")

# y_train debe quedar como Serie con la columna 'helada'
if "helada" in y_train.columns:
    y_train = y_train["helada"]
else:
    y_train = y_train.iloc[:, 0]

# Asegurar que X sea numérico
X_train = X_train.astype(float)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}\n")

# ------------ 3. Calcular scale_pos_weight (desbalanceo) ------------

# pos = cantidad de días con helada (1)
# neg = cantidad de días sin helada (0)
pos = y_train.sum()
neg = len(y_train) - pos
scale = neg / pos

print("Distribución en y_train:")
print(f"  Helada = 1 : {pos}")
print(f"  Helada = 0 : {neg}")
print(f"scale_pos_weight = {scale:.2f}\n")

# ------------ 4. Definir y entrenar el modelo XGBoost ------------

model = XGBClassifier(
    n_estimators=400,      # número de árboles
    learning_rate=0.05,   # tasa de aprendizaje
    max_depth=6,          # profundidad máxima de cada árbol
    subsample=0.8,        # fracción de filas usada por árbol
    colsample_bytree=0.8, # fracción de columnas usada por árbol
    eval_metric="logloss",
    tree_method="hist",   # más rápido
    scale_pos_weight=scale  # corrige el desbalanceo de clases
)

print("Entrenando modelo XGBoost...\n")
model.fit(X_train, y_train)
print("✅ Modelo entrenado correctamente.\n")

# ------------ 5. Guardar el modelo entrenado ------------

json_path = MODELS_DIR / "modelo_heladas.json"
pkl_path = MODELS_DIR / "modelo_heladas.pkl"

model.save_model(json_path)
with open(pkl_path, "wb") as f:
    pickle.dump(model, f)

print("Modelo guardado en:")
print(f"  - {json_path}")
print(f"  - {pkl_path}\n")

# ------------ 6. Guardar importancia de variables ------------

importancia = pd.DataFrame({
    "variable": X_train.columns,
    "importancia": model.feature_importances_
}).sort_values(by="importancia", ascending=False)

imp_path = REPORTS_DIR / "importancia_variables.csv"
importancia.to_csv(imp_path, index=False)

print("Importancia de variables guardada en:")
print(f"  - {imp_path}")

print("\n✅ P2 FINAL (entrenamiento) completado.")
