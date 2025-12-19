"""
P6 - VALIDACIÓN CRUZADA DEL MODELO XGBOOST (SALIDA EN TXT)
----------------------------------------------------------

Este script ejecuta validación cruzada K-FOLD (k=5) y genera:

- Resultados por cada fold (accuracy, precision, recall, f1, auc)
- Promedios de cada métrica
- Un archivo TXT estéticamente presentado en /reports
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBClassifier

# ---------------------------------------------------------
# 1. Rutas
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 2. Cargar datos
# ---------------------------------------------------------

print("Cargando datos X_train y y_train...")

X_train = pd.read_csv(DATA_PROCESSED / "X_train.csv")
y_train = pd.read_csv(DATA_PROCESSED / "y_train.csv").iloc[:, 0]

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}\n")

# ---------------------------------------------------------
# 3. Modelo XGBoost
# ---------------------------------------------------------

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist"
)

# ---------------------------------------------------------
# 4. Validación cruzada
# ---------------------------------------------------------

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "auc": "roc_auc"
}

print("Ejecutando validación cruzada k=5...\n")

resultados = cross_validate(
    model,
    X_train,
    y_train,
    cv=kfold,
    scoring=scoring,
    return_train_score=False,
    n_jobs=1
)

# ---------------------------------------------------------
# 5. Crear DataFrame
# ---------------------------------------------------------

df_resultados = pd.DataFrame({
    "accuracy": resultados["test_accuracy"],
    "precision": resultados["test_precision"],
    "recall": resultados["test_recall"],
    "f1": resultados["test_f1"],
    "auc": resultados["test_auc"]
})

# Promedios
promedios = df_resultados.mean()

# ---------------------------------------------------------
# 6. Generar archivo TXT estético
# ---------------------------------------------------------

output_path = REPORTS_DIR / "validacion_cruzada.txt"

with open(output_path, "w", encoding="utf-8") as f:

    f.write("VALIDACIÓN CRUZADA DEL MODELO XGBOOST (K=5)\n")
    f.write("------------------------------------------------------\n\n")
    
    f.write("RESULTADOS POR FOLD:\n")
    f.write("------------------------------------------------------\n")
    
    for i, row in df_resultados.iterrows():
        f.write(f"Fold {i+1}:\n")
        f.write(f"  Accuracy : {row['accuracy']:.6f}\n")
        f.write(f"  Precision: {row['precision']:.6f}\n")
        f.write(f"  Recall   : {row['recall']:.6f}\n")
        f.write(f"  F1-score : {row['f1']:.6f}\n")
        f.write(f"  AUC      : {row['auc']:.6f}\n")
        f.write("\n")

    f.write("PROMEDIOS GENERALES:\n")
    f.write("------------------------------------------------------\n")
    f.write(f"Accuracy promedio : {promedios['accuracy']:.6f}\n")
    f.write(f"Precision promedio: {promedios['precision']:.6f}\n")
    f.write(f"Recall promedio   : {promedios['recall']:.6f}\n")
    f.write(f"F1-score promedio : {promedios['f1']:.6f}\n")
    f.write(f"AUC promedio      : {promedios['auc']:.6f}\n")

print(f"\nArchivo guardado en: {output_path}")
print("✔ Validación cruzada completada y guardada en TXT.")

