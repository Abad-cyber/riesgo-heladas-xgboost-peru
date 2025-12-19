"""
P7 - COMPARACIÓN RÁPIDA DE MODELOS (Salida TXT)
-----------------------------------------------

Comparación entre:
- Regresión Logística
- Random Forest (optimizado y ligero)
- XGBoost

Uso de un SUBSAMPLE para acelerar el proceso sin perder validez científica.
Resultados guardados en /reports/comparacion_modelos.txt
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

print("Cargando X_train y y_train...")

X_train_full = pd.read_csv(DATA_PROCESSED / "X_train.csv")
y_train_full = pd.read_csv(DATA_PROCESSED / "y_train.csv").iloc[:, 0]

print(f"Dataset completo: {X_train_full.shape}")

# ---------------------------------------------------------
# 3. Usar SUBSAMPLE para acelerar comparación
# ---------------------------------------------------------

N = 200_000  # Puedes cambiar a 150k si lo deseas aún más rápido

print(f"\nTomando SUBSAMPLE de {N} filas para comparación...\n")
X_train = X_train_full.sample(N, random_state=42)
y_train = y_train_full.loc[X_train.index]

print(f"Subsample final: {X_train.shape}")

# ---------------------------------------------------------
# 4. Definir modelos (versiones rápidas)
# ---------------------------------------------------------

modelos = {
    "LogisticRegression": LogisticRegression(
        max_iter=800,
        solver="saga",
        n_jobs=1
    ),

    "RandomForest": RandomForestClassifier(
        n_estimators=80,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=1
    )
}

# ---------------------------------------------------------
# 5. Configurar validación cruzada rápida
# ---------------------------------------------------------

kfold = KFold(n_splits=3, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "auc": "roc_auc"
}

# ---------------------------------------------------------
# 6. Ejecutar comparación
# ---------------------------------------------------------

resultados_finales = []

for nombre, modelo in modelos.items():

    print(f"Evaluando modelo: {nombre}...\n")

    cv_result = cross_validate(
        modelo,
        X_train,
        y_train,
        cv=kfold,
        scoring=scoring,
        n_jobs=1
    )

    resultados_finales.append({
        "modelo": nombre,
        "accuracy": cv_result["test_accuracy"].mean(),
        "precision": cv_result["test_precision"].mean(),
        "recall": cv_result["test_recall"].mean(),
        "f1": cv_result["test_f1"].mean(),
        "auc": cv_result["test_auc"].mean()
    })

# Convertir a DataFrame
df_comp = pd.DataFrame(resultados_finales)

print("\n===== RESULTADOS DE COMPARACIÓN =====\n")
print(df_comp)

# ---------------------------------------------------------
# 7. Guardar TXT estético
# ---------------------------------------------------------

output_path = REPORTS_DIR / "comparacion_modelos.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.write("COMPARACIÓN DE MODELOS (Logistic, RandomForest, XGBoost)\n")
    f.write("------------------------------------------------------------\n\n")

    for _, row in df_comp.iterrows():
        f.write(f"Modelo: {row['modelo']}\n")
        f.write(f"  Accuracy : {row['accuracy']:.6f}\n")
        f.write(f"  Precision: {row['precision']:.6f}\n")
        f.write(f"  Recall   : {row['recall']:.6f}\n")
        f.write(f"  F1-score : {row['f1']:.6f}\n")
        f.write(f"  AUC      : {row['auc']:.6f}\n")
        f.write("\n")

print(f"\nArchivo TXT guardado en: {output_path}")
print("✔ Comparación rápida completada.")

