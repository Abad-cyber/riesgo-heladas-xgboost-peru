"""
P3 - EVALUACIÓN DEL MODELO XGBOOST (VERSIÓN FINAL)
--------------------------------------------------

- Carga X_test y y_test desde data/processed.
- Carga el modelo entrenado desde /models.
- Calcula:
    * Accuracy
    * Matriz de confusión
    * Reporte de clasificación (precision, recall, f1)
    * AUC-ROC
- Guarda:
    * reporte_clasificacion.txt
    * matriz_confusion.csv
    * curva_roc.png
    * matriz_confusion.png
  en la carpeta /reports.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt

# ------------ 1. Rutas ------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------ 2. Cargar X_test, y_test y modelo ------------

print("Cargando datos de prueba...")

X_test = pd.read_csv(DATA_PROCESSED / "X_test.csv")
y_test = pd.read_csv(DATA_PROCESSED / "y_test.csv")

if "helada" in y_test.columns:
    y_test = y_test["helada"]
else:
    y_test = y_test.iloc[:, 0]

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}\n")

print("Cargando modelo...\n")
with open(MODELS_DIR / "modelo_heladas.pkl", "rb") as f:
    model = pickle.load(f)

# ------------ 3. Predicciones ------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # probabilidad de helada=1

# ------------ 4. Métricas ------------

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
reporte = classification_report(y_test, y_pred, digits=4)

try:
    auc = roc_auc_score(y_test, y_prob)
except ValueError:
    auc = np.nan

print("===== MÉTRICAS DEL MODELO =====")
print(f"Accuracy : {acc:.4f}")
print(f"AUC-ROC  : {auc:.4f}")
print("\nMatriz de confusión:")
print(cm)
print("\nReporte de clasificación:")
print(reporte)

# ------------ 5. Guardar matriz de confusión (CSV) ------------

cm_df = pd.DataFrame(
    cm,
    index=["Real_0_NoHelada", "Real_1_Helada"],
    columns=["Pred_0_NoHelada", "Pred_1_Helada"],
)
cm_path = REPORTS_DIR / "matriz_confusion.csv"
cm_df.to_csv(cm_path, index=True)

# ------------ 6. Guardar reporte de clasificación (TXT) ------------

reporte_path = REPORTS_DIR / "reporte_clasificacion.txt"
with open(reporte_path, "w") as f:
    f.write("MÉTRICAS DE EVALUACIÓN DEL MODELO DE HELADAS\n\n")
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"AUC-ROC: {auc:.4f}\n\n")
    f.write("Matriz de confusión:\n")
    f.write(str(cm_df))
    f.write("\n\nReporte de clasificación:\n")
    f.write(reporte)

# ------------ 7. Curva ROC (PNG) ------------

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC - Modelo de heladas")
plt.legend(loc="lower right")
roc_path = REPORTS_DIR / "curva_roc.png"
plt.savefig(roc_path, dpi=300)
plt.close()

# ------------ 8. Imagen de matriz de confusión (PNG) ------------

plt.figure()
plt.imshow(cm, interpolation="nearest")
plt.title("Matriz de confusión")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["No helada (0)", "Helada (1)"], rotation=45)
plt.yticks(tick_marks, ["No helada (0)", "Helada (1)"])

thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

plt.tight_layout()
plt.ylabel("Etiqueta real")
plt.xlabel("Etiqueta predicha")
cm_img_path = REPORTS_DIR / "matriz_confusion.png"
plt.savefig(cm_img_path, dpi=300)
plt.close()

print("\nArchivos guardados en /reports:")
print(f" - {cm_path.name}")
print(f" - {reporte_path.name}")
print(f" - {roc_path.name}")
print(f" - {cm_img_path.name}")
print("\n✅ P3 (evaluación) FINAL completado.")

