import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score, roc_auc_score
)

# ============================================================
# CONFIGURACIÓN CENTRALIZADA
# ============================================================
SEMILLA = 42
TEST_SIZE = 0.30
N_FOLDS = 5
RECALL_MINIMO = 0.95
THRESHOLDS_BARRIDO = [0.30, 0.40, 0.50, 0.60, 0.70]

RUTA_CSV = "datos/invernadero_cascada.csv"
TARGET = "estado_riego"

# ============================================================
# 1. CARGAR Y PARTICIONAR EL DATASET
# ============================================================
print("=" * 70)
print("  SISTEMA DE ENTRENAMIENTO - INVERNADERO ML")
print("=" * 70)

print("\n[1/8] Cargando el archivo CSV...")
df = pd.read_csv(RUTA_CSV)
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# Partición estratificada: mantiene la proporción de clases en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEMILLA, stratify=y
)
print(f"      Train: {len(X_train)} registros | Test: {len(X_test)} registros")
print(f"      Distribución train: {dict(y_train.value_counts())}")
print(f"      Distribución test:  {dict(y_test.value_counts())}")

# ============================================================
# 2. ESCALADO DE FEATURES (Feature Engineering - Sesión 6)
# ============================================================
# KNN y SVM se basan en distancias/kernels -> REQUIEREN escalado.
# El Árbol de Decisión NO lo requiere (invariante a la escala),
# pero lo entrenamos con datos sin escalar para respetar esto.
print("\n[2/8] Escalando features (StandardScaler)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler_cascada.pkl')
print("      Scaler guardado: scaler_cascada.pkl")

# ============================================================
# 3. DEFINICIÓN DE MODELOS Y GRIDS (Sesión 5 - Benchmarking)
# ============================================================
# Cada modelo tiene su grid de hiperparámetros para buscar la
# mejor combinación con GridSearchCV + validación cruzada.
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)

modelos_config = {
    "KNN": {
        "estimador": KNeighborsClassifier(),
        "grid": {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights':     ['uniform', 'distance'],
            'metric':      ['euclidean', 'manhattan'],
        },
        "requiere_escala": True,
    },
    "DecisionTree": {
        "estimador": DecisionTreeClassifier(random_state=SEMILLA),
        "grid": {
            'max_depth':        [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'criterion':        ['gini', 'entropy'],
        },
        "requiere_escala": False,
    },
    "SVM": {
        "estimador": SVC(random_state=SEMILLA, probability=True),
        "grid": {
            'C':      [1, 10, 50],
            'gamma':  ['scale', 0.1, 0.01],
            'kernel': ['rbf'],
        },
        "requiere_escala": True,
    },
}

# ============================================================
# 4. BENCHMARKING - ENTRENAR Y EVALUAR LOS 3 MODELOS
# ============================================================
print("\n[3/8] Ejecutando benchmarking de modelos...")
print("-" * 70)

resultados_benchmark = []
modelos_entrenados = {}

for nombre, config in modelos_config.items():
    # Elegir datos escalados o sin escalar según el modelo
    Xtr = X_train_scaled if config["requiere_escala"] else X_train
    Xte = X_test_scaled if config["requiere_escala"] else X_test

    print(f"\n  Entrenando {nombre}...")
    t0 = time.perf_counter()

    # GridSearchCV con F1 como métrica (evita depender solo de accuracy)
    grid = GridSearchCV(
        estimator=config["estimador"],
        param_grid=config["grid"],
        scoring='f1',
        cv=cv,
        n_jobs=-1
    )
    grid.fit(Xtr, y_train)
    tiempo_train = time.perf_counter() - t0

    modelo = grid.best_estimator_
    y_pred = modelo.predict(Xte)
    y_proba = modelo.predict_proba(Xte)[:, 1]

    # Estabilidad: F1 en validación cruzada (Sesión 4 - Modelo Fiable)
    f1_cv = cross_val_score(modelo, Xtr, y_train, cv=cv,
                            scoring='f1', n_jobs=-1)

    fila = {
        "modelo":              nombre,
        "mejores_params":      str(grid.best_params_),
        "accuracy":            round(accuracy_score(y_test, y_pred), 4),
        "precision":           round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":              round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":                  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":             round(roc_auc_score(y_test, y_proba), 4),
        "f1_cv_mean":          round(float(np.mean(f1_cv)), 4),
        "f1_cv_std":           round(float(np.std(f1_cv)), 4),
        "tiempo_entrenamiento": round(tiempo_train, 2),
    }
    resultados_benchmark.append(fila)
    modelos_entrenados[nombre] = modelo

    print(f"    ✓ {nombre}: Acc={fila['accuracy']} | Prec={fila['precision']} | "
          f"Rec={fila['recall']} | F1={fila['f1']} | AUC={fila['roc_auc']}")
    print(f"      CV F1: {fila['f1_cv_mean']} ± {fila['f1_cv_std']}")
    print(f"      Mejores parámetros: {grid.best_params_}")
    print(f"      Tiempo: {fila['tiempo_entrenamiento']}s")

# ============================================================
# 5. TABLA COMPARATIVA DEL BENCHMARKING
# ============================================================
df_benchmark = pd.DataFrame(resultados_benchmark)
print("\n" + "=" * 70)
print("  TABLA COMPARATIVA - BENCHMARKING DE MODELOS")
print("=" * 70)
cols_mostrar = ['modelo', 'accuracy', 'precision', 'recall', 'f1',
                'roc_auc', 'f1_cv_mean', 'f1_cv_std']
print(df_benchmark[cols_mostrar].to_string(index=False))

# Guardar tabla como CSV
df_benchmark.to_csv('benchmark_resultados.csv', index=False)
print("\nTabla guardada: benchmark_resultados.csv")

# ============================================================
# 6. GUARDAR LOS 3 MODELOS ENTRENADOS
# ============================================================
print("\n[4/8] Guardando modelos entrenados...")
joblib.dump(modelos_entrenados["KNN"],          'modelo_knn_cascada.pkl')
joblib.dump(modelos_entrenados["DecisionTree"],  'modelo_tree_cascada.pkl')
joblib.dump(modelos_entrenados["SVM"],           'modelo_svm_cascada.pkl')
print("      ✓ modelo_knn_cascada.pkl")
print("      ✓ modelo_tree_cascada.pkl")
print("      ✓ modelo_svm_cascada.pkl")

# ============================================================
# 7. CLASSIFICATION REPORT DE CADA MODELO
# ============================================================
print("\n[5/8] Reportes de clasificación detallados...")
for nombre, config in modelos_config.items():
    Xte = X_test_scaled if config["requiere_escala"] else X_test
    modelo = modelos_entrenados[nombre]
    y_pred = modelo.predict(Xte)
    print(f"\n----- {nombre} (umbral por defecto 0.5) -----")
    print(classification_report(y_test, y_pred,
          target_names=["No Regar (0)", "Iniciar Ciclo (1)"]))

# ==============================================================
# 8. THRESHOLD TUNING SOBRE SVM (Sesión 7)
# ==============================================================
# El umbral por defecto (0.5) RARA VEZ es el óptimo.
# En este proyecto la asimetría de costos del dominio justifica
# priorizar recall sobre precision:
#   - Falso Negativo (FN) = NO regar cuando se necesitaba -> planta dañada
#   - Falso Positivo (FP) = regar cuando NO hacía falta  -> agua desperdiciada
# El FN es mucho más costoso.
print("\n" + "=" * 70)
print("  BARRIDO DE THRESHOLDS - SVM (Sesión 7)")
print("=" * 70)

svm = modelos_entrenados["SVM"]
y_probs_svm = svm.predict_proba(X_test_scaled)[:, 1]

print(f"\n{'Threshold':<12}{'Precision':>12}{'Recall':>12}{'F1':>10}"
      f"{'TP':>6}{'FP':>6}{'FN':>6}{'TN':>6}")
print("-" * 70)

filas_threshold = []
for t in THRESHOLDS_BARRIDO:
    y_pred_t = (y_probs_svm >= t).astype(int)
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    rec = recall_score(y_test, y_pred_t, zero_division=0)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t, labels=[0, 1]).ravel()

    print(f"{t:<12.2f}{prec:>12.4f}{rec:>12.4f}{f1:>10.4f}"
          f"{tp:>6}{fp:>6}{fn:>6}{tn:>6}")

    filas_threshold.append({
        'threshold': t, 'precision': prec, 'recall': rec,
        'f1': f1, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
    })

df_umbrales = pd.DataFrame(filas_threshold)

# ----- ELECCIÓN DEL MEJOR UMBRAL -----
# Estrategia: recall >= 0.95, y entre los candidatos, el de mayor precision.
candidatos = df_umbrales[df_umbrales['recall'] >= RECALL_MINIMO]
if not candidatos.empty:
    mejor = candidatos.loc[candidatos['precision'].idxmax()]
    criterio = f"recall>={RECALL_MINIMO}, max precision"
else:
    mejor = df_umbrales.loc[df_umbrales['recall'].idxmax()]
    criterio = "fallback: max recall (ningún umbral cumplió el mínimo)"

UMBRAL_ELEGIDO = float(mejor['threshold'])

print(f"\n----- JUSTIFICACIÓN DE LA DECISIÓN -----")
print(f"Umbral elegido: {UMBRAL_ELEGIDO:.2f}")
print(f"  - Precision : {mejor['precision']:.4f}")
print(f"  - Recall    : {mejor['recall']:.4f}")
print(f"  - F1        : {mejor['f1']:.4f}")
print(f"Criterio: {criterio}")
print(f"Razón de dominio: en un sistema de riego automático, un Falso Negativo")
print(f"(no regar cuando se necesita) puede dañar el cultivo, mientras que un")
print(f"Falso Positivo (regar de más) solo desperdicia agua.")

# Reporte con umbral ajustado
y_pred_svm_tuned = (y_probs_svm >= UMBRAL_ELEGIDO).astype(int)
print(f"\n===== SVM con umbral ajustado ({UMBRAL_ELEGIDO:.2f}) =====")
print("Accuracy:", accuracy_score(y_test, y_pred_svm_tuned))
print(classification_report(y_test, y_pred_svm_tuned,
      target_names=["No Regar (0)", "Iniciar Ciclo (1)"]))

# Guardar umbral + metadatos
svm_params = modelos_config["SVM"]["estimador"].get_params()
# Extraer los mejores params reales del grid
svm_best_params = resultados_benchmark[2]["mejores_params"]

with open('umbral_optimo.json', 'w') as f:
    json.dump({
        'umbral': UMBRAL_ELEGIDO,
        'criterio': criterio,
        'modelo': 'SVM',
        'mejores_params_svm': svm_best_params,
        'precision_en_test': round(float(mejor['precision']), 4),
        'recall_en_test': round(float(mejor['recall']), 4),
        'f1_en_test': round(float(mejor['f1']), 4),
    }, f, indent=2)
print("Umbral guardado: umbral_optimo.json")

# ==============================================================
# 9. GRÁFICOS (Sesión 6 - Visualización)
# ==============================================================
print("\n[7/8] Generando gráficos...")

# --- Gráfico A: Análisis de Umbrales SVM (2x2) ---
fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# (a) Matriz de confusión con el umbral elegido
cm = confusion_matrix(y_test, y_pred_svm_tuned)
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Regar (0)", "Iniciar Ciclo (1)"]
).plot(cmap=plt.cm.Blues, ax=axes[0, 0], colorbar=False)
axes[0, 0].set_title(f"Matriz de Confusión SVM (umbral={UMBRAL_ELEGIDO:.2f})")

# (b) Métricas vs umbral
axes[0, 1].plot(df_umbrales['threshold'],
                df_umbrales['precision'], marker='o', label='Precision')
axes[0, 1].plot(df_umbrales['threshold'], df_umbrales['recall'],
                marker='o', label='Recall')
axes[0, 1].plot(df_umbrales['threshold'], df_umbrales['f1'],
                marker='o', label='F1')
axes[0, 1].axvline(UMBRAL_ELEGIDO, color='red', linestyle='--',
                   label=f'Elegido={UMBRAL_ELEGIDO:.2f}')
axes[0, 1].axvline(0.5, color='gray', linestyle=':', label='Defecto=0.5')
axes[0, 1].set_xlabel('Umbral')
axes[0, 1].set_ylabel('Métrica')
axes[0, 1].set_title('Trade-off Precision/Recall vs Umbral - SVM')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# (c) Curva Precision-Recall
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_probs_svm)
axes[1, 0].plot(rec_curve, prec_curve)
axes[1, 0].set_xlabel('Recall')
axes[1, 0].set_ylabel('Precision')
axes[1, 0].set_title('Curva Precision-Recall - SVM')
axes[1, 0].grid(alpha=0.3)

# (d) Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_probs_svm)
roc_auc_val = auc(fpr, tpr)
axes[1, 1].plot(fpr, tpr, label=f'AUC = {roc_auc_val:.3f}')
axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
axes[1, 1].set_xlabel('FPR')
axes[1, 1].set_ylabel('TPR')
axes[1, 1].set_title('Curva ROC - SVM')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('analisis_umbrales.png', dpi=120, bbox_inches='tight')
print("      ✓ analisis_umbrales.png")

# --- Gráfico B: Benchmarking Comparativo ---
fig2, ax = plt.subplots(figsize=(10, 6))
metricas_graf = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
x = range(len(df_benchmark))
ancho = 0.15
colores = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
for i, m in enumerate(metricas_graf):
    ax.bar([xi + i * ancho for xi in x], df_benchmark[m], ancho,
           label=m, color=colores[i])
ax.set_xticks([xi + ancho * 2 for xi in x])
ax.set_xticklabels(df_benchmark['modelo'], fontsize=12)
ax.set_ylabel('Valor', fontsize=12)
ax.set_title('Benchmarking de Modelos - Métricas en Test', fontsize=14)
ax.legend(loc='lower right')
ax.grid(alpha=0.3, axis='y')
ax.set_ylim(0, 1.08)
plt.tight_layout()
plt.savefig('benchmark_comparativo.png', dpi=120, bbox_inches='tight')
print("      ✓ benchmark_comparativo.png")

plt.show()

# ==============================================================
# RESUMEN FINAL
# ==============================================================
print("\n" + "=" * 70)
print("  ✅ ENTRENAMIENTO FINALIZADO")
print("=" * 70)
print(f"  Modelo de producción: SVM")
print(f"  Umbral óptimo:        {UMBRAL_ELEGIDO:.2f}")
print(f"  Criterio:             {criterio}")
print(f"\n  Archivos generados:")
print(f"    - scaler_cascada.pkl")
print(f"    - modelo_knn_cascada.pkl")
print(f"    - modelo_tree_cascada.pkl")
print(f"    - modelo_svm_cascada.pkl")
print(f"    - umbral_optimo.json")
print(f"    - benchmark_resultados.csv")
print(f"    - analisis_umbrales.png")
print(f"    - benchmark_comparativo.png")
print("=" * 70)
