import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

print("Cargando el archivo CSV...")

# 1. Leer el dataset
ruta_csv = "datos/invernadero_cascada.csv"
df = pd.read_csv(ruta_csv)

# 2. Separar variables (X) y el objetivo (y)
X = df.drop("estado_riego", axis=1)
y = df["estado_riego"]

# 3. Dividir datos (70% entrenamiento, 30% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 4. Escalar los datos y GUARDAR el scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler_cascada.pkl')

# ==============================
# ENTRENAMIENTO BÁSICO
# ==============================
print("Entrenando KNN base...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

print("Entrenando Árbol de Decisión base...")
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

# ==============================
# MODELO PRINCIPAL: SVM OPTIMIZADO
# ==============================
print("\nOptimizando SVM (GridSearchCV)... Esto tomará unos segundos.")
param_grid = {
    'C': [1, 10, 50],
    'gamma': ['scale', 0.1, 0.01],
    'kernel': ['rbf']
}

svm_base = SVC(random_state=42)
grid_search = GridSearchCV(
    estimator=svm_base, param_grid=param_grid, scoring='f1', cv=3, n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Extraer el mejor modelo y GUARDARLO
mejor_svm = grid_search.best_estimator_
y_pred_svm = mejor_svm.predict(X_test_scaled)

print(f"\n¡SVM Optimizado! Mejores parámetros: {grid_search.best_params_}")
joblib.dump(mejor_svm, 'modelo_svm_cascada.pkl')
print("Modelo guardado exitosamente como 'modelo_svm_cascada.pkl'")

# ==============================
# REPORTES DE RESULTADOS
# ==============================
print("\n===== RESULTADO KNN =====")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))

print("\n===== RESULTADO ARBOL DE DECISION =====")
print("Accuracy:", accuracy_score(y_test, y_pred_tree))

print("\n===== RESULTADO SVM OPTIMIZADO =====")
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# ==============================
# VISUALIZACIÓN
# ==============================
print("\nMostrando Matriz de Confusión... (Cierra la ventana gráfica para finalizar el script)")

cm = confusion_matrix(y_test, y_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Regar (0)", "Iniciar Ciclo (1)"])

fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Matriz de Confusión - Sistema en Cascada (SVM)")
plt.show()
