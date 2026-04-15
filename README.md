# 🌿 Proyecto Invernadero: Modelo ML para Riego en Cascada (SVM + Threshold Tuning)

Sistema de Machine Learning que predice cuándo activar el riego de un invernadero a partir de datos de sensores (temperatura, humedad del aire, humedad de la tierra y horas desde el último riego).

Se realiza un **benchmarking completo** de tres modelos (KNN, Árbol de Decisión, SVM) y se elige **SVM (Support Vector Machine)** como modelo de producción por su superior rendimiento en las métricas clave. Adicionalmente, se realiza un **barrido de thresholds** para encontrar el umbral óptimo según el criterio del dominio (priorizar recall: no dejar el cultivo sin riego).

---

## 📦 Estructura del proyecto

```
invernadero-ml/
├── datos/
│   └── invernadero_cascada.csv       # Dataset generado (10,000 registros)
├── generar_dataset.py                # Genera el CSV simulado
├── entrenar_modelo.py                # Entrena 3 modelos, benchmarking, threshold tuning
├── predecir_riego.py                 # Predicción puntual desde consola
├── api_riego.py                      # API REST con FastAPI
├── simular_sensor.py                 # Simulador IoT que envía lecturas a la API
├── requerimientos.txt                # Dependencias Python
├── scaler_cascada.pkl                # (generado) escalador StandardScaler
├── modelo_svm_cascada.pkl            # (generado) SVM entrenado (producción)
├── modelo_knn_cascada.pkl            # (generado) KNN entrenado (benchmarking)
├── modelo_tree_cascada.pkl           # (generado) Árbol entrenado (benchmarking)
├── umbral_optimo.json                # (generado) umbral óptimo + parámetros SVM
├── benchmark_resultados.csv          # (generado) tabla comparativa de modelos
├── analisis_umbrales.png             # (generado) gráfico de thresholds
└── benchmark_comparativo.png         # (generado) gráfico comparativo de modelos
```

---

## 🚀 Instrucciones de Instalación y Ejecución

### Paso 1 — Clonar el repositorio
```bash
git clone <URL_DEL_REPOSITORIO>
cd invernadero-ml
```

### Paso 2 — Crear y activar el entorno virtual

**Windows (PowerShell):**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Paso 3 — Instalar dependencias
```bash
python -m pip install --upgrade pip
pip install -r requerimientos.txt
```

### Paso 4 — Generar el dataset simulado
```bash
python generar_dataset.py
```
Crea `datos/invernadero_cascada.csv` con 10,000 registros sintéticos.

### Paso 5 — Entrenar los modelos y calcular el umbral óptimo
```bash
python entrenar_modelo.py
```
Este script:
1. Carga el dataset y lo divide (train 70% / test 30%, estratificado).
2. Escala con `StandardScaler` y guarda `scaler_cascada.pkl`.
3. **Benchmarking**: entrena KNN, Árbol de Decisión y SVM con `GridSearchCV` y validación cruzada estratificada.
4. Imprime tabla comparativa con Accuracy, Precision, Recall, F1 y ROC-AUC.
5. **Threshold tuning** sobre SVM: barrido de umbrales [0.3, 0.4, 0.5, 0.6, 0.7].
6. Elige el umbral óptimo con el criterio: *recall ≥ 0.95, max precision*.
7. Guarda todos los artefactos (.pkl, .json, .csv, .png).

### Paso 6 — Probar una predicción puntual (opcional)
```bash
python predecir_riego.py
```

### Paso 7 — Levantar la API REST
```bash
uvicorn api_riego:app --reload
```
- `GET /` — estado del servidor
- `POST /predecir` — recibe lectura y devuelve decisión
- Swagger: `http://127.0.0.1:8000/docs`

### Paso 8 — Simulador de sensores
En **otra terminal** (con venv activo y API corriendo):
```bash
python simular_sensor.py
python simular_sensor.py --intervalo 10   # cada 10 segundos
```

---

## 🔁 Orden de ejecución resumido

```bash
# Una sola vez
python -m venv venv
venv\Scripts\activate
pip install -r requerimientos.txt

# Cada vez que quieras regenerar todo
python generar_dataset.py            # 1) crea CSV
python entrenar_modelo.py            # 2) entrena modelos + umbral
python predecir_riego.py             # 3) (opcional) prueba puntual

# Sistema en vivo (2 terminales)
uvicorn api_riego:app --reload       # Terminal A
python simular_sensor.py             # Terminal B
```

---

## 🧠 Sobre el modelo elegido: SVM

Tras el benchmarking de los tres modelos, se seleccionó **SVM (Support Vector Machine)** porque:
- Obtuvo el mejor **accuracy** y **F1 Score** en el conjunto de test.
- Demostró resultados **consistentes** en la validación cruzada (bajo `f1_cv_std`).
- Su capacidad para encontrar el hiperplano óptimo de separación lo hace potente con datos complejos.

## 🎯 Sobre el threshold tuning (Sesión 7)

El umbral por defecto (0.5) **rara vez es el óptimo**. Se prueban varios umbrales y se elige el que cumple:
- **Recall ≥ 0.95** sobre la clase "Iniciar Riego"
- y, dentro de los candidatos, el de **mayor Precision**.

**Justificación de dominio:** un Falso Negativo (no regar cuando hace falta) puede dañar el cultivo, mientras que un Falso Positivo (regar de más) solo desperdicia agua. Por eso es preferible un umbral que se incline a regar.

El umbral elegido se persiste en `umbral_optimo.json` y la API lo carga automáticamente al iniciar.

