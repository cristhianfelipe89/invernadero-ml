# 🌿 Proyecto Invernadero Inteligente V3: Arquitectura Híbrida (SVM + Deep Learning)

Este proyecto implementa un sistema inteligente de toma de decisiones para la activación del riego en un invernadero simulado en Cali, Colombia. Evolucionando desde modelos clásicos de Machine Learning, esta versión introduce una **Arquitectura Híbrida (Ensemble)** que combina sensores locales con pronósticos climáticos de internet mediante Deep Learning.

Desarrollado para la electiva de **Profundización II - Machine Learning**, por Cristhian Felipe Herrera y Gonzalo Afanador Ochoa.

---

## 🧠 Arquitectura del Sistema (4 Capas)

El sistema opera mediante una API REST en FastAPI que evalúa las condiciones en cascada:

1. **Capa 1 (Hardware / Fallback):** Si el sensor local falla, el sistema no se detiene; transfiere el control 100% a la Red Neuronal (Deep Learning) usando datos de internet.
2. **Capa 2 (Alertas Climáticas):** Detecta temperaturas extremas y genera advertencias preventivas.
3. **Capa 3 (Modelo Local - SVM):** Evalúa el presente inmediato (humedad de la tierra y horas sin riego) con un umbral optimizado para priorizar el cuidado de la planta (Recall).
4. **Capa 4 (Modelo Externo - ANN):** Evalúa el futuro cercano consultando la API de OpenMeteo y pasando los datos por una Red Neuronal de Keras/TensorFlow para vetar o confirmar el riego según la probabilidad de lluvia y humedad externa.

---

## 📦 Estructura del Proyecto

```text
invernadero-ml/
├── .streamlit/
│   └── config.toml                   # Fuerza el modo claro y colores del Dashboard
├── datos/
│   ├── invernadero_cascada.csv       # Dataset original simulado (10,000 registros)
│   └── logs_riego.csv                # Registro histórico de decisiones (Auditoría)
├── src_modelos/                      # (Modelos generados tras entrenamiento)
│   ├── scaler_cascada.pkl            # Estandarizador de datos
│   ├── modelo_svm_cascada.pkl        # Modelo ML clásico
│   ├── modelo_dl_cascada.h5          # Modelo Deep Learning (Keras)
│   └── umbral_optimo.json            # Umbral calibrado (ej. 0.30)
├── generar_dataset.py                # Script generador de datos sintéticos
├── entrenar_modelo.py                # Entrena SVM, hace benchmarking y threshold tuning
├── entrenar_dl.py                    # Entrena la Red Neuronal Artificial (ANN)
├── api_riego.py                      # (BACKEND) API FastAPI Híbrida
├── dashboard.py                      # (FRONTEND) Interfaz en Streamlit con KPIs
├── simular_sensor.py                 # (IOT) Simulador de envío de datos y fallas
└── requerimientos.txt                # Dependencias del proyecto
```

---

## 🚀 Instrucciones de Ejecución

El sistema opera con base en microservicios, por lo que requiere levantar el backend (API) y el frontend (Dashboard) por separado. 

Elige la opción que corresponda a tu caso:

### Opción A: Ejecución desde cero (Primera vez)
Sigue estos pasos si acabas de clonar el repositorio y necesitas construir y entrenar los modelos de Inteligencia Artificial:

**1. Crear entorno e instalar dependencias:**
```bash
python -m venv venv
# En Windows: venv\Scripts\activate 
# En Mac/Linux: source venv/bin/activate
pip install -r requerimientos.txt
```

**2. Generar el dataset sintético (10.000 registros):**
```bash
python generar_dataset.py
```

**3. Entrenar el modelo local (SVM) y generar el escalador:**
```bash
python entrenar_modelo.py
```

**4. Entrenar la Red Neuronal (Deep Learning):**
```bash
python entrenar_dl.py
```

*Una vez creados los archivos `.pkl` y `.h5`, pasa a la Opción B para encender el sistema.*

---

### Opción B: Arranque del Sistema en Vivo (Showcase)
Si ya tienes los modelos entrenados y los archivos generados, necesitas abrir **3 terminales separadas** (asegúrate de que el entorno `venv` esté activo en las tres):

**Terminal 1: El Cerebro Híbrido (Backend)**
Levanta la API REST que orquesta el SVM, la Red Neuronal y la API de clima en internet.
```bash
uvicorn api_riego:app --reload
```
*(Documentación interactiva disponible en: http://127.0.0.1:8000/docs)*

**Terminal 2: El Dashboard (Frontend)**
Levanta la interfaz gráfica interactiva (Business Intelligence). Se abrirá automáticamente en tu navegador web.
```bash
streamlit run dashboard.py
```

**Terminal 3: Simulador IoT (Sensores en Tiempo Real)**
Inicia el envío continuo de datos hacia la API cada 5 segundos. Este script inyecta aleatoriamente "fallas de sensor" para demostrar cómo el modelo de Deep Learning entra al rescate (Fallback).
```bash
python simular_sensor.py
```

---

## 🛠️ Tecnologías Utilizadas
* **Machine Learning Clásico:** `scikit-learn` (SVM, Decision Trees, KNN).
* **Deep Learning:** `TensorFlow` y `Keras` (ANN).
* **Backend y API:** `FastAPI`, `Uvicorn`, `Pydantic`.
* **Frontend y Visualización:** `Streamlit`, `Matplotlib`, `Pandas`.
* **Conectividad Externa:** `requests` (OpenMeteo API).