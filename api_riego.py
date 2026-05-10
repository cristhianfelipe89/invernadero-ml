"""
API REST del invernadero inteligente evolucionada (FastAPI).

El sistema opera en 4 CAPAS EN CASCADA:
  1) Validación de hardware (Sensor dañado -> Fallback a Deep Learning).
  2) Alertas climáticas (Temperatura extrema -> Avisar).
  3) Modelo SVM Local + Umbral Óptimo (Criterio de raíz).
  4) Modelo Deep Learning + Internet (Criterio de entorno y futuro).

Ejecutar:
    uvicorn api_riego:app --reload
"""

import json
import os
import csv
import joblib
import pandas as pd
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model

# ============================================================
# INICIALIZACIÓN Y CARGA DE MODELOS [cite: 463, 464]
# ============================================================
app = FastAPI(
    title="API Invernadero - Sistema Híbrido SVM & Deep Learning",
    description="Predicción de riego combinando sensores locales y datos de internet",
    version="3.0"
)

print("Cargando artefactos de inteligencia...")
try:
    # Carga de modelos clásicos (SVM) [cite: 464]
    scaler = joblib.load('scaler_cascada.pkl')
    modelo_svm = joblib.load('modelo_svm_cascada.pkl')

    # Carga del umbral óptimo (0.30) [cite: 464, 541]
    with open('umbral_optimo.json', 'r') as f:
        config_umbral = json.load(f)
    UMBRAL = float(config_umbral['umbral'])

    # Carga del nuevo modelo de Deep Learning (Keras)
    modelo_dl = load_model('modelo_dl_cascada.h5')

    print(f"✅ ¡Sistemas listos! Umbral SVM: {UMBRAL}")
except Exception as e:
    print(f"❌ Error crítico al cargar modelos: {e}")
    UMBRAL = 0.5  # Fallback de seguridad

# Configuración del archivo de logs para auditoría y Dashboard
archivo_logs = "datos/logs_riego.csv"
if not os.path.exists(archivo_logs):
    os.makedirs("datos", exist_ok=True)
    with open(archivo_logs, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'temp_local', 'humedad_tierra',
                        'temp_internet', 'prob_svm', 'prob_dl', 'decision_final', 'metodo'])

# ============================================================
# ESQUEMA DE DATOS [cite: 465, 466]
# ============================================================


class LecturaSensores(BaseModel):
    temperatura: float = Field(..., description="Temperatura ambiente en °C")
    humedad_aire: float = Field(...,
                                description="Humedad relativa del aire (%)")
    humedad_tierra_base: float = Field(...,
                                       description="Humedad de la tierra (%)")
    horas_desde_ultimo_riego: float = Field(..., ge=0,
                                            description="Horas desde el último riego")

# ============================================================
# ENDPOINTS
# ============================================================


@app.post("/predecir")
def predecir_riego(lectura: LecturaSensores):
    """
    Recibe lectura de sensores y decide el riego usando un Ensemble de SVM y Deep Learning.
    """
    try:
        # --------------------------------------------------------
        # VALIDACIÓN DE INTERNET (Datos Externos)
        # --------------------------------------------------------
        clima_internet = {"temperatura": lectura.temperatura,
                          "humedad": lectura.humedad_aire}
        try:
            # Consulta a OpenMeteo (Cali, Valle del Cauca)
            url_clima = "https://api.open-meteo.com/v1/forecast?latitude=3.4372&longitude=-76.5225&current=temperature_2m,relative_humidity_2m"
            resp = requests.get(url_clima, timeout=3).json()
            clima_internet["temperatura"] = resp["current"]["temperature_2m"]
            clima_internet["humedad"] = resp["current"]["relative_humidity_2m"]
        except Exception:
            print("⚠️ No se pudo conectar a la API de clima, usando sensores locales.")

        # --------------------------------------------------------
        # CAPA 1: VALIDAR HARDWARE (Sensor dañado) [cite: 467, 468]
        # --------------------------------------------------------
        sensor_danado = False
        if (lectura.humedad_aire < 0 or lectura.humedad_aire > 100 or
                lectura.humedad_tierra_base < 0 or lectura.humedad_tierra_base > 100):
            sensor_danado = True
            # En lugar de bloquear, marcamos para usar Fallback de Deep Learning

        # --------------------------------------------------------
        # CAPA 2: ALERTAS CLIMÁTICAS [cite: 469, 470, 471]
        # --------------------------------------------------------
        alerta_clima = "Clima estable."
        if lectura.temperatura > 40.0:
            alerta_clima = "⚠️ ALERTA: Calor extremo detectado."
        elif lectura.temperatura < 15.0:
            alerta_clima = "⚠️ ALERTA: Temperatura baja detectada."

        # --------------------------------------------------------
        # CAPA 3 & 4: MODELO HÍBRIDO (SVM + DL) [cite: 472]
        # --------------------------------------------------------

        # Preparación para SVM (Datos de sensores locales)
        df_local = pd.DataFrame([lectura.model_dump()])
        X_svm = scaler.transform(df_local)
        proba_svm = float(modelo_svm.predict_proba(X_svm)[0, 1])

        # Preparación para Deep Learning (Datos de Internet + Tierra)
        datos_dl = [[clima_internet["temperatura"], clima_internet["humedad"],
                     lectura.humedad_tierra_base if not sensor_danado else 40.0,
                     lectura.horas_desde_ultimo_riego]]
        X_dl = scaler.transform(pd.DataFrame(
            datos_dl, columns=df_local.columns))
        proba_dl = float(modelo_dl.predict(X_dl, verbose=0)[0][0])

        # LÓGICA DE DECISIÓN FINAL (Ensemble)
        decision_final = 0
        metodo_usado = ""
        mensaje_riego = ""

        if sensor_danado:
            # CASO A: Fallback por sensor dañado
            decision_final = 1 if proba_dl >= UMBRAL else 0
            metodo_usado = "DEEP_LEARNING_FALLBACK"
            mensaje_riego = "🚨 Sensor local falló. Decisión tomada por DL basado en Internet."
        else:
            # CASO B: Sistema sano (Ensemble)
            dec_svm = 1 if proba_svm >= UMBRAL else 0
            dec_dl = 1 if proba_dl >= 0.40 else 0  # Umbral adaptativo para DL

            if dec_svm == 1 and dec_dl == 1:
                decision_final = 1
                metodo_usado = "ENSEMBLE_CONFIRMADO"
                mensaje_riego = "💧 Humedad baja y clima seco. Riego autorizado."
            elif dec_svm == 1 and dec_dl == 0:
                decision_final = 0
                metodo_usado = "DEEP_LEARNING_VETO"
                mensaje_riego = "☀️ SVM detectó tierra seca, pero DL bloqueó por pronóstico húmedo."
            else:
                decision_final = 0
                metodo_usado = "SVM_OPTIMO"
                mensaje_riego = "☀️ Condiciones estables. No se requiere riego."

        # --------------------------------------------------------
        # REGISTRO Y RESPUESTA [cite: 473, 474, 475]
        # --------------------------------------------------------
        with open(archivo_logs, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                lectura.temperatura, lectura.humedad_tierra_base, clima_internet["temperatura"],
                round(proba_svm, 3), round(
                    proba_dl, 3), decision_final, metodo_usado
            ])

        return {
            "estado_riego": decision_final,
            "probabilidades": {
                "svm_local": round(proba_svm, 4),
                "dl_internet": round(proba_dl, 4)
            },
            "metodo_decision": metodo_usado,
            "accion": "INICIAR_CICLO" if decision_final == 1 else "NO_REGAR",
            "mensaje": mensaje_riego,
            "alerta_climatica": alerta_clima
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def estado_servidor():
    """Verifica que el servidor esté activo y los modelos cargados[cite: 476]."""
    return {
        "mensaje": "🌱 Servidor Invernadero V3 (Híbrido) Activo",
        "modelos": ["SVM Scikit-Learn", "ANN TensorFlow/Keras"],
        "umbral_configurado": UMBRAL,
        "logs": "datos/logs_riego.csv"
    }
