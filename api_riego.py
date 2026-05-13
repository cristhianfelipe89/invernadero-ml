"""
API REST del invernadero inteligente evolucionada (FastAPI).
Incluye filtro "Debouncing" para fallos de hardware (5 lecturas consecutivas)
y prevención de data basura.
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

app = FastAPI(title="API Invernadero - Híbrido SVM & DL", version="3.1")

# ============================================================
# VARIABLES GLOBALES DE ESTADO (Debouncing)
# ============================================================
contador_fallos = 0
ultima_humedad_valida = 40.0  # Valor lógico por defecto para inicializar

# ============================================================
# CARGA DE MODELOS
# ============================================================
print("Cargando artefactos de inteligencia...")
try:
    scaler = joblib.load('scaler_cascada.pkl')
    modelo_svm = joblib.load('modelo_svm_cascada.pkl')
    with open('umbral_optimo.json', 'r') as f:
        config_umbral = json.load(f)
    UMBRAL = float(config_umbral['umbral'])
    modelo_dl = load_model('modelo_dl_cascada.h5')
    print(f"✅ ¡Sistemas listos! Umbral SVM: {UMBRAL}")
except Exception as e:
    print(f"❌ Error crítico al cargar modelos: {e}")
    UMBRAL = 0.5  # Fallback de seguridad

# Configuración del CSV
archivo_logs = "datos/logs_riego.csv"
if not os.path.exists(archivo_logs):
    os.makedirs("datos", exist_ok=True)
    with open(archivo_logs, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'temp_local', 'humedad_tierra', 'temp_internet', 'prob_svm', 'prob_dl', 'decision_final', 'metodo'])

# ============================================================
# ESQUEMA DE DATOS
# ============================================================
class LecturaSensores(BaseModel):
    temperatura: float = Field(..., description="Temperatura ambiente en °C")
    humedad_aire: float = Field(..., description="Humedad relativa del aire (%)")
    humedad_tierra_base: float = Field(..., description="Humedad de la tierra (%)")
    horas_desde_ultimo_riego: float = Field(..., ge=0, description="Horas desde el último riego")

# ============================================================
# ENDPOINTS
# ============================================================
@app.post("/predecir")
def predecir_riego(lectura: LecturaSensores):
    global contador_fallos, ultima_humedad_valida
    try:
        # 1. Consulta del Clima (Internet)
        clima_internet = {"temperatura": lectura.temperatura, "humedad": lectura.humedad_aire}
        try:
            url_clima = "https://api.open-meteo.com/v1/forecast?latitude=3.4372&longitude=-76.5225&current=temperature_2m,relative_humidity_2m"
            resp = requests.get(url_clima, timeout=3).json()
            clima_internet["temperatura"] = resp["current"]["temperature_2m"]
            clima_internet["humedad"] = resp["current"]["relative_humidity_2m"]
        except Exception:
            pass # Falla silenciosa: si no hay internet, usa valores del sensor

        # 2. VALIDACIÓN DE HARDWARE Y ANTIRREBOTE (Evitar data basura)
        es_dato_basura = (lectura.humedad_tierra_base < 0 or lectura.humedad_tierra_base > 100)

        if es_dato_basura:
            contador_fallos += 1
            # LIMPIEZA: Ignoramos la basura y usamos el último valor lógico conocido
            humedad_limpia = ultima_humedad_valida 
        else:
            contador_fallos = 0
            ultima_humedad_valida = lectura.humedad_tierra_base
            humedad_limpia = lectura.humedad_tierra_base

        sensor_danado_critico = (contador_fallos >= 5)

        # 3. ALERTAS CLIMÁTICAS (Restauradas para el simulador)
        alerta_clima = "Ninguna. Clima estable."
        if lectura.temperatura > 40.0:
            alerta_clima = "⚠️ ALERTA: Calor extremo (>40°C). Riesgo alto para el cultivo."
        elif lectura.temperatura < 15.0:
            alerta_clima = "⚠️ ALERTA: Temperatura inusualmente baja para Cali."

        # 4. Preparación de Datos (Usando humedad_limpia SIEMPRE)
        df_local = pd.DataFrame([[lectura.temperatura, lectura.humedad_aire, humedad_limpia, lectura.horas_desde_ultimo_riego]], 
                                columns=LecturaSensores.model_fields.keys())
        X_svm = scaler.transform(df_local)
        proba_svm = float(modelo_svm.predict_proba(X_svm)[0, 1])

        datos_dl = [[clima_internet["temperatura"], clima_internet["humedad"], humedad_limpia, lectura.horas_desde_ultimo_riego]]
        X_dl = scaler.transform(pd.DataFrame(datos_dl, columns=df_local.columns))
        proba_dl = float(modelo_dl.predict(X_dl, verbose=0)[0][0])

        # 5. LÓGICA DE DECISIÓN
        if sensor_danado_critico:
            decision_final = 1 if proba_dl >= UMBRAL else 0
            metodo_usado = "DEEP_LEARNING_FALLBACK"
            mensaje_riego = "🚨 Fallo Crítico (5 lecturas malas). Decisión 100% por Internet (DL)."
        elif contador_fallos > 0:
            # Hay ruido, pero aún no es crítico. Se usa el Ensemble con el último dato bueno.
            decision_final = 1 if proba_svm >= UMBRAL else 0
            metodo_usado = "IGNORANDO_RUIDO"
            mensaje_riego = f"⚠️ Pico de ruido detectado. Puenteando con último valor: {humedad_limpia}%"
        else:
            # Operación Normal Híbrida (Ensemble)
            dec_svm = 1 if proba_svm >= UMBRAL else 0
            dec_dl = 1 if proba_dl >= 0.40 else 0
            
            if dec_svm == 1 and dec_dl == 1:
                decision_final = 1
                metodo_usado = "ENSEMBLE_CONFIRMADO"
                mensaje_riego = "💧 SVM y DL coinciden. Riego autorizado."
            elif dec_svm == 1 and dec_dl == 0:
                decision_final = 0
                metodo_usado = "DEEP_LEARNING_VETO"
                mensaje_riego = "☀️ DL bloqueó riego por pronóstico húmedo externo."
            else:
                decision_final = 0
                metodo_usado = "SVM_OPTIMO"
                mensaje_riego = "☀️ Tierra óptima. No se requiere riego."

        # 6. GUARDADO SEGURO (El CSV nunca verá el 150%)
        with open(archivo_logs, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                lectura.temperatura, humedad_limpia, clima_internet["temperatura"],
                round(proba_svm, 3), round(proba_dl, 3), decision_final, metodo_usado
            ])

        # RESPUESTA JSON COMPLETA (Compatible con simular_sensor.py y el Dashboard)
        return {
            "estado_riego": decision_final, 
            "accion": "INICIAR_CICLO" if decision_final == 1 else "NO_REGAR",
            "mensaje": mensaje_riego,
            "alerta_climatica": alerta_clima,
            "probabilidad_riego": round(proba_svm, 4),
            "umbral_usado": UMBRAL,
            "metodo": metodo_usado, 
            "probabilidades": {
                "svm_local": round(proba_svm, 4),
                "dl_internet": round(proba_dl, 4)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def estado_servidor():
    """Verifica que el servidor esté activo y los modelos cargados."""
    return {
        "mensaje": "🌱 Servidor Invernadero V3 (Híbrido) Activo",
        "modelos": ["SVM Scikit-Learn", "ANN TensorFlow/Keras"],
        "umbral_configurado": UMBRAL,
        "logs": "datos/logs_riego.csv"
    }