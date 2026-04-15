"""
API REST del invernadero inteligente (FastAPI).

El sistema opera en 3 CAPAS EN CASCADA:
  1) Validación de hardware (sensor dañado → bloquear).
  2) Alertas climáticas (temperatura extrema → avisar).
  3) Modelo SVM + umbral óptimo → decidir riego.

Ejecutar:
    uvicorn api_riego:app --reload

Documentación interactiva:
    http://127.0.0.1:8000/docs
"""
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# ============================================================
# INICIALIZACIÓN
# ============================================================
app = FastAPI(
    title="API Invernadero - Sistema en Cascada con SVM",
    description="Predicción de riego con SVM optimizado y umbral ajustado",
    version="2.0"
)

print("Cargando modelo SVM, escalador y umbral óptimo...")
try:
    scaler = joblib.load('scaler_cascada.pkl')
    modelo_svm = joblib.load('modelo_svm_cascada.pkl')

    with open('umbral_optimo.json', 'r') as f:
        config_umbral = json.load(f)
    UMBRAL = float(config_umbral['umbral'])
    print(f"✅ ¡Modelos listos! Umbral óptimo cargado = {UMBRAL:.2f}")
except Exception as e:
    print(f"❌ Error al cargar los modelos: {e}")
    UMBRAL = 0.5  # fallback

# ============================================================
# ESQUEMA DE DATOS
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
    """Recibe lectura de sensores y devuelve la decisión de riego."""
    try:
        # ==========================================
        # CAPA 1: VALIDAR HARDWARE (Sensor dañado)
        # ==========================================
        if (lectura.humedad_aire < 0 or lectura.humedad_aire > 100 or
                lectura.humedad_tierra_base < 0 or lectura.humedad_tierra_base > 100):
            return {
                "estado_riego": -1,
                "accion": "DETENER_SISTEMA",
                "mensaje": "🚨 ALERTA CRÍTICA: Lectura de humedad irreal.",
                "detalle": "Revisar cableado del sensor. El SVM no se ejecutará por seguridad."
            }

        # ==========================================
        # CAPA 2: ALERTAS CLIMÁTICAS
        # ==========================================
        alerta_clima = "Ninguna. Clima estable."
        if lectura.temperatura > 40.0:
            alerta_clima = "⚠️ ALERTA: Calor extremo (>40°C). Riesgo alto para el cultivo."
        elif lectura.temperatura < 15.0:
            alerta_clima = "⚠️ ALERTA: Temperatura inusualmente baja para Cali."

        # ==========================================
        # CAPA 3: MODELO SVM + UMBRAL ÓPTIMO
        # ==========================================
        datos = lectura.model_dump()
        df_lectura = pd.DataFrame([datos])
        lectura_escalada = scaler.transform(df_lectura)

        # Probabilidad de la clase positiva (regar)
        proba = float(modelo_svm.predict_proba(lectura_escalada)[0, 1])

        # Aplicar el UMBRAL ÓPTIMO (ajustado en threshold tuning)
        resultado = int(proba >= UMBRAL)

        # ==========================================
        # RESPUESTA FINAL
        # ==========================================
        if resultado == 1:
            accion_texto = "INICIAR_CICLO"
            mensaje_riego = "💧 SVM ordenó abrir válvula. Tierra seca o tiempo cumplido."
        else:
            accion_texto = "NO_REGAR"
            mensaje_riego = "☀️ SVM ordenó esperar. Condiciones de humedad óptimas."

        return {
            "estado_riego": resultado,
            "probabilidad_riego": round(proba, 4),
            "umbral_usado": UMBRAL,
            "accion": accion_texto,
            "mensaje": mensaje_riego,
            "alerta_climatica": alerta_clima
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def estado_servidor():
    """Verifica que el servidor esté activo y el modelo cargado."""
    return {
        "mensaje": "🌱 Servidor del invernadero activo",
        "modelo": "SVM optimizado",
        "umbral_optimo": UMBRAL
    }
