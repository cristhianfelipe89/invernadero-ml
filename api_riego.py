from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Inicializar la aplicación API
app = FastAPI(title="API Invernadero - Sistema en Cascada con Alertas")

print("Cargando modelo SVM y escalador...")
try:
    scaler = joblib.load('scaler_cascada.pkl')
    modelo_svm = joblib.load('modelo_svm_cascada.pkl')
    print("✅ ¡Modelos listos para recibir peticiones!")
except Exception as e:
    print(f"❌ Error al cargar los modelos: {e}")

# Definir la estructura de los datos
class LecturaSensores(BaseModel):
    temperatura: float
    humedad_aire: float
    humedad_tierra_base: float
    horas_desde_ultimo_riego: float

@app.post("/predecir")
def predecir_riego(lectura: LecturaSensores):
    try:
        # ==========================================
        # 1. CAPA DE SEGURIDAD: Validar Hardware
        # ==========================================
        # Si la humedad es menor a 0 o mayor a 100, el sensor se desconectó o se dañó.
        if (lectura.humedad_aire < 0 or lectura.humedad_aire > 100 or 
            lectura.humedad_tierra_base < 0 or lectura.humedad_tierra_base > 100):
            return {
                "estado_riego": -1, # -1 significa ERROR/BLOQUEO
                "accion": "DETENER_SISTEMA",
                "mensaje": "🚨 ALERTA CRÍTICA: Lectura de humedad irreal.",
                "detalle": "Revisar cableado del sensor. El SVM no se ejecutará por seguridad."
            }

        # ==========================================
        # 2. CAPA DE ALERTAS CLIMÁTICAS
        # ==========================================
        alerta_clima = "Ninguna. Clima estable."
        if lectura.temperatura > 40.0:
            alerta_clima = "⚠️ ALERTA: Calor extremo (>40°C). Riesgo alto para el cultivo."
        elif lectura.temperatura < 15.0:
            alerta_clima = "⚠️ ALERTA: Temperatura inusualmente baja para Cali."

        # ==========================================
        # 3. EJECUCIÓN DEL MODELO SVM (Inteligencia)
        # ==========================================
        datos_diccionario = lectura.model_dump()
        df_nueva_lectura = pd.DataFrame([datos_diccionario])
        
        lectura_escalada = scaler.transform(df_nueva_lectura)
        prediccion = modelo_svm.predict(lectura_escalada)
        resultado = int(prediccion[0])
        
        # ==========================================
        # 4. RESPUESTA FINAL
        # ==========================================
        if resultado == 1:
            accion_texto = "INICIAR_CICLO"
            mensaje_riego = "💧 SVM ordenó abrir válvula. Tierra seca o tiempo cumplido."
        else:
            accion_texto = "NO_REGAR"
            mensaje_riego = "☀️ SVM ordenó esperar. Condiciones de humedad óptimas."
            
        return {
            "estado_riego": resultado,
            "accion": accion_texto,
            "mensaje": mensaje_riego,
            "alerta_climatica": alerta_clima
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def estado_servidor():
    return {"mensaje": "El servidor del invernadero está funcionando y validando datos 🌱"}