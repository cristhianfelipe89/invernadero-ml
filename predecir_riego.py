import joblib
import pandas as pd

print("🌱 Iniciando Sistema de Predicción en Vivo...")

# 1. Cargar el modelo y el escalador previamente entrenados
try:
    scaler = joblib.load('scaler_cascada.pkl')
    modelo_svm = joblib.load('modelo_svm_cascada.pkl')
    print("✅ Modelo y escalador cargados correctamente.\n")
except FileNotFoundError:
    print("❌ Error: No se encontraron los archivos .pkl.")
    print("Por favor, ejecuta primero 'python entrenar_modelo.py' para generarlos.")
    exit()

# 2. Simular una lectura en vivo de los sensores (ej. datos llegando de un Arduino/ESP32)
# Supongamos que hace mucho calor, la tierra base se secó y pasaron 5 horas.
lectura_sensores = {
    'temperatura': [26.0],             # Un clima fresco para Cali
    'humedad_aire': [65.0],            # Aire húmedo
    'humedad_tierra_base': [80.0],     # ¡Tierra súper mojada! (80%)
    'horas_desde_ultimo_riego': [0.5]  # Lo regamos hace apenas media hora
}

##
##lectura_sensores = {
##    'temperatura': [26.0],             # Un clima fresco para Cali
##    'humedad_aire': [65.0],            # Aire húmedo
##    'humedad_tierra_base': [80.0],     # ¡Tierra súper mojada! (80%)
##    'horas_desde_ultimo_riego': [0.5]  # Lo regamos hace apenas media hora
##}

# Convertimos el diccionario a un DataFrame de Pandas (el formato que le gusta al modelo)
df_nueva_lectura = pd.DataFrame(lectura_sensores)

print("📊 Nueva lectura de sensores recibida:")
print(df_nueva_lectura.to_string(index=False))

# 3. Escalar los datos nuevos
# ¡Súper importante! Usamos transform() y NO fit_transform().
# Esto aplica exactamente la misma escala matemática que aprendió en el entrenamiento.
lectura_escalada = scaler.transform(df_nueva_lectura)

# 4. Hacer la predicción
prediccion = modelo_svm.predict(lectura_escalada)

# 5. Interpretar la decisión final para el usuario (o para el relé de la bomba de agua)
print("\n🤖 DECISIÓN DEL MODELO SVM:")
print("-" * 40)
if prediccion[0] == 1:
    print("💧 ACCIÓN: ¡INICIAR CICLO DE CASCADA! (Válvula ABIERTA)")
else:
    print("☀️ ACCIÓN: No regar por ahora. (Válvula CERRADA)")
print("-" * 40)
