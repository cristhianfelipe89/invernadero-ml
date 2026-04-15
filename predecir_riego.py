import json
import joblib
import pandas as pd

print("🌱 Iniciando Sistema de Predicción en Vivo (SVM + Umbral Óptimo)...")

# 1. Cargar el modelo, el escalador y el umbral previamente entrenados
try:
    scaler = joblib.load('scaler_cascada.pkl')
    modelo_svm = joblib.load('modelo_svm_cascada.pkl')
    with open('umbral_optimo.json', 'r') as f:
        config_umbral = json.load(f)
    UMBRAL = float(config_umbral['umbral'])
    print(
        f"✅ Modelo SVM, escalador y umbral cargados (umbral = {UMBRAL:.2f}).\n")
except FileNotFoundError:
    print("❌ Error: No se encontraron los archivos .pkl o el umbral_optimo.json.")
    print("Por favor, ejecuta primero 'python entrenar_modelo.py' para generarlos.")
    exit()

# 2. Simular una lectura en vivo de los sensores
lectura_sensores = {
    'temperatura': [36.0],             # Clima caluroso en Cali
    'humedad_aire': [55.0],            # Aire moderado
    'humedad_tierra_base': [28.0],     # Tierra seca (< 35%)
    'horas_desde_ultimo_riego': [5.0]  # 5 horas sin regar
}

# Convertimos a DataFrame
df_nueva_lectura = pd.DataFrame(lectura_sensores)

print("📊 Nueva lectura de sensores recibida:")
print(df_nueva_lectura.to_string(index=False))

# 3. Escalar los datos nuevos (transform, NO fit_transform)
lectura_escalada = scaler.transform(df_nueva_lectura)

# 4. Predicción usando probabilidad + umbral óptimo
proba = float(modelo_svm.predict_proba(lectura_escalada)[0, 1])
prediccion = int(proba >= UMBRAL)

# 5. Interpretar la decisión final
print("\n🤖 DECISIÓN DEL MODELO SVM:")
print("-" * 45)
print(f"Probabilidad de regar : {proba:.4f}")
print(f"Umbral aplicado       : {UMBRAL:.2f}")
print(f"¿Supera el umbral?    : {'SÍ' if prediccion == 1 else 'NO'}")
print("-" * 45)
if prediccion == 1:
    print("💧 ACCIÓN: ¡INICIAR CICLO DE CASCADA! (Válvula ABIERTA)")
else:
    print("☀️ ACCIÓN: No regar por ahora. (Válvula CERRADA)")
print("-" * 45)
