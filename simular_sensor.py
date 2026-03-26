import requests
import time
import random

URL_API = "http://127.0.0.1:8000/predecir"

print("📡 Iniciando simulador de sensor IoT (Con simulación de fallas)...")
print("Presiona Ctrl+C para detenerlo.\n")

while True:
    # --- SIMULACIÓN DE CONDICIONES ---
    
    # 10% de probabilidad de simular un sensor dañado (Cortocircuito)
    if random.random() < 0.10:
        humedad_tierra = 150.0  # ¡Dato irreal!
        print("\n⚡ [SIMULANDO FALLA FÍSICA EN EL SENSOR] ⚡")
    else:
        humedad_tierra = round(random.uniform(20.0, 90.0), 1)

    # 10% de probabilidad de simular calor extremo
    if random.random() < 0.10:
        temperatura = 42.5 # ¡Calor peligroso!
    else:
        temperatura = round(random.uniform(25.0, 38.0), 1)

    # Armamos el paquete de datos
    datos_sensor = {
        "temperatura": temperatura,
        "humedad_aire": round(random.uniform(40.0, 80.0), 1),
        "humedad_tierra_base": humedad_tierra,
        "horas_desde_ultimo_riego": round(random.uniform(0.1, 8.0), 1)
    }

    print(f"📤 Enviando lectura: Temp:{datos_sensor['temperatura']}°C | Tierra:{datos_sensor['humedad_tierra_base']}%")

    # --- ENVIAR A LA API Y LEER RESPUESTA ---
    try:
        respuesta = requests.post(URL_API, json=datos_sensor)
        
        if respuesta.status_code == 200:
            decision = respuesta.json()
            estado = decision.get('estado_riego')
            
            # Leer la decisión principal
            if estado == 1:
                print(f"   🟢 ACCIÓN: {decision['accion']} -> {decision['mensaje']}")
            elif estado == 0:
                print(f"   ⚪ ACCIÓN: {decision['accion']} -> {decision['mensaje']}")
            elif estado == -1:
                print(f"   🔴 ACCIÓN: {decision['accion']} -> {decision['mensaje']}")
                print(f"      {decision.get('detalle')}")
            
            # Leer si hay alertas climáticas extras
            alerta = decision.get('alerta_climatica')
            if alerta and alerta != "Ninguna. Clima estable.":
                print(f"   {alerta}")
                
        else:
            print(f"   ⚠️ Error del servidor: {respuesta.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Error: No se encuentra la API. Verifica que Uvicorn esté corriendo.")

    print("-" * 60)
    time.sleep(30) # Espera 4 segundos