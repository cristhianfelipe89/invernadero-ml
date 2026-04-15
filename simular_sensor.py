import argparse
import random
import sys
import time
import requests

# ============================================================
# CONFIGURACIÓN POR DEFECTO
# ============================================================
URL_API_DEFAULT = "http://127.0.0.1:8000/predecir"
INTERVALO_DEFAULT = 5  # segundos entre lecturas


def generar_lectura(prob_falla_sensor=0.10, prob_calor_extremo=0.10) -> dict:
    """
    Genera una lectura simulada de sensores.
    Incluye probabilidad de simular fallas (sensor dañado, calor extremo).
    """
    # 10% de probabilidad: sensor dañado (cortocircuito)
    if random.random() < prob_falla_sensor:
        humedad_tierra = 150.0  # ¡Dato irreal!
        print("   ⚡ [SIMULANDO FALLA FÍSICA EN EL SENSOR] ⚡")
    else:
        humedad_tierra = round(random.uniform(20.0, 90.0), 1)

    # 10% de probabilidad: calor extremo
    if random.random() < prob_calor_extremo:
        temperatura = 42.5
    else:
        temperatura = round(random.uniform(25.0, 38.0), 1)

    return {
        "temperatura": temperatura,
        "humedad_aire": round(random.uniform(40.0, 80.0), 1),
        "humedad_tierra_base": humedad_tierra,
        "horas_desde_ultimo_riego": round(random.uniform(0.1, 8.0), 1),
    }


def imprimir_decision(decision: dict) -> None:
    """Imprime la respuesta de la API de forma legible."""
    estado = decision.get('estado_riego')

    if estado == 1:
        print(f"   🟢 ACCIÓN: {decision['accion']} -> {decision['mensaje']}")
    elif estado == 0:
        print(f"   ⚪ ACCIÓN: {decision['accion']} -> {decision['mensaje']}")
    elif estado == -1:
        print(f"   🔴 ACCIÓN: {decision['accion']} -> {decision['mensaje']}")
        print(f"      {decision.get('detalle', '')}")

    # Alertas climáticas
    alerta = decision.get('alerta_climatica', '')
    if alerta and alerta != "Ninguna. Clima estable.":
        print(f"   {alerta}")

    # Probabilidad y umbral (si están disponibles)
    if 'probabilidad_riego' in decision:
        print(f"   📊 Probabilidad: {decision['probabilidad_riego']} | "
              f"Umbral: {decision.get('umbral_usado', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Simulador de sensores IoT para el invernadero"
    )
    parser.add_argument(
        "--url", default=URL_API_DEFAULT,
        help=f"URL del endpoint de predicción (default: {URL_API_DEFAULT})"
    )
    parser.add_argument(
        "--intervalo", type=float, default=INTERVALO_DEFAULT,
        help=f"Segundos entre lecturas (default: {INTERVALO_DEFAULT})"
    )
    args = parser.parse_args()

    print(f"📡 Iniciando simulador de sensor IoT")
    print(f"   URL: {args.url}")
    print(f"   Intervalo: {args.intervalo}s")
    print(f"   Presiona Ctrl+C para detenerlo.\n")

    ciclo = 0
    try:
        while True:
            ciclo += 1
            datos = generar_lectura()
            print(f"📤 [{ciclo}] Temp:{datos['temperatura']}°C | "
                  f"Aire:{datos['humedad_aire']}% | "
                  f"Tierra:{datos['humedad_tierra_base']}% | "
                  f"Horas:{datos['horas_desde_ultimo_riego']}h")

            try:
                respuesta = requests.post(args.url, json=datos, timeout=5)
                if respuesta.status_code == 200:
                    imprimir_decision(respuesta.json())
                else:
                    print(
                        f"   ⚠️ Error del servidor: HTTP {respuesta.status_code}")
            except requests.exceptions.ConnectionError:
                print("   ❌ No se pudo conectar a la API. "
                      "Verifica que uvicorn esté corriendo.")
            except requests.exceptions.Timeout:
                print("   ⏰ Timeout: la API no respondió a tiempo.")

            print("-" * 65)
            time.sleep(args.intervalo)

    except KeyboardInterrupt:
        print(f"\n👋 Simulador detenido después de {ciclo} lecturas.")
        sys.exit(0)


if __name__ == "__main__":
    main()
