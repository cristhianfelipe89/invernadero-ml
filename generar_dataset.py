import pandas as pd
import numpy as np
import random
import os

print("Generando dataset de 10,000 registros para sistema en cascada...")

# Asegurarnos de que la carpeta 'datos' exista
if not os.path.exists('datos'):
    os.makedirs('datos')

# Semilla para que los resultados sean reproducibles
np.random.seed(42)
n_registros = 10000

# 1. Generar variables simuladas (clima de Cali aprox)
temperatura = np.round(np.random.uniform(22.0, 40.0, n_registros), 1)
humedad_aire = np.round(np.random.uniform(30.0, 90.0, n_registros), 1)
humedad_tierra_base = np.round(np.random.uniform(10.0, 95.0, n_registros), 1)
horas_desde_ultimo_riego = np.round(
    np.random.uniform(0.5, 24.0, n_registros), 1)

estado_riego = []

# 2. Lógica del sistema en cascada
for i in range(n_registros):
    temp = temperatura[i]
    tierra = humedad_tierra_base[i]
    horas = horas_desde_ultimo_riego[i]

    # REGLA: Tierra seca (<35%) O (Tierra medio seca + Calor + Varias horas sin regar)
    if tierra < 35.0 or (tierra < 50.0 and temp > 32.0 and horas > 4.0):
        # 95% de probabilidad de Iniciar Ciclo (1)
        estado = 1 if random.random() > 0.05 else 0
    else:
        # 95% de probabilidad de No Regar (0)
        estado = 0 if random.random() > 0.05 else 1

    estado_riego.append(estado)

# 3. Crear el DataFrame
df = pd.DataFrame({
    'temperatura': temperatura,
    'humedad_aire': humedad_aire,
    'humedad_tierra_base': humedad_tierra_base,
    'horas_desde_ultimo_riego': horas_desde_ultimo_riego,
    'estado_riego': estado_riego
})

# 4. Guardar físicamente el archivo CSV
ruta_archivo = 'datos/invernadero_cascada.csv'
df.to_csv(ruta_archivo, index=False)

print(f"¡Éxito! Archivo guardado en: {ruta_archivo}")
print("\nDistribución de las clases generadas:")
print(df['estado_riego'].value_counts())
