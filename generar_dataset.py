import os
import random
import numpy as np
import pandas as pd

# ============================================================
# CONFIGURACIÓN
# ============================================================
SEMILLA = 42
N_REGISTROS = 10_000
RUTA_CSV = "datos/invernadero_cascada.csv"

FEATURES = [
    "temperatura",
    "humedad_aire",
    "humedad_tierra_base",
    "horas_desde_ultimo_riego",
]
TARGET = "estado_riego"


def generar_dataset(n_registros: int = N_REGISTROS, semilla: int = SEMILLA) -> pd.DataFrame:
    """Genera el dataset simulado del invernadero."""
    np.random.seed(semilla)
    random.seed(semilla)

    temperatura = np.round(np.random.uniform(22.0, 40.0, n_registros), 1)
    humedad_aire = np.round(np.random.uniform(30.0, 90.0, n_registros), 1)
    humedad_tierra = np.round(np.random.uniform(10.0, 95.0, n_registros), 1)
    horas_riego = np.round(np.random.uniform(0.5, 24.0, n_registros), 1)

    estado = []
    for t, h_t, h_r in zip(temperatura, humedad_tierra, horas_riego):
        # REGLA DE DOMINIO (sistema en cascada):
        # Tierra seca (<35%) O (tierra medio-seca + calor + varias horas sin regar)
        if h_t < 35.0 or (h_t < 50.0 and t > 32.0 and h_r > 4.0):
            estado.append(1 if random.random() > 0.05 else 0)  # 95% -> regar
        else:
            estado.append(0 if random.random() > 0.05 else 1)  # 95% -> no regar

    df = pd.DataFrame({
        "temperatura": temperatura,
        "humedad_aire": humedad_aire,
        "humedad_tierra_base": humedad_tierra,
        "horas_desde_ultimo_riego": horas_riego,
        TARGET: estado,
    })

    # Validar que las columnas estén en el orden esperado
    assert list(df.columns) == FEATURES + [TARGET], "Error en el orden de columnas"
    return df


def main():
    # Crear carpeta si no existe
    if not os.path.exists("datos"):
        os.makedirs("datos")

    print("Generando dataset de 10,000 registros para sistema en cascada...")
    df = generar_dataset()
    df.to_csv(RUTA_CSV, index=False)

    print(f"✅ Archivo guardado en: {RUTA_CSV}")
    print(f"Total de registros: {len(df)}")
    print(f"\nDistribución de clases:")
    print(df[TARGET].value_counts())
    print(f"\nClase 0 (No regar): {(df[TARGET]==0).sum()}")
    print(f"Clase 1 (Regar):    {(df[TARGET]==1).sum()}")


if __name__ == "__main__":
    main()

