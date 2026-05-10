import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Invernadero Inteligente", layout="wide")

st.title("🌱 Dashboard: Invernadero Inteligente (Híbrido SVM + DL)")
st.markdown("Monitor de decisiones y KPIs de salud del sistema en tiempo real.")

archivo_logs = "datos/logs_riego.csv"

def cargar_datos():
    if os.path.exists(archivo_logs):
        try:
            df = pd.read_csv(archivo_logs)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# Función para crear las tarjetas HTML/CSS
def crear_tarjeta(titulo, valor, icono, color):
    html_tarjeta = f"""
    <div style="
        border-radius: 10px;
        border-left: 6px solid {color};
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        background-color: rgba(128, 128, 128, 0.1);
        margin-bottom: 20px;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;">
        <p style="margin: 0; font-size: 1.1rem; opacity: 0.8;">{icono} {titulo}</p>
        <h2 style="margin: 5px 0 0 0; font-size: 1.8rem; color: {color};">{valor}</h2>
    </div>
    """
    return html_tarjeta

# Diccionario para mapeo amigable
MAPEO_MOTOR = {
    "DEEP_LEARNING_FALLBACK": "DL (Fallo Sensor)",
    "ENSEMBLE_CONFIRMADO": "Ensemble (SVM + DL)",
    "DEEP_LEARNING_VETO": "DL (Veto Clima)",
    "SVM_OPTIMO": "SVM (Local)"
}

placeholder = st.empty()

while True:
    df = cargar_datos()
    
    with placeholder.container():
        if not df.empty:
            hoy = datetime.now().date()
            df['fecha'] = df['timestamp'].dt.date
            df_hoy = df[df['fecha'] == hoy]
            ultima_lectura = df.iloc[-1]

            # --- ALERTA DE FALLO ---
            motor_crudo = str(ultima_lectura.get('metodo', 'Desconocido'))
            if motor_crudo == "DEEP_LEARNING_FALLBACK":
                st.error("🚨 ¡ALERTA DE HARDWARE! Usando Fallback de Deep Learning.", icon="🚨")

            # --- SECCIÓN 1: TEMPERATURAS EN TARJETAS SEPARADAS ---
            st.subheader("🌡️ Monitor de Temperaturas")
            t_col1, t_col2 = st.columns(2)
            
            with t_col1:
                st.markdown(crear_tarjeta("Temperatura Sensor Local", f"{ultima_lectura['temp_local']} °C", "🌡️", "#FF9800"), unsafe_allow_html=True)
            with t_col2:
                st.markdown(crear_tarjeta("Temperatura Internet (API)", f"{ultima_lectura['temp_internet']} °C", "🌐", "#2196F3"), unsafe_allow_html=True)

            st.divider()

            # --- SECCIÓN 2: KPIs DE ESTADO Y SALUD ---
            st.subheader("📊 Estado del Sistema y Salud")
            k1, k2, k3, k4 = st.columns(4)
            
            # Formateo
            veces_regado_hoy = df_hoy[df_hoy['decision_final'] == 1].shape[0]
            fallos_sensor_hoy = df_hoy[df_hoy['metodo'] == 'DEEP_LEARNING_FALLBACK'].shape[0]
            
            estado_txt = "REGANDO" if ultima_lectura['decision_final'] == 1 else "EN ESPERA"
            color_est = "#4CAF50" if ultima_lectura['decision_final'] == 1 else "#FFC107"
            motor_txt = MAPEO_MOTOR.get(motor_crudo, motor_crudo)
            color_mot = "#F44336" if motor_crudo == "DEEP_LEARNING_FALLBACK" else "#00BCD4"

            with k1:
                st.markdown(crear_tarjeta("Estado Riego", estado_txt, "💧", color_est), unsafe_allow_html=True)
            with k2:
                st.markdown(crear_tarjeta("Motor de Decisión", motor_txt, "🧠", color_mot), unsafe_allow_html=True)
            with k3:
                st.markdown(crear_tarjeta("Riegos Hoy", str(veces_regado_hoy), "🔄", "#9C27B0"), unsafe_allow_html=True)
            with k4:
                st.markdown(crear_tarjeta("Fallos Sensor", str(fallos_sensor_hoy), "⚠️", "#F44336" if fallos_sensor_hoy > 0 else "#4CAF50"), unsafe_allow_html=True)

            st.divider()

            # --- SECCIÓN 3: LOG DE AUDITORÍA ---
            st.subheader("📋 Últimas Decisiones Registradas")
            df_log = df[['timestamp', 'temp_local', 'temp_internet', 'metodo', 'decision_final']].copy()
            df_log['metodo'] = df_log['metodo'].map(MAPEO_MOTOR).fillna(df_log['metodo'])
            st.dataframe(df_log.tail(8).sort_values(by='timestamp', ascending=False), use_container_width=True)

            # --- SECCIÓN 4: COMPARATIVA GRÁFICA (AL FINAL) ---
            st.subheader("📈 Comparativa Histórica: Sensor vs Internet")
            df_chart = df[['timestamp', 'temp_local', 'temp_internet']].set_index('timestamp')
            df_chart = df_chart.rename(columns={'temp_local': 'Sensor Local (°C)', 'temp_internet': 'Internet API (°C)'})
            st.line_chart(df_chart)
            
        else:
            st.info("Esperando datos del simulador...")
            
    time.sleep(5.0)