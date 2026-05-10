import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Invernadero Inteligente", layout="wide")

st.title("🌱 Dashboard: Invernadero Inteligente (Híbrido SVM + DL)")
st.markdown("Monitor de decisiones, KPIs de salud del sistema y comparativas en tiempo real.")

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
        <h2 style="margin: 5px 0 0 0; font-size: 1.5rem; color: {color};">{valor}</h2>
    </div>
    """
    return html_tarjeta

# Diccionario para mapear las variables del motor a texto legible en el dashboard
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
            
            # --- ALERTA VISUAL SI EL SENSOR ESTÁ FALLANDO AHORA MISMO ---
            motor_crudo = str(ultima_lectura.get('metodo', 'Desconocido'))
            if motor_crudo == "DEEP_LEARNING_FALLBACK":
                st.error("🚨 ¡ALERTA CRÍTICA DE HARDWARE! El sensor de humedad de tierra está enviando valores irreales. El sistema SVM fue desactivado y el riego se está operando 100% con la Red Neuronal (Clima de Internet).", icon="🚨")
            
            # Cálculos de KPIs
            veces_regado_hoy = df_hoy[df_hoy['decision_final'] == 1].shape[0]
            fallos_sensor_hoy = df_hoy[df_hoy['metodo'] == 'DEEP_LEARNING_FALLBACK'].shape[0]
            
            # Formateo de Tarjeta 1: Estado Actual
            estado_actual = "REGANDO" if ultima_lectura['decision_final'] == 1 else "EN ESPERA"
            color_estado = "#4CAF50" if ultima_lectura['decision_final'] == 1 else "#FFC107"
            
            # Formateo de Tarjeta 2: Motor Actual (Pintar de rojo si falla)
            motor_legible = MAPEO_MOTOR.get(motor_crudo, motor_crudo)
            color_motor = "#F44336" if motor_crudo == "DEEP_LEARNING_FALLBACK" else "#2196F3"
            
            # Formateo de Tarjeta 4: Fallos Hoy
            color_fallos = "#F44336" if fallos_sensor_hoy > 0 else "#4CAF50"
            
            # --- FILA DE KPIs ---
            st.subheader("📊 KPIs del Día (Rendimiento y Salud)")
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            
            with kpi1:
                st.markdown(crear_tarjeta("Estado Actual", estado_actual, "💧", color_estado), unsafe_allow_html=True)
            with kpi2:
                st.markdown(crear_tarjeta("Motor Actual", motor_legible, "🧠", color_motor), unsafe_allow_html=True)
            with kpi3:
                st.markdown(crear_tarjeta("Riegos Ejecutados Hoy", str(veces_regado_hoy), "🔄", "#9C27B0"), unsafe_allow_html=True)
            with kpi4:
                st.markdown(crear_tarjeta("Fallos Sensor Hoy", str(fallos_sensor_hoy), "⚠️", color_fallos), unsafe_allow_html=True)
            
            st.divider()
            
            # --- GRÁFICOS COMPARATIVOS ---
            st.subheader("📈 Comparativas: Local vs Internet")
            col_graf1, col_graf2 = st.columns(2)
            
            with col_graf1:
                st.markdown("**Temperatura: Sensor Local vs API Web**")
                df_temp = df[['timestamp', 'temp_local', 'temp_internet']].set_index('timestamp')
                df_temp = df_temp.rename(columns={'temp_local': 'Sensor Local (°C)', 'temp_internet': 'Internet API (°C)'})
                st.line_chart(df_temp)
                
            with col_graf2:
                st.markdown("**Probabilidades de Riego: SVM vs Deep Learning**")
                df_probs = df[['timestamp', 'prob_svm', 'prob_dl']].set_index('timestamp')
                df_probs = df_probs.rename(columns={'prob_svm': 'SVM (Local)', 'prob_dl': 'DL (Internet)'})
                st.line_chart(df_probs)

            # --- LOG DE AUDITORÍA ---
            st.subheader("📋 Últimas Decisiones (Auditoría)")
            df_mostrar = df[['timestamp', 'temp_local', 'humedad_tierra', 'temp_internet', 'metodo', 'decision_final']]
            # Aplicamos el mismo mapeo amigable a la tabla para que se vea limpio
            df_mostrar['metodo'] = df_mostrar['metodo'].map(MAPEO_MOTOR).fillna(df_mostrar['metodo'])
            st.dataframe(df_mostrar.tail(10).sort_values(by='timestamp', ascending=False), use_container_width=True)
            
        else:
            st.info("Esperando datos del simulador... Asegúrate de iniciar 'api_riego.py' y 'simular_sensor.py'")
            
    time.sleep(5.0)