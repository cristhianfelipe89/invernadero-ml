import streamlit as st
import pandas as pd
import altair as alt
import time
import os
from datetime import datetime

# Configuración de la página
st.set_page_config(page_title="Invernadero Inteligente V3", layout="wide", initial_sidebar_state="collapsed")

# 1. CSS: Ocultar header, espaciado corregido y fuentes gigantes (Versión Limpia)
st.markdown("""
    <style>
    .stApp { background-color: #f1f5f9; }
    #MainMenu {visibility: hidden;} 
    header {visibility: hidden;} 
    footer {visibility: hidden;}
    
    /* Padding superior aumentado a 5rem para despegarlo totalmente del borde */
    .block-container { 
        padding-top: 5rem; 
        padding-bottom: 3rem; 
        max-width: 95%; 
    }
    
    .section-header {
        color: #475569;
        font-size: 16px;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 12px;
        border-bottom: 2px solid #cbd5e1;
        padding-bottom: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

archivo_logs = "datos/logs_riego.csv"

def cargar_datos():
    if os.path.exists(archivo_logs):
        try:
            df = pd.read_csv(archivo_logs)
            if not df.empty: df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except: return pd.DataFrame()
    return pd.DataFrame()

def crear_tarjeta_gigante(titulo, valor, icono, color_borde):
    return f"""
    <div style="
        background-color: #ffffff;
        border-radius: 12px;
        border-bottom: 10px solid {color_borde};
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 15px;
        text-align: center;">
        <div style="font-size: 14px; color: #64748b; font-weight: 700; text-transform: uppercase; margin-bottom: 10px;">{titulo}</div>
        <div style="font-size: 34px; font-weight: 900; color: #1e293b; line-height: 1.1;">
            <span style="font-size: 30px; margin-right: 10px;">{icono}</span>{valor}
        </div>
    </div>
    """

MAPEO_MOTOR = {
    "DEEP_LEARNING_FALLBACK": "⚠️ FALLBACK DL",
    "IGNORANDO_RUIDO": "🛡️ FILTRO RUIDO",
    "ENSEMBLE_CONFIRMADO": "🤝 ENSEMBLE",
    "DEEP_LEARNING_VETO": "🛑 VETO CLIMA",
    "SVM_OPTIMO": "🤖 SVM LOCAL",
    "SVM_NO_REGAR": "🤖 SVM LOCAL"
}

placeholder = st.empty()

while True:
    df = cargar_datos()
    
    with placeholder.container():
        if not df.empty:
            ultima_lectura = df.iloc[-1]
            ultimo_timestamp = ultima_lectura['timestamp']
            
            hoy = datetime.now().date()
            df['fecha'] = df['timestamp'].dt.date
            df_hoy = df[df['fecha'] == hoy]
            alertas_hoy = df_hoy[df_hoy['metodo'] == 'DEEP_LEARNING_FALLBACK'].shape[0]
            riegos_hoy = df_hoy[df_hoy['decision_final'] == 1].shape[0]

            # --- ENCABEZADO (Sin la campana) ---
            st.markdown("<h1 style='color: #0f172a; margin-top: 0; margin-bottom: 0;'>🌿 Dashboard Invernadero IA</h1>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: #64748b; margin-top: 5px; font-weight: 600;'>Última actualización: {ultimo_timestamp.strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)

            # --- TARJETAS AGRUPADAS ---
            st.markdown("<br>", unsafe_allow_html=True)
            col_izq, col_cen, col_der = st.columns(3)

            with col_izq:
                st.markdown("<div class='section-header'>⚙️ PANEL DE CONTROL</div>", unsafe_allow_html=True)
                motor = MAPEO_MOTOR.get(str(ultima_lectura['metodo']), str(ultima_lectura['metodo']))
                color_motor = "#ef4444" if "FALLBACK" in motor else "#6366f1"
                st.markdown(crear_tarjeta_gigante("Motor Activo", motor, "", color_motor), unsafe_allow_html=True)
                estado_val = "ABIERTA" if ultima_lectura['decision_final'] == 1 else "CERRADA"
                color_val = "#10b981" if ultima_lectura['decision_final'] == 1 else "#94a3b8"
                st.markdown(crear_tarjeta_gigante("Estado Válvula", estado_val, "🚿", color_val), unsafe_allow_html=True)

            with col_cen:
                st.markdown("<div class='section-header'>📍 MÉTRICAS LOCALES</div>", unsafe_allow_html=True)
                st.markdown(crear_tarjeta_gigante("Humedad Tierra", f"{ultima_lectura['humedad_tierra']:.0f}%", "💧", "#3b82f6"), unsafe_allow_html=True)
                st.markdown(crear_tarjeta_gigante("Temperatura Local", f"{ultima_lectura['temp_local']:.1f}°C", "🌡️", "#f59e0b"), unsafe_allow_html=True)

            with col_der:
                st.markdown("<div class='section-header'>☁️ DEEP LEARNING (WEB)</div>", unsafe_allow_html=True)
                prob_dl = float(ultima_lectura.get('prob_dl', 0))
                st.markdown(crear_tarjeta_gigante("Prob. Lluvia", f"{prob_dl * 100:.0f}%", "🌧️", "#8b5cf6"), unsafe_allow_html=True)
                st.markdown(crear_tarjeta_gigante("Temperatura Web", f"{ultima_lectura['temp_internet']:.1f}°C", "🌐", "#0ea5e9"), unsafe_allow_html=True)

            # --- CONTEOS DIARIOS ---
            st.markdown("<br>", unsafe_allow_html=True)
            kpi_col1, kpi_col2 = st.columns(2)
            with kpi_col1:
                st.markdown(crear_tarjeta_gigante("Riegos Ejecutados Hoy", str(riegos_hoy), "🔄", "#10b981"), unsafe_allow_html=True)
            with kpi_col2:
                color_falla = "#ef4444" if alertas_hoy > 0 else "#10b981"
                st.markdown(crear_tarjeta_gigante("Fallas Críticas Sensor", str(alertas_hoy), "⚠️", color_falla), unsafe_allow_html=True)

            # --- GRÁFICA CORREGIDA ---
            st.markdown("<br>", unsafe_allow_html=True)
            df_2h = df[df['timestamp'] >= (ultimo_timestamp - pd.Timedelta(hours=2))].copy()
            df_2h['Local'] = df_2h['temp_local'].rolling(window=10, min_periods=1).mean()
            df_2h['Internet'] = df_2h['temp_internet'].rolling(window=10, min_periods=1).mean()
            df_melted = df_2h.melt(id_vars=['timestamp'], value_vars=['Local', 'Internet'], var_name='Origen', value_name='Temp')
            
            # La leyenda se configura de forma nativa en alt.Legend (Sin configure_legend que daba el error)
            chart = alt.Chart(df_melted).mark_line(strokeWidth=4, interpolate='monotone').encode(
                x=alt.X('timestamp:T', title='', axis=alt.Axis(format='%H:%M', grid=False, labelFontSize=14)),
                y=alt.Y('Temp:Q', scale=alt.Scale(zero=False), title='Celsius (°C)', axis=alt.Axis(grid=True)),
                color=alt.Color('Origen:N', 
                                scale=alt.Scale(domain=['Local', 'Internet'], range=['#f59e0b', '#3b82f6']),
                                legend=alt.Legend(
                                    title=None, 
                                    orient='top',           # Coloca la leyenda arriba (generalmente se auto-centra)
                                    direction='horizontal', # Las pone una al lado de la otra
                                    labelFontSize=15, 
                                    symbolSize=120, 
                                    padding=10
                                )),
                tooltip=['timestamp:T', 'Origen:N', alt.Tooltip('Temp:Q', format='.1f')]
            ).properties(
                title=alt.TitleParams(text='📉 Comparativa Térmica (Últimas 2 Horas)', fontSize=22, anchor='start', offset=30),
                height=400
            ).configure_view(strokeOpacity=0)
            
            st.markdown('<div style="background-color: #ffffff; border-radius: 15px; padding: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.05);">', unsafe_allow_html=True)
            st.altair_chart(chart, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # --- LOG ---
            st.markdown("<br>", unsafe_allow_html=True)
            st.subheader("📋 Log de Auditoría Operativa")
            df_log = df.tail(8).copy().sort_values(by='timestamp', ascending=False)
            def formato_accion(f):
                m = MAPEO_MOTOR.get(f['metodo'], f['metodo'])
                return f"{m} ➔ {'Regar' if f['decision_final']==1 else 'Espera'}"
            df_log['Evento'] = df_log.apply(formato_accion, axis=1)
            st.dataframe(df_log[['timestamp', 'Evento', 'metodo']].rename(columns={'timestamp': 'Fecha/Hora'}), use_container_width=True, hide_index=True)
            
        else:
            st.info("Sincronizando con el servidor...")
            
    time.sleep(2.5)