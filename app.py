import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.interpolate import griddata
from influxdb_client import InfluxDBClient # Ya no se necesita Point, WritePrecision, SYNCHRONOUS para solo lectura
from datetime import timedelta
from streamlit_autorefresh import st_autorefresh # Opcional para refrescar auto

# Refrescar cada 30s
st_autorefresh(interval=30000, key="refresh")

# --- CONFIGURACIÓN DE INFLUXDB ---
# ¡IMPORTANTE! VERIFICA ESTOS VALORES CON TU CONFIGURACIÓN DE INFLUXDB REAL
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "0Gogft785BaN9fzPYk3OdVcO8Qlrt3Y39dA3Ug2IwiJk2TDadgIwmc13AFEMoeakBqkmv08zdr7di072VuMICQ=="
# Podría ser "CADI" o "PI" - ¡Asegúrate de usar el correcto!
INFLUXDB_ORG = "PI"
# Podría ser "Ruido" o "ruido_uam_azc_e311" - ¡Asegúrate de usar el correcto!
INFLUXDB_BUCKET = "ruido_uam_azc_e311"

# --- FUNCIÓN DE CONSULTA A INFLUXDB MEJORADA ---
@st.cache_data(ttl=10) # Se mantiene el caché para rendimiento
def obtener_datos(nodos, tipo_rango="ultimos_minutos", minutos=10, fecha_inicio=None, fecha_fin=None):
    """
    Obtiene datos de InfluxDB para nodos seleccionados.
    Permite consultar los últimos N minutos o un rango de fecha/hora específico.
    """
    try:
        client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )
        query_api = client.query_api()
        dfs = []

        for nodo in nodos:
            # Construir el rango de tiempo dinámicamente
            if tipo_rango == "ultimos_minutos":
                range_query = f'range(start: -{minutos}m)'
            elif tipo_rango == "rango_fechas" and fecha_inicio and fecha_fin:
                # InfluxDB espera formato ISO 8601 con Zulu time (UTC)
                start_str = fecha_inicio.isoformat(timespec='seconds') + "Z"
                stop_str = fecha_fin.isoformat(timespec='seconds') + "Z"
                range_query = f'range(start: {start_str}, stop: {stop_str})'
            else:
                st.warning("Tipo de rango no válido o fechas no proporcionadas para la consulta.")
                return pd.DataFrame()

            query = f'''
                from(bucket: "{INFLUXDB_BUCKET}")
                    |> {range_query}
                    |> filter(fn: (r) => r["_measurement"] == "leq")
                    |> filter(fn: (r) => r["nodo"] == "{nodo}")
                    |> aggregateWindow(every: 1s, fn: last, createEmpty: false)
                    |> yield(name: "last")
            '''
            result = query_api.query(org=INFLUXDB_ORG, query=query)

            data = []
            for table in result:
                for record in table.records:
                    data.append({
                        "time": record.get_time(),
                        "nodo": nodo,
                        "value": record.get_value()
                    })

            df_nodo = pd.DataFrame(data)
            dfs.append(df_nodo)

        if dfs:
            df_total = pd.concat(dfs, ignore_index=True)
            # Ajuste de zona horaria: InfluxDB usualmente guarda en UTC.
            # Aquí se resta 6 horas para ajustar a CST (UTC-6), si es tu caso.
            df_total["time"] = pd.to_datetime(df_total["time"]) - pd.Timedelta(hours=6)
            return df_total
        else:
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error al consultar InfluxDB: {e}")
        return pd.DataFrame()

# --- CONFIGURACIÓN DE STREAMLIT ---
st.set_page_config(page_title="Visualización de Niveles de Sonido", layout="wide")

st.markdown("""
    <style>
        header {visibility: hidden;}
        .block-container {padding-top: 1rem;}
        h2 { font-size: 16px !important; color: red !important; }
        .subheader { color: #333; }
        .menu-button {
            background-color: #004080;
            color: white;
            padding: 10px 25px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            border: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- TÍTULO E IMAGEN ---
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.title("**Investigación del comportamiento del ruido en un ambiente universitario**")
    st.image("UAMAZC.jpg", use_container_width=True)

# --- MENÚ DE NAVEGACIÓN ---
if "seccion" not in st.session_state:
    st.session_state.seccion = "Introducción"

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Introducción", use_container_width=True):
        st.session_state.seccion = "Introducción"
with col2:
    if st.button("Objetivo", use_container_width=True):
        st.session_state.seccion = "Objetivo"
with col3:
    if st.button("Desarrollo", use_container_width=True):
        st.session_state.seccion = "Desarrollo"
with col4:
    if st.button("Resultados", use_container_width=True):
        st.session_state.seccion = "Resultados"

seccion_activa = st.session_state.seccion
st.markdown('<p class="subheader">Aplicación de análisis acústico para investigación técnica</p>', unsafe_allow_html=True)

# --- SECCIONES DE CONTENIDO ---
if seccion_activa == "Introducción":
    st.markdown("### Introducción")
    st.write("Aquí puedes agregar el texto de introducción sobre tu investigación.")
    st.write("Esta aplicación permite visualizar y analizar los datos de nivel de sonido (Leq) recolectados por una red de sonómetros en un ambiente universitario.")

elif seccion_activa == "Objetivo":
    st.markdown("### Objetivo")
    st.write("Describe aquí el objetivo principal de tu investigación.")
    st.write("El objetivo es investigar el comportamiento del ruido para identificar patrones y posibles zonas de alto impacto acústico, contribuyendo a la creación de entornos más saludables.")

elif seccion_activa == "Desarrollo":
    st.markdown("### Desarrollo")
    st.write("Detalla el proceso de desarrollo de tu sistema, incluyendo la metodología, hardware y software utilizados.")
    st.write("Se utilizó una red de sonómetros basados en LoRa32 para la recolección de datos, los cuales se almacenan en una base de datos InfluxDB para su posterior análisis.")

elif seccion_activa == "Resultados":
    st.markdown("### Resultados")

    # --- Sidebar para filtros de datos ---
    with st.sidebar:
        st.header("Filtros de datos")
        nodos_disponibles = [1, 2, 3, 4]
        nodos_seleccionados = st.multiselect("Selecciona los nodos:", nodos_disponibles, default=nodos_disponibles)

        opcion_tiempo = st.radio(
            "Selecciona el rango de tiempo:",
            ("Últimos minutos", "Rango de fecha específico"),
            key="tiempo_radio"
        )

        df_filtrado = pd.DataFrame() # Inicializar para evitar errores si no hay datos

        if opcion_tiempo == "Últimos minutos":
            minutos = st.slider("Últimos minutos a consultar", 1, 60, 10, key="minutos_slider")
            df_filtrado = obtener_datos(nodos_seleccionados, tipo_rango="ultimos_minutos", minutos=minutos)
        else:
            # Obtener la fecha y hora actuales para los valores por defecto
            now = pd.Timestamp.now()
            # Valores por defecto para el rango de fecha: hoy de 00:00 a 23:59
            default_start_date = now.date()
            default_end_date = now.date()
            default_start_time = pd.to_datetime("00:00").time()
            default_end_time = pd.to_datetime("23:59").time()


            st.subheader("Seleccionar rango de fecha y hora")
            col_fecha_inicio, col_fecha_fin = st.columns(2)
            with col_fecha_inicio:
                fecha_inicio = st.date_input("Fecha de inicio", value=default_start_date, key="fecha_inicio")
                hora_inicio = st.time_input("Hora de inicio", value=default_start_time, key="hora_inicio")
            with col_fecha_fin:
                fecha_fin = st.date_input("Fecha de fin", value=default_end_date, key="fecha_fin")
                hora_fin = st.time_input("Hora de fin", value=default_end_time, key="hora_fin")

            # Combinar fecha y hora para crear objetos datetime
            datetime_inicio = pd.to_datetime(f"{fecha_inicio} {hora_inicio}")
            datetime_fin = pd.to_datetime(f"{fecha_fin} {hora_fin}")

            # Ajuste de zona horaria si tus inputs son locales y InfluxDB espera UTC
            # Si tus inputs ya son UTC o el ajuste se hace en la función de consulta, esto puede omitirse o ajustarse
            # Por simplicidad, la función `obtener_datos` ya tiene un ajuste de salida.
            # Para la entrada a InfluxDB, los objetos datetime son pasados, y InfluxDB maneja su conversión a UTC si no tienen zona horaria.

            if datetime_inicio >= datetime_fin:
                st.warning("La fecha/hora de inicio debe ser anterior a la fecha/hora de fin. Por favor, corrige las selecciones.")
            else:
                df_filtrado = obtener_datos(nodos_seleccionados, tipo_rango="rango_fechas", fecha_inicio=datetime_inicio, fecha_fin=datetime_fin)

    # --- Contenido principal de Resultados ---
    if not df_filtrado.empty:
        # Botón de descarga CSV
        st.download_button(
            label="Descargar datos como CSV",
            data=df_filtrado.to_csv(index=False).encode('utf-8'),
            file_name=f'datos_ruido_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )

        # Funciones auxiliares para clasificación
        def clasificar_riesgo(db):
            if db < 85:
                return "Seguro"
            elif db < 100:
                return "Riesgo moderado"
            else:
                return "Peligroso"

        def clasificar_rango(db):
            if db < 30:
                return "0–30 dB: Sin riesgo"
            elif db < 60:
                return "30–60 dB: Sin riesgo"
            elif db < 85:
                return "60–85 dB: Riesgo leve"
            elif db < 100:
                return "85–100 dB: Riesgo moderado"
            else:
                return "100–120+ dB: Peligroso"

        df_filtrado["riesgo"] = df_filtrado["value"].apply(clasificar_riesgo)
        df_filtrado["hora"] = df_filtrado["time"].dt.hour

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Mapa de Sonido",
            "📈 Gráficos por nodo",
            "🧩 Comparación general",
            "📊 Análisis estadístico",
            "🧨 Riesgo por hora"
        ])

        with tab1:
            st.markdown("### 💥 Mapa de niveles de sonido")
            # Para el mapa de calor, necesitamos suficientes puntos para interpolar.
            # Si se consulta un rango muy corto, puede que no haya suficientes datos para una buena interpolación.
            if len(df_filtrado) > 10 and len(df_filtrado['nodo'].unique()) > 1: # Asegurar suficientes datos para el mapa
                X = df_filtrado['nodo'].astype(int).values
                # Para un rango de fechas flexible, la base de tiempo debe ser dinámica.
                # Usa el primer tiempo de los datos para la referencia.
                fecha_base = df_filtrado["time"].min()
                tiempos_segundos = (df_filtrado['time'] - fecha_base).dt.total_seconds().values
                Z = df_filtrado['value'].astype(float).values

                # Asegurarse de que hay suficientes puntos únicos para meshgrid y griddata
                if len(np.unique(X)) > 1 and len(np.unique(tiempos_segundos)) > 1:
                    x_unique = np.unique(X)
                    y_unique = np.unique(tiempos_segundos)
                    X_grid, Y_grid = np.meshgrid(x_unique, y_unique)

                    # Si hay duplicados en (X, tiempos_segundos), griddata puede fallar.
                    # Asegurémonos de tener puntos únicos para la interpolación.
                    points = np.column_stack((X, tiempos_segundos))
                    values = Z
                    # Si hay puntos duplicados, se puede promediar los valores o tomar el primero/último
                    # Aquí, asumimos que los datos ya están en su mayoría únicos por la agregación de 1s en InfluxDB
                    # Si sigue habiendo errores aquí, podrías necesitar agrupar y promediar `df_filtrado` antes de este paso.

                    Z_grid = griddata(points, values, (X_grid, Y_grid), method='linear')
                    Z_grid = np.nan_to_num(Z_grid, nan=np.nanmin(Z_grid) if np.nanmin(Z_grid) is not np.nan else 0) # Manejo de NaN

                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Ajustar yticks para que sean legibles en un rango de tiempo largo
                    # Selecciona un número razonable de etiquetas o espaciado
                    num_yticks = min(10, len(y_unique)) # Limita el número de etiquetas
                    if num_yticks > 0:
                        # Asegúrate de que y_unique tenga al menos 2 elementos para linspace funcionar
                        if len(y_unique) > 1:
                            yticks_indices = np.linspace(0, len(y_unique) - 1, num=num_yticks, dtype=int)
                            # Asegúrate de que los índices sean válidos antes de acceder
                            yticks_indices = yticks_indices[yticks_indices < len(y_unique)]
                            yticklabels = [(fecha_base + pd.Timedelta(seconds=y_unique[i])).strftime('%H:%M') for i in yticks_indices]
                            ax.set_yticks(yticks_indices)
                            ax.set_yticklabels(yticklabels)
                        else: # Solo un punto de tiempo único
                            ax.set_yticks([0])
                            ax.set_yticklabels([(fecha_base + pd.Timedelta(seconds=y_unique[0])).strftime('%H:%M')])

                        sb.heatmap(Z_grid, cmap='jet', xticklabels=x_unique, yticklabels=ax.get_yticklabels(), ax=ax)
                        ax.invert_yaxis() # Invertir el eje Y para que el tiempo vaya de abajo hacia arriba o viceversa
                        ax.set_xlabel("Nodos")
                        ax.set_ylabel("Hora (HH:MM)")
                        st.pyplot(fig)
                    else:
                        st.info("No hay suficientes puntos de tiempo únicos para generar el mapa de calor.")
                else:
                    st.info("No hay suficientes datos únicos de nodos o tiempo para generar el mapa de calor.")
            else:
                st.info("No hay suficientes datos o nodos seleccionados para generar el mapa de calor. Asegúrate de tener al menos dos nodos y datos significativos.")


        with tab2:
            st.markdown("#### Evolución temporal por nodo")
            # Ordenar los nodos para una visualización consistente
            for nodo in sorted(df_filtrado["nodo"].unique()):
                st.subheader(f"Nodo {nodo}")
                datos_nodo = df_filtrado[df_filtrado["nodo"] == nodo]
                if not datos_nodo.empty:
                    st.line_chart(datos_nodo.set_index("time")["value"], height=200)
                else:
                    st.info(f"No hay datos para el Nodo {nodo} en el rango seleccionado.")

        with tab3:
            st.markdown("### Comparación general de nodos")
            # Pivotear los datos para tener los nodos como columnas y el tiempo como índice
            df_pivot = df_filtrado.pivot(index='time', columns='nodo', values='value').sort_index()
            if not df_pivot.empty:
                st.line_chart(df_pivot, height=300)
            else:
                st.info("No hay datos para comparar entre nodos en el rango seleccionado.")


        with tab4:
            st.markdown("### Análisis estadístico por nodo")
            if not df_filtrado.empty:
                resumen = df_filtrado.groupby("nodo")["value"].agg(["min", "max", "mean", "median", "count"]).round(2)
                st.dataframe(resumen)

                st.markdown("#### Valor máximo de Leq por nodo")
                # Asegurar que los datos para el bar_chart sean numéricos y no estén vacíos
                if not resumen["max"].empty:
                    st.bar_chart(resumen["max"])
                else:
                    st.info("No hay datos para mostrar el valor máximo por nodo.")
            else:
                st.info("No hay datos para realizar el análisis estadístico.")

        with tab5:
            st.markdown("### Distribución por riesgo auditivo")
            # Aplicar la función de clasificación de rango
            df_filtrado["rango"] = df_filtrado["value"].apply(clasificar_rango)

            if not df_filtrado.empty:
                # Obtener horas únicas disponibles en los datos filtrados
                horas_disponibles = sorted(df_filtrado["hora"].unique())
                if horas_disponibles:
                    hora_sel = st.selectbox("Selecciona la hora (0–23):", horas_disponibles, key="hora_riesgo_select")
                    df_hora = df_filtrado[df_filtrado["hora"] == hora_sel]
                    conteo = df_hora["rango"].value_counts().reindex([
                        "0–30 dB: Sin riesgo",
                        "30–60 dB: Sin riesgo",
                        "60–85 dB: Riesgo leve",
                        "85–100 dB: Riesgo moderado",
                        "100–120+ dB: Peligroso"
                    ], fill_value=0) # Asegura que todos los rangos aparezcan, incluso si no tienen datos

                    colores = {
                        "0–30 dB: Sin riesgo": "#b3d9ff",    # Azul claro
                        "30–60 dB: Sin riesgo": "#80bfff",    # Azul medio
                        "60–85 dB: Riesgo leve": "#ffcc80",   # Naranja claro
                        "85–100 dB: Riesgo moderado": "#ff9966", # Naranja oscuro
                        "100–120+ dB: Peligroso": "#ff4d4d"  # Rojo
                    }

                    # Filtrar conteo para eliminar etiquetas con 0 si no se desean en el pie chart
                    conteo_validos = conteo[conteo > 0]
                    if not conteo_validos.empty:
                        fig, ax = plt.subplots()
                        ax.pie(
                            conteo_validos,
                            labels=conteo_validos.index,
                            autopct="%1.1f%%",
                            colors=[colores.get(k, "#ccc") for k in conteo_validos.index],
                            startangle=90
                        )
                        ax.set_title(f"{hora_sel}:00 hrs — Niveles de sonido por rango de riesgo")
                        ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
                        st.pyplot(fig)
                    else:
                        st.info(f"No hay datos de ruido para la hora {hora_sel}:00 hrs con el rango de riesgo seleccionado.")
                else:
                    st.info("No hay horas disponibles en los datos filtrados para mostrar la distribución por riesgo.")
            else:
                st.info("No hay datos disponibles para mostrar la distribución por riesgo auditivo.")

    else:
        st.warning("No hay datos disponibles en este momento para mostrar o descargar. Por favor, ajusta los filtros o verifica la conexión a InfluxDB.")
    else:
        st.warning("No hay datos para los parámetros seleccionados.")
