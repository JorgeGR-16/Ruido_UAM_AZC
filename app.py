import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.interpolate import griddata
from influxdb_client import InfluxDBClient
from influxdb_client import InfluxDBClient
from datetime import datetime, timedelta


# --- CONFIGURACIÓN DE INFLUXDB ---
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "AQUÍ_TU_TOKEN"
INFLUXDB_ORG = "CADI"
INFLUXDB_BUCKET = "Ruido

# --- CLIENTE ---
client = InfluxDBClient(
    url=INFLUXDB_URL,
    token=INFLUXDB_TOKEN,
    org=INFLUXDB_ORG
)
query_api = client.query_api()


st.set_page_config(page_title="Visualización de Niveles de Sonido", layout="wide")


@st.cache_data(ttl=10)  # Cachea por 10 segundos para evitar saturar Influx
def obtener_datos_realtime(nodos, minutos=10):
    try:
        client = InfluxDBClient(
            url=INFLUXDB_URL,
            token=INFLUXDB_TOKEN,
            org=INFLUXDB_ORG
        )
        query_api = client.query_api()
        dfs = []

        for nodo in nodos:
            query = f'''
                from(bucket: "{INFLUXDB_BUCKET}")
                    |> range(start: -{minutos}m)
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

        df_total = pd.concat(dfs, ignore_index=True)
        df_total["time"] = pd.to_datetime(df_total["time"]) - pd.Timedelta(hours=6)  # Ajuste de UTC a tu zona
        return df_total

    except Exception as e:
        st.error(f"Error al consultar InfluxDB: {e}")
        return pd.DataFrame()


# --- ESTILO PERSONALIZADO ---
st.markdown("""
    <style>
        .stApp {
            padding-top: 0rem !important;
            margin-top: 0rem !important;
        }
        header { 
            visibility: hidden;
        }
        .block-container {
            padding-top: 1rem;
        }
        h1 {
            margin-top: -2rem;
        }
        h2 {
            font-size: 16px !important;
            color: red !important;
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }
        h3, h4, h5, h6 {
            color: black;
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }
        .subheader {
            color: #333;
        }
        .menu-button {
            background-color: #004080;
            color: white;
            padding: 10px 25px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            border: none;
        }
        .menu-button:hover {
            background-color: #0059b3;
        }
    </style>
""", unsafe_allow_html=True)

# --- TÍTULO GENERAL ---
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.title("**Investigación del comportamiento del ruido en un ambiente universitario**")

# --- IMAGEN PRINCIPAL ---
col1, col2, col3 = st.columns([1, 4, 1])
with col2:
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

# --- SECCIONES ---
if seccion_activa == "Introducción":
    st.markdown("### Introducción")
    st.markdown("""
    <div style='text-align: justify;'>
     El presente proyecto tiene como objetivo investigar cómo afecta el ruido ambiental en una zona específica de la universidad mediante la instalación y uso de sonómetros para medir los niveles sonoros.
     El ruido es un factor ambiental que puede influir negativamente en la calidad de vida, el rendimiento académico y la salud de estudiantes y personal universitario...
    
    </div>
    """, unsafe_allow_html=True)


    #st.markdown("""
    #<div style='text-align: justify;'>
    #El sonómetro es un instrumento utilizado para medir el nivel de presión sonora, es decir, la intensidad del sonido en el ambiente. 
    #Sirve para medir la intensidad del sonido, expresada en decibeles (dB). Se utiliza para cuantificar el nivel de ruido en un lugar determinado, ya sea en control de ruido ambiental o laboral, o para evaluar la exposición sonora a la que están sometidas las personas.
    #</div>
    #""", unsafe_allow_html=True)
    
    #st.markdown("""
    #<div style='text-align: justify;'>
    #Está diseñado para simular la respuesta del oído humano mediante filtros (como el filtro A, dBA), que ponderan las frecuencias del sonido.Su importancia radica en que permite
    #cuantificar el ruido ambiental, evaluar el cumplimiento de normativas acústicas, diseñar políticas de control y mitigación del ruido, y proteger la salud pública y el bienestar social.
    #Medir el ruido con sonómetros es fundamental para diagnosticar problemas de contaminación acústica, realizar mapas de ruido en zonas urbanas, evaluar el impacto ambiental de proyectos de infraestructura, proteger zonas sensibles como hospitales, escuelas o áreas naturales y realizar controles laborales en entornos con maquinaria ruidosa.
    #</div>
    #""", unsafe_allow_html=True)
    


    
    st.markdown("### 1.1 Principio de funcionamiento")
    st.markdown("""
    <div style='text-align: justify;'>
    **1. Captación del sonido:**  
    El sonido ambiente es captado por un micrófono de condensador...
    </div>
    """, unsafe_allow_html=True)

    st.latex(r'''
        \text{Nivel de presión sonora (dB)} = 20 \cdot \log_{10} \left(\frac{P}{P_0}\right)
    ''')

    st.markdown("""
        Donde:  
        - \( P \): presión sonora medida  
        - \( P_0 = 20\,\mu\text{Pa} \): presión sonora de referencia
    """, unsafe_allow_html=True)

    st.markdown("### 1.2 Diagrama del dispositivo.")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("Diagrama.png", use_container_width=True)

elif seccion_activa == "Objetivo":
    st.markdown("### Objetivo")
    st.markdown("* Visualizar el comportamiento del sonido en una área específica...")

    #st.markdown("### 2.1 Objetivo General")
    #st.markdown("Diseñar y construir un sonómetro digital que permita medir niveles de presión sonora en tiempo real, facilitando el monitoreo del ruido ambiental con precisión.")
    
    #st.markdown("### 2.2 Objetivos específicos")
    #st.markdown("* Seleccionar y calibrar un sensor  de sonido compatible con microcontroladores.")
    #st.markdown("* Programar el microcontrolador para interpretar los datos de decibeles(dB) y mostrarlos en una interfaz digital.")
    #st.markdown("* Integrar un sistema de visualización en pantalla.")
    #st.markdown("* Evaluar el desempeño del prototipo frente a un sonómetro comercial.")
    #st.markdown("* Medir los niveles de ruido en diferentes puntos del área usando un sonómetro de clase adecuada.")
    #st.markdown("* Registrar y analizar los datos obtenidos para identificar zonas con niveles de ruido.")
    #st.markdown("* Comparar los resultados con los límites establecidos en las normas oficiales.")
    #st.markdown("* Fomentar la concientización sobre la importancia del control del ruido en espacios públicos, escolares o laborales.")

elif seccion_activa == "Desarrollo":
    st.markdown("### Desarrollo del prototipo")
    st.header("*En esta parte veremos el desarrollo del prototipo y su construcción.*")

    #st.markdown("""
    #<div style='text-align: justify;'>
     #La construcción de un sonómetro es un proceso complejo que involucra varias partes, tanto electrónicas como mecánicas, que trabajan juntas para medir el sonido de manera precisa.
     #A continuación, se explican en detalle los elementos que componen un sonómetro.
     #- **Micrófono:** se encarga de captar las ondas sonoras del ambiente y convertirlas en una señal eléctrica.
     #- **Amplificador:** La señal eléctrica generada por el micrófono es extremadamente débil, por lo que debe ser amplificada para que sea procesada correctamente. Este proceso lo lleva a cabo el pre-amplificador, que amplifica la señal de manera lineal sin distorsionarla.
     #- **Filtros de frecuencia:** simula la percepción del oído humano o adaptarse a diferentes tipos de medición.
     #- **Circuito de procesamiento de señales:** cuando la señal ha sido amplificada y filtrada, pasa al circuito de procesamiento que se encarga de convertir la señal analógica en digital y realizar los cálculos necesarios para determinar el nivel de presión sonora.
     #- **Pantalla de visualización:**  es el componente que permite visualizar los resultados de las mediciones. Dependiendo del modelo del sonómetro, puede ser una pantalla LCD o LED.
     #- **Controladores y botones:** tiene una serie de botones o controles para que el usuario ajuste las opciones según sus necesidades.
     #- **Fuente de alimentación:** funcionan con baterías recargables o pilas de 9V. Algunos modelos más grandes pueden tener una fuente de alimentación externa. La duración de la batería es crucial para la portabilidad del sonómetro, especialmente en mediciones de campo.

     #Lo siguiente es mostrar un manual para construir un sonómetro y su diseño.
    #</div>
    #""", unsafe_allow_html=True)
    
    
    #st.markdown("### 3.1 Diseño del modelo ESP32")
    #col1, col2, col3 = st.columns([1, 3, 1])
    #with col2:
        #st.image("ESP32.jpg", use_container_width=True)
        
    #st.markdown("### 3.2 Construcción del sonómetro")
    #st.markdown("### Materiales necesarios")
    #st.markdown("""
            #| Componente     | Descripción                            
            #|----------------|-------------------------------------|
            #| ESP32 T3 V1.6.1        | Microcontrolador                | 
            #| Sensor de sonido (micrófono)      | Detecta presión sonora para convertirla a señal analógica                 | 
            #| Pantalla OLED       | Muestra el nivel de decibeles en tiempo real          | 
            #| Jumpers hembra-hembra/ macho-hembra  | Para las conexiones entre módulos                     | 
            #| Pulsador (botón de control) | Encendido, reinicio o cambio de modo |
            #| Caja impresa en 3D | Para encapsular el dispositivo |
            #| Fuente de alimentación (batería o alimentación USB) | Para darle energía al ESP32 | 
    #""")
    #st.markdown("### Procedimiento de armado")
    #st.markdown("""
    #<div style='text-align: justify;'>
        #1. Conexión del sensor de sonido
        #2. Conexión de la pantalla OLED
        #3. Botón de control
        #4. Código en Arduino
        #5. Montaje físico y carcas
    #</div>
     #""", unsafe_allow_html=True)
    
    

elif seccion_activa == "Resultados":
    st.markdown("### Resultados")

    with st.sidebar:
        st.header("Parámetros de entrada")
        with st.sidebar:
        st.header("Datos en tiempo real desde InfluxDB")
    
        nodos_disponibles = [1, 2, 3, 4]
        nodos_seleccionados = st.multiselect("Selecciona los nodos:", nodos_disponibles, default=nodos_disponibles)
    
        minutos = st.slider("Últimos minutos a consultar", 1, 60, 10)
        
        # Esta función ya tiene manejo de errores por dentro
        df_filtrado = obtener_datos_realtime(nodos_seleccionados, minutos)

            columnas_requeridas = ['_time', 'nodo', '_value']

            if not all(col in df.columns for col in columnas_requeridas):
                st.error("El archivo no contiene las columnas necesarias.")
                df_filtrado = pd.DataFrame()
            else:
                df['_time'] = pd.to_datetime(df['_time'], utc=True, errors='coerce')

                if df['_time'].isna().all():
                    st.error("No se pudieron interpretar las fechas.")
                    df_filtrado = pd.DataFrame()
                else:
                    tiempo_min = df['_time'].min()
                    tiempo_max = df['_time'].max()

                    fecha = st.date_input("Fecha", value=tiempo_min.date(), min_value=tiempo_min.date(), max_value=tiempo_max.date())
                    hora_inicio = st.time_input("Hora de inicio", value=pd.to_datetime('00:00').time())
                    hora_fin = st.time_input("Hora de fin", value=pd.to_datetime('23:59').time())

                    nodos_disponibles = sorted(df["nodo"].unique())
                    nodos_seleccionados = st.multiselect(
                        "Selecciona los nodos:",
                        options=nodos_disponibles,
                        default=nodos_disponibles
                    )

                    fecha_inicio = pd.to_datetime(f"{fecha} {hora_inicio}").tz_localize('UTC')
                    fecha_fin = pd.to_datetime(f"{fecha} {hora_fin}").tz_localize('UTC')

                    df_filtrado = df[
                        (df['_time'] >= fecha_inicio) &
                        (df['_time'] <= fecha_fin) &
                        (df['nodo'].isin(nodos_seleccionados))
                    ]

        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            df_filtrado = pd.DataFrame()

    if not df_filtrado.empty:
        df_filtrado = df_filtrado.copy()

        # Clasificar riesgo
        def clasificar_riesgo(db):
            if db < 85:
                return "Seguro"
            elif db < 100:
                return "Riesgo moderado"
            else:
                return "Peligroso"

        df_filtrado["riesgo"] = df_filtrado["_value"].apply(clasificar_riesgo)
        df_filtrado["hora"] = df_filtrado["_time"].dt.hour

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Mapa de Sonido", 
            "📈 Gráficos por nodo", 
            "🧩 Comparación general", 
            "📊 Análisis estadístico",
            "🧨 Riesgo por hora"
        ])

        with tab1:
            st.markdown("### 💥 Mapa de niveles de sonido ")
    
            st.markdown("""
            Este mapa de calor representa la intensidad del ruido registrado por cada nodo (sensor) a lo largo del tiempo en un día específico.
            
            - **Eje horizontal:** representa los nodos o sensores distribuidos en la zona de medición.
            - **Eje vertical:** representa la hora del día (formato HH:MM).
            - **Colores:** indican el nivel de sonido en decibeles (dB); colores más cálidos (rojos) indican niveles más altos.
            
            Este gráfico permite identificar fácilmente en qué momentos y en qué ubicaciones se presentan niveles de ruido elevados.
            """)
            
            X = df_filtrado['nodo'].astype(int).values
            fecha_base = pd.Timestamp(fecha).tz_localize('UTC')
            tiempos_segundos = (df_filtrado['_time'] - fecha_base).dt.total_seconds().values
            Z = df_filtrado['_value'].astype(float).values

            x_unique = np.unique(X)
            y_unique = np.unique(tiempos_segundos)
            X_grid, Y_grid = np.meshgrid(x_unique, y_unique)
            Z_grid = griddata((X, tiempos_segundos), Z, (X_grid, Y_grid), method='linear')
            Z_grid = np.nan_to_num(Z_grid, nan=np.nanmin(Z_grid))

            fig, ax = plt.subplots(figsize=(10, 6))
            yticks = np.linspace(0, len(y_unique) - 1, num=10, dtype=int)
            yticklabels = [pd.to_datetime(y_unique[i], unit='s').strftime('%H:%M') for i in yticks]

            sb.heatmap(Z_grid, cmap='jet', xticklabels=x_unique, yticklabels=False, ax=ax)
            ax.invert_yaxis()
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels, rotation=0)
            ax.set_xlabel("Nodos")
            ax.set_ylabel("Hora (HH:MM)")
            st.pyplot(fig)

        with tab2:
            st.markdown("""
            En esta sección se muestra la evolución del nivel de ruido a lo largo del tiempo para cada nodo seleccionado.
            Esto permite observar tendencias, picos o patrones específicos de ruido en cada sensor.
            """)
            st.markdown("#### Evolución temporal por nodo")
            for nodo in sorted(df_filtrado["nodo"].unique()):
                st.subheader(f"Nodo {nodo}")
                datos_nodo = df_filtrado[df_filtrado["nodo"] == nodo]
                st.line_chart(datos_nodo.set_index("_time")["_value"], height=200, use_container_width=True)

        with tab3:
            st.markdown("""
            Aquí se visualizan todos los nodos juntos para comparar sus niveles de ruido en el tiempo.
            Esto facilita detectar diferencias o similitudes en el comportamiento acústico entre distintas áreas.
            """)
            st.markdown("### Comparación general de nodos en un solo gráfico")
            df_pivot = df_filtrado.pivot(index='_time', columns='nodo', values='_value').sort_index()
            st.line_chart(df_pivot, height=300, use_container_width=True)

        with tab4:
            st.markdown("### Análisis estadístico básico por nodo")
            resumen_estadistico = df_filtrado.groupby("nodo")["_value"].agg(
                Minimo="min",
                Maximo="max",
                Media="mean",
                Mediana="median",
                Conteo="count"
            ).round(2)
            st.dataframe(resumen_estadistico, use_container_width=True)
            st.markdown("### Gráfico de valores máximos por nodo")
            st.bar_chart(resumen_estadistico["Maximo"])
        with tab5:
            st.markdown("### **Efectos del ruido en la audición**")
            st.markdown("""
                <div style='text-align: justify;'>
                La sensibilidad al ruido varía de persona a persona. Algunas personas tienen oídos más sensibles, especialmente a ciertas frecuencias (es decir, qué tan graves o agudos son los sonidos). Sin embargo, cualquier sonido lo suficientemente fuerte y prolongado puede dañar la audición, provocando una pérdida auditiva temporal o permanente.
                Proteger tus oídos es clave para mantener una buena salud auditiva, especialmente en ambientes ruidosos o con exposición prolongada.
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🔊 **Rangos de niveles de sonido (dB)**")

            st.markdown("""
            | Nivel (dB)     | Ejemplo                            | Efecto sobre la salud                                  |
            |----------------|-------------------------------------|--------------------------------------------------------|
            | 0–30 dB        | Biblioteca, susurros                | Sin riesgo                                             |
            | 30–60 dB       | Conversación normal                 | Sin riesgo                                             |
            | 60–85 dB       | Tráfico denso, aspiradora          | Riesgo leve si exposición prolongada                   |
            | **85–100 dB**  | Moto, concierto                     | **Puede causar daño si hay exposición prolongada (>8h)** |
            | **100–120 dB** | Sirena ambulancia, martillo neumático | **Daño auditivo posible en minutos**                  |
            """)
            st.markdown("### Distribución de niveles de sonido por hora (clasificados por riesgo auditivo)")
        
            # Clasificación personalizada
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
        
            df_filtrado["rango"] = df_filtrado["_value"].apply(clasificar_rango)
            df_filtrado["hora"] = df_filtrado["_time"].dt.hour
            horas_disponibles = sorted(df_filtrado["hora"].unique())
            
            # Selector de una sola hora (por ejemplo: 13, 14, 15...)
            hora_seleccionada = st.selectbox(
                "Selecciona la hora que deseas visualizar (formato 24h):",
                options=horas_disponibles,  # debe ser una lista de enteros (0 a 23, por ejemplo)
                index=0  # opcional, elige cuál aparece por defecto
            )
            
            # Filtrar datos por la hora seleccionada
            df_hora = df_filtrado[df_filtrado["hora"] == hora_seleccionada]
            conteo = df_hora["rango"].value_counts().sort_index()
            
            # Colores personalizados por rango de riesgo
            colores = {
                "0–30 dB: Sin riesgo": "#b3d9ff",
                "30–60 dB: Sin riesgo": "#80bfff",
                "60–85 dB: Riesgo leve": "#ffcc80",
                "85–100 dB: Riesgo moderado": "#ff9966",
                "100–120+ dB: Peligroso": "#ff4d4d"
            }
            
            # Crear gráfico de pastel
            fig, ax = plt.subplots()
            ax.pie(
                conteo,
                labels=conteo.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=[colores.get(cat, "#cccccc") for cat in conteo.index]
            )
            ax.set_title(f"{hora_seleccionada}:00 hrs — Niveles de sonido por rango")
            st.pyplot(fig)
           

    else:
        st.warning("No hay datos para los parámetros seleccionados.")
