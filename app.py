import os
import base64
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import psycopg2
import streamlit as st

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Visor de nivel y predicción",
    layout="wide",
    initial_sidebar_state="expanded",
)

TZ_LOCAL = ZoneInfo("America/Guayaquil")

UMBRAL_MONITOREO = 8.0
UMBRAL_AMARILLA = 8.3
UMBRAL_NARANJA = 8.6
UMBRAL_PELIGRO = 8.9

DEFAULT_ID_ESTACION = int(os.getenv("DEFAULT_ID_ESTACION", "64383"))
DEFAULT_NOMBRE_ESTACION = os.getenv("DEFAULT_NOMBRE_ESTACION", "San Mateo")
DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", "0.7760"))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", "-79.6530"))

OBS_DB = {
    "HOST": os.getenv("OBS_DB_HOST", "10.0.153.201"),
    "PORT": int(os.getenv("OBS_DB_PORT", "5432")),
    "DATABASE": os.getenv("OBS_DB_NAME", "bandahm"),
    "USER": os.getenv("OBS_DB_USER", "jupyter"),
    "PASSWORD": os.getenv("OBS_DB_PASSWORD", "zagG3@rcah"),
}

PRON_DB = {
    "HOST": os.getenv("PRON_DB_HOST", "10.0.153.95"),
    "PORT": int(os.getenv("PRON_DB_PORT", "5432")),
    "DATABASE": os.getenv("PRON_DB_NAME", "bandahm_pron"),
    "USER": os.getenv("PRON_DB_USER", "dpa_pronostico_wrf"),
    "PASSWORD": os.getenv("PRON_DB_PASSWORD", "fR8#x2Lq"),
}

TABLA_OBS = 'automaticas."_014101601h"'
TABLA_PRED = 'hm_model_forecast."_017146601h"'

PDF_BOLETIN_1 = "Boletines/boletin_1.pdf"
PDF_BOLETIN_2 = "Boletines/boletin_2.pdf"


# ============================================================
# FUNCIONES DE ACCESO A DATOS
# ============================================================
def validar_config(params: dict) -> None:
    faltantes = [k for k, v in params.items() if v in (None, "")]
    if faltantes:
        raise ValueError(f"Faltan parámetros de conexión: {', '.join(faltantes)}")


def query_to_dataframe(query: str, params: dict) -> pd.DataFrame:
    validar_config(params)
    conn = None
    cursor = None

    try:
        conn = psycopg2.connect(
            host=params["HOST"],
            port=params["PORT"],
            database=params["DATABASE"],
            user=params["USER"],
            password=params["PASSWORD"],
            connect_timeout=8,
        )
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=colnames)

    except Exception as e:
        raise RuntimeError(
            f"Fallo de conexión/consulta a {params['HOST']}:{params['PORT']} / {params['DATABASE']}: {e}"
        ) from e

    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()


def query_observados(id_estacion: int, fecha_inicio: str, fecha_fin: str) -> str:
    return f"""
    SELECT
        fecha_toma_dato,
        id_estacion,
        "1h" AS observado_horario
    FROM {TABLA_OBS}
    WHERE id_estacion = {id_estacion}
      AND fecha_toma_dato >= '{fecha_inicio}'
      AND fecha_toma_dato <= '{fecha_fin}'
    ORDER BY fecha_toma_dato ASC
    """


def query_predicciones(id_estacion: int, fecha_inicio: str, fecha_fin: str) -> str:
    return f"""
    SELECT
        fecha_toma_dato,
        id_estacion,
        pred_3h
    FROM {TABLA_PRED}
    WHERE id_estacion = {id_estacion}
      AND fecha_toma_dato >= '{fecha_inicio}'
      AND fecha_toma_dato <= '{fecha_fin}'
    ORDER BY fecha_toma_dato ASC
    """


@st.cache_data(ttl=300, show_spinner=False)
def cargar_observados(id_estacion: int, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
    df = query_to_dataframe(query_observados(id_estacion, fecha_inicio, fecha_fin), OBS_DB)

    if df.empty:
        return df

    df["fecha_toma_dato"] = pd.to_datetime(df["fecha_toma_dato"], utc=True, errors="coerce")
    df["observado_horario"] = pd.to_numeric(df["observado_horario"], errors="coerce")
    df = df.dropna(subset=["fecha_toma_dato"]).copy()
    df["fecha_local"] = df["fecha_toma_dato"].dt.tz_convert(TZ_LOCAL)

    return df


@st.cache_data(ttl=300, show_spinner=False)
def cargar_predicciones(id_estacion: int, fecha_inicio: str, fecha_fin: str) -> pd.DataFrame:
    df = query_to_dataframe(query_predicciones(id_estacion, fecha_inicio, fecha_fin), PRON_DB)

    if df.empty:
        return df

    df["fecha_toma_dato"] = pd.to_datetime(df["fecha_toma_dato"], utc=True, errors="coerce")
    df["pred_3h"] = pd.to_numeric(df["pred_3h"], errors="coerce")
    df = df.dropna(subset=["fecha_toma_dato"]).copy()
    df["fecha_emision_local"] = df["fecha_toma_dato"].dt.tz_convert(TZ_LOCAL)
    df["fecha_valida_utc"] = df["fecha_toma_dato"] + pd.Timedelta(hours=3)
    df["fecha_valida_local"] = df["fecha_valida_utc"].dt.tz_convert(TZ_LOCAL)

    return df


# ============================================================
# PROCESAMIENTO
# ============================================================
def calcular_estado(valor: float) -> str:
    if pd.isna(valor):
        return "Sin dato"
    if valor >= UMBRAL_PELIGRO:
        return "Peligro"
    if valor >= UMBRAL_NARANJA:
        return "Alerta naranja"
    if valor >= UMBRAL_AMARILLA:
        return "Alerta amarilla"
    if valor >= UMBRAL_MONITOREO:
        return "Monitoreo"
    return "Normal"


def construir_comparacion(df_obs: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    pred = pd.DataFrame()

    if not df_pred.empty:
        pred = df_pred[["fecha_valida_utc", "fecha_valida_local", "pred_3h"]].copy()
        pred = pred.rename(
            columns={
                "fecha_valida_utc": "fecha_toma_dato",
                "fecha_valida_local": "fecha_local",
            }
        )

    if pred.empty:
        return pd.DataFrame(columns=["fecha_local", "obs_3h", "pred_3h", "error_pred_obs"])

    if not df_obs.empty:
        obs = df_obs[["fecha_toma_dato", "observado_horario"]].copy()
        obs = obs.rename(columns={"observado_horario": "obs_horario"}).sort_values("fecha_toma_dato")
        obs["obs_3h"] = obs["obs_horario"].rolling(window=3, min_periods=3).mean()
        obs = obs[["fecha_toma_dato", "obs_3h"]].copy()
        out = pd.merge(pred, obs, on="fecha_toma_dato", how="left")
    else:
        out = pred.copy()
        out["obs_3h"] = np.nan

    out["error_pred_obs"] = out["pred_3h"] - out["obs_3h"]
    out = out[["fecha_local", "obs_3h", "pred_3h", "error_pred_obs"]]

    return out.sort_values("fecha_local").reset_index(drop=True)


def calcular_metricas(df_comp: pd.DataFrame) -> dict:
    if df_comp.empty:
        return {"mae": np.nan, "rmse": np.nan, "sesgo": np.nan, "n_comp": 0}

    validos = df_comp[["obs_3h", "pred_3h", "error_pred_obs"]].dropna()
    if validos.empty:
        return {"mae": np.nan, "rmse": np.nan, "sesgo": np.nan, "n_comp": 0}

    mae = validos["error_pred_obs"].abs().mean()
    rmse = float(np.sqrt((validos["error_pred_obs"] ** 2).mean()))
    sesgo = validos["error_pred_obs"].mean()
    n_comp = int(validos.shape[0])

    return {"mae": mae, "rmse": rmse, "sesgo": sesgo, "n_comp": n_comp}


def construir_figura(df_obs: pd.DataFrame, df_pred: pd.DataFrame, nombre_estacion: str) -> go.Figure:
    fig = go.Figure()

    max_y_obs = (
        float(df_obs["observado_horario"].max())
        if not df_obs.empty and df_obs["observado_horario"].notna().any()
        else UMBRAL_PELIGRO + 1.5
    )
    max_y_pred = (
        float(df_pred["pred_3h"].max())
        if not df_pred.empty and df_pred["pred_3h"].notna().any()
        else UMBRAL_PELIGRO + 1.5
    )
    max_y = max(max_y_obs, max_y_pred, UMBRAL_PELIGRO + 1.5)

    fig.add_hrect(y0=UMBRAL_MONITOREO, y1=UMBRAL_AMARILLA, fillcolor="rgba(17,17,17,0.07)", line_width=0)
    fig.add_hrect(y0=UMBRAL_AMARILLA, y1=UMBRAL_NARANJA, fillcolor="rgba(242,201,76,0.15)", line_width=0)
    fig.add_hrect(y0=UMBRAL_NARANJA, y1=UMBRAL_PELIGRO, fillcolor="rgba(245,158,11,0.14)", line_width=0)
    fig.add_hrect(y0=UMBRAL_PELIGRO, y1=max_y + 0.2, fillcolor="rgba(217,45,32,0.12)", line_width=0)

    if not df_obs.empty:
        fig.add_trace(
            go.Scatter(
                x=df_obs["fecha_local"],
                y=df_obs["observado_horario"],
                mode="lines+markers",
                name="Observado horario",
                line=dict(width=2.8, color="#1f77b4"),
                marker=dict(size=5, color="#1f77b4"),
            )
        )

    if not df_pred.empty:
        fig.add_trace(
            go.Scatter(
                x=df_pred["fecha_valida_local"],
                y=df_pred["pred_3h"],
                mode="lines+markers",
                name="Predicción válida +3h",
                line=dict(width=2.8, color="#7c68e3", dash="dot"),
                marker=dict(size=6, color="#7c68e3"),
            )
        )

    for nivel, nombre, color in [
        (UMBRAL_MONITOREO, "Monitoreo", "#111827"),
        (UMBRAL_AMARILLA, "Alerta amarilla", "#b45309"),
        (UMBRAL_NARANJA, "Alerta naranja", "#c2410c"),
        (UMBRAL_PELIGRO, "Peligro", "#b91c1c"),
    ]:
        fig.add_hline(
            y=nivel,
            line_dash="dot",
            line_color=color,
            annotation_text=nombre,
            annotation_position="top left",
        )

    fig.update_layout(
        title=f"Nivel observado vs predicción - {nombre_estacion}",
        xaxis_title="Fecha/hora local",
        yaxis_title="Nivel (m)",
        hovermode="x unified",
        height=560,
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )

    return fig


def construir_mapa(nombre_estacion: str, lat: float, lon: float, ultimo_valor: float, estado: str) -> go.Figure:
    color_estado = {
        "Peligro": "#d92d20",
        "Alerta naranja": "#f59e0b",
        "Alerta amarilla": "#f2c94c",
        "Monitoreo": "#111827",
        "Normal": "#2563eb",
        "Sin dato": "#9aa0a6",
    }.get(estado, "#2563eb")

    texto = (
        f"{nombre_estacion}<br>Último nivel: {ultimo_valor:.3f} m"
        if pd.notna(ultimo_valor)
        else f"{nombre_estacion}<br>Sin dato"
    )

    fig = go.Figure(
        go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode="markers+text",
            text=[nombre_estacion],
            textposition="top right",
            marker=dict(size=18, color=color_estado),
            hovertemplate=texto + "<br>Estado: " + estado + "<extra></extra>",
        )
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=9,
        mapbox_center={"lat": lat, "lon": lon},
        margin=dict(l=0, r=0, t=0, b=0),
        height=360,
        paper_bgcolor="white",
    )

    return fig


# ============================================================
# UTILIDADES PDF
# ============================================================
def leer_pdf_bytes(ruta_pdf: str):
    if not os.path.exists(ruta_pdf):
        return None
    with open(ruta_pdf, "rb") as f:
        return f.read()


def mostrar_pdf(pdf_bytes: bytes, height: int = 700):
    base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="{height}"
        type="application/pdf">
    </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)


def mostrar_bloque_pdf(titulo: str, descripcion: str, ruta_pdf: str):
    st.markdown(f"### {titulo}")
    st.write(descripcion)

    pdf_bytes = leer_pdf_bytes(ruta_pdf)

    if pdf_bytes is None:
        st.warning(f"No se encontró el archivo: {ruta_pdf}")
        return

    mostrar_pdf(pdf_bytes, height=800)


# ============================================================
# INTERFAZ
# ============================================================
st.title("Visor de nivel y predicción de nivel a 3 horas")
st.caption(
    f"Estación {DEFAULT_NOMBRE_ESTACION}. Monitoreo operativo, resumen del proyecto, automatización y boletines."
)

lat_estacion = DEFAULT_LAT
lon_estacion = DEFAULT_LON
nombre_estacion = DEFAULT_NOMBRE_ESTACION
id_estacion = DEFAULT_ID_ESTACION

with st.sidebar:
    st.header("Panel de control")

    ventana_horas = st.selectbox(
        "Ventana de análisis",
        options=[24, 48, 72, 96, 168],
        index=1,
        format_func=lambda x: f"{x} horas ({x/24:.0f} días)" if x % 24 == 0 else f"{x} horas",
    )

    refrescar = st.button("Actualizar datos", use_container_width=True)

    st.markdown("---")
    st.markdown(f"**Estación:** {nombre_estacion}")
    st.markdown(f"**ID:** {id_estacion}")

ahora_local = pd.Timestamp.now(tz=TZ_LOCAL)
fecha_fin_local = ahora_local
fecha_inicio_local = ahora_local - pd.Timedelta(hours=int(ventana_horas))

fecha_inicio_obs = fecha_inicio_local.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
fecha_fin = fecha_fin_local.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")
fecha_inicio_pred = (fecha_inicio_local - pd.Timedelta(hours=3)).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S")

if refrescar:
    st.cache_data.clear()

# ============================================================
# CARGA ROBUSTA DE DATOS
# ============================================================
df_obs = pd.DataFrame()
df_pred = pd.DataFrame()
df_comp = pd.DataFrame()
metricas = {"mae": np.nan, "rmse": np.nan, "sesgo": np.nan, "n_comp": 0}

error_obs = None
error_pred = None
error_comp = None

with st.spinner("Consultando datos..."):
    try:
        df_obs = cargar_observados(int(id_estacion), fecha_inicio_obs, fecha_fin)
    except Exception as e:
        error_obs = str(e)

    try:
        df_pred = cargar_predicciones(int(id_estacion), fecha_inicio_pred, fecha_fin)
    except Exception as e:
        error_pred = str(e)

    try:
        df_comp = construir_comparacion(df_obs, df_pred)
        metricas = calcular_metricas(df_comp)
    except Exception as e:
        error_comp = str(e)
        df_comp = pd.DataFrame()
        metricas = {"mae": np.nan, "rmse": np.nan, "sesgo": np.nan, "n_comp": 0}

ultimo_obs = df_obs.iloc[-1]["observado_horario"] if not df_obs.empty else np.nan
ultima_pred = df_pred.iloc[-1]["pred_3h"] if not df_pred.empty else np.nan
ultimo_estado = calcular_estado(ultimo_obs)
ultimo_error = (ultima_pred - ultimo_obs) if pd.notna(ultimo_obs) and pd.notna(ultima_pred) else np.nan

tab1, tab2, tab3, tab4 = st.tabs([
    "Monitoreo operativo",
    "Resumen del proyecto",
    "Automatización",
    "Boletines"
])

with tab1:
    st.subheader("Monitoreo operativo")

    if error_obs:
        st.warning(f"No fue posible consultar datos observados. {error_obs}")

    if error_pred:
        st.warning(f"No fue posible consultar predicciones. {error_pred}")

    if error_comp:
        st.warning(f"No fue posible construir la comparación observación-predicción. {error_comp}")

    if df_obs.empty and df_pred.empty:
        st.info(
            "No hay datos operativos disponibles en este momento. "
            "El resto del dashboard continúa accesible."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Último observado", f"{ultimo_obs:.3f} m" if pd.notna(ultimo_obs) else "Sin dato")
        c2.metric("Última predicción +3h", f"{ultima_pred:.3f} m" if pd.notna(ultima_pred) else "Sin dato")
        c3.metric("Error pred - obs", f"{ultimo_error:+.3f} m" if pd.notna(ultimo_error) else "Sin dato")
        c4.metric("Estado actual", ultimo_estado)

        top_left, top_right = st.columns([2.8, 1.2])

        with top_left:
            try:
                st.plotly_chart(
                    construir_figura(df_obs, df_pred, DEFAULT_NOMBRE_ESTACION),
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"No fue posible construir la gráfica principal: {e}")

        with top_right:
            st.subheader("Estación hidrológica")
            st.write(f"**Nombre:** {DEFAULT_NOMBRE_ESTACION}")
            st.write(f"**ID:** {int(id_estacion)}")

            try:
                st.plotly_chart(
                    construir_mapa(
                        DEFAULT_NOMBRE_ESTACION,
                        float(lat_estacion),
                        float(lon_estacion),
                        ultimo_obs,
                        ultimo_estado
                    ),
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"No fue posible construir el mapa: {e}")

            st.write("**Lectura rápida**")
            st.write(f"Nivel observado: {f'{ultimo_obs:.3f} m' if pd.notna(ultimo_obs) else 'Sin dato'}")
            st.write(f"Predicción +3h: {f'{ultima_pred:.3f} m' if pd.notna(ultima_pred) else 'Sin dato'}")
            st.write(f"Estado actual: {ultimo_estado}")

        st.markdown("### Métricas generales del modelo")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Comparaciones válidas", f"{metricas['n_comp']}")
        m2.metric("MAE", f"{metricas['mae']:.3f} m" if pd.notna(metricas["mae"]) else "Sin dato")
        m3.metric("RMSE", f"{metricas['rmse']:.3f} m" if pd.notna(metricas["rmse"]) else "Sin dato")
        m4.metric("Sesgo", f"{metricas['sesgo']:+.3f} m" if pd.notna(metricas["sesgo"]) else "Sin dato")

        with st.expander("Ver tabla comparativa cada 3 horas", expanded=True):
            if df_comp.empty:
                st.info("No hay comparación disponible para el rango seleccionado.")
            else:
                mostrar = df_comp.copy()
                mostrar["fecha_local"] = pd.to_datetime(mostrar["fecha_local"], errors="coerce")
                st.dataframe(
                    mostrar.sort_values("fecha_local", ascending=False),
                    use_container_width=True,
                    hide_index=True,
                )

with tab2:
    st.subheader("Resumen del proyecto")

    r1, r2 = st.columns(2)

    with r1:
        st.markdown("**Objetivo**")
        st.write(
            "Fortalecer el monitoreo y la alerta temprana en la cuenca del río Esmeraldas "
            "mediante la integración de información hidrometeorológica, modelación predictiva "
            "y visualización operativa."
        )

    with r2:
        st.markdown("**Enfoque del trabajo**")
        st.write(
            "El desarrollo integra consulta de datos, procesamiento, predicción, comparación "
            "con observaciones y visualización en un solo entorno, facilitando el seguimiento "
            "y la comunicación de resultados."
        )

    st.markdown("### Componentes principales")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**Datos**")
        st.write(
            "Integración de información observada y de apoyo para consolidar una base útil para análisis y operación."
        )

    with c2:
        st.markdown("**Modelo**")
        st.write(
            "Generación de predicción de nivel a corto plazo como apoyo al seguimiento hidrológico."
        )

    with c3:
        st.markdown("**Visualización**")
        st.write(
            "Construcción de un visor que permite revisar nivel observado, predicción, estado y métricas."
        )

    with c4:
        st.markdown("**Aplicación**")
        st.write(
            "Uso del sistema como insumo para monitoreo, análisis técnico y apoyo a la toma de decisiones."
        )

    st.markdown("### ¿Qué se ha trabajado?")
    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown("**Desarrollo técnico**")
        st.markdown(
            "- Consulta y organización de datos\n"
            "- Preparación de información para el modelo\n"
            "- Generación de predicción operativa\n"
            "- Comparación con observaciones"
        )

    with a2:
        st.markdown("**Visualización**")
        st.markdown(
            "- Diseño de dashboard\n"
            "- Métricas resumidas\n"
            "- Gráfico de seguimiento\n"
            "- Ubicación geográfica de la estación"
        )

    with a3:
        st.markdown("**Valor agregado**")
        st.markdown(
            "- Comunicación más clara de resultados\n"
            "- Información centralizada\n"
            "- Base para automatización\n"
            "- Soporte a productos institucionales"
        )

    st.info(
        "El proyecto integra datos, procesamiento, predicción y visualización en una sola herramienta, "
        "permitiendo transformar información dispersa en un apoyo operativo más claro y oportuno."
    )

with tab3:
    st.subheader("Automatización del proceso")

    st.markdown("### Flujo general automatizado")
    st.write(
        "Datos observados → Procesamiento → Modelo predictivo → "
        "Almacenamiento → Visualización → Soporte a decisiones"
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.markdown("**1. Ingesta**")
        st.write(
            "Consulta de datos desde las fuentes disponibles y recuperación de información necesaria para el análisis."
        )

    with p2:
        st.markdown("**2. Procesamiento**")
        st.write(
            "Transformación temporal, validación y organización de los datos para su uso en el flujo operativo."
        )

    with p3:
        st.markdown("**3. Predicción**")
        st.write(
            "Ejecución del modelo para estimar el nivel esperado a 3 horas y generar una salida útil para seguimiento."
        )

    with p4:
        st.markdown("**4. Visualización y uso**")
        st.write(
            "Actualización del dashboard, revisión de métricas y disponibilidad de resultados para apoyo técnico."
        )

    st.markdown("### Beneficios operativos")
    b1, b2, b3, b4 = st.columns(4)

    with b1:
        st.write("Reduce trabajo manual")

    with b2:
        st.write("Mejora la oportunidad de la información")

    with b3:
        st.write("Centraliza el seguimiento en un solo visor")

    with b4:
        st.write("Sirve como base para escalar a otros productos")

    z1, z2 = st.columns([1.45, 1])

    with z1:
        st.markdown("**¿Qué permite esta automatización?**")
        st.write(
            "Permite que la consulta de datos, el procesamiento, la generación de predicciones "
            "y la visualización de resultados se integren en un flujo más ordenado, reduciendo "
            "dependencia de pasos manuales y facilitando el monitoreo operativo."
        )

    with z2:
        st.markdown("**Proyección**")
        st.write(
            "Esta base puede ampliarse para fortalecer la actualización periódica, la generación "
            "de productos técnicos y el soporte a procesos institucionales de alerta y seguimiento."
        )

    st.success(
        "La automatización mejora la eficiencia del flujo técnico y también fortalece la comunicación "
        "de resultados y su aprovechamiento operativo."
    )

with tab4:
    st.subheader("Boletines automatizados")

    mostrar_bloque_pdf(
        titulo="Boletín Monitoreo Hidrometeorológico - Esmeraldas diario",
        descripcion=(
            "En el marco del Proyecto AdaptaClima, desde el Instituto Nacional de Meteorología "
            "e Hidrología (INAMHI) nos permitimos remitir el Boletín de Monitoreo Hidrometeorológico "
            "de la provincia de Esmeraldas, el cual presenta información actualizada sobre el "
            "comportamiento de las variables meteorológicas e hidrológicas registradas en diversas "
            "estaciones de la provincia.\n\n"
            "El boletín incluye además el pronóstico de lluvias derivado del modelo numérico WRF, "
            "como insumo técnico para el seguimiento de las condiciones atmosféricas y la planificación "
            "de acciones preventivas."
        ),
        ruta_pdf=PDF_BOLETIN_1
    )

    st.markdown("---")

    mostrar_bloque_pdf(
        titulo="Boletín Monitoreo Hidrometeorológico - Esmeraldas eventual",
        descripcion=(
            "Este boletín eventual comparte información de las últimas 6 horas, siempre y cuando "
            "el nivel del río supere un umbral de precaución.\n\n"
            "En este producto se integra una tabla con valores cada 30 minutos correspondientes a "
            "las últimas 6 horas, junto con la imagen del comportamiento observado del nivel del río, "
            "como apoyo al seguimiento de la situación hidrológica."
        ),
        ruta_pdf=PDF_BOLETIN_2
    )
