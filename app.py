# -*- coding: utf-8 -*-
"""
OBSERVATORIO DEPARTAMENTAL DE COLOMBIA  ·  versión Streamlit
Datos oficiales de datos.gov.co · EDA · PCA · Ridge/Lasso · SVR · MLP · VAR · VECM · Mapas
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

warnings.filterwarnings("ignore")

# set_page_config debe ser el PRIMER comando de Streamlit
st.set_page_config(
    page_title="Observatorio Departamental Colombia",
    page_icon="🇨🇴",
    layout="wide",
)

# ============================================================
# 1. NORMALIZACIÓN DE DEPARTAMENTOS
# ============================================================
def quitar_tildes(texto):
    if pd.isna(texto) or texto is None:
        return ""
    t = str(texto).strip().lower()
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    t = t.replace(",", "").replace(".", " ").replace("  ", " ").strip()
    return t

DEPARTAMENTOS_CANONICOS = {
    "amazonas": "Amazonas", "antioquia": "Antioquia", "arauca": "Arauca",
    "atlantico": "Atlántico", "bogota": "Bogotá D.C.", "bogota d c": "Bogotá D.C.",
    "bogota dc": "Bogotá D.C.", "distrito capital": "Bogotá D.C.",
    "bolivar": "Bolívar", "boyaca": "Boyacá", "caldas": "Caldas",
    "caqueta": "Caquetá", "casanare": "Casanare", "cauca": "Cauca",
    "cesar": "Cesar", "choco": "Chocó", "cordoba": "Córdoba",
    "cundinamarca": "Cundinamarca", "guainia": "Guainía", "guaviare": "Guaviare",
    "huila": "Huila", "la guajira": "La Guajira", "magdalena": "Magdalena",
    "meta": "Meta", "narino": "Nariño", "norte de santander": "Norte de Santander",
    "putumayo": "Putumayo", "quindio": "Quindío", "risaralda": "Risaralda",
    "san andres": "San Andrés y Providencia", "san andres y providencia": "San Andrés y Providencia",
    "archipielago de san andres": "San Andrés y Providencia",
    "santander": "Santander", "sucre": "Sucre", "tolima": "Tolima",
    "valle": "Valle del Cauca", "valle del cauca": "Valle del Cauca",
    "vaupes": "Vaupés", "vichada": "Vichada",
}

CODIGOS_DANE = {
    "05": "Antioquia", "5": "Antioquia", "08": "Atlántico", "8": "Atlántico",
    "11": "Bogotá D.C.", "13": "Bolívar", "15": "Boyacá", "17": "Caldas",
    "18": "Caquetá", "19": "Cauca", "20": "Cesar", "23": "Córdoba",
    "25": "Cundinamarca", "27": "Chocó", "41": "Huila", "44": "La Guajira",
    "47": "Magdalena", "50": "Meta", "52": "Nariño", "54": "Norte de Santander",
    "63": "Quindío", "66": "Risaralda", "68": "Santander", "70": "Sucre",
    "73": "Tolima", "76": "Valle del Cauca", "81": "Arauca", "85": "Casanare",
    "86": "Putumayo", "88": "San Andrés y Providencia", "91": "Amazonas",
    "94": "Guainía", "95": "Guaviare", "97": "Vaupés", "99": "Vichada",
}

def normalizar_departamento(nombre):
    if pd.isna(nombre):
        return None
    val = str(nombre).strip()
    if val in CODIGOS_DANE:
        return CODIGOS_DANE[val]
    clave = quitar_tildes(val)
    return DEPARTAMENTOS_CANONICOS.get(clave, val.title() if val else None)

# ============================================================
# 2. CARGA DE DATOS OFICIALES
# ============================================================
def cargar_pib():
    try:
        url = "https://www.datos.gov.co/resource/kgyi-qc7j.json?$limit=50000&$order=a_o DESC"
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df["anio"] = pd.to_numeric(df["a_o"], errors="coerce")
        df["valor"] = pd.to_numeric(df["valor_miles_de_millones_de"], errors="coerce")
        df["departamento"] = df["departamento"].apply(normalizar_departamento)
        df = df[df["tipo_de_precios"].str.contains("constantes", case=False, na=False)]
        agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum().dropna()
        agg["indicador"] = "PIB (miles de millones, constantes 2015)"
        agg["fuente"] = "datos.gov.co/kgyi-qc7j (DANE)"
        print(f"  PIB: {len(agg)} filas | {int(agg['anio'].min())}–{int(agg['anio'].max())}")
        return agg
    except Exception as e:
        print(f"[ERROR PIB] {e}")
        return pd.DataFrame()

def cargar_homicidios():
    try:
        url = "https://www.datos.gov.co/resource/m8fd-ahd9.json?$limit=150000"
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df["fecha"] = pd.to_datetime(df["fecha_hecho"], errors="coerce")
        df["anio"] = df["fecha"].dt.year
        df["departamento"] = df["departamento"].apply(normalizar_departamento)
        df["cantidad"] = pd.to_numeric(df.get("cantidad"), errors="coerce").fillna(1)
        agg = (df.groupby(["departamento", "anio"], as_index=False)["cantidad"]
               .sum().rename(columns={"cantidad": "valor"}).dropna())
        agg["indicador"] = "Homicidios (víctimas)"
        agg["fuente"] = "datos.gov.co/m8fd-ahd9 (MinDefensa)"
        print(f"  Homicidios: {len(agg)} filas | {int(agg['anio'].min())}–{int(agg['anio'].max())}")
        return agg
    except Exception as e:
        print(f"[ERROR HOMICIDIOS] {e}")
        return pd.DataFrame()

def cargar_dnp():
    try:
        url = "https://www.datos.gov.co/resource/wc86-9j4a.json?$limit=80000"
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        df["departamento"] = df["departamento"].apply(normalizar_departamento)
        df["anio"] = pd.to_numeric(df.get("vigencia"), errors="coerce")
        valor_col = next((c for c in ["valorpagado", "valorobligado", "valorcomprometido"] if c in df.columns), None)
        if not valor_col:
            return pd.DataFrame()
        df["valor"] = pd.to_numeric(df[valor_col], errors="coerce")
        agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum().dropna()
        agg["indicador"] = "Recursos públicos DNP (COP)"
        agg["fuente"] = "datos.gov.co/wc86-9j4a (DNP)"
        print(f"  DNP: {len(agg)} filas | {int(agg['anio'].min())}–{int(agg['anio'].max())}")
        return agg
    except Exception as e:
        print(f"[ERROR DNP] {e}")
        return pd.DataFrame()

def cargar_internet():
    try:
        url = "https://www.datos.gov.co/resource/hwrp-3su3.json?$limit=100000"
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        if "departamento" in df.columns:
            df["departamento"] = df["departamento"].apply(normalizar_departamento)
        elif "codigo_dane_departamento" in df.columns:
            df["departamento"] = df["codigo_dane_departamento"].astype(str).apply(normalizar_departamento)
        else:
            col = next((c for c in df.columns if "departamento" in c.lower() or "dane" in c.lower()), None)
            if col:
                df["departamento"] = df[col].astype(str).apply(normalizar_departamento)
            else:
                return pd.DataFrame()
        col_anio = next((c for c in df.columns if c.lower() in ["anno", "anio", "año", "a_o", "year"]), None)
        col_valor = next((c for c in df.columns if "acceso" in c.lower() or "cantidad" in c.lower() or "no_de_accesos" in c.lower()), None)
        if not col_anio or not col_valor:
            return pd.DataFrame()
        df["anio"] = pd.to_numeric(df[col_anio], errors="coerce")
        df["valor"] = pd.to_numeric(df[col_valor], errors="coerce")
        agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum().dropna()
        agg["indicador"] = "Accesos fijos a Internet"
        agg["fuente"] = "MinTIC/hwrp-3su3"
        print(f"  Internet: {len(agg)} filas | {int(agg['anio'].min())}–{int(agg['anio'].max())}")
        return agg
    except Exception as e:
        print(f"[ERROR INTERNET] {e}")
        return pd.DataFrame()

def cargar_victimas():
    try:
        url = "https://www.datos.gov.co/resource/u8ku-2jh2.json?$limit=100000"
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        col_depto = next((c for c in df.columns if "depto" in c.lower() or "departamento" in c.lower()), None)
        if not col_depto:
            return pd.DataFrame()
        df["departamento"] = df[col_depto].apply(normalizar_departamento)
        if "fecha_corte" in df.columns:
            df["anio"] = pd.to_datetime(df["fecha_corte"], errors="coerce").dt.year
        else:
            df["anio"] = 2024
        col_valor = next((c for c in df.columns if "per_" in c.lower() or "victima" in c.lower() or "personas" in c.lower()), None)
        df["valor"] = pd.to_numeric(df[col_valor], errors="coerce") if col_valor else 1
        agg = df.groupby(["departamento", "anio"], as_index=False)["valor"].sum().dropna()
        agg["indicador"] = "Víctimas del conflicto armado"
        agg["fuente"] = "UARIV/u8ku-2jh2"
        print(f"  Víctimas: {len(agg)} filas")
        return agg
    except Exception as e:
        print(f"[ERROR VÍCTIMAS] {e}")
        return pd.DataFrame()

@st.cache_data(ttl=6 * 3600, show_spinner="Descargando datos oficiales de datos.gov.co (puede tardar 1-2 min la primera vez)…")
def construir_maestro():
    print("=== Cargando fuentes oficiales ===")
    funciones = [cargar_pib, cargar_homicidios, cargar_dnp, cargar_internet, cargar_victimas]
    # Las 5 descargas se hacen en paralelo para reducir el tiempo de carga
    with ThreadPoolExecutor(max_workers=5) as ex:
        resultados = list(ex.map(lambda f: f(), funciones))
    partes = [d for d in resultados if not d.empty]
    if not partes:
        return pd.DataFrame(columns=["departamento", "anio", "valor", "indicador", "fuente"])
    maestro = pd.concat(partes, ignore_index=True)
    maestro["anio"] = maestro["anio"].astype(int)
    maestro = maestro.dropna(subset=["departamento", "anio", "valor"])
    maestro = maestro[maestro["valor"] > 0]
    maestro = maestro[maestro["anio"].between(2005, 2025)]
    maestro = maestro[~maestro["departamento"].isin(["Nacional", "0", None])]
    maestro = maestro[maestro["departamento"].isin(set(DEPARTAMENTOS_CANONICOS.values()))]
    print(f"\n=== MAESTRO FINAL: {len(maestro)} registros ===")
    print(maestro.groupby("indicador").size())
    return maestro.sort_values(["indicador", "departamento", "anio"]).reset_index(drop=True)


DF = construir_maestro()


if DF.empty:
    st.error("No se pudieron descargar los datos de datos.gov.co. Intenta recargar en unos minutos.")
    if st.button("🔄 Reintentar"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

lista_deptos = sorted(DF["departamento"].dropna().unique())
lista_inds = sorted(DF["indicador"].unique())
amin = int(DF["anio"].min())
amax = int(DF["anio"].max())

INDS_LARGOS = [
    "PIB (miles de millones, constantes 2015)",
    "Homicidios (víctimas)",
    "Recursos públicos DNP (COP)"
]

def filtrar(df, deptos, indicador, anio_min, anio_max):
    d = df.copy()
    if deptos and "Todos" not in deptos:
        d = d[d["departamento"].isin(deptos)]
    if indicador and indicador != "Todos":
        d = d[d["indicador"] == indicador]
    d = d[(d["anio"] >= anio_min) & (d["anio"] <= anio_max)]
    return d

def stats_descriptivas(df, deptos, indicador, anio_min, anio_max):
    d = filtrar(df, deptos, indicador, anio_min, anio_max)
    if d.empty:
        return "Sin datos para los filtros seleccionados."
    res = d.groupby("indicador")["valor"].agg(
        n="count", media="mean", mediana="median", std="std",
        min="min", max="max",
        q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
    ).round(2).reset_index()
    return res.to_markdown(index=False)

def tendencia(df, deptos, indicador, anio_min, anio_max):
    d = filtrar(df, deptos, indicador, anio_min, anio_max)
    if d.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    fig = px.line(d, x="anio", y="valor", color="departamento", markers=True,
                  title=f"Tendencia – {indicador}", template="plotly_white")
    fig.update_layout(hovermode="x unified", height=500)
    return fig

def barras_top(df, indicador, anio, top_n=15):
    d = df[(df["indicador"] == indicador) & (df["anio"] == anio)]
    if d.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    d = d.nlargest(top_n, "valor")
    fig = px.bar(d, x="valor", y="departamento", orientation="h",
                 title=f"Top {top_n} – {indicador} ({anio})", template="plotly_white",
                 color="valor", color_continuous_scale="Tealgrn")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
    return fig

def preparar_panel_ml(df, anio_min=2015, anio_max=2023):
    d = df[df["anio"].between(anio_min, anio_max)].copy()
    pivot = d.pivot_table(index=["departamento", "anio"], columns="indicador",
                          values="valor", aggfunc="mean").reset_index()
    pivot = pivot.dropna(thresh=len(pivot.columns) - 2)
    return pivot

def aplicar_pca(df, n_components=2, anio_min=2015, anio_max=2023):
    panel = preparar_panel_ml(df, anio_min, anio_max)
    cols = [c for c in panel.columns if c not in ["departamento", "anio"]]
    if len(cols) < 2:
        return None, "Se necesitan ≥2 indicadores"
    X = panel[cols].fillna(panel[cols].median())
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(n_components, len(cols)))
    comp = pca.fit_transform(X_scaled)
    result = panel[["departamento", "anio"]].copy()
    for i in range(comp.shape[1]):
        result[f"PC{i+1}"] = comp[:, i]
    loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(comp.shape[1])], index=cols)
    return result, {"varianza_explicada": pca.explained_variance_ratio_, "loadings": loadings}

def regressiones_regularizadas(df, target, anio_min=2015, anio_max=2023):
    panel = preparar_panel_ml(df, anio_min, anio_max)
    cols = [c for c in panel.columns if c not in ["departamento", "anio"]]
    if target not in cols:
        return None, f"Target '{target}' no disponible"
    preds = [c for c in cols if c != target]
    data = panel.dropna(subset=[target] + preds)
    X, y = data[preds], data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(X_train, y_train)
    lasso = LassoCV(alphas=[0.1, 1.0, 10.0, 100.0], max_iter=5000).fit(X_train, y_train)
    return {
        "Ridge": {"alpha": ridge.alpha_, "coef": dict(zip(preds, ridge.coef_.round(4))),
                  "R2": round(r2_score(y_test, ridge.predict(X_test)), 4),
                  "RMSE": round(np.sqrt(mean_squared_error(y_test, ridge.predict(X_test))), 2)},
        "Lasso": {"alpha": lasso.alpha_, "coef": dict(zip(preds, lasso.coef_.round(4))),
                  "R2": round(r2_score(y_test, lasso.predict(X_test)), 4),
                  "RMSE": round(np.sqrt(mean_squared_error(y_test, lasso.predict(X_test))), 2)}
    }, None

def svm_y_dl(df, target, anio_min=2015, anio_max=2023):
    panel = preparar_panel_ml(df, anio_min, anio_max)
    cols = [c for c in panel.columns if c not in ["departamento", "anio"]]
    if target not in cols:
        return None, "Target no encontrado"
    preds = [c for c in cols if c != target]
    data = panel.dropna(subset=[target] + preds)
    X, y = data[preds], data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    svr = SVR(kernel="rbf", C=10, epsilon=0.1).fit(X_train_s, y_train)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42).fit(X_train_s, y_train)
    return {
        "SVR": {"R2": round(r2_score(y_test, svr.predict(X_test_s)), 4),
                "RMSE": round(np.sqrt(mean_squared_error(y_test, svr.predict(X_test_s))), 2)},
        "MLP (Deep Learning)": {"R2": round(r2_score(y_test, mlp.predict(X_test_s)), 4),
                                "RMSE": round(np.sqrt(mean_squared_error(y_test, mlp.predict(X_test_s))), 2)}
    }, None

def estimar_var(df, deptos, inds, maxlags=2, anio_min=2015, anio_max=2023):
    inds = [i for i in inds if i in INDS_LARGOS]
    if len(inds) < 2:
        return None, "Usa al menos 2 de: PIB, Homicidios, Recursos DNP (tienen serie larga)"
    d = df[(df["departamento"].isin(deptos)) & (df["indicador"].isin(inds)) &
           (df["anio"].between(anio_min, anio_max))]
    serie = d.groupby(["anio", "indicador"])["valor"].mean().unstack().dropna()
    if len(serie) < maxlags + 5:
        return None, "Datos insuficientes (necesitas ≥10 años solapados)"
    model = VAR(serie)
    res = model.fit(model.select_order(maxlags=maxlags).aic)
    forecast = res.forecast(serie.values[-res.k_ar:], steps=3)
    fdf = pd.DataFrame(forecast, columns=serie.columns,
                       index=range(serie.index[-1]+1, serie.index[-1]+4))
    return {"lags": res.k_ar, "aic": res.aic, "forecast": fdf.round(2)}, None

def estimar_vecm(df, deptos, inds, anio_min=2015, anio_max=2023):
    inds = [i for i in inds if i in INDS_LARGOS]
    if len(inds) < 2:
        return None, "Usa PIB + Homicidios o PIB + DNP"
    d = df[(df["departamento"].isin(deptos)) & (df["indicador"].isin(inds)) &
           (df["anio"].between(anio_min, anio_max))]
    serie = d.groupby(["anio", "indicador"])["valor"].mean().unstack().dropna()
    if len(serie) < 10:
        return None, "Se necesitan ≥10 observaciones"
    joh = coint_johansen(serie, det_order=0, k_ar_diff=1)
    n_coint = sum(joh.lr1 > joh.cvt[:, 1])
    res = {"n_coint": int(n_coint), "traza": joh.lr1.round(3).tolist(),
           "criticos": joh.cvt[:, 1].round(3).tolist()}
    if n_coint >= 1:
        vecm = VECM(serie, k_ar_diff=1, coint_rank=n_coint, deterministic="ci").fit()
        res["summary"] = str(vecm.summary())
    else:
        res["summary"] = "No hay cointegración al 5%. No se estima VECM."
    return res, None

# ============================================================
# MAPA COROPLÉTICO ROBUSTO (corregido)
# ============================================================

@st.cache_resource(ttl=24 * 3600, show_spinner=False)
def obtener_geojson():
    """Carga el GeoJSON más confiable disponible"""
    urls = [
        "https://raw.githubusercontent.com/santiblanko/colombia.geojson/master/depto.json",
        "https://raw.githubusercontent.com/caticoa3/colombia_mapa/master/co_2018_MGN_DPTO_POLITICO.geojson",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=25)
            if r.status_code == 200:
                data = r.json()
                print(f"✅ GeoJSON cargado desde: {url}")
                return data
        except Exception as e:
            print(f"Error con {url}: {e}")
            continue
    print("⚠️ No se pudo cargar ningún GeoJSON → se usará ranking de barras")
    return None


geojson_col = obtener_geojson()

# Mapeo de nombres de nuestro DataFrame → nombres que suelen venir en los GeoJSON
MAPEO_NOMBRES_GEOJSON = {
    "Amazonas": "AMAZONAS",
    "Antioquia": "ANTIOQUIA",
    "Arauca": "ARAUCA",
    "Atlántico": "ATLANTICO",
    "Bogotá D.C.": "BOGOTA D.C.",
    "Bolívar": "BOLIVAR",
    "Boyacá": "BOYACA",
    "Caldas": "CALDAS",
    "Caquetá": "CAQUETA",
    "Casanare": "CASANARE",
    "Cauca": "CAUCA",
    "Cesar": "CESAR",
    "Chocó": "CHOCO",
    "Córdoba": "CORDOBA",
    "Cundinamarca": "CUNDINAMARCA",
    "Guainía": "GUAINIA",
    "Guaviare": "GUAVIARE",
    "Huila": "HUILA",
    "La Guajira": "LA GUAJIRA",
    "Magdalena": "MAGDALENA",
    "Meta": "META",
    "Nariño": "NARIÑO",
    "Norte de Santander": "NORTE DE SANTANDER",
    "Putumayo": "PUTUMAYO",
    "Quindío": "QUINDIO",
    "Risaralda": "RISARALDA",
    "San Andrés y Providencia": "ARCHIPIELAGO DE SAN ANDRES",
    "Santander": "SANTANDER",
    "Sucre": "SUCRE",
    "Tolima": "TOLIMA",
    "Valle del Cauca": "VALLE DEL CAUCA",
    "Vaupés": "VAUPES",
    "Vichada": "VICHADA",
}

def mapa_coropletico(indicador, anio):
    d = DF[(DF["indicador"] == indicador) & (DF["anio"] == anio)].copy()

    if d.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Sin datos para ese indicador y año",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#64748b")
        )
        fig.update_layout(template="plotly_white", height=520, paper_bgcolor="#f8fafc")
        return fig

    # Intentar mapa coroplético
    if geojson_col is not None:
        try:
            # Detectar la clave correcta del GeoJSON
            props = geojson_col["features"][0]["properties"]

            if "NOMBRE_DPT" in props:
                key = "NOMBRE_DPT"
            elif "DPTO_CNMBR" in props:
                key = "DPTO_CNMBR"
            else:
                key = list(props.keys())[0]

            # Crear columna temporal con el nombre que espera el GeoJSON
            d["nombre_geo"] = d["departamento"].map(MAPEO_NOMBRES_GEOJSON).fillna(
                d["departamento"].str.upper()
            )

            fig = px.choropleth(
                d,
                geojson=geojson_col,
                locations="nombre_geo",
                featureidkey=f"properties.{key}",
                color="valor",
                color_continuous_scale="Tealgrn",
                title=f"<b>{indicador}</b><br><sup>Año {anio}</sup>",
                hover_name="departamento",
                labels={"valor": "Valor"}
            )

            fig.update_geos(
                fitbounds="locations",
                visible=False,
                bgcolor="rgba(0,0,0,0)"
            )

            fig.update_layout(
                height=620,
                margin=dict(r=10, t=60, l=10, b=10),
                paper_bgcolor="#f8fafc",
                plot_bgcolor="#f8fafc",
                font=dict(family="Segoe UI, sans-serif", size=13),
                title_x=0.5
            )
            return fig

        except Exception as e:
            print(f"Choropleth falló: {e}")

    # ========== FALLBACK PROFESIONAL (siempre funciona) ==========
    d = d.sort_values("valor", ascending=True)

    fig = px.bar(
        d,
        x="valor",
        y="departamento",
        orientation="h",
        color="valor",
        color_continuous_scale="Tealgrn",
        title=f"<b>{indicador}</b> — Ranking departamental {anio}",
        labels={"valor": "Valor", "departamento": ""}
    )

    fig.update_layout(
        height=650,
        template="plotly_white",
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f8fafc",
        font=dict(family="Segoe UI, sans-serif"),
        title_x=0.5,
        margin=dict(l=10, r=30, t=70, b=20),
        coloraxis_colorbar=dict(title="Valor")
    )
    fig.update_yaxes(categoryorder="total ascending")

    return fig


# ============================================================
# 5. FUNCIONES DE PRESENTACIÓN (resultados en texto)
# ============================================================
def resultado_pca(n_comp, a_ini, a_fin):
    result, info = aplicar_pca(DF, n_components=n_comp, anio_min=a_ini, anio_max=a_fin)
    if result is None:
        return str(info), None, None
    var_txt = "  \n".join(f"PC{i+1}: {v*100:.1f}%" for i, v in enumerate(info["varianza_explicada"]))
    loadings_md = info["loadings"].round(3).to_markdown()
    texto = f"**Varianza explicada:**  \n{var_txt}\n\n**Loadings:**\n\n{loadings_md}"
    if "PC3" in result.columns:
        fig = px.scatter_3d(result, x="PC1", y="PC2", z="PC3", color="departamento",
                            hover_data=["anio"], title="PCA – Componentes Principales")
    else:
        fig = px.scatter(result, x="PC1", y="PC2", color="departamento", hover_data=["anio"],
                         title="PCA – Componentes Principales", template="plotly_white")
    return texto, fig, result.head(40)

def resultado_ml(target, a_ini, a_fin):
    res_reg, err = regressiones_regularizadas(DF, target, a_ini, a_fin)
    if err:
        return f"❌ {err}"
    res_dl, _ = svm_y_dl(DF, target, a_ini, a_fin)
    txt = "### Ridge Regression\n"
    txt += f"- Alpha: {res_reg['Ridge']['alpha']} | R²: {res_reg['Ridge']['R2']} | RMSE: {res_reg['Ridge']['RMSE']}\n"
    txt += f"- Coeficientes (interpretabilidad): {res_reg['Ridge']['coef']}\n\n"
    txt += "### Lasso Regression\n"
    txt += f"- Alpha: {res_reg['Lasso']['alpha']} | R²: {res_reg['Lasso']['R2']} | RMSE: {res_reg['Lasso']['RMSE']}\n"
    txt += f"- Coeficientes (selección de variables): {res_reg['Lasso']['coef']}\n\n"
    if res_dl:
        txt += "### SVR + Deep Learning (MLP)\n"
        txt += f"- SVR → R²: {res_dl['SVR']['R2']} | RMSE: {res_dl['SVR']['RMSE']}\n"
        txt += f"- MLP → R²: {res_dl['MLP (Deep Learning)']['R2']} | RMSE: {res_dl['MLP (Deep Learning)']['RMSE']}"
    return txt

def resultado_var(deptos, inds, a_ini, a_fin):
    if not deptos or not inds or len(inds) < 2:
        return "Selecciona ≥1 departamento y ≥2 indicadores (preferiblemente PIB + Homicidios o DNP)."
    res, err = estimar_var(DF, deptos, inds, anio_min=max(a_ini, 2015), anio_max=min(a_fin, 2023))
    if err:
        return f"❌ {err}"
    txt = f"**Lags (AIC):** {res['lags']} | **AIC:** {res['aic']:.2f}\n\n"
    txt += "### Forecast 3 períodos (predicción)\n\n" + res["forecast"].to_markdown()
    return txt

def resultado_vecm(deptos, inds, a_ini, a_fin):
    if not deptos or not inds or len(inds) < 2:
        return "Selecciona ≥1 departamento y ≥2 indicadores de serie larga.", None
    res, err = estimar_vecm(DF, deptos, inds, anio_min=max(a_ini, 2015), anio_max=min(a_fin, 2023))
    if err:
        return f"❌ {err}", None
    txt = f"**Relaciones de cointegración (5%):** {res['n_coint']}\n\n"
    txt += f"Traza: {res['traza']}  \nCríticos 5%: {res['criticos']}"
    return txt, str(res.get("summary", ""))[:1800]

def rango_default(lo, hi):
    """Valores por defecto de un slider de rango, ajustados a los años disponibles."""
    return (max(amin, lo), min(amax, hi))

def defaults_en(opciones, preferidos):
    return [p for p in preferidos if p in opciones]

# ============================================================
# 6. INTERFAZ STREAMLIT
# ============================================================
st.markdown(
    """
<div style="background: linear-gradient(135deg, #0f766e 0%, #0e7490 50%, #1e40af 100%);
            padding: 24px 28px; border-radius: 14px; color: white; margin-bottom: 16px;">
<h1 style="margin:0; font-size: 1.8rem; color: white;">🇨🇴 Observatorio Departamental de Colombia</h1>
<p style="margin: 6px 0 0 0; opacity: 0.95;">
Datos 100% reales de datos.gov.co · EDA · PCA · Ridge/Lasso · SVM · Deep Learning · VAR · VECM · Mapas
</p>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### ⚙️ Datos")
    st.caption(f"{len(DF):,} registros · {amin}–{amax}")
    if st.button("🔄 Recargar datos"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

tab_pres, tab_eda, tab_pca, tab_ml, tab_var, tab_vecm, tab_mapa = st.tabs([
    "🏠 Presentación", "📊 Análisis Exploratorio", "🧩 PCA",
    "🤖 ML + Deep Learning", "📈 Modelo VAR", "🔗 Modelo VECM", "🗺️ Mapas Coropléticos",
])

# ---------- 0. PRESENTACIÓN ----------
with tab_pres:
    st.markdown("""
## Bienvenidas y bienvenidos al Observatorio Departamental de Colombia

Este observatorio es una herramienta interactiva de **análisis territorial** construida con datos 100% oficiales de **datos.gov.co**. Su propósito es facilitar la exploración, comparación y modelado de indicadores económicos, sociales y de seguridad a nivel departamental.

### ¿Qué contiene el observatorio?

| Indicador | Fuente | Cobertura temporal aproximada |
|-----------|--------|-------------------------------|
| **PIB** (precios constantes 2015) | DANE | 2005 – 2023 |
| **Homicidios** (víctimas) | MinDefensa / Medicina Legal | 2015 – 2025 |
| **Recursos públicos DNP** | Departamento Nacional de Planeación | 2008 – 2025 |
| **Accesos fijos a Internet** | MinTIC | 2023 |
| **Víctimas del conflicto armado** | UARIV | 2024 – 2025 |

### Técnicas disponibles

- **Análisis Exploratorio (EDA)**: estadísticas descriptivas, tendencias temporales y rankings.
- **PCA**: reducción de dimensionalidad y visualización de componentes principales.
- **Machine Learning**: Ridge, Lasso, Support Vector Regression y redes neuronales (MLP).
- **Series temporales**: modelos **VAR** (predicción) y **VECM** (cointegración de Johansen).
- **Mapas coropléticos** (o ranking territorial cuando el GeoJSON no está disponible).

### Propósito

El observatorio busca **democratizar el acceso a datos territoriales** y ofrecer un espacio de exploración rigurosa para investigadores, estudiantes, tomadores de decisión y ciudadanía interesada en comprender las dinámicas económicas y sociales de los departamentos de Colombia.

---

### Diseñadoras del Observatorio
""")
    st.markdown(
        """
<div style="background: #f0fdfa; border-left: 5px solid #0f766e; padding: 18px 22px; border-radius: 8px; margin: 10px 0 20px 0;">
<p style="margin:0; font-size: 1.15rem; font-weight: 600; color: #0f766e;">
Jimena Busaquillo &nbsp;·&nbsp; María Fernanda González &nbsp;·&nbsp; Angie Cabezas
</p>
<p style="margin: 8px 0 0 0; color: #475569; font-size: 0.95rem;">Seminario de Investigación</p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.info("**Nota metodológica**: Todos los datos provienen de fuentes oficiales abiertas. No se utilizan datos sintéticos. "
            "La normalización de nombres de departamentos incluye códigos DANE y variantes ortográficas comunes.")

# ---------- 1. EDA ----------
with tab_eda:
    col_f, col_r = st.columns([1, 2])
    with col_f:
        deptos_eda = st.multiselect(
            "Departamento(s)", ["Todos"] + lista_deptos,
            default=defaults_en(lista_deptos, ["Antioquia", "Bogotá D.C.", "Valle del Cauca"]),
            key="eda_deptos")
        ind_eda = st.selectbox("Indicador", ["Todos"] + lista_inds, index=1, key="eda_ind")
        rango_eda = st.slider("Rango de años", amin, amax, rango_default(2015, amax), key="eda_rango")
        anio_rank = st.slider("Año ranking", amin, amax, min(amax, 2023), key="eda_rank")
    with col_r:
        st.markdown(stats_descriptivas(DF, deptos_eda, ind_eda, rango_eda[0], rango_eda[1]))
        sub_t, sub_r, sub_d = st.tabs(["Tendencias", "Ranking", "Datos"])
        with sub_t:
            st.plotly_chart(tendencia(DF, deptos_eda, ind_eda, rango_eda[0], rango_eda[1]), key="eda_tend")
        with sub_r:
            ind_barras = ind_eda if ind_eda != "Todos" else lista_inds[0]
            st.plotly_chart(barras_top(DF, ind_barras, anio_rank), key="eda_barras")
        with sub_d:
            st.dataframe(filtrar(DF, deptos_eda, ind_eda, rango_eda[0], rango_eda[1]).head(300))

# ---------- 2. PCA ----------
with tab_pca:
    c1, c2 = st.columns([1, 2])
    with c1:
        n_comp = st.slider("Componentes", 2, 3, 2, key="pca_n")
        rango_pca = st.slider("Rango de años", amin, amax, rango_default(2015, 2023), key="pca_rango")
    texto_pca, fig_pca, df_pca = resultado_pca(n_comp, rango_pca[0], rango_pca[1])
    with c2:
        st.markdown(texto_pca)
    if fig_pca is not None:
        st.plotly_chart(fig_pca, key="pca_fig")
        st.dataframe(df_pca)

# ---------- 3. ML ----------
with tab_ml:
    st.markdown("### Ridge · Lasso · SVR · MLP (Deep Learning) · Coeficientes interpretables")
    target_ml = st.selectbox("Variable objetivo", lista_inds, key="ml_target")
    rango_ml = st.slider("Rango de años", amin, amax, rango_default(2015, 2023), key="ml_rango")
    if st.button("Entrenar modelos", type="primary", key="ml_btn"):
        with st.spinner("Entrenando modelos…"):
            st.markdown(resultado_ml(target_ml, rango_ml[0], rango_ml[1]))

# ---------- 4. VAR ----------
with tab_var:
    deptos_var = st.multiselect("Departamento(s)", lista_deptos,
                                default=defaults_en(lista_deptos, ["Antioquia"]), key="var_deptos")
    inds_var = st.multiselect("Indicadores (≥2 de serie larga)", lista_inds,
                              default=defaults_en(lista_inds, INDS_LARGOS[:2]), key="var_inds")
    rango_var = st.slider("Rango de años", amin, amax, rango_default(2015, 2023), key="var_rango")
    if st.button("Estimar VAR + Forecast", type="primary", key="var_btn"):
        with st.spinner("Estimando VAR…"):
            st.markdown(resultado_var(deptos_var, inds_var, rango_var[0], rango_var[1]))

# ---------- 5. VECM ----------
with tab_vecm:
    deptos_vecm = st.multiselect("Departamento(s)", lista_deptos,
                                 default=defaults_en(lista_deptos, ["Antioquia", "Bogotá D.C."]), key="vecm_deptos")
    inds_vecm = st.multiselect("Indicadores (≥2 de serie larga)", lista_inds,
                               default=defaults_en(lista_inds, INDS_LARGOS[:2]), key="vecm_inds")
    rango_vecm = st.slider("Rango de años", amin, amax, rango_default(2015, 2023), key="vecm_rango")
    if st.button("Estimar VECM (Johansen)", type="primary", key="vecm_btn"):
        with st.spinner("Estimando VECM…"):
            txt_vecm, resumen_vecm = resultado_vecm(deptos_vecm, inds_vecm, rango_vecm[0], rango_vecm[1])
        st.markdown(txt_vecm)
        if resumen_vecm:
            st.code(resumen_vecm)

# ---------- 6. MAPAS ----------
with tab_mapa:
    st.markdown("### Visualización territorial (choropleth o ranking si el GeoJSON no carga)")
    c1, c2 = st.columns(2)
    with c1:
        ind_mapa = st.selectbox("Indicador", lista_inds, key="mapa_ind")
    with c2:
        anio_mapa = st.slider("Año", amin, amax, min(amax, 2023), key="mapa_anio")
    st.plotly_chart(mapa_coropletico(ind_mapa, anio_mapa), key="mapa_fig")

st.markdown(
    """
<div style="text-align:center; margin-top: 20px; color: #64748b; font-size: 0.85rem;">
Datos oficiales · Seminario de Investigación · Diseñado por Jimena Busaquillo, María Fernanda González y Angie Cabezas
</div>
""",
    unsafe_allow_html=True,
)
