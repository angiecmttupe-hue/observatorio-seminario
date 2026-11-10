# 🇨🇴 Observatorio Departamental de Colombia

Herramienta interactiva de **análisis territorial** construida con datos oficiales de [datos.gov.co](https://www.datos.gov.co). Permite explorar, comparar y modelar indicadores económicos, sociales y de seguridad a nivel departamental.

**Diseñado por:** Jimena Busaquillo · María Fernanda González · Angie Cabezas
**Contexto:** Seminario de Investigación

---

## Propósito

Democratizar el acceso a datos territoriales y ofrecer un espacio de exploración rigurosa para investigadores, estudiantes, tomadores de decisión y ciudadanía interesada en comprender las dinámicas económicas y sociales de los departamentos de Colombia.

## Fuentes de datos

Todos los datos se descargan en tiempo real desde la API de datos abiertos (Socrata). No se utilizan datos sintéticos.

| Indicador | Fuente | Dataset | Cobertura aprox. |
|-----------|--------|---------|------------------|
| PIB (miles de millones, precios constantes 2015) | DANE | `kgyi-qc7j` | 2005 – 2023 |
| Homicidios (víctimas) | MinDefensa | `m8fd-ahd9` | 2015 – 2025 |
| Recursos públicos DNP (COP) | DNP | `wc86-9j4a` | 2008 – 2025 |
| Accesos fijos a Internet | MinTIC | `hwrp-3su3` | 2023 |
| Víctimas del conflicto armado | UARIV | `u8ku-2jh2` | 2024 – 2025 |

Endpoint base: `https://www.datos.gov.co/resource/<dataset>.json`

## Funcionalidades

La aplicación (Streamlit) se organiza en pestañas:

1. **🏠 Presentación**: descripción, fuentes, propósito y créditos.
2. **📊 Análisis Exploratorio (EDA)**: estadísticas descriptivas (n, media, mediana, desviación, mínimo, máximo, cuartiles), tendencias temporales por departamento, ranking Top 15 y tabla de datos.
3. **🧩 PCA**: reducción de dimensionalidad (2 o 3 componentes), varianza explicada, *loadings* y gráfico de dispersión.
4. **🤖 ML + Deep Learning**: Ridge y Lasso (con validación de `alpha`, coeficientes interpretables), SVR (kernel RBF) y MLP (capas 64-32). Métricas: R² y RMSE sobre un 25 % de prueba.
5. **📈 Modelo VAR**: selección de rezagos por AIC y pronóstico a 3 períodos.
6. **🔗 Modelo VECM**: prueba de cointegración de Johansen y estimación del VECM cuando existe al menos una relación de cointegración (5 %).
7. **🗺️ Mapas coropléticos**: mapa por departamento y año; si el GeoJSON no está disponible, muestra un ranking de barras como alternativa.

## Estructura del código

| Sección | Descripción |
|---------|-------------|
| 1. Normalización | `quitar_tildes`, `normalizar_departamento`, diccionarios `DEPARTAMENTOS_CANONICOS` y `CODIGOS_DANE` (nombres, variantes ortográficas y códigos DANE). |
| 2. Carga de datos | `cargar_pib`, `cargar_homicidios`, `cargar_dnp`, `cargar_internet`, `cargar_victimas` y `construir_maestro`, que unifica todo en un DataFrame con columnas `departamento`, `anio`, `valor`, `indicador`, `fuente`. |
| 3. Análisis | `filtrar`, `stats_descriptivas`, `tendencia`, `barras_top`, `aplicar_pca`, `regressiones_regularizadas`, `svm_y_dl`, `estimar_var`, `estimar_vecm`. |
| 4. Mapas | `obtener_geojson`, `MAPEO_NOMBRES_GEOJSON`, `mapa_coropletico`. |
| 5. Presentación | `resultado_pca`, `resultado_ml`, `resultado_var`, `resultado_vecm`. |
| 6. Interfaz | Streamlit con pestañas (`st.tabs`), datos en caché (`st.cache_data`) y descarga de las 5 fuentes en paralelo. |

## Procesamiento de datos

- Filtro del maestro: años entre 2005 y 2025, valores > 0 y solo departamentos canónicos (se excluye "Nacional").
- PIB: solo registros a precios constantes, agregados por departamento y año.
- Homicidios: suma de víctimas por año de `fecha_hecho`.
- VAR y VECM usan únicamente indicadores de serie larga: PIB, Homicidios y Recursos DNP.

## Requisitos

- Python 3.9 o superior
- Dependencias listadas en `requirements.txt`

## Instalación y ejecución

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Despliegue en Streamlit Community Cloud:** sube `app.py` y `requirements.txt` a la raíz del repositorio de GitHub y apunta la app a `app.py`. La primera carga descarga los datos (1–2 min); luego quedan en caché por 6 horas. El botón *Recargar datos* de la barra lateral fuerza una nueva descarga.

Requiere conexión a internet para consultar datos.gov.co y el GeoJSON.

## Limitaciones

- Dependen de la disponibilidad de la API de datos.gov.co (timeout de 90 s por fuente).
- *Accesos a Internet* y *Víctimas del conflicto* tienen cobertura temporal corta, por lo que no se usan en VAR/VECM.
- VAR y VECM requieren al menos 10 años de datos solapados y 2 indicadores de serie larga.
- El rango de años de los modelos de series de tiempo se limita a 2015–2023.
- Los resultados son exploratorios y no constituyen evidencia causal.

## Nota metodológica

Todos los datos provienen de fuentes oficiales abiertas. La normalización de nombres de departamentos incluye códigos DANE y variantes ortográficas comunes.
