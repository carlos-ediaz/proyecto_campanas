import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# 1. Carga y limpieza de datos
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
NOMBRE_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def localizar_csv():
    for c in [os.environ.get("DATA_PATH", ""), ROOT / "data" / NOMBRE_CSV, ROOT / NOMBRE_CSV]:
        if c and Path(c).is_file():
            return Path(c)
    for carpeta in (ROOT / "data", ROOT):
        csvs = sorted(carpeta.glob("*.csv"))
        if csvs:
            return csvs[0]
    raise FileNotFoundError(f"No se encontro el CSV. Colocalo en {ROOT / 'data' / NOMBRE_CSV} o define DATA_PATH.")


RENOMBRAR = {
    "customerID": "id_cliente", "gender": "genero", "SeniorCitizen": "adulto_mayor", "Partner": "pareja",
    "Dependents": "dependientes", "tenure": "antiguedad_meses", "InternetService": "internet",
    "OnlineSecurity": "seguridad_online", "TechSupport": "soporte_tecnico", "Contract": "contrato",
    "PaperlessBilling": "factura_digital", "PaymentMethod": "metodo_pago", "MonthlyCharges": "cargo_mensual",
    "TotalCharges": "cargo_total", "Churn": "churn",
}

RUTA = localizar_csv()
BRUTO = pd.read_csv(RUTA)
faltan = sorted(set(RENOMBRAR) - set(BRUTO.columns))
if faltan:
    raise ValueError(f"Al CSV {RUTA.name} le faltan las columnas: {faltan}")
DATA = BRUTO[list(RENOMBRAR)].rename(columns=RENOMBRAR).copy()

# Registro de cada paso de limpieza (se muestra en la pestaña "Contexto y datos")
LIMPIEZA = [("Carga del archivo", f"{len(BRUTO):,} clientes y {BRUTO.shape[1]} columnas originales; se usan {len(RENOMBRAR)}.")]

# 1) Espacios sobrantes en textos
for c in DATA.columns:
    if DATA[c].dtype == object or str(DATA[c].dtype).startswith("str"):
        DATA[c] = DATA[c].str.strip()

# 2) Duplicados
n_dup = int(DATA.duplicated().sum()) + int(DATA.id_cliente.duplicated().sum())
DATA = DATA.drop_duplicates(subset="id_cliente")
LIMPIEZA.append(("Duplicados", f"{n_dup} registros duplicados (por fila o por id_cliente)."))

# 3) TotalCharges viene como texto y tiene espacios en blanco -> numerico
DATA["cargo_total"] = pd.to_numeric(DATA.cargo_total, errors="coerce")
sin_cargo = DATA.cargo_total.isna()
n_blank = int(sin_cargo.sum())
n_blank_nuevos = int((sin_cargo & (DATA.antiguedad_meses == 0)).sum())
# Son clientes con 0 meses de antigüedad (aun no se les factura): total = antiguedad x cargo mensual = 0
DATA.loc[sin_cargo, "cargo_total"] = DATA.loc[sin_cargo, "antiguedad_meses"] * DATA.loc[sin_cargo, "cargo_mensual"]
LIMPIEZA.append(("Tipo y vacíos de TotalCharges",
                 f"Columna de texto convertida a número. {n_blank} valores en blanco ({n_blank_nuevos} de ellos con antigüedad 0, "
                 "clientes nuevos sin facturar): se imputan con antigüedad × cargo mensual (= 0) en lugar de eliminar los registros."))

# 4) Otros nulos y rangos validos
n_nulos = int(DATA.isna().sum().sum())
DATA = DATA.dropna()
validos = DATA.antiguedad_meses.between(0, 72) & (DATA.cargo_mensual > 0) & (DATA.cargo_total >= 0)
n_invalidos = int((~validos).sum())
DATA = DATA[validos]
LIMPIEZA.append(("Nulos y rangos", f"{n_nulos} nulos restantes y {n_invalidos} registros fuera de rango "
                 "(antigüedad 0-72 meses, cargos positivos)."))

# 5) Variable objetivo y categorias legibles
DATA["churn"] = (DATA.churn == "Yes").astype(int)
DATA["adulto_mayor"] = DATA.adulto_mayor.map({0: "No", 1: "Sí"})
SI_NO = {"Yes": "Sí", "No": "No", "No internet service": "Sin internet"}
for c in ["pareja", "dependientes", "factura_digital", "seguridad_online", "soporte_tecnico"]:
    DATA[c] = DATA[c].map(SI_NO)
DATA["genero"] = DATA.genero.map({"Female": "Mujer", "Male": "Hombre"})
DATA["internet"] = DATA.internet.map({"DSL": "DSL", "Fiber optic": "Fibra óptica", "No": "Sin internet"})
DATA["contrato"] = DATA.contrato.map({"Month-to-month": "Mensual", "One year": "1 año", "Two year": "2 años"})
DATA["metodo_pago"] = DATA.metodo_pago.map({
    "Electronic check": "Cheque electrónico", "Mailed check": "Cheque por correo",
    "Bank transfer (automatic)": "Transferencia automática", "Credit card (automatic)": "Tarjeta automática"})
DATA["tramo_antiguedad"] = pd.cut(DATA.antiguedad_meses, [-1, 12, 24, 48, 72],
                                  labels=["0-12 meses", "13-24 meses", "25-48 meses", "49-72 meses"])
DATA = DATA.reset_index(drop=True)
TOTAL = len(DATA)
TASA = float(DATA.churn.mean())
LIMPIEZA.append(("Transformaciones", "Churn Yes/No → 1/0; categorías traducidas al español; SeniorCitizen 0/1 → No/Sí; "
                 "antigüedad agrupada en 4 tramos para el análisis."))
LIMPIEZA.append(("Resultado final", f"{TOTAL:,} clientes listos para analizar; {TASA:.1%} abandonó (clases desbalanceadas)."))
ETIQUETA = {1: "Abandona", 0: "Se queda"}

# Variables predictoras: caracteristicas del cliente y del servicio (no incluyen el resultado)
NUM = ["antiguedad_meses", "cargo_mensual", "cargo_total"]
CAT = ["genero", "adulto_mayor", "pareja", "dependientes", "internet", "seguridad_online", "soporte_tecnico",
       "contrato", "factura_digital", "metodo_pago"]
MODA = {c: DATA[c].mode()[0] for c in CAT}

COLORS = {"navy": "#12304A", "blue": "#1976D2", "cyan": "#26A6A1", "gold": "#F4B942", "red": "#D95D5D", "paper": "#F4F7FA"}
PALETA = [COLORS["blue"], COLORS["red"]]


# ---------------------------------------------------------------------------
# 2. Modelos de clasificacion binaria: probabilidades fuera de muestra (validacion cruzada)
# ---------------------------------------------------------------------------
def prep():
    return ColumnTransformer(
        [("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)]
    )


MODELOS = {
    "v1 Regresion logistica (antigüedad)": Pipeline(
        [("sel", ColumnTransformer([("num", StandardScaler(), ["antiguedad_meses"])])),
         ("m", LogisticRegression(max_iter=1000, class_weight="balanced"))]
    ),
    "v2 Regresion logistica multiple": Pipeline(
        [("prep", prep()), ("m", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
    "v3 Random Forest": Pipeline(
        [("prep", prep()), ("m", RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced_subsample",
                                                        random_state=42, n_jobs=-1))]
    ),
}
MODELO_DEFECTO = "v2 Regresion logistica multiple"

X, y = DATA[NUM + CAT], DATA["churn"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
PROB = pd.DataFrame(
    {nombre: cross_val_predict(m, X, y, cv=skf, method="predict_proba")[:, 1] for nombre, m in MODELOS.items()},
    index=DATA.index)


def metricas(real, prob):
    """Metricas de clasificacion con umbral 0.5 sobre la probabilidad de abandono."""
    real = np.asarray(real)
    pred = (np.asarray(prob) >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(real, pred),
        "precision": precision_score(real, pred, zero_division=0),
        "recall": recall_score(real, pred, zero_division=0),
        "f1": f1_score(real, pred, zero_division=0),
        "auc": roc_auc_score(real, prob) if len(np.unique(real)) == 2 else np.nan,
    }


COMPARACION = pd.DataFrame([{"modelo": n, **metricas(y, PROB[n])} for n in MODELOS])
MODELO_DEFECTO = COMPARACION.sort_values("auc", ascending=False).iloc[0].modelo  # el mejor por ROC-AUC

# Importancia por permutacion (modelo ajustado en 80 %, evaluado en 20 %) con AUC
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
_mod = clone(MODELOS[MODELO_DEFECTO]).fit(X_tr, y_tr)
_imp = permutation_importance(_mod, X_te, y_te, n_repeats=10, random_state=42, scoring="roc_auc")
IMPORTANCIA = (
    pd.DataFrame({"variable": X.columns, "importancia": _imp.importances_mean}).sort_values("importancia", ascending=False)
)

# Modelo final (todos los datos) para el simulador
MODELO_FINAL = clone(MODELOS[MODELO_DEFECTO]).fit(X, y)

# ---------------------------------------------------------------------------
# 3. Aplicacion e interfaz
# ---------------------------------------------------------------------------
app = Dash(__name__, title="Abandono de clientes - Telecomunicaciones", suppress_callback_exceptions=True)
server = app.server

ETIQ_COL = {
    "id_cliente": "ID cliente", "contrato": "Contrato", "internet": "Internet", "metodo_pago": "Método de pago",
    "antiguedad_meses": "Antigüedad (meses)", "cargo_mensual": "Cargo mensual ($)", "cargo_total": "Cargo total ($)",
    "adulto_mayor": "Adulto mayor", "clase_real": "Clase real", "clase_predicha": "Clase predicha",
    "prob_churn": "Prob. de abandono", "resultado": "Resultado",
}


def card(title, value_id, note):
    return html.Div([html.P(title, className="kpi-title"), html.H2(id=value_id), html.P(note, className="kpi-note")], className="kpi-card")


def filtro(titulo, id_, valores, placeholder):
    return [html.Label(titulo), dcc.Dropdown(valores, multi=True, id=id_, placeholder=placeholder)]


STYLE_TABLA = dict(
    style_table={"overflowX": "auto"}, style_cell={"fontFamily": "Arial", "fontSize": 13, "padding": "8px"},
    style_header={"backgroundColor": COLORS["navy"], "color": "white", "fontWeight": "bold"})


def tab_contexto():
    filas = [{"paso": p, "detalle": d} for p, d in LIMPIEZA]
    return html.Div([
        html.Div([
            html.H3("Problema de negocio"),
            html.P("Captar un cliente nuevo cuesta más que conservar uno actual. Una empresa de telecomunicaciones necesita saber "
                   "qué clientes están en riesgo de cancelar su servicio (churn) para actuar antes de perderlos."),
            html.H3("Objetivo"),
            html.P("Identificar qué características del cliente y del servicio se asocian con el abandono y construir un modelo de "
                   "clasificación binaria (¿abandona o se queda?) que permita priorizar acciones de retención."),
            html.H3("Datos"),
            html.P(f"Dataset «Telco Customer Churn» de Kaggle (IBM): {len(BRUTO):,} clientes con datos demográficos, servicios contratados, "
                   "tipo de contrato, facturación y si el cliente abandonó el último mes. Es relevante porque combina variables que la empresa "
                   "puede influir (contrato, método de pago, soporte) con el resultado que quiere evitar."),
        ], className="narrative"),
        html.Div([
            html.H3("Limpieza y preparación de datos"),
            dash_table.DataTable(
                data=filas, columns=[{"name": "Paso", "id": "paso"}, {"name": "Qué se hizo / qué se encontró", "id": "detalle"}],
                style_table={"overflowX": "auto"}, style_header=STYLE_TABLA["style_header"],
                style_cell={"fontFamily": "Arial", "fontSize": 13, "padding": "8px", "textAlign": "left",
                            "whiteSpace": "normal", "height": "auto"},
                style_cell_conditional=[{"if": {"column_id": "paso"}, "width": "220px", "fontWeight": "bold"}]),
        ], className="narrative"),
    ])


def tab_simulador():
    def campo(titulo, id_, opciones, valor):
        return html.Div([html.Label(titulo), dcc.Dropdown(sorted(opciones), value=valor, clearable=False, id=id_)],
                        style={"marginBottom": "10px"})
    return html.Div([
        html.P("Elige el perfil de un cliente y el modelo estima su probabilidad de abandono. Las demás características se fijan en el valor más frecuente."),
        html.Div([
            html.Div([
                campo("Contrato", "sim-contrato", DATA.contrato.unique(), "Mensual"),
                campo("Servicio de internet", "sim-internet", DATA.internet.unique(), "Fibra óptica"),
                campo("Método de pago", "sim-pago", DATA.metodo_pago.unique(), "Cheque electrónico"),
                campo("Soporte técnico", "sim-soporte", DATA.soporte_tecnico.unique(), "No"),
                campo("Seguridad online", "sim-seguridad", DATA.seguridad_online.unique(), "No"),
                campo("Adulto mayor", "sim-adulto", DATA.adulto_mayor.unique(), "No"),
                html.Label("Antigüedad (meses)"),
                dcc.Slider(0, 72, 1, value=6, id="sim-ant", marks={v: str(v) for v in range(0, 73, 12)},
                           tooltip={"placement": "bottom"}),
                html.Br(),
                html.Label("Cargo mensual ($)"),
                dcc.Slider(18, 120, 1, value=70, id="sim-cargo", marks={v: f"${v}" for v in range(20, 121, 20)},
                           tooltip={"placement": "bottom"}),
            ], style={"flex": "1", "minWidth": "280px", "paddingRight": "24px"}),
            html.Div([dcc.Graph(id="sim-graf"), html.Div(id="sim-texto", className="narrative")],
                     style={"flex": "1", "minWidth": "300px"}),
        ], style={"display": "flex", "flexWrap": "wrap"}),
    ])


app.layout = html.Div([
    html.Header([
        html.Div([
            html.P("CASO DE ESTUDIO · TELECOMUNICACIONES", className="eyebrow"),
            html.H1("Abandono de clientes: ¿quién se va y por qué?"),
            html.P(f"Clasificación binaria sobre {TOTAL:,} clientes ({TASA:.1%} abandonó). Modelos evaluados con validación cruzada. Uso académico.",
                   className="subtitle"),
        ]),
        html.Div("DASH + PLOTLY", className="badge"),
    ], className="hero"),
    html.Div([
        html.Div([
            html.H3("Filtros interactivos"),
            *filtro("Contrato", "contrato", sorted(DATA.contrato.unique()), "Todos los contratos"),
            *filtro("Servicio de internet", "internet", sorted(DATA.internet.unique()), "Todos los servicios"),
            *filtro("Método de pago", "pago", sorted(DATA.metodo_pago.unique()), "Todos los métodos"),
            *filtro("Adulto mayor", "adulto", sorted(DATA.adulto_mayor.unique()), "Todos"),
            html.Label("Antigüedad (meses)"),
            dcc.RangeSlider(0, 72, 1, value=[0, 72], id="antig", marks={v: str(v) for v in range(0, 73, 12)}),
            html.Hr(),
            html.Label("Modelo a evaluar"),
            dcc.Dropdown(list(MODELOS), value=MODELO_DEFECTO, clearable=False, id="modelo"),
            html.P("Probabilidad de abandono con validación cruzada estratificada (5 pliegues): cada cliente se clasifica con un modelo que no lo vio al entrenar. Umbral de decisión: 0.5.", className="help"),
            html.Button("Restablecer filtros", id="reset", n_clicks=0),
        ], className="sidebar"),
        html.Main([
            html.Div([
                card("Clientes", "kpi-n", "población filtrada"),
                card("Tasa de abandono", "kpi-churn", "porcentaje que se fue"),
                card("Cargo mensual medio", "kpi-cargo", "dólares por cliente"),
                card("Antigüedad media", "kpi-ant", "meses"),
                card("Ingreso mensual en riesgo", "kpi-riesgo", "cargos de quienes abandonaron"),
            ], className="kpi-grid"),
            dcc.Tabs([
                dcc.Tab(label="Contexto y datos", children=[tab_contexto()]),
                dcc.Tab(label="Panorama", children=[
                    html.Div([dcc.Graph(id="contrato-chart"), dcc.Graph(id="pago-chart")], className="grid-2"),
                    html.Div([dcc.Graph(id="tramo-chart"), dcc.Graph(id="hist-chart")], className="grid-2"),
                    html.Div(id="insights", className="narrative"),
                ]),
                dcc.Tab(label="Rendimiento del modelo", children=[
                    html.Div(id="metrics-strip", className="metric-strip"),
                    html.Div([dcc.Graph(id="cm-chart"), dcc.Graph(id="roc-chart")], className="grid-2"),
                    html.Div([dcc.Graph(id="model-chart"), dcc.Graph(id="importance-chart")], className="grid-2"),
                    html.Div(id="narrative", className="narrative"),
                ]),
                dcc.Tab(label="Resultados predictivos", children=[
                    html.H3("Clientes activos con mayor riesgo de abandono (prioridad de retención)"),
                    dash_table.DataTable(id="tabla-riesgo", page_size=10, sort_action="native", filter_action="native", **STYLE_TABLA),
                    html.H3("Clientes que el modelo clasifica peor (errores primero)"),
                    dash_table.DataTable(
                        id="tabla-errores", page_size=10, sort_action="native", filter_action="native", **STYLE_TABLA,
                        style_data_conditional=[
                            {"if": {"filter_query": '{resultado} = "Error"', "column_id": "resultado"}, "color": COLORS["red"], "fontWeight": "bold"},
                            {"if": {"filter_query": '{resultado} = "Acierto"', "column_id": "resultado"}, "color": COLORS["cyan"]},
                        ]),
                ]),
                dcc.Tab(label="Simulador", children=[tab_simulador()]),
            ]),
            html.Footer(f"Fuente: Telco Customer Churn, Kaggle ({RUTA.name}). Clasificación fuera de muestra con validación cruzada estratificada de 5 pliegues."),
        ], className="content"),
    ], className="shell"),
], className="app")


# ---------------------------------------------------------------------------
# 4. Callbacks
# ---------------------------------------------------------------------------
def tasa_por(d, col):
    g = d.groupby(col, observed=True, as_index=False).agg(pct=("churn", "mean"), clientes=("id_cliente", "count"))
    g["pct"] = g.pct * 100
    return g


def barra(g, x, titulo, color, ordenar=True):
    if ordenar:
        g = g.sort_values("pct", ascending=False)
    fig = px.bar(g, x=x, y="pct", title=titulo, color_discrete_sequence=[color], hover_data={"clientes": True})
    fig.add_hline(y=TASA * 100, line_dash="dash", line_color=COLORS["gold"], annotation_text=f"Promedio global {TASA:.1%}")
    fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    tope = max(10.0, float(g.pct.max()) * 1.3) if len(g) else 100
    fig.update_layout(template="plotly_white", yaxis_title="% de abandono", xaxis_title="", yaxis_range=[0, tope])
    return fig


def extremos(d, col):
    """Categoria con mas y con menos abandono (solo si hay 2+ categorias con 30+ clientes)."""
    g = tasa_por(d, col)
    g = g[g.clientes >= 30]
    if len(g) < 2:
        return None
    a, b = g.loc[g.pct.idxmax()], g.loc[g.pct.idxmin()]
    return a[col], a.pct / 100, b[col], b.pct / 100


@app.callback(
    Output("contrato", "value"), Output("internet", "value"), Output("pago", "value"), Output("adulto", "value"),
    Output("antig", "value"), Output("modelo", "value"),
    Input("reset", "n_clicks"), prevent_initial_call=True)
def reset_filters(_):
    return [], [], [], [], [0, 72], MODELO_DEFECTO


@app.callback(
    Output("kpi-n", "children"), Output("kpi-churn", "children"), Output("kpi-cargo", "children"),
    Output("kpi-ant", "children"), Output("kpi-riesgo", "children"),
    Output("contrato-chart", "figure"), Output("pago-chart", "figure"), Output("tramo-chart", "figure"), Output("hist-chart", "figure"),
    Output("insights", "children"),
    Output("metrics-strip", "children"), Output("cm-chart", "figure"), Output("roc-chart", "figure"),
    Output("model-chart", "figure"), Output("importance-chart", "figure"), Output("narrative", "children"),
    Output("tabla-riesgo", "data"), Output("tabla-riesgo", "columns"),
    Output("tabla-errores", "data"), Output("tabla-errores", "columns"),
    Input("contrato", "value"), Input("internet", "value"), Input("pago", "value"), Input("adulto", "value"),
    Input("antig", "value"), Input("modelo", "value"))
def update(contratos, internets, pagos, adultos, antig, modelo):
    d = DATA
    if contratos: d = d[d.contrato.isin(contratos)]
    if internets: d = d[d.internet.isin(internets)]
    if pagos: d = d[d.metodo_pago.isin(pagos)]
    if adultos: d = d[d.adulto_mayor.isin(adultos)]
    d = d[d.antiguedad_meses.between(antig[0], antig[1])]
    template = "plotly_white"
    hay = len(d) > 0
    tasa = d.churn.mean() if hay else np.nan

    # --- Panorama -----------------------------------------------------------
    fig_contrato = barra(tasa_por(d, "contrato"), "contrato", "% de abandono por tipo de contrato", COLORS["blue"])
    fig_pago = barra(tasa_por(d, "metodo_pago"), "metodo_pago", "% de abandono por método de pago", COLORS["cyan"])
    fig_tramo = barra(tasa_por(d, "tramo_antiguedad"), "tramo_antiguedad", "% de abandono según antigüedad", COLORS["red"], ordenar=False)

    fig_hist = px.histogram(d.assign(estado=d.churn.map(ETIQUETA)), x="cargo_mensual", color="estado", nbins=30, barmode="overlay",
                            opacity=0.65, title="Distribución del cargo mensual",
                            color_discrete_map={"Abandona": COLORS["red"], "Se queda": COLORS["blue"]})
    fig_hist.update_layout(template=template, xaxis_title="Cargo mensual ($)", yaxis_title="Clientes", legend_title="")

    insights = [html.H3("Hallazgos y recomendaciones (según los filtros activos)")]
    if hay:
        riesgo = d.loc[d.churn == 1, "cargo_mensual"].sum()
        items = [html.Li(f"La vista tiene {len(d):,} clientes y {tasa:.1%} abandonó; esos clientes representaban ${riesgo:,.0f} de ingreso mensual.")]
        e = extremos(d, "contrato")
        if e:
            items.append(html.Li(f"Contrato: con «{e[0]}» abandona {e[1]:.0%} frente a {e[3]:.0%} con «{e[2]}». "
                                 "Recomendación: ofrecer incentivos para migrar a contratos más largos."))
        e = extremos(d, "tramo_antiguedad")
        if e:
            items.append(html.Li(f"Antigüedad: el tramo «{e[0]}» abandona {e[1]:.0%} y «{e[2]}» solo {e[3]:.0%}. "
                                 "Recomendación: reforzar el acompañamiento en los primeros meses del cliente."))
        e = extremos(d, "metodo_pago")
        if e:
            items.append(html.Li(f"Pago: con «{e[0]}» abandona {e[1]:.0%} frente a {e[3]:.0%} con «{e[2]}». "
                                 "Recomendación: promover el pago automático."))
        e = extremos(d, "internet")
        if e:
            items.append(html.Li(f"Internet: «{e[0]}» abandona {e[1]:.0%} y «{e[2]}» {e[3]:.0%}. "
                                 "Recomendación: revisar calidad y precio del servicio con mayor abandono."))
        e = extremos(d, "soporte_tecnico")
        if e:
            items.append(html.Li(f"Soporte técnico: «{e[0]}» abandona {e[1]:.0%} frente a {e[3]:.0%} con «{e[2]}»."))
        insights += [html.Ul(items), html.P("Estos factores muestran asociación, no causalidad.")]
    else:
        insights.append(html.P("No hay registros con estos filtros."))

    # --- Modelo -------------------------------------------------------------
    p = d.assign(prob_churn=PROB.loc[d.index, modelo])
    p["pred"] = (p.prob_churn >= 0.5).astype(int)
    if len(p) >= 3:
        m = metricas(p.churn, p.prob_churn)
        auc_txt = "--" if np.isnan(m["auc"]) else f"{m['auc']:.3f}"
        strip = [html.Div([html.Span("Exactitud (accuracy)"), html.Strong(f"{m['accuracy']:.1%}")]),
                 html.Div([html.Span("Precisión"), html.Strong(f"{m['precision']:.1%}")]),
                 html.Div([html.Span("Sensibilidad (recall)"), html.Strong(f"{m['recall']:.1%}")]),
                 html.Div([html.Span("F1"), html.Strong(f"{m['f1']:.3f}")]),
                 html.Div([html.Span("ROC-AUC"), html.Strong(auc_txt)])]
    else:
        m = {k: np.nan for k in ("accuracy", "precision", "recall", "f1", "auc")}
        strip = [html.Div([html.Span("Sin muestra suficiente"), html.Strong("--")])]

    cm = confusion_matrix(p.churn, p.pred, labels=[0, 1]) if len(p) else np.zeros((2, 2), dtype=int)
    corto = modelo.split(" ")[0]
    fig_cm = px.imshow(cm, text_auto=True, x=["Se queda", "Abandona"], y=["Se queda", "Abandona"],
                       labels=dict(x="Clase predicha", y="Clase real", color="Clientes"),
                       color_continuous_scale=[[0, "#EAF2FB"], [1, COLORS["blue"]]],
                       title=f"Matriz de confusión ({corto})")
    fig_cm.update_layout(template=template, coloraxis_showscale=False)

    fig_roc = go.Figure()
    if p.churn.nunique() == 2:
        fpr, tpr, _ = roc_curve(p.churn, p.prob_churn)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{corto} (AUC {m['auc']:.2f})",
                                     line=dict(color=COLORS["blue"], width=3)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar (AUC 0.50)", line=dict(dash="dash", color=COLORS["navy"])))
    fig_roc.update_layout(template=template, title="Curva ROC", xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")

    comp = COMPARACION.copy()
    comp["seleccionado"] = np.where(comp.modelo == modelo, "Seleccionado", "Otros")
    fig_models = px.bar(comp, x="modelo", y="auc", color="seleccionado", text=comp.auc.map(lambda v: f"{v:.3f}"),
                        title="Comparación de iteraciones (ROC-AUC, todos los clientes)",
                        color_discrete_map={"Seleccionado": COLORS["blue"], "Otros": "#B9C7D4"})
    fig_models.add_hline(y=0.5, line_dash="dash", line_color=COLORS["gold"], annotation_text="azar")
    fig_models.update_layout(template=template, xaxis_title="", yaxis_title="ROC-AUC", legend_title="")

    imp = IMPORTANCIA.sort_values("importancia")
    fig_imp = px.bar(imp, x="importancia", y="variable", orientation="h", title=f"Importancia por permutación ({MODELO_DEFECTO.split(' ')[0]})",
                     color_discrete_sequence=[COLORS["cyan"]])
    fig_imp.update_layout(template=template, xaxis_title="Caída media en ROC-AUC", yaxis_title="")

    top_var = IMPORTANCIA.iloc[0].variable
    narrativa = [
        html.H3("Lectura ejecutiva"),
        html.P(f"El modelo {modelo} clasifica bien {m['accuracy']:.1%} de los clientes (ROC-AUC: {'--' if np.isnan(m['auc']) else format(m['auc'], '.2f')}). "
               f"Detecta {m['recall']:.0%} de quienes realmente abandonan (recall) y, de los que marca como riesgo, acierta {m['precision']:.0%} (precisión)."
               if not np.isnan(m["accuracy"]) else "La selección es muy pequeña para calcular métricas."),
        html.P(f"Como solo {TASA:.0%} de los clientes abandona, la exactitud por sí sola engaña: un modelo que dijera siempre «se queda» acertaría {1 - TASA:.0%}. "
               "Por eso se usan el recall, el F1 y el ROC-AUC, y los modelos se entrenan con clases balanceadas."),
        html.P(f"La variable con mayor peso predictivo es «{top_var}». Un AUC cercano a 0.50 equivale a adivinar al azar; cuanto más cerca de 1.00, mejor separa a quienes se van de quienes se quedan."),
        html.P("Los factores muestran asociación predictiva, no causalidad."),
    ]

    # --- Tablas -------------------------------------------------------------
    p["clase_real"] = p.churn.map(ETIQUETA)
    p["clase_predicha"] = p.pred.map(ETIQUETA)
    p["resultado"] = np.where(p.pred == p.churn, "Acierto", "Error")
    p["distancia"] = (p.churn - p.prob_churn).abs()
    p["prob_churn"] = p.prob_churn.round(3)

    cols_r = ["id_cliente", "contrato", "internet", "metodo_pago", "antiguedad_meses", "cargo_mensual", "prob_churn"]
    t_riesgo = p[p.churn == 0].sort_values("prob_churn", ascending=False)[cols_r].head(100)
    cols_e = ["id_cliente", "contrato", "internet", "antiguedad_meses", "cargo_mensual", "clase_real", "clase_predicha", "prob_churn", "resultado"]
    t_err = p.sort_values("distancia", ascending=False)[cols_e].head(100)

    return (f"{len(d):,}",
            f"{tasa:.1%}" if hay else "--",
            f"${d.cargo_mensual.mean():,.2f}" if hay else "--",
            f"{d.antiguedad_meses.mean():.1f}" if hay else "--",
            f"${d.loc[d.churn == 1, 'cargo_mensual'].sum():,.0f}" if hay else "--",
            fig_contrato, fig_pago, fig_tramo, fig_hist, insights,
            strip, fig_cm, fig_roc, fig_models, fig_imp, narrativa,
            t_riesgo.to_dict("records"), [{"name": ETIQ_COL[c], "id": c} for c in cols_r],
            t_err.to_dict("records"), [{"name": ETIQ_COL[c], "id": c} for c in cols_e])


@app.callback(
    Output("sim-graf", "figure"), Output("sim-texto", "children"),
    Input("sim-contrato", "value"), Input("sim-internet", "value"), Input("sim-pago", "value"), Input("sim-soporte", "value"),
    Input("sim-seguridad", "value"), Input("sim-adulto", "value"), Input("sim-ant", "value"), Input("sim-cargo", "value"))
def simular(contrato, internet, pago, soporte, seguridad, adulto, ant, cargo):
    fila = {**MODA, "contrato": contrato, "internet": internet, "metodo_pago": pago, "soporte_tecnico": soporte,
            "seguridad_online": seguridad, "adulto_mayor": adulto,
            "antiguedad_meses": ant, "cargo_mensual": cargo, "cargo_total": ant * cargo}
    prob = float(MODELO_FINAL.predict_proba(pd.DataFrame([fila])[NUM + CAT])[0, 1])
    nivel, color = ("bajo", COLORS["cyan"]) if prob < 0.3 else ("medio", COLORS["gold"]) if prob < 0.6 else ("alto", COLORS["red"])
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": " %"}, title={"text": "Probabilidad de abandono"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 30], "color": "#E3F4F3"}, {"range": [30, 60], "color": "#FCF1D8"}, {"range": [60, 100], "color": "#F8E0E0"}]}))
    fig.update_layout(template="plotly_white", height=320, margin=dict(t=60, b=10))
    texto = [html.H3(f"Riesgo {nivel}"),
             html.P(f"Para este perfil, el modelo {MODELO_DEFECTO.split(' ')[0]} estima {prob:.0%} de probabilidad de abandono (promedio real: {TASA:.0%})."),
             html.P("Al entrenar con clases balanceadas la probabilidad tiende a sobrestimar el riesgo; úsala para comparar perfiles entre sí.")]
    return fig, texto


# ---------------------------------------------------------------------------
# 5. Inicio del servidor
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
