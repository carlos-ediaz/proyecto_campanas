"""Dashboard interactivo: clasificacion binaria de campañas de marketing (ROI alto / ROI bajo)."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, dash_table, dcc, html
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
# 1. Carga de datos en memoria
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
NOMBRE_CSV = "marketing_campaign_dataset.csv"
# El dataset original tiene ~200,000 filas: se trabaja con una muestra aleatoria para que
# el entrenamiento sea rapido. Se puede cambiar con la variable de entorno MAX_FILAS.
MAX_FILAS = int(os.environ.get("MAX_FILAS", 20000))


def localizar_csv():
    """Busca el CSV: DATA_PATH, data/<nombre>, <nombre> y, si no, el primer .csv de data/ o de la carpeta."""
    for c in [os.environ.get("DATA_PATH", ""), ROOT / "data" / NOMBRE_CSV, ROOT / NOMBRE_CSV]:
        if c and Path(c).is_file():
            return Path(c)
    for carpeta in (ROOT / "data", ROOT):
        csvs = sorted(carpeta.glob("*.csv"))
        if csvs:
            return csvs[0]
    raise FileNotFoundError(f"No se encontro el CSV. Colocalo en {ROOT / 'data' / NOMBRE_CSV} o define DATA_PATH.")


RENOMBRAR = {
    "Campaign_ID": "id_campana", "Company": "empresa", "Campaign_Type": "tipo_campana",
    "Target_Audience": "audiencia", "Duration": "duracion_dias", "Channel_Used": "canal",
    "Conversion_Rate": "tasa_conversion", "Acquisition_Cost": "costo_adquisicion", "ROI": "roi",
    "Location": "ubicacion",
}

RUTA = localizar_csv()
BRUTO = pd.read_csv(RUTA)
faltan = sorted(set(RENOMBRAR) - set(BRUTO.columns))
if faltan:
    raise ValueError(f"Al CSV {RUTA.name} le faltan las columnas: {faltan}")
DATA = BRUTO[list(RENOMBRAR)].rename(columns=RENOMBRAR)

# Limpieza de tipos: "$16,174.00" -> 16174.0 ; "30 days" -> 30
DATA["costo_adquisicion"] = pd.to_numeric(
    DATA.costo_adquisicion.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
DATA["duracion_dias"] = pd.to_numeric(DATA.duracion_dias.astype(str).str.extract(r"(\d+)")[0], errors="coerce")
DATA["tasa_conversion"] = pd.to_numeric(DATA.tasa_conversion, errors="coerce")
DATA["roi"] = pd.to_numeric(DATA.roi, errors="coerce")
DATA = DATA.dropna(subset=["costo_adquisicion", "duracion_dias", "tasa_conversion", "roi"])
TOTAL_FILAS = len(DATA)

# Variable objetivo binaria: 1 = ROI alto (ROI >= mediana), 0 = ROI bajo.
# La mediana se calcula con todas las filas, antes de tomar la muestra.
UMBRAL = float(DATA.roi.median())
if len(DATA) > MAX_FILAS:
    DATA = DATA.sample(MAX_FILAS, random_state=42)
DATA = DATA.reset_index(drop=True)
DATA["roi_alto"] = (DATA.roi >= UMBRAL).astype(int)
ETIQUETA = {1: "Alto", 0: "Bajo"}

# Variables predictoras: se conocen al planear la campaña.
# Se excluye tasa_conversion porque es un resultado de la campaña (igual que el ROI).
NUM = ["costo_adquisicion", "duracion_dias"]
CAT = ["empresa", "tipo_campana", "audiencia", "canal", "ubicacion"]

COLORS = {"navy": "#12304A", "blue": "#1976D2", "cyan": "#26A6A1", "gold": "#F4B942", "red": "#D95D5D", "paper": "#F4F7FA"}
PALETA = [COLORS["blue"], COLORS["cyan"], COLORS["gold"], COLORS["red"], "#7E57C2", "#8D6E63"]


# ---------------------------------------------------------------------------
# 2. Modelos de clasificacion binaria: probabilidades fuera de muestra (validacion cruzada)
# ---------------------------------------------------------------------------
def prep():
    return ColumnTransformer(
        [("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)]
    )


MODELOS = {
    "v1 Regresion logistica (costo)": Pipeline(
        [("sel", ColumnTransformer([("num", StandardScaler(), ["costo_adquisicion"])])),
         ("m", LogisticRegression(max_iter=1000))]
    ),
    "v2 Regresion logistica multiple": Pipeline([("prep", prep()), ("m", LogisticRegression(max_iter=1000))]),
    "v3 Random Forest": Pipeline(
        [("prep", prep()), ("m", RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=42, n_jobs=-1))]
    ),
}
MODELO_DEFECTO = "v3 Random Forest"

X, y = DATA[NUM + CAT], DATA["roi_alto"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
PROB = pd.DataFrame(
    {nombre: cross_val_predict(m, X, y, cv=skf, method="predict_proba")[:, 1] for nombre, m in MODELOS.items()},
    index=DATA.index)


def metricas(real, prob):
    """Metricas de clasificacion con umbral 0.5 sobre la probabilidad de ROI alto."""
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

# Importancia por permutacion del Random Forest (ajustado en 80 %, evaluado en 20 %) con AUC
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
_rf = MODELOS[MODELO_DEFECTO].fit(X_tr, y_tr)
_imp = permutation_importance(_rf, X_te, y_te, n_repeats=10, random_state=42, scoring="roc_auc")
IMPORTANCIA = (
    pd.DataFrame({"variable": X.columns, "importancia": _imp.importances_mean}).sort_values("importancia", ascending=False)
)

# ---------------------------------------------------------------------------
# 3. Aplicacion e interfaz
# ---------------------------------------------------------------------------
app = Dash(__name__, title="Clasificacion de campañas de marketing", suppress_callback_exceptions=True)
server = app.server

COSTO_MIN = int(DATA.costo_adquisicion.min() // 1000 * 1000)
COSTO_MAX = int(-(-DATA.costo_adquisicion.max() // 1000) * 1000)
PASO_MARCA = max(1000, int(round((COSTO_MAX - COSTO_MIN) / 5 / 1000)) * 1000)
ETIQ_COL = {
    "id_campana": "ID campaña", "empresa": "Empresa", "tipo_campana": "Tipo de campaña", "audiencia": "Audiencia",
    "duracion_dias": "Duración (días)", "canal": "Canal", "ubicacion": "Ubicación", "costo_adquisicion": "Costo de adquisición ($)",
    "tasa_conversion": "Tasa de conversión", "roi": "ROI", "clase_real": "Clase real", "clase_predicha": "Clase predicha",
    "prob_alto": "Prob. ROI alto", "resultado": "Resultado",
}


def card(title, value_id, note):
    return html.Div([html.P(title, className="kpi-title"), html.H2(id=value_id), html.P(note, className="kpi-note")], className="kpi-card")


def filtro(titulo, id_, valores, placeholder):
    return [html.Label(titulo), dcc.Dropdown(valores, multi=True, id=id_, placeholder=placeholder)]


app.layout = html.Div([
    html.Header([
        html.Div([
            html.P("CASO DE ESTUDIO · MARKETING DIGITAL", className="eyebrow"),
            html.H1("Clasificacion de campañas: ¿ROI alto o bajo?"),
            html.P(f"Modelos de clasificacion binaria: ROI alto = ROI de {UMBRAL:.2f} o mas (mediana de {TOTAL_FILAS:,} campañas). "
                   f"Los modelos se entrenan con una muestra aleatoria de {len(DATA):,} campañas. Uso academico.", className="subtitle"),
        ]),
        html.Div("DASH + PLOTLY", className="badge"),
    ], className="hero"),
    html.Div([
        html.Div([
            html.H3("Filtros interactivos"),
            *filtro("Canal", "canal", sorted(DATA.canal.unique()), "Todos los canales"),
            *filtro("Tipo de campaña", "tipo", sorted(DATA.tipo_campana.unique()), "Todos los tipos"),
            *filtro("Ubicacion", "ubicacion", sorted(DATA.ubicacion.unique()), "Todas las ubicaciones"),
            *filtro("Audiencia objetivo", "audiencia", sorted(DATA.audiencia.unique()), "Todas las audiencias"),
            html.Label("Duracion"),
            dcc.Dropdown([{"label": f"{int(v)} dias", "value": int(v)} for v in sorted(DATA.duracion_dias.unique())],
                         multi=True, id="duracion", placeholder="Todas las duraciones"),
            html.Label("Costo de adquisicion ($)"),
            dcc.RangeSlider(COSTO_MIN, COSTO_MAX, 500, value=[COSTO_MIN, COSTO_MAX], id="costo",
                            marks={v: f"{v / 1000:.0f}k" for v in range(COSTO_MIN, COSTO_MAX + 1, PASO_MARCA)}),
            html.Hr(),
            html.Label("Modelo a evaluar"),
            dcc.Dropdown(list(MODELOS), value=MODELO_DEFECTO, clearable=False, id="modelo"),
            html.P("Probabilidad de ROI alto con validacion cruzada estratificada (5 pliegues): cada campaña se clasifica con un modelo que no la vio al entrenar. Umbral de decision: 0.5.", className="help"),
            html.Button("Restablecer filtros", id="reset", n_clicks=0),
        ], className="sidebar"),
        html.Main([
            html.Div([
                card("Campañas", "kpi-n", "poblacion filtrada"),
                card("ROI medio", "kpi-roi", "retorno por campaña"),
                card("Costo de adquisicion medio", "kpi-costo", "dolares por campaña"),
                card("Tasa de conversion media", "kpi-conv", "porcentaje"),
                card("Campañas con ROI alto", "kpi-alto", "porcentaje del total filtrado"),
            ], className="kpi-grid"),
            dcc.Tabs([
                dcc.Tab(label="Panorama", children=[
                    html.Div([dcc.Graph(id="canal-chart"), dcc.Graph(id="tipo-chart")], className="grid-2"),
                    html.Div([dcc.Graph(id="scatter-chart"), dcc.Graph(id="hist-chart")], className="grid-2"),
                ]),
                dcc.Tab(label="Rendimiento del modelo", children=[
                    html.Div(id="metrics-strip", className="metric-strip"),
                    html.Div([dcc.Graph(id="cm-chart"), dcc.Graph(id="roc-chart")], className="grid-2"),
                    html.Div([dcc.Graph(id="model-chart"), dcc.Graph(id="importance-chart")], className="grid-2"),
                    html.Div(id="narrative", className="narrative"),
                ]),
                dcc.Tab(label="Errores de clasificacion", children=[
                    html.P("Campañas que el modelo seleccionado clasifica peor (errores primero, ordenados por la distancia entre la probabilidad estimada y la clase real)."),
                    dash_table.DataTable(
                        id="tabla", page_size=12, sort_action="native", filter_action="native",
                        style_table={"overflowX": "auto"}, style_cell={"fontFamily": "Arial", "fontSize": 13, "padding": "8px"},
                        style_header={"backgroundColor": COLORS["navy"], "color": "white", "fontWeight": "bold"},
                        style_data_conditional=[
                            {"if": {"filter_query": '{resultado} = "Error"', "column_id": "resultado"}, "color": COLORS["red"], "fontWeight": "bold"},
                            {"if": {"filter_query": '{resultado} = "Acierto"', "column_id": "resultado"}, "color": COLORS["cyan"]},
                        ]),
                ]),
            ]),
            html.Footer(f"Fuente: dataset de campañas de marketing ({RUTA.name}). Clasificacion fuera de muestra con validacion cruzada estratificada de 5 pliegues."),
        ], className="content"),
    ], className="shell"),
], className="app")


# ---------------------------------------------------------------------------
# 4. Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("canal", "value"), Output("tipo", "value"), Output("ubicacion", "value"), Output("audiencia", "value"),
    Output("duracion", "value"), Output("costo", "value"), Output("modelo", "value"),
    Input("reset", "n_clicks"), prevent_initial_call=True)
def reset_filters(_):
    return [], [], [], [], [], [COSTO_MIN, COSTO_MAX], MODELO_DEFECTO


@app.callback(
    Output("kpi-n", "children"), Output("kpi-roi", "children"), Output("kpi-costo", "children"),
    Output("kpi-conv", "children"), Output("kpi-alto", "children"),
    Output("canal-chart", "figure"), Output("tipo-chart", "figure"), Output("scatter-chart", "figure"), Output("hist-chart", "figure"),
    Output("metrics-strip", "children"), Output("cm-chart", "figure"), Output("roc-chart", "figure"),
    Output("model-chart", "figure"), Output("importance-chart", "figure"), Output("narrative", "children"),
    Output("tabla", "data"), Output("tabla", "columns"),
    Input("canal", "value"), Input("tipo", "value"), Input("ubicacion", "value"), Input("audiencia", "value"),
    Input("duracion", "value"), Input("costo", "value"), Input("modelo", "value"))
def update(canales, tipos, ubicaciones, audiencias, duraciones, costo, modelo):
    d = DATA
    if canales: d = d[d.canal.isin(canales)]
    if tipos: d = d[d.tipo_campana.isin(tipos)]
    if ubicaciones: d = d[d.ubicacion.isin(ubicaciones)]
    if audiencias: d = d[d.audiencia.isin(audiencias)]
    if duraciones: d = d[d.duracion_dias.isin(duraciones)]
    d = d[d.costo_adquisicion.between(costo[0], costo[1])]
    template = "plotly_white"
    hay = len(d) > 0

    # --- Panorama -----------------------------------------------------------
    por_canal = d.groupby("canal", as_index=False).agg(roi=("roi", "mean"), campanas=("id_campana", "count")).sort_values("roi", ascending=False)
    fig_canal = px.bar(por_canal, x="canal", y="roi", title="ROI medio por canal", color_discrete_sequence=[COLORS["blue"]],
                       hover_data={"campanas": True})
    fig_canal.update_traces(texttemplate="%{y:.2f}", textposition="outside")
    fig_canal.update_layout(template=template, yaxis_title="ROI medio", xaxis_title="")

    por_tipo = d.groupby("tipo_campana", as_index=False).agg(pct_alto=("roi_alto", "mean"), campanas=("id_campana", "count"))
    por_tipo["pct_alto"] = por_tipo.pct_alto * 100
    fig_tipo = px.bar(por_tipo.sort_values("pct_alto", ascending=False), x="tipo_campana", y="pct_alto",
                      title="% de campañas con ROI alto por tipo de campaña", color_discrete_sequence=[COLORS["cyan"]],
                      hover_data={"campanas": True})
    fig_tipo.add_hline(y=50, line_dash="dash", line_color=COLORS["gold"], annotation_text="50 % (referencia)")
    fig_tipo.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
    fig_tipo.update_layout(template=template, yaxis_title="% con ROI alto", xaxis_title="", yaxis_range=[0, 100])

    muestra = d.sample(min(1500, len(d)), random_state=7) if hay else d
    fig_scatter = px.scatter(muestra, x="costo_adquisicion", y="roi", color="canal", opacity=0.6,
                             title="Costo de adquisicion vs. ROI (muestra)", color_discrete_sequence=PALETA)
    fig_scatter.update_layout(template=template, xaxis_title="Costo de adquisicion ($)", yaxis_title="ROI")

    fig_hist = px.histogram(d, x="roi", nbins=30, title="Distribucion del ROI", color_discrete_sequence=[COLORS["cyan"]])
    fig_hist.add_vline(x=UMBRAL, line_dash="dash", line_color=COLORS["gold"], annotation_text="umbral ROI alto")
    fig_hist.update_layout(template=template, xaxis_title="ROI", yaxis_title="Frecuencia")

    # --- Modelo -------------------------------------------------------------
    p = d.assign(prob_alto=PROB.loc[d.index, modelo])
    p["pred"] = (p.prob_alto >= 0.5).astype(int)
    if len(p) >= 3:
        m = metricas(p.roi_alto, p.prob_alto)
        auc_txt = "--" if np.isnan(m["auc"]) else f"{m['auc']:.3f}"
        strip = [html.Div([html.Span("Exactitud (accuracy)"), html.Strong(f"{m['accuracy']:.1%}")]),
                 html.Div([html.Span("Precision"), html.Strong(f"{m['precision']:.1%}")]),
                 html.Div([html.Span("Sensibilidad (recall)"), html.Strong(f"{m['recall']:.1%}")]),
                 html.Div([html.Span("F1"), html.Strong(f"{m['f1']:.3f}")]),
                 html.Div([html.Span("ROC-AUC"), html.Strong(auc_txt)])]
    else:
        m = {k: np.nan for k in ("accuracy", "precision", "recall", "f1", "auc")}
        strip = [html.Div([html.Span("Sin muestra suficiente"), html.Strong("--")])]

    cm = confusion_matrix(p.roi_alto, p.pred, labels=[0, 1]) if len(p) else np.zeros((2, 2), dtype=int)
    fig_cm = px.imshow(cm, text_auto=True, x=["Bajo", "Alto"], y=["Bajo", "Alto"],
                       labels=dict(x="Clase predicha", y="Clase real", color="Campañas"),
                       color_continuous_scale=[[0, "#EAF2FB"], [1, COLORS["blue"]]],
                       title=f"Matriz de confusion ({modelo.split(' ')[0]})")
    fig_cm.update_layout(template=template, coloraxis_showscale=False)

    fig_roc = go.Figure()
    if p.roi_alto.nunique() == 2:
        fpr, tpr, _ = roc_curve(p.roi_alto, p.prob_alto)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{modelo.split(' ')[0]} (AUC {m['auc']:.2f})",
                                     line=dict(color=COLORS["blue"], width=3)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar (AUC 0.50)", line=dict(dash="dash", color=COLORS["navy"])))
    fig_roc.update_layout(template=template, title="Curva ROC", xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")

    comp = COMPARACION.copy()
    comp["seleccionado"] = np.where(comp.modelo == modelo, "Seleccionado", "Otros")
    fig_models = px.bar(comp, x="modelo", y="auc", color="seleccionado", text=comp.auc.map(lambda v: f"{v:.3f}"),
                        title="Comparacion de iteraciones (ROC-AUC, muestra completa)",
                        color_discrete_map={"Seleccionado": COLORS["blue"], "Otros": "#B9C7D4"})
    fig_models.add_hline(y=0.5, line_dash="dash", line_color=COLORS["gold"], annotation_text="azar")
    fig_models.update_layout(template=template, xaxis_title="", yaxis_title="ROC-AUC", legend_title="")

    imp = IMPORTANCIA.sort_values("importancia")
    fig_imp = px.bar(imp, x="importancia", y="variable", orientation="h", title="Importancia por permutacion (Random Forest)",
                     color_discrete_sequence=[COLORS["cyan"]])
    fig_imp.update_layout(template=template, xaxis_title="Caida media en ROC-AUC", yaxis_title="")

    top_var = IMPORTANCIA.iloc[0].variable
    pct_alto = d.roi_alto.mean() if hay else np.nan
    narrativa = [
        html.H3("Lectura ejecutiva"),
        html.P(f"La vista contiene {len(d):,} campañas; {pct_alto:.0%} tienen ROI alto (≥ {UMBRAL:.2f}). ROI medio: {d.roi.mean():.2f}; "
               f"costo de adquisicion medio: ${d.costo_adquisicion.mean():,.0f}." if hay else "No hay registros con estos filtros."),
        html.P(f"El modelo {modelo} clasifica bien {m['accuracy']:.1%} de las campañas (ROC-AUC: {'--' if np.isnan(m['auc']) else format(m['auc'], '.2f')}). "
               "Un AUC cercano a 0.50 equivale a adivinar al azar." if not np.isnan(m["accuracy"]) else "La seleccion es muy pequeña para calcular metricas."),
        html.P(f"La variable con mayor peso predictivo en el Random Forest es «{top_var}». Si la importancia es cercana a cero, las variables disponibles aportan poca informacion para separar un ROI alto de uno bajo."),
        html.P("Los factores muestran asociacion predictiva, no causalidad."),
    ]

    # --- Tabla --------------------------------------------------------------
    p["clase_real"] = p.roi_alto.map(ETIQUETA)
    p["clase_predicha"] = p.pred.map(ETIQUETA)
    p["resultado"] = np.where(p.pred == p.roi_alto, "Acierto", "Error")
    p["distancia"] = (p.roi_alto - p.prob_alto).abs()
    p["prob_alto"] = p.prob_alto.round(3)
    p["tasa_conversion"] = p.tasa_conversion.round(3)
    cols = ["id_campana", "empresa", "tipo_campana", "canal", "ubicacion", "audiencia", "duracion_dias", "costo_adquisicion",
            "tasa_conversion", "roi", "clase_real", "clase_predicha", "prob_alto", "resultado"]
    t = p.sort_values("distancia", ascending=False)[cols].head(100)
    columnas = [{"name": ETIQ_COL[c], "id": c} for c in cols]

    return (f"{len(d):,}",
            f"{d.roi.mean():.2f}" if hay else "--",
            f"${d.costo_adquisicion.mean():,.0f}" if hay else "--",
            f"{d.tasa_conversion.mean():.1%}" if hay else "--",
            f"{pct_alto:.1%}" if hay else "--",
            fig_canal, fig_tipo, fig_scatter, fig_hist,
            strip, fig_cm, fig_roc, fig_models, fig_imp, narrativa,
            t.to_dict("records"), columnas)


# ---------------------------------------------------------------------------
# 5. Inicio del servidor
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
