from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import io
from pathlib import Path
import joblib
from functools import lru_cache

from backend.model_utils import load_data, predict_from_dataframe, evaluate_model, parse_month
from backend.feature_engineering import build_features

# -------------------------------------------------------
# Configuración de FastAPI
# -------------------------------------------------------
app = FastAPI(
    title="Sales Forecasting API",
    description="API para predicciones de Profit/Quantity y KPIs de ventas",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------
# Paths globales
# -------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# Carpeta de modelos y features pickle
MODELS_DIR = BASE_DIR / "models_features"

# CSV de entrenamiento subido
uploaded_csv_path: Path | None = None

# Frontend estático
FRONTEND_DIR = PROJECT_DIR / "frontend"
SRC_DIR      = FRONTEND_DIR / "src"
CSS_DIR      = FRONTEND_DIR / "css"
JS_DIR       = FRONTEND_DIR / "js"

# -------------------------------------------------------
# Carga dinámica de modelos con caché
# -------------------------------------------------------
@lru_cache()
def load_profit_model():
    return joblib.load(MODELS_DIR / "model_Profit.pkl")

@lru_cache()
def load_quantity_model():
    return joblib.load(MODELS_DIR / "model_Quantity.pkl")

# -------------------------------------------------------
# Pydantic schemas
# -------------------------------------------------------
class PredictionIn(BaseModel):
    features: List[float]

class PredictionOut(BaseModel):
    prediction: float

# -------------------------------------------------------
# Montaje de estáticos y frontend
# -------------------------------------------------------
app.mount("/static/css", StaticFiles(directory=str(CSS_DIR)), name="static_css")
app.mount("/static/js",  StaticFiles(directory=str(JS_DIR)),  name="static_js")

@app.get("/")
def serve_frontend():
    index_path = SRC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "index.html no encontrado en frontend/src/")
    return FileResponse(str(index_path))

# -------------------------------------------------------
# Util: cargar DataFrame desde CSV subido
# -------------------------------------------------------
def _get_df() -> pd.DataFrame:
    if not uploaded_csv_path:
        raise HTTPException(400, "No se ha subido ningún CSV de entrenamiento.")
    return pd.read_csv(str(uploaded_csv_path), encoding="latin1")

# -------------------------------------------------------
# 1) CSV upload
# -------------------------------------------------------
@app.post("/upload_csv")
def upload_training_csv(file: UploadFile = File(...)):
    global uploaded_csv_path
    try:
        data = file.file.read()
        target = PROJECT_DIR / "stores_sales_forecasting.csv"
        with open(target, "wb") as f:
            f.write(data)
        uploaded_csv_path = target
        return {"detail": f"CSV cargado en {target.name}"}
    except Exception as e:
        raise HTTPException(500, str(e))

# -------------------------------------------------------
# 2) Métricas XGBoost
# -------------------------------------------------------
@app.get("/metrics_xgb")
def metrics_xgb_endpoint():
    try:
        df = load_data(str(uploaded_csv_path)) if uploaded_csv_path else _get_df()
        metrics = evaluate_model(df)
        return {"metrics": metrics}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

# -------------------------------------------------------
# 3) Predicción batch CSV
# -------------------------------------------------------
@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):
    try:
        raw = file.file.read()
        df  = pd.read_csv(io.BytesIO(raw), encoding="latin1")
        preds = predict_from_dataframe(df)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(400, str(e))

# -------------------------------------------------------
# 4) Predicción JSON genérico
# -------------------------------------------------------
@app.post("/predict")
def predict_json(data: List[dict]):
    try:
        df = pd.DataFrame(data)
        preds = predict_from_dataframe(df)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(400, str(e))

# -------------------------------------------------------
# 5) KPIs con filtros
# -------------------------------------------------------
@app.get("/kpis")
def get_kpis(
    month: str = Query(None, description="Mes YYYY-MM, opcional"),
    vendor: str = Query("Todos", description="Customer Name"),
    product: str = Query("Todos", description="Product Name")
):
    df = _get_df()
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    if month:
        df = df[df["Order Date"].dt.to_period("M") == pd.Period(month)]
    if vendor != "Todos":
        df = df[df["Customer Name"] == vendor]
    if product != "Todos":
        df = df[df["Product Name"] == product]
    for col in ("Sales","Profit"):
        if col not in df.columns:
            raise HTTPException(500, f"Falta columna '{col}'")
    total_sales    = df["Sales"].sum()
    avg_profit_pct = (df["Profit"]/df["Sales"]).mean() if total_sales else 0
    count          = len(df)
    avg_sales      = df["Sales"].mean() if count else 0
    return {
        "total_sales": float(total_sales),
        "avg_profit_pct": float(avg_profit_pct),
        "sale_count": count,
        "avg_sales": float(avg_sales)
    }

# -------------------------------------------------------
# 6) Datos agrupados
# -------------------------------------------------------
@app.get("/grouped")
def get_grouped_data(
    field: str = Query(..., description="Campo a agrupar"),
    month: str = Query(None),
    vendor: str = Query("Todos"),
    product: str = Query("Todos")
):
    df = _get_df()
    if "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    if month:
        df = df[df["Order Date"].dt.to_period("M") == pd.Period(month)]
    if vendor != "Todos":
        df = df[df["Customer Name"] == vendor]
    if product != "Todos":
        df = df[df["Product Name"] == product]
    if field not in df.columns:
        raise HTTPException(400, f"Campo '{field}' no existe")
    for col in ("Sales","Quantity","Discount","Profit"):
        if col not in df.columns:
            raise HTTPException(500, f"Falta columna '{col}'")
    grouped = (
        df.groupby(field, dropna=False)
          .agg(
            total_sales   = ("Sales", "sum"),
            total_quantity= ("Quantity","sum"),
            avg_discount  = ("Discount","mean"),
            total_profit  = ("Profit","sum")
          )
          .reset_index()
          .rename(columns={field:"group"})
          .sort_values("total_sales", ascending=False)
    )
    return {"data": grouped.to_dict("records")}

# -------------------------------------------------------
# 7) Predicción “baja nivel” (listas de features)
# -------------------------------------------------------
@app.post("/predict/profit", response_model=PredictionOut)
def predict_profit(payload: PredictionIn):
    from backend.model_utils import load_feature_names  # local import
    cols  = load_feature_names()  # features_Profit.pkl
    if len(payload.features) != len(cols):
        raise HTTPException(400, f"Se esperaban {len(cols)} features")
    df = pd.DataFrame([payload.features], columns=cols)
    pred = load_profit_model().predict(df)[0]
    return {"prediction": float(pred)}

@app.post("/predict/quantity", response_model=PredictionOut)
def predict_quantity(payload: PredictionIn):
    from backend.model_utils import load_feature_names  # local import
    cols  = load_feature_names()  # features_Quantity.pkl
    if len(payload.features) != len(cols):
        raise HTTPException(400, f"Se esperaban {len(cols)} features")
    df = pd.DataFrame([payload.features], columns=cols)
    pred = load_quantity_model().predict(df)[0]
    return {"prediction": float(pred)}

# -------------------------------------------------------
# 8) Endpoint de configuración de features
# -------------------------------------------------------
@app.get("/predict/config")
def get_prediction_config():
    # Para formularios dinámicos, aunque ya no se usa en by_fields
    profit_feats   = joblib.load(MODELS_DIR / "features_Profit.pkl")
    quantity_feats = joblib.load(MODELS_DIR / "features_Quantity.pkl")
    return {"profit": profit_feats, "quantity": quantity_feats}

# -------------------------------------------------------
# 11) Endpoints para poblar dropdowns
# -------------------------------------------------------

@app.get("/metadata/regions")
def get_regions():
    """
    Devuelve la lista de todas las regiones únicas del CSV subido.
    """
    df = _get_df()
    if "Region" not in df.columns:
        raise HTTPException(500, "No existe la columna 'Region'.")
    regions = sorted(df["Region"].dropna().unique().tolist())
    return regions

@app.get("/metadata/products")
def get_products():
    """
    Devuelve la lista de todos los nombres de producto únicos.
    """
    df = _get_df()
    if "Product Name" not in df.columns:
        raise HTTPException(500, "No existe la columna 'Product Name'.")
    prods = sorted(df["Product Name"].dropna().unique().tolist())
    return prods

@app.get("/metadata/subcategories")
def get_subcategories():
    """
    Devuelve la lista de todas las sub-categorías únicas.
    """
    df = _get_df()
    if "Sub-Category" not in df.columns:
        raise HTTPException(500, "No existe la columna 'Sub-Category'.")
    subs = sorted(df["Sub-Category"].dropna().unique().tolist())
    return subs

# -------------------------------------------------------
# 9) Predicción a partir de campos sencillos
# -------------------------------------------------------
@app.post("/predict/by_fields", response_model=PredictionOut)
def predict_by_fields(
    region:      str = Query(..., description="Región"),
    product_id:  str = Query(..., description="Product ID"),
    sub_category:str = Query(..., description="Sub-Category"),
    order_date:  str = Query(..., description="Fecha (YYYY-MM-DD)"),
    model:       str = Query("profit", regex="^(profit|quantity)$")
):
    try:
        df_feat = build_features(region, product_id, sub_category, order_date, model_type=model)
        mdl     = load_profit_model() if model=="profit" else load_quantity_model()
        pred    = mdl.predict(df_feat)[0]
        return {"prediction": float(pred)}
    except Exception as e:
        raise HTTPException(400, detail=str(e))

# -------------------------------------------------------
# 10) Tendencia de ventas (sales_trend)
# -------------------------------------------------------
@app.get("/sales_trend")
def sales_trend(
    year:   int = Query(2020, description="Año (p.ej. 2020)"),
    month:  str = Query(None, description="Mes YYYY-MM, opcional"),
    vendor: str = Query("Todos", description="Customer Name")
):
    df = _get_df()
    if "Order Date" not in df.columns:
        raise HTTPException(500, "No existe 'Order Date'")
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df[df["Order Date"].dt.year == year]

    if month:
        periodo = pd.Period(month)
        df = df[df["Order Date"].dt.to_period("M")==periodo]
        if vendor!="Todos":
            df = df[df["Customer Name"]==vendor]
        df["Day"] = df["Order Date"].dt.day
        pivot = df.groupby(["Day","Customer Name"])["Sales"].sum().unstack(fill_value=0)
        days = list(range(1, periodo.days_in_month+1))
        labels = [f"{month}-{d:02d}" for d in days]
        return {
            "labels": labels,
            "datasets": [
                {"vendor": v, "values": pivot.get(v, pd.Series([0]*len(days))).tolist()}
                for v in pivot.columns
            ]
        }

    if vendor!="Todos":
        df = df[df["Customer Name"]==vendor]
    df["YearMonth"] = df["Order Date"].dt.to_period("M").astype(str)
    pivot = df.groupby(["YearMonth","Customer Name"])["Sales"].sum().unstack(fill_value=0)
    months = [f"{year}-{m:02d}" for m in range(1,13)]
    pivot = pivot.reindex(months, fill_value=0)
    return {
        "labels": months,
        "datasets": [
            {"vendor": v, "values": pivot.get(v, pd.Series([0]*12)).tolist()}
            for v in pivot.columns
        ]
    }

