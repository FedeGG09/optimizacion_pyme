# backend/feature_engineering.py

import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# Donde tengas guardados tus models_features
BASE_DIR      = Path(__file__).resolve().parent
MODELS_DIR    = BASE_DIR / "models_features"
TRAIN_CSV     = BASE_DIR.parent / "stores_sales_forecasting.csv"

# Carga las listas de features directamente
def _load_features(model_type: str) -> list[str]:
    if model_type == "profit":
        return joblib.load(MODELS_DIR / "features_Profit.pkl")
    else:
        return joblib.load(MODELS_DIR / "features_Quantity.pkl")

def build_features(region: str,
                   product_id: str,
                   sub_category: str,
                   order_date: str,
                   model_type: str = "profit") -> pd.DataFrame:
    """
    Construye el vector de 1 fila con todas las columnas que espera el modelo.
    """
    # 1) Leer CSV de entrenamiento
    df_all = pd.read_csv(TRAIN_CSV, encoding="latin1")

    # 2) Parsear fecha
    od = pd.to_datetime(order_date)

    # 3) Empezar con los inputs 'simples'
    base = {
        "Order Date":    od,
        "Region":        region,
        "Product ID":    product_id,
        "Sub-Category":  sub_category,
        "Month":         od.month,
        "DayOfWeek":     od.weekday(),
    }

    # 4) Agregaciones por región (profit)
    reg = df_all.groupby("Region")["Profit"].agg(["mean","min","max"])
    reg.columns = ["Profit_Region_Mean","Profit_Region_Min","Profit_Region_Max"]
    if region in reg.index:
        for c in reg.columns:
            base[c] = reg.loc[region,c]
    else:
        for c in reg.columns:
            base[c] = 0.0

    # 5) Agregaciones por sub-category (profit & count)
    sub = df_all.groupby("Sub-Category")["Profit"].agg(["count","mean"])
    sub.columns = ["Sub-Category_Count","Profit_Sub-Category_Mean"]
    if sub_category in sub.index:
        for c in sub.columns:
            base[c] = sub.loc[sub_category,c]
    else:
        for c in sub.columns:
            base[c] = 0.0

    # 6) Estadísticos de Quantity por Product ID
    prod = df_all.groupby("Product ID")["Quantity"].agg(["mean","std","median","max"])
    prod.columns = ["Quantity_ProductID_Mean","Quantity_ProductID_Std",
                    "Quantity_ProductID_Median","Quantity_ProductID_Max"]
    if product_id in prod.index:
        for c in prod.columns:
            base[c] = prod.loc[product_id,c]
    else:
        for c in prod.columns:
            base[c] = 0.0

    # 7) Montar DataFrame y hacer get_dummies
    df_feat = pd.DataFrame([base])
    df_feat = pd.get_dummies(df_feat, drop_first=True)

    # 8) Alinear al orden del modelo
    cols = _load_features(model_type)
    df_aligned = df_feat.reindex(columns=cols, fill_value=0)
    return df_aligned

