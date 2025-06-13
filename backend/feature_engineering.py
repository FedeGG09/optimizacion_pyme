# backend/feature_engineering.py

import pandas as pd
import joblib
from pathlib import Path

# -------------------------------------------------------------------
# Configuración de rutas
# -------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models_features"
TRAIN_CSV  = BASE_DIR.parent / "stores_sales_forecasting.csv"

# -------------------------------------------------------------------
# Carga de lista de features
# -------------------------------------------------------------------
def _load_features(model_type: str) -> list[str]:
    """
    Devuelve la lista de features esperados por el modelo "profit" o "quantity".
    """
    if model_type == "profit":
        return joblib.load(MODELS_DIR / "features_Profit.pkl")
    else:
        return joblib.load(MODELS_DIR / "features_Quantity.pkl")

# -------------------------------------------------------------------
# Construcción de features a partir de inputs sencillos
# -------------------------------------------------------------------
def build_features(region: str,
                   product_name: str,
                   sub_category: str,
                   order_date: str,
                   model_type: str = "profit") -> pd.DataFrame:
    """
    Dado:
      - region: nombre de la región (p.ej. "West")
      - product_name: nombre del producto (p.ej. "iPhone 12")
      - sub_category: sub-categoría (p.ej. "Phones")
      - order_date: fecha en formato "YYYY-MM-DD"
      - model_type: "profit" o "quantity"
    Construye un DataFrame de 1 fila con todas las columnas (dummies + agregaciones)
    que el modelo espera, alineado al pickle de features.
    """
    # 1) Cargar CSV completo de entrenamiento
    df_all = pd.read_csv(TRAIN_CSV, encoding="latin1")

    # 2) Parsear la fecha
    od = pd.to_datetime(order_date)

    # 3) Inputs base
    base = {
        "Order Date":    od,
        "Region":        region,
        "Product Name":  product_name,
        "Sub-Category":  sub_category,
        "Month":         od.month,
        "DayOfWeek":     od.weekday(),
    }

    # 4) Agregados por región (Profit)
    reg = (
        df_all
        .groupby("Region")["Profit"]
        .agg(["mean", "min", "max"])
        .rename(columns={
            "mean": "Profit_Region_Mean",
            "min":  "Profit_Region_Min",
            "max":  "Profit_Region_Max"
        })
    )
    if region in reg.index:
        for c in reg.columns:
            base[c] = reg.at[region, c]
    else:
        for c in reg.columns:
            base[c] = 0.0

    # 5) Agregados por sub-categoría (Profit & Count)
    sub = (
        df_all
        .groupby("Sub-Category")["Profit"]
        .agg(["count", "mean"])
        .rename(columns={
            "count": "Sub-Category_Count",
            "mean":  "Profit_Sub-Category_Mean"
        })
    )
    if sub_category in sub.index:
        for c in sub.columns:
            base[c] = sub.at[sub_category, c]
    else:
        for c in sub.columns:
            base[c] = 0.0

    # 6) Estadísticos de Quantity por nombre de producto
    prod = (
        df_all
        .groupby("Product Name")["Quantity"]
        .agg(["mean", "std", "median", "max"])
        .rename(columns={
            "mean":   "Quantity_ProductName_Mean",
            "std":    "Quantity_ProductName_Std",
            "median": "Quantity_ProductName_Median",
            "max":    "Quantity_ProductName_Max"
        })
    )
    if product_name in prod.index:
        for c in prod.columns:
            base[c] = prod.at[product_name, c]
    else:
        for c in prod.columns:
            base[c] = 0.0

    # 7) Crear DataFrame y codificar categóricas
    df_feat = pd.DataFrame([base])
    df_feat = pd.get_dummies(df_feat, drop_first=True)

    # 8) Alinear columnas al orden que espera el modelo
    cols       = _load_features(model_type)
    df_aligned = df_feat.reindex(columns=cols, fill_value=0)

    return df_aligned


