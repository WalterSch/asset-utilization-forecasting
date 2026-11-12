# ==============================================================
# Feature Engineering para Análisis de Utilización de Flota
# ==============================================================
#
# Objetivo:
# Construir atributos para:
#
# - Predicción de horas futuras
# - Predicción de kilómetros futuros
#
# ==============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

# --------------------------------------------------------------
# LOAD DATA, TARGET = uso del año siguiente
# --------------------------------------------------------------

print("Cargando dataset...")

df = pd.read_pickle(
    "../../data/interim/02_outliers_removed_chauvenets.pkl"
)

df = df.dropna()
df.info()

# --------------------------------------------------------------
# Agregacion inicial: km
# --------------------------------------------------------------

print("Agregación inicial...")
annual = (
    df.groupby(["sigla", "bit-year"], as_index=False)
      .agg(
          uso_anual=("delta", "sum"),
          dias_utilizados=("delta", lambda x: (x > 0).sum()),
          uso_promedio=("delta", "mean"),
          uso_std=("delta", "std"),
          uso_max=("delta", "max")
      )
)

# --------------------------------------------------------------
# Features del equipo
# --------------------------------------------------------------
print("Features del equipo...")
# Edad
year = df.groupby("sigla")["year"].max().reset_index()
annual = pd.merge(left=annual, right=year, how='left', left_on='sigla', right_on='sigla')
annual["edad"] = annual["bit-year"].astype(int) - annual["year"].astype(int)
annual = annual.dropna()
annual = annual[annual["edad"] >= 0]
del annual["year"]

# Medidor acumulado
ulectura = df.groupby("sigla")["ulectura"].max().reset_index()
annual = pd.merge(left=annual, right=ulectura, how='left', left_on='sigla', right_on='sigla')
annual["log_ulectura"] = np.log1p(annual["ulectura"])
annual = annual.dropna()
annual = annual[annual["edad"] >= 0]
del annual["ulectura"]

# location
loc = df.groupby("sigla")["loc"].last().reset_index()
annual = pd.merge(left=annual, right=loc, how='left', left_on='sigla', right_on='sigla')
annual = annual.dropna()

# type
types = df.groupby("sigla")["type"].last().reset_index()
annual = pd.merge(left=annual, right=types, how='left', left_on='sigla', right_on='sigla')
annual = annual.dropna()

# --------------------------------------------------------------
# Uso promedio por tipología en la provincia
# --------------------------------------------------------------
type_loc_df = annual.groupby(["type", "loc"])["uso_anual"].mean().reset_index()
annual = pd.merge(left=annual, right=type_loc_df, how='left', left_on=["type", "loc"], right_on=["type", "loc"])
annual = annual.rename(columns={"uso_anual_x": "uso_anual","uso_anual_y": "type_loc_mean_usage"})

# --------------------------------------------------------------
# Uso promedio histórico de la tipología
# --------------------------------------------------------------
type_df = annual.groupby("type")["uso_anual"].mean().reset_index()
annual = pd.merge(left=annual, right=type_df, how='left', left_on='type', right_on='type')
annual = annual.rename(columns={"uso_anual_x": "uso_anual","uso_anual_y": "type_mean_usage"})

# --------------------------------------------------------------
# Historial de utilización
# --------------------------------------------------------------
print("Historial de utilización...")

annual = annual.sort_values(["sigla", "bit-year"])

# Uso años anteriores
annual["uso_t_1"] = (annual.groupby("sigla")["uso_anual"].shift(1))
annual["uso_t_2"] = (annual.groupby("sigla")["uso_anual"].shift(2))
annual["uso_t_3"] = (annual.groupby("sigla")["uso_anual"].shift(3))

# Días utilizados años anteriores
annual["dias_t_1"] = (annual.groupby("sigla")["dias_utilizados"].shift(1))
annual["dias_t_2"] = (annual.groupby("sigla")["dias_utilizados"].shift(2))

# --------------------------------------------------------------
# Promedios móviles
# --------------------------------------------------------------
print("Promedios móviles...")

annual["media_2_anos"] = (annual.groupby("sigla")["uso_anual"].transform(lambda x: x.shift(1).rolling(2).mean()))
annual["media_3_anos"] = (annual.groupby("sigla")["uso_anual"].transform(lambda x: x.shift(1).rolling(3).mean()))
annual["media_5_anos"] = (annual.groupby("sigla")["uso_anual"].transform(lambda x: x.shift(1).rolling(5).mean()))

# --------------------------------------------------------------
# Tendencias
# --------------------------------------------------------------
print("Tendencias...")

annual["trend_1"] = (annual["uso_t_1"] - annual["uso_t_2"])
annual["trend_2"] = (annual["uso_t_1"]- annual["media_3_anos"])

# --------------------------------------------------------------
# Variación porcentual
# --------------------------------------------------------------
print("Variación porcentual...")
annual["growth_pct"] = (annual["uso_t_1"] /annual["uso_t_2"]- 1)

# --------------------------------------------------------------
# Estabilidad histórica
# --------------------------------------------------------------
print("Estabilidad histórica...")
annual["std_3_anos"] = (annual.groupby("sigla")["uso_anual"].transform(lambda x: x.shift(1).rolling(3).std()))
annual["std_5_anos"] = (annual.groupby("sigla")["uso_anual"].transform(lambda x: x.shift(1).rolling(5).std()))

# --------------------------------------------------------------
# Intensidad operacional
# --------------------------------------------------------------
print("Intensidad operacional...")
annual["uso_por_dia"] = (annual["uso_anual"] /annual["dias_utilizados"])


# --------------------------------------------------------------
# Ranking
# --------------------------------------------------------------
print("Ranking...")
annual["uso_vs_media"] = (annual["uso_t_1"] /annual["media_3_anos"])

# --------------------------------------------------------------
# Separar por odómetro/horómetro
# --------------------------------------------------------------
med = df.groupby("sigla")["medidor"].max().reset_index()
annual = pd.merge(left=annual, right=med, how='left', left_on='sigla', right_on='sigla')

annual_hr = annual[annual["medidor"] == "HOROMET"]
annual_km = annual[annual["medidor"] == "ODOMKM"]

# --------------------------------------------------------------
# Target
# --------------------------------------------------------------
annual_hr["target"] = (annual_hr.groupby("sigla")["uso_anual"].shift(-1))
annual_km["target"] = (annual_km.groupby("sigla")["uso_anual"].shift(-1))

# --------------------------------------------------------------
# Export
# --------------------------------------------------------------

annual_hr.to_pickle("../../data/interim/03_data_features_hr.pkl")
annual_hr.to_csv("../../data/interim/03_data_features_hr.csv")

annual_km.to_pickle("../../data/interim/03_data_features_km.pkl")
annual_km.to_csv("../../data/interim/03_data_features_km.csv")
