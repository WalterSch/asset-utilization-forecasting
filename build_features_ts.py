# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:34:09 2026

@author: walter.schneider
"""

import pandas as pd
import numpy as np
import seaborn as sns
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

print(f"Total numbers of uinque days in dataset: {df.index.nunique()}")
print(f"Total numbers of uinque days in dataset: {len(df.index)}")

# Equipos con mayores registros
df.reset_index().groupby('sigla')['FECHA'].count().reset_index().sort_values('FECHA', ascending=False).head(50)

import logging
logging.getLogger().setLevel(logging.CRITICAL)

df_aux = df[df['sigla'] == '4D-MMKO-493'].reset_index()
df_aux = df_aux[df_aux['bit-year'] == '2025'].reset_index()
plt.rcParams.update({'figure.figsize': (17,3), 'figure.dpi': 300})
fig, ax = plt.subplots()
sns.lineplot(data = df_aux, x='FECHA', y='delta')
ax.tick_params(axis='x', rotation=90)

# Too less daily delta to bo bother forecasting. Let's look at monthly order volume instead
sampling = {
    'sigla': "last",
    'medidor': "last",
    'ini': "first",
    'fin': "last",
    'region': "last",
    'bit-year': "last",
    'set': "last",
    'type': "last",
    'loc': "last",
    'year': "last",
    'ulectura': "last",
    'delta': "sum"}

df_resampled = (
    df.groupby([pd.Grouper(freq="ME"), "activo"])
    .agg(sampling)
    .dropna()
    .reset_index()
)

# Rango completo de meses
full_dates = pd.date_range(
    start=df_resampled["FECHA"].min(),
    end=df_resampled["FECHA"].max(),
    freq="ME"
)

# Todos los equipos
all_siglas = df_resampled["sigla"].unique()

# Grilla completa
full_index = pd.MultiIndex.from_product(
    [all_siglas, full_dates],
    names=["sigla", "FECHA"]
)

# Expandir
df_resampled = (
    df_resampled
    .set_index(["sigla", "FECHA"])
    .reindex(full_index)
    .reset_index()
)



df_resampled["edad"] = pd.to_numeric(df_resampled["bit-year"])-df_resampled["year"]

del df_resampled["activo"]
del df_resampled["ini"]
del df_resampled["fin"]
del df_resampled["region"]
del df_resampled["bit-year"]
del df_resampled["set"]
del df_resampled["year"]
del df_resampled["ulectura"]

# Visualiza un ativo
df_aux = df_resampled[df_resampled['sigla'] == '5F-CPFR-1174'].reset_index()
plt.rcParams.update({'figure.figsize': (17,3), 'figure.dpi': 300})
fig, ax = plt.subplots()
sns.lineplot(data = df_aux, x='FECHA', y='delta')
ax.tick_params(axis='x', rotation=90)


# Calendar Features: enero ≈ diciembre; mejor que un número entero.
df_resampled["month"] = df_resampled["FECHA"].dt.month
df_resampled["quarter"] = df_resampled["FECHA"].dt.quarter
df_resampled["year_feature"] = df_resampled["FECHA"].dt.year

# Features cíclicas
df_resampled["month_sin"] = np.sin(2*np.pi*df_resampled["month"]/12)
df_resampled["month_cos"] = np.cos(2*np.pi*df_resampled["month"]/12)

# Lag Features
lags = [
    1,
    2,
    3,
    6,
    12
]

for lag in lags:
    df_resampled[f"lag_{lag}"] = (
        df_resampled
        .groupby("sigla")["delta"]
        .shift(lag)
    )

# Rolling Means Features
rollings = [
    1,
    2,
    3,
    6,
    12
]

for rolling in rollings:
    df_resampled[f"rolling_mean_{rolling}"] = (
        df_resampled
        .groupby("sigla")["delta"]
        .shift(1).rolling(rolling).mean()
    )

# Rolling Std Features
stds = [
    3,
    6,
    12
]

for std in stds:
    df_resampled[f"rolling_std_{std}"] = (
        df_resampled
        .groupby("sigla")["delta"]
        .shift(1).rolling(std).std()
    )

# Target
df_resampled["target"] = (df_resampled.groupby("sigla")["delta"].shift(-1))

df_resampled = df_resampled.dropna()

df_resampled.to_pickle("../../data/interim/03_data_features_ts.pkl")
df_resampled.to_csv("../../data/interim/03_data_features_ts.csv")

