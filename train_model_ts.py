# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:44:03 2026

@author: walter.schneider
"""

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

df = pd.read_pickle("../../data/interim/03_data_features_ts.pkl")

# Set Activo
sigla = '6G-KDMA-1608'
df_set = df[df['sigla'] ==  sigla]

del df_set["sigla"]
del df_set["medidor"]
del df_set["type"]
del df_set["loc"]

# Model
features = list(df_set.columns[1:21])
label = 'target'
test_df = df_set[df_set['FECHA'] >= '2026-01-31']
train_df = df_set[df_set['FECHA'] < '2026-01-31']

X_train, y_train = train_df[features], train_df[label]
X_test, y_test = test_df[features], test_df[label]

# XGBoost
reg = XGBRegressor(n_estimators=500, learning_rate=0.01, eval_metric='mae')
reg.fit(X_train, y_train, eval_set=[(X_train, y_train),(X_test, y_test)])

# Making Predictions
predictions = reg.predict(X_test)

test_df = test_df.reset_index().drop('index', axis=1)
test_df["predictions"] = pd.Series(predictions)

plt.rcParams.update({"figure.figsize": (17, 5),"figure.dpi": 300,})
fig, ax = plt.subplots()
sns.lineplot(data=df_set, x='FECHA', y='target')
sns.lineplot(data=test_df, x='FECHA', y='predictions')
ax.tick_params(axis='x', rotation=90)

# ==========================================
# Predicción recursiva mes a mes para el año 2027
# ==========================================

# Creamos una copia de trabajo con el histórico disponible hasta octubre de 2026

end = df_set["FECHA"].max()

df_future_work = df_set[(df_set['FECHA'] <= end)].copy()

# Definimos las fechas a predecir (de Noviembre 2026 a Diciembre 2027)
fechas_2027 = pd.date_range(start=end + pd.offsets.MonthEnd(-1), end='2027-12-31', freq='ME')

predictions_2027 = []

# Una sola fecha en adelante
fecha = fechas_2027[0]
nueva_fila = pd.DataFrame({'FECHA': [fecha]})
temp_df = pd.concat([df_future_work, nueva_fila], ignore_index=True)

temp_df.loc[temp_df['FECHA'] == fecha, 'delta'] = temp_df.loc[temp_df['FECHA'] == fecha + pd.offsets.MonthEnd(-1), 'target'].values[0]
temp_df.loc[temp_df['FECHA'] == fecha, 'edad'] = fecha.year - (temp_df.loc[temp_df['FECHA'] == fecha + pd.offsets.MonthEnd(-1), 'year_feature'].values[0] - temp_df.loc[temp_df['FECHA'] == fecha + pd.offsets.MonthEnd(-1), 'edad'].values[0])
temp_df.loc[temp_df['FECHA'] == fecha, 'year_feature'] = fecha.year
temp_df.loc[temp_df['FECHA'] == fecha, 'month'] = fecha.month
temp_df.loc[temp_df['FECHA'] == fecha, 'quarter'] = fecha.quarter
temp_df.loc[temp_df['FECHA'] == fecha, 'month_sin'] = np.sin(2 * np.pi * fecha.month / 12)
temp_df.loc[temp_df['FECHA'] == fecha, 'month_cos'] = np.cos(2 * np.pi * fecha.month / 12)

for lag in [1, 2, 3, 6, 12]:
    temp_df.loc[temp_df['FECHA'] == fecha, f"lag_{lag}"] = temp_df['delta'].shift(lag)

    # Recalculamos Rolling Means
for rolling in [1, 2, 3, 6, 12]:
    temp_df.loc[temp_df['FECHA'] == fecha, f"rolling_mean_{rolling}"] = temp_df['delta'].shift(1).rolling(rolling).mean()

    # Recalculamos Rolling Stds
for std in [3, 6, 12]:
    temp_df.loc[temp_df['FECHA'] == fecha, f"rolling_std_{std}"] = temp_df['delta'].shift(1).rolling(std).std()

X_row = temp_df[temp_df['FECHA'] == fecha][features]
pred_val = reg.predict(X_row)
temp_df.loc[temp_df['FECHA'] == fecha, 'target'] = pred_val

temp_df = pd.concat([df_future_work, nueva_fila], ignore_index=True)

# Hasta Diciembre 2027
for fecha in fechas_2027:
    # 1. Creamos una fila temporal para la fecha a predecir
    nueva_fila = pd.DataFrame({'FECHA': [fecha]})
    
    # Concatenamos temporalmente para calcular features de forma sencilla usando todo el historial acumulado
    temp_df = pd.concat([temp_df, nueva_fila], ignore_index=True)
    
    #delta
    temp_df.loc[temp_df['FECHA'] == fecha, 'delta'] = temp_df.loc[temp_df['FECHA'] == fecha + pd.offsets.MonthEnd(-1), 'target'].values[0]
    
    #edad
    temp_df.loc[temp_df['FECHA'] == fecha, 'edad'] = fecha.year - (temp_df.loc[temp_df['FECHA'] == fecha + pd.offsets.MonthEnd(-1), 'year_feature'].values[0] - temp_df.loc[temp_df['FECHA'] == fecha + pd.offsets.MonthEnd(-1), 'edad'].values[0])
    
    # Rellenamos las variables de tiempo y fecha
    temp_df.loc[temp_df['FECHA'] == fecha, 'year_feature'] = fecha.year
    temp_df.loc[temp_df['FECHA'] == fecha, 'month'] = fecha.month
    temp_df.loc[temp_df['FECHA'] == fecha, 'quarter'] = fecha.quarter
    temp_df.loc[temp_df['FECHA'] == fecha, 'month_sin'] = np.sin(2 * np.pi * fecha.month / 12)
    temp_df.loc[temp_df['FECHA'] == fecha, 'month_cos'] = np.cos(2 * np.pi * fecha.month / 12)
    
    # Si tienes la columna 'edad' u otras dependientes, asegúrate de actualizar su lógica aquí:
    # temp_df.loc[temp_df['FECHA'] == fecha, 'edad'] = ... 

    # Recalculamos Lags sobre el DataFrame temporal acumulado
    for lag in [1, 2, 3, 6, 12]:
        temp_df.loc[temp_df['FECHA'] == fecha, f"lag_{lag}"] = temp_df['delta'].shift(lag)

    # Recalculamos Rolling Means
    for rolling in [1, 2, 3, 6, 12]:
        temp_df.loc[temp_df['FECHA'] == fecha, f"rolling_mean_{rolling}"] = temp_df['delta'].shift(1).rolling(rolling).mean()

    # Recalculamos Rolling Stds
    for std in [3, 6, 12]:
        temp_df.loc[temp_df['FECHA'] == fecha, f"rolling_std_{std}"] = temp_df['delta'].shift(1).rolling(std).std()

    # 2. Extraemos los features para la fecha actual de predicción
    X_row = temp_df[temp_df['FECHA'] == fecha][features]
    
    # 3. Predecimos el valor para este mes
    pred_val = reg.predict(X_row)
    temp_df.loc[temp_df['FECHA'] == fecha, 'target'] = pred_val


# ==========================================
# Visualización final (Histórico + Test + 2027)
# ==========================================
plt.rcParams.update({"figure.figsize": (17, 5), "figure.dpi": 300})
fig, ax = plt.subplots()
# Histórico real
sns.lineplot(data=df_set, x='FECHA', y='target', label='Histórico Real', ax=ax, color='blue')
# Predicciones del conjunto de prueba (2026)
sns.lineplot(data=test_df, x='FECHA', y='predictions', label='Predicción Test (2026)', ax=ax, color='orange')
# Predicciones futuras (2027)
sns.lineplot(data=temp_df[temp_df['FECHA'] >= end], x='FECHA', y='target', label='Predicción Futura (2027)', ax=ax, color='green', linestyle='--',marker="o")
# Etiquetas de datos para 2027
for _, row in df_set.iterrows():

    ax.annotate(
        f"{int(round(row['delta']))}",
        (row["FECHA"], row["delta"]),
        textcoords="offset points",
        xytext=(-15, 0),
        ha="center",
        va = "top",
        fontsize=8,
        color="blue"
    )

ax.tick_params(axis='x', rotation=90)
plt.legend()
plt.title(f'El equipo {sigla} tiene un uso esperado de {int(temp_df[temp_df['FECHA'] >= '2027-01-31']['delta'].sum())} para el año 2027')
plt.show()

print(f'El equipo {sigla} tiene un uso esperado de {int(temp_df[temp_df['FECHA'] >= '2027-01-31']['delta'].sum())} para el año 2027')

