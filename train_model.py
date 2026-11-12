import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#from LearningAlgorithms import ClassificationAlgorithms
import seaborn as sns
import itertools
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostRegressor

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

df = pd.read_pickle("../../data/interim/03_data_features_hr.pkl")
# --------------------------------------------------------------
# Create a training and test set
# --------------------------------------------------------------
df_set = df.dropna()
df_set.info()
df_set = df_set[df_set["bit-year"] != "2026"]
df_set = df_set.drop(["sigla", "medidor", "bit-year"], axis=1)

X = df_set.drop("target", axis=1)
y = df_set["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10
)

# Histograma
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df_set["target"], bins=30, alpha=0.5, label="Total")
ax.hist(y_train, bins=30, alpha=0.5, label="Train")
ax.hist(y_test, bins=30, alpha=0.5, label="Test")
ax.set_xlabel("Target")
ax.set_ylabel("Frecuencia")
ax.legend()
plt.show()

# Densidad
fig, ax = plt.subplots(figsize=(10, 5))
df_set["target"].plot(kind="density", ax=ax, label="Total")
y_train.plot(kind="density", ax=ax, label="Train")
y_test.plot(kind="density", ax=ax, label="Test")
plt.legend()
plt.show()

# Boxplot
plot_df = pd.DataFrame(
    {"Total": df_set["target"], "Train": pd.Series(y_train), "Test": pd.Series(y_test)}
)
plot_df.boxplot(figsize=(8, 5))
plt.ylabel("Target")
plt.show()

# --------------------------------------------------------------
# Split feature subsets
# --------------------------------------------------------------

# Seleccion de variables
numeric_features = [
    "edad",
    "log_ulectura",
    "uso_t_1",
    "uso_t_2",
    "uso_t_3",
    "media_2_anos",
    "media_3_anos",
    "media_5_anos",
    "trend_1",
    "trend_2",
    "growth_pct",
    "std_3_anos",
    "std_5_anos",
    "uso_por_dia",
    "uso_vs_media",
    "type_mean_usage",
    "type_loc_mean_usage",
]

categorical_features = ["type", "loc"]

# Feature sets

feature_set_1 = numeric_features

feature_set_2 = [
    "uso_t_1",
    "uso_t_2",
    "uso_t_3",
    "media_3_anos",
    "trend_1",
    "std_3_anos",
]

feature_set_3 = numeric_features + categorical_features

feature_set_4 = [
    "edad",
    "log_ulectura",
    "uso_t_1",
    "media_3_anos",
    "trend_1",
    "std_3_anos",
    "type",
    "loc",
]

feature_set_5 = [
    "uso_anual",
    "dias_utilizados",
    "uso_promedio",
    "edad",
    "loc",
    "type",
    "type_loc_mean_usage",
    "uso_t_1",
    "uso_t_2",
    "uso_t_3",
    "dias_t_1",
    "dias_t_2",
    "media_3_anos",
    
]


# --------------------------------------------------------------
# Feature Selection
# --------------------------------------------------------------

X_train_num = X_train[numeric_features]

rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X_train_num, y_train)

importance_df = pd.DataFrame(
    {"feature": X_train_num.columns, "importance": rf.feature_importances_}
)

importance_df = importance_df.sort_values("importance", ascending=False)

print(importance_df.head(10))
top10_features = importance_df.head(10)["feature"].tolist()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

selector = SelectKBest(score_func=f_regression, k=10)

selector.fit(X_train_num, y_train)

selector = SelectKBest(score_func=f_regression, k=10)

selector.fit(X_train_num, y_train)

feature_scores = pd.DataFrame(
    {"feature": X_train_num.columns, "score": selector.scores_}
)

feature_scores = feature_scores.sort_values("score", ascending=False)

feature_scores_list = feature_scores["feature"].head(10).tolist()

print(feature_scores.head(10))

possible_feature_sets = [
    feature_set_1,
    feature_set_2,
    feature_set_3,
    feature_set_4,
    feature_set_5,
    top10_features,
    feature_scores_list,
]

feature_names = [
    "Numerical Features",
    "Historical Usage",
    "All Features",
    "Reduced Set",
    "Selected features",
    "Top 10",
    "KBest",
]

# --------------------------------------------------------------
# Compare different models and feature sets
# --------------------------------------------------------------
score_df = pd.DataFrame()

for features, feature_name in zip(
    possible_feature_sets,
    feature_names,
):
    print(f"\nFeature Set: {feature_name}")

    train_X = X_train[features].copy()
    test_X = X_test[features].copy()

    # ----------------------------------------------------------
    # One-hot encoding if categorical variables exist
    # ----------------------------------------------------------

    has_cat = any(col in categorical_features for col in features)

    if has_cat:
        train_X = pd.get_dummies(
            train_X,
            drop_first=True,
        )

        test_X = pd.get_dummies(
            test_X,
            drop_first=True,
        )

        train_X, test_X = train_X.align(
            test_X,
            join="left",
            axis=1,
            fill_value=0,
        )

        # ----------------------------------------------------------
    # Scaling for Neural Network
    # ----------------------------------------------------------

    scaler = StandardScaler()

    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)

    # ----------------------------------------------------------
    # Regression models
    # ----------------------------------------------------------
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Neural Network": MLPRegressor(
            hidden_layer_sizes=(8, 8),
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            max_iter=2000,
            early_stopping=True,
            random_state=42,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.03,
            loss_function="RMSE",
            verbose=False,
            random_state=42,
        ),
    }

    # ----------------------------------------------------------
    # Train and evaluate
    # ----------------------------------------------------------

    for model_name, model in models.items():
        print(f"Training {model_name}")

        # Red neuronal usa datos escalados
        if model_name == "Neural Network":
            model.fit(train_X_scaled, y_train)

            pred = model.predict(test_X_scaled)

        # CatBoost
        elif model_name == "CatBoost":
            model.fit(train_X, y_train)

            pred = model.predict(test_X)

        # Resto de modelos
        else:
            model.fit(train_X, y_train)

            pred = model.predict(test_X)

        mae = mean_absolute_error(y_test, pred)

        rmse = np.sqrt(mean_squared_error(y_test, pred))

        r2 = r2_score(y_test, pred)

        result = pd.DataFrame(
            {
                "feature_set": [feature_name],
                "model": [model_name],
                "MAE": [mae],
                "RMSE": [rmse],
                "R2": [r2],
            }
        )

        score_df = pd.concat([score_df, result], ignore_index=True)

score_df.sort_values("MAE")
score_df.sort_values("R2", ascending=False)

plt.figure(figsize=(10, 10))
sns.barplot(x="model", y="MAE", hue="feature_set", data=score_df)
plt.xlabel("Model")
plt.ylabel("MAE")
plt.ylim(0, 200)
plt.legend(loc="lower right")
plt.show()

plt.figure(figsize=(12, 5))
sns.barplot(
    data=score_df.sort_values("R2", ascending=False),
    x="model",
    y="R2",
    hue="feature_set",
)
plt.title("Model Comparison (R²)")
plt.ylim(0, 1)
plt.show()

score_summary = score_df.sort_values(["R2", "MAE"], ascending=[False, True])
print(score_summary)

best_model_row = score_df.loc[score_df["R2"].idxmax()]

print("\nBest model:")
print(best_model_row)

# --------------------------------------------------------------
# Select best model and evaluate results
# --------------------------------------------------------------

scaler = StandardScaler()
train_X = X_train[top10_features].copy()
test_X = X_test[top10_features].copy()

train_X_scaled = scaler.fit_transform(train_X)
test_X_scaled = scaler.transform(test_X)


best_model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    learning_rate="adaptive",
    max_iter=2000,
    early_stopping=True,
    random_state=42,
)


best_model.fit(train_X_scaled, y_train)
pred = best_model.predict(test_X_scaled)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print(f"MAE : {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²  : {r2:.4f}")

plt.figure(figsize=(8, 8))
plt.scatter(y_test, pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Uso Real")
plt.ylabel("Uso Predicho")
plt.title(f"Real vs Predicho\nR²={r2:.3f}")
plt.show()


errors = pred - y_test
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=30)
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Error")
plt.ylabel("Frecuencia")
plt.title("Distribución de errores")
plt.show()

print("Error promedio:")
print(errors.mean())

print()

print("Desviación estándar del error:")
print(errors.std())


results = pd.DataFrame(
    {
        "real": y_test,
        "pred": pred,
    }
)

results = results.sort_values("real")
plt.figure(figsize=(15, 5))
plt.plot(results["real"].values, label="Real")
plt.plot(results["pred"].values, label="Predicho")
plt.legend()
plt.title("Uso anual real vs predicho")
plt.show()
