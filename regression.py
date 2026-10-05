import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# =========================
# 1. LOAD DATA
# =========================

df = pd.read_csv("rental_info.csv")


# =========================
# 2. FEATURE ENGINEERING
# =========================

df["return_date"] = pd.to_datetime(df["return_date"])
df["rental_date"] = pd.to_datetime(df["rental_date"])

df["rental_length_days"] = (
    df["return_date"] - df["rental_date"]
).dt.days


df["deleted_scenes"] = (
    df["special_features"]
    .str.contains("Deleted Scenes", na=False)
    .astype(int)
)

df["behind_the_scenes"] = (
    df["special_features"]
    .str.contains("Behind the Scenes", na=False)
    .astype(int)
)


# =========================
# 3. FEATURES AND TARGET
# =========================

X = df[[
    "amount",
    "release_year",
    "rental_rate",
    "length",
    "replacement_cost",
    "NC-17",
    "PG",
    "PG-13",
    "R",
    "amount_2",
    "length_2",
    "rental_rate_2",
    "deleted_scenes",
    "behind_the_scenes"
]]

y = df["rental_length_days"]


# =========================
# 4. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=9
)


# =========================
# 5. LINEAR REGRESSION
# =========================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

mse_linear = mean_squared_error(y_test, linear_pred)

print("Linear Regression MSE:", mse_linear)


# =========================
# 6. RANDOM FOREST
# =========================

param_dist = {
    "n_estimators": np.arange(1, 101),
    "max_depth": np.arange(1, 11)
}

rf = RandomForestRegressor(random_state=9)

rand_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    cv=5,
    random_state=9
)

rand_search.fit(X_train, y_train)

hyper_params = rand_search.best_params_

print("\nRandom Forest best parameters:", hyper_params)

rf = RandomForestRegressor(
    n_estimators=hyper_params["n_estimators"],
    max_depth=hyper_params["max_depth"],
    random_state=9
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

mse_rf = mean_squared_error(y_test, rf_pred)

print("Random Forest MSE:", mse_rf)


# =========================
# 7. GRADIENT BOOSTING
# =========================

gb = GradientBoostingRegressor(random_state=9)

gb.fit(X_train, y_train)

gb_pred = gb.predict(X_test)

mse_gb = mean_squared_error(y_test, gb_pred)

print("Gradient Boosting MSE:", mse_gb)


# =========================
# 8. COMPARE MODELS
# =========================

models = {
    "Linear Regression": mse_linear,
    "Random Forest": mse_rf,
    "Gradient Boosting": mse_gb
}

print("\n===== MODEL COMPARISON =====")

for model, mse in models.items():
    print(f"{model}: {mse}")


# =========================
# 9. DETERMINE BEST MODEL
# =========================

best_model_name = min(models, key=models.get)
best_mse = models[best_model_name]

print("\n===== BEST MODEL =====")
print("Best model:", best_model_name)
print("Best MSE:", best_mse)