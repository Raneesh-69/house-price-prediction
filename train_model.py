import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from utils import preprocess_data

# ==========================================
# Load Dataset
# ==========================================

data = pd.read_csv("data.csv")

print("Dataset Loaded Successfully!")

# ==========================================
# Data Preprocessing
# ==========================================

data = preprocess_data(data)

print("Data Preprocessing Completed!")

# ==========================================
# Features & Target
# ==========================================

X = data.drop("price", axis=1)
y = data["price"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train-Test Split Completed!")

# ==========================================
# Models
# ==========================================

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)
}

results = {}

best_model = None
best_score = -999

print("\nTraining Models...\n")

# ==========================================
# Train All Models
# ==========================================

for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    mae = mean_absolute_error(y_test, prediction)

    mse = mean_squared_error(y_test, prediction)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, prediction)

    cv = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="r2"
    ).mean()

    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "CV Score": cv
    }

    print("=" * 50)
    print(name)
    print(f"MAE      : {mae:.2f}")
    print(f"RMSE     : {rmse:.2f}")
    print(f"R2 Score : {r2:.4f}")
    print(f"CV Score : {cv:.4f}")

    if r2 > best_score:
        best_score = r2
        best_model = model

# ==========================================
# Hyperparameter Tuning
# ==========================================

print("\nOptimizing Random Forest...\n")

params = {

    "n_estimators": [100, 200, 300],

    "max_depth": [5, 10, 15, 20, None],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 4]

}

rf = RandomForestRegressor(random_state=42)

random_search = RandomizedSearchCV(

    estimator=rf,

    param_distributions=params,

    cv=5,

    random_state=42,

    n_iter=10,

    n_jobs=-1

)

random_search.fit(X_train, y_train)

tuned_rf = random_search.best_estimator_

prediction = tuned_rf.predict(X_test)

mae = mean_absolute_error(y_test, prediction)

mse = mean_squared_error(y_test, prediction)

rmse = mse ** 0.5

r2 = r2_score(y_test, prediction)

cv = cross_val_score(
    tuned_rf,
    X,
    y,
    cv=5,
    scoring="r2"
).mean()

results["Tuned Random Forest"] = {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "CV Score": cv
}

print("=" * 50)
print("Tuned Random Forest")
print(random_search.best_params_)
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")
print(f"R2 Score : {r2:.4f}")
print(f"CV Score : {cv:.4f}")

if r2 > best_score:
    best_model = tuned_rf
    best_score = r2

print("\nBest Model Selected Successfully!")

# ==========================================
# Save Model
# ==========================================

pickle.dump(
    best_model,
    open("model.pkl", "wb")
)

pickle.dump(
    results,
    open("model_metrics.pkl", "wb")
)

prediction = best_model.predict(X_test)

pickle.dump(
    (y_test, prediction),
    open("predictions.pkl", "wb")
)

print("\nFiles Saved Successfully!")

print("model.pkl")

print("model_metrics.pkl")

print("predictions.pkl")

print("\nProject Completed Successfully!")