# --- RELEVENT LIBRARIES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- GBM LIBRARIES ---
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# --- IMPORT AND CLEAN DATA ---
df = pd.read_csv("insurance.csv")

X = df.drop("charges", axis=1) # charges = target, y
y = df["charges"]

cat_cols = ["sex", "smoker", "region"] # categorical vars
num_cols = ["age", "bmi", "children"]


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
) # # categorical vars -> numerical vals


# --- MODELS ---
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

lgb_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1  
)

cb_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="RMSE",
    random_seed=42,
    verbose=False
)
# CatBoost can natively handle categoricals, but to keep parity with the others
# we’ll feed it the one-hot-encoded features via the same preprocessor



# --- SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) # 80/20

# --- PIPELINES - APPLY MODELS TO DATA ---
xgb_pipeline = Pipeline([("preprocessor", preprocessor), ("model", xgb_model)])
lgb_pipeline = Pipeline([("preprocessor", preprocessor), ("model", lgb_model)])
cb_pipeline  = Pipeline([("preprocessor", preprocessor), ("model", cb_model)])


# --- APPLY TRAINING DATA ---
xgb_pipeline.fit(X_train, y_train)
lgb_pipeline.fit(X_train, y_train)
cb_pipeline.fit(X_train, y_train)


# --- PREDICT CHARGES WITH TEST DATA ---
y_pred_xgb = xgb_pipeline.predict(X_test)
y_pred_lgb = lgb_pipeline.predict(X_test)
y_pred_cb  = cb_pipeline.predict(X_test)


# --- ERROR CALC FUNC ---
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n{name} Performance:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.3f}")


# --- PRINT RESULTS ---
evaluate("XGBoost",   y_test, y_pred_xgb)
evaluate("LightGBM",  y_test, y_pred_lgb)
evaluate("CatBoost",  y_test, y_pred_cb)


# --- PLOT A - DENSITY PLOT ---
residuals_df = pd.DataFrame({
    "XGBoost": y_test - y_pred_xgb,
    "LightGBM": y_test - y_pred_lgb,
    "CatBoost": y_test - y_pred_cb
})

plt.figure(figsize=(10, 6))
for model in residuals_df.columns:
    sns.kdeplot(residuals_df[model], label=model, fill=True, alpha=0.4)

plt.title("Residual Distribution by Model")
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Density")
plt.legend()
plt.savefig("residual_distribution.png", dpi=300, bbox_inches="tight")


# PLOT B - BOX PLOT
plt.figure(figsize=(8, 6))
sns.boxplot(data=residuals_df, orient="v", palette="Set2")
plt.title("Residual Spread Comparison by Model")
plt.ylabel("Residual (Actual - Predicted)")
plt.xlabel("Model")
plt.savefig("residual_boxplot.png", dpi=300, bbox_inches="tight")

# ==== FAST + FIXED OPTUNA TUNING ====
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import clone
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

CV = KFold(n_splits=3, shuffle=True, random_state=42)
SCORING = "neg_root_mean_squared_error"
N_TRIALS = 12
sampler = TPESampler(seed=42)

def objective(trial, base_pipeline, model_name):
    p = {}
    if model_name == "xgb":
        p.update({
            "model__n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "model__max_depth": trial.suggest_int("max_depth", 3, 6),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "model__subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "model__random_state": 42,
            "model__objective": "reg:squarederror",
        })
    elif model_name == "lgb":
        p.update({
            "model__n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "model__num_leaves": trial.suggest_int("num_leaves", 20, 48),
            "model__max_depth": trial.suggest_int("max_depth", 4, 10),
            "model__subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "model__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "model__random_state": 42,
            "model__verbose": -1,
        })
    else:  # "cb"
        p.update({
            "model__iterations": trial.suggest_int("iterations", 300, 800),
            "model__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "model__depth": trial.suggest_int("depth", 5, 8),
            "model__l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 8.0),
            "model__bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 0.8),
            "model__random_seed": 42,
            "model__loss_function": "RMSE",
            "model__verbose": False,
        })

    pipe = clone(base_pipeline)
    pipe.set_params(**p)

    scores = cross_val_score(pipe, X, y, cv=CV, scoring=SCORING, n_jobs=-1)
    return -scores.mean()

def tune_and_refit(pipeline, name):
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(lambda t: objective(t, pipeline, name), n_trials=N_TRIALS)
    best_params = {f"model__{k}": v for k, v in study.best_params.items()}
    tuned = clone(pipeline).set_params(**best_params).fit(X_train, y_train)
    return tuned

xgb_tuned = tune_and_refit(xgb_pipeline, "xgb")
lgb_tuned = tune_and_refit(lgb_pipeline, "lgb")
cb_tuned  = tune_and_refit(cb_pipeline,  "cb")

def print_metrics(name, model):
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae  = mean_absolute_error(y_test, pred)
    r2   = r2_score(y_test, pred)
    print(f"{name} (tuned) -> RMSE: {rmse:.2f} | MAE: {mae:.2f} | R²: {r2:.3f}")

print("\n# Tuned models (test set)")
print_metrics("XGBoost",  xgb_tuned)
print_metrics("LightGBM", lgb_tuned)
print_metrics("CatBoost", cb_tuned)

# ==== METRICS + ONE FINAL TABLE ====
import os
os.makedirs("outputs", exist_ok=True)

def compute_metrics(model_name, y_true, y_pred, phase):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {
        "Model": model_name,
        "Phase": phase,           # "Baseline" or "Tuned"
        "RMSE":  round(rmse, 2),
        "MAE":   round(mae, 2),
        "R2":    round(r2, 3),
    }

# --- Baseline metrics (no printing here) ---
baseline_rows = []
baseline_rows.append(compute_metrics("XGBoost",   y_test, y_pred_xgb, "Baseline"))
baseline_rows.append(compute_metrics("LightGBM",  y_test, y_pred_lgb, "Baseline"))
baseline_rows.append(compute_metrics("CatBoost",  y_test, y_pred_cb,  "Baseline"))

# --- Tuned models (already fitted above) ---
y_pred_xgb_t = xgb_tuned.predict(X_test)
y_pred_lgb_t = lgb_tuned.predict(X_test)
y_pred_cb_t  = cb_tuned.predict(X_test)

tuned_rows = []
tuned_rows.append(compute_metrics("XGBoost",   y_test, y_pred_xgb_t, "Tuned"))
tuned_rows.append(compute_metrics("LightGBM",  y_test, y_pred_lgb_t, "Tuned"))
tuned_rows.append(compute_metrics("CatBoost",  y_test, y_pred_cb_t,  "Tuned"))

# --- Combine, sort, print once ---
summary_df = pd.DataFrame(baseline_rows + tuned_rows)
summary_df = summary_df.sort_values(["Model","Phase"], ascending=[True, True])

print("\n=== GBM Results Summary (Test Set) ===")
print(summary_df.to_string(index=False))

# Save for later use
summary_df.to_csv("outputs/results_summary.csv", index=False)
print("\nSaved: outputs/results_summary.csv")
