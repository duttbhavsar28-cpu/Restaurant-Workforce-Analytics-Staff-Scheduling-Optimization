"""
demand_forecasting.py
=====================
Machine Learning Customer Demand Forecasting for Restaurant Workforce Optimization.

Builds, evaluates, and compares multiple predictive models to forecast shift customer covers:
- Baseline Model: Linear Regression
- Ensemble Model: Random Forest Regressor
- Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R² Score
- Feature Importance extraction and automated dynamic model selection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Path Configuration & Python Path Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_analysis import save_feature_importance_chart


# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"
CHARTS_DIR = OUTPUTS_DIR / "charts"
DATA_FILE = DATA_DIR / "restaurant_workforce.csv"


def prepare_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract shift-level dataset with calendar and operational features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or engineered workforce DataFrame.

    Returns
    -------
    pd.DataFrame
        Deduplicated shift-level DataFrame with features:
        ['month', 'day_of_month', 'day_of_week', 'weekend', 'holiday',
         'special_event', 'weather', 'shift', 'covers']
    """
    # Group or drop duplicates by (date, shift) to isolate shift-level demand
    shift_df = df.drop_duplicates(subset=["date", "shift"]).copy()

    # Feature extraction from date
    date_series = pd.to_datetime(shift_df["date"])
    shift_df["month"] = date_series.dt.month
    shift_df["day_of_month"] = date_series.dt.day

    feature_cols = [
        "month",
        "day_of_month",
        "day_of_week",
        "weekend",
        "holiday",
        "special_event",
        "weather",
        "shift",
        "covers",
    ]
    return shift_df[feature_cols].copy()


def build_preprocessor() -> ColumnTransformer:
    """Construct ColumnTransformer for numerical scaling and categorical one-hot encoding."""
    numeric_features = ["month", "day_of_month", "day_of_week"]
    binary_features = ["weekend", "holiday", "special_event"]
    categorical_features = ["weather", "shift"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("bin", "passthrough", binary_features),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_features),
        ]
    )
    return preprocessor


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Train pipeline and compute regression evaluation metrics."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    return {
        "model": name,
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "pipeline": pipeline,
        "predictions": y_pred,
    }


def train_and_compare_models(
    shift_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Train Linear Regression and Random Forest Regressor, compare their metrics,
    dynamically select the best model, and extract feature importances.
    """
    X = shift_df.drop(columns=["covers"])
    y = shift_df["covers"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # 1. Linear Regression Pipeline
    lr_preprocessor = build_preprocessor()
    lr_pipeline = Pipeline([
        ("preprocessor", lr_preprocessor),
        ("regressor", LinearRegression()),
    ])
    lr_results = evaluate_model("Linear Regression", lr_pipeline, X_train, y_train, X_test, y_test)

    # 2. Random Forest Regressor Pipeline
    rf_preprocessor = build_preprocessor()
    rf_pipeline = Pipeline([
        ("preprocessor", rf_preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=150, max_depth=12, random_state=random_state)),
    ])
    rf_results = evaluate_model("Random Forest", rf_pipeline, X_train, y_train, X_test, y_test)

    # Compile metrics table
    models_comparison = [
        {"Model": lr_results["model"], "MAE": lr_results["mae"], "RMSE": lr_results["rmse"], "R2": lr_results["r2"]},
        {"Model": rf_results["model"], "MAE": rf_results["mae"], "RMSE": rf_results["rmse"], "R2": rf_results["r2"]},
    ]
    metrics_df = pd.DataFrame(models_comparison)

    # Dynamic model selection (lowest RMSE, tiebreak on higher R2)
    candidate_results = [lr_results, rf_results]
    best_result = min(candidate_results, key=lambda res: (res["rmse"], -res["r2"]))

    # Extract Feature Importances from Random Forest
    rf_fitted = rf_pipeline.named_steps["regressor"]
    cat_encoder = rf_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(["weather", "shift"]).tolist()
    all_feature_names = ["month", "day_of_month", "day_of_week", "weekend", "holiday", "special_event"] + cat_feature_names

    importances = rf_fitted.feature_importances_
    fi_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Save outputs
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    fi_df.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    # Render Chart 10
    save_feature_importance_chart(fi_df)

    selection_summary = {
        "selected_model_name": best_result["model"],
        "best_pipeline": best_result["pipeline"],
        "mae": best_result["mae"],
        "rmse": best_result["rmse"],
        "r2": best_result["r2"],
        "all_results": {lr_results["model"]: lr_results, rf_results["model"]: rf_results},
    }

    return selection_summary, metrics_df, fi_df


def run_forecasting() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Load data, execute demand forecasting pipeline, and return trained models."""
    df = pd.read_csv(DATA_FILE)
    shift_df = prepare_demand_features(df)
    selection_summary, metrics_df, fi_df = train_and_compare_models(shift_df)
    return selection_summary, metrics_df, fi_df


if __name__ == "__main__":
    summary, metrics, fi = run_forecasting()
    print("Demand Forecasting Pipeline Complete:")
    print(metrics.to_string(index=False))
    print(f"\nDynamically Selected Best Model: {summary['selected_model_name']} (RMSE: {summary['rmse']}, R2: {summary['r2']})")
    print("\nTop 5 Feature Importances:")
    print(fi.head(5).to_string(index=False))
