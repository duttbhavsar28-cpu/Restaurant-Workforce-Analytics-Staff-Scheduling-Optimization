"""
test_pipeline.py
================
Automated Test Suite for Restaurant Workforce Analytics & Staff Scheduling Optimization.

Verifies:
1. Data generation integrity (row counts, column schemas, value ranges)
2. Feature engineering mathematical correctness
3. Machine learning model convergence and evaluation validity
4. Staffing optimization ratios across all operational roles
5. Persistence of all 10 visual charts and analytical reports
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
import pandas as pd

# Setup pathing
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generate_data import generate_workforce_data, ROLES
from src.data_analysis import (
    engineer_features,
    compute_summary_metrics,
    CHARTS_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR,
)
from src.demand_forecasting import (
    prepare_demand_features,
    train_and_compare_models,
)
from src.staffing_optimizer import (
    calculate_required_staff,
    generate_7_day_recommendations,
    ROLE_PRODUCTIVITY,
)


def test_synthetic_data_integrity():
    """Verify that generated workforce data satisfies all structural and domain constraints."""
    df = generate_workforce_data(start_date="2024-01-01", days=365, random_seed=42)

    # Check minimum record requirement (>= 1,500 records)
    assert len(df) >= 1500, f"Expected at least 1,500 records, got {len(df)}"
    assert len(df) == 3650, f"Expected 3,650 records (365 days x 2 shifts x 5 roles), got {len(df)}"

    # Check required columns
    required_cols = [
        "date", "day_of_week", "day_name", "weekend", "holiday", "special_event",
        "weather", "shift", "role", "covers", "employees_required",
        "employees_scheduled", "absent", "actual_present", "hours_per_employee",
        "hourly_wage", "labor_cost", "service_score"
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Check value bounds
    assert df["covers"].min() > 0, "Covers must be strictly positive"
    assert df["actual_present"].min() >= 1, "At least 1 employee must be present per shift"
    assert df["service_score"].between(1.0, 5.0).all(), "Service scores must be bounded between 1.0 and 5.0"
    assert df["labor_cost"].min() > 0, "Labor costs must be strictly positive"


def test_feature_engineering_correctness():
    """Verify feature engineering formulas for staffing gaps, flags, and absence rates."""
    df_raw = generate_workforce_data(start_date="2024-01-01", days=30, random_seed=123)
    df = engineer_features(df_raw)

    # Check staffing gap formula
    expected_gap = df["actual_present"] - df["employees_required"]
    pd.testing.assert_series_equal(df["staffing_gap"], expected_gap, check_names=False)

    # Check understaffed / overstaffed boolean flags
    assert ((df["staffing_gap"] < 0) == (df["understaffed"] == 1)).all()
    assert ((df["staffing_gap"] > 0) == (df["overstaffed"] == 1)).all()

    # Check absence rate bounds
    assert df["absence_rate"].between(0.0, 1.0).all()

    # Check metrics computation returns valid non-empty dict
    metrics = compute_summary_metrics(df)
    assert metrics["total_records"] == len(df)
    assert metrics["total_labor_cost"] > 0
    assert 0 <= metrics["understaffing_percentage"] <= 100


def test_demand_forecasting_pipeline():
    """Verify ML models train successfully and meet minimum R2 threshold."""
    df_raw = generate_workforce_data(start_date="2024-01-01", days=180, random_seed=99)
    shift_df = prepare_demand_features(df_raw)

    selection_summary, metrics_df, fi_df = train_and_compare_models(shift_df)

    assert "Linear Regression" in metrics_df["Model"].values
    assert "Random Forest" in metrics_df["Model"].values

    # Check that model exhibits strong predictive power (R2 > 0.60 on test set)
    assert selection_summary["r2"] > 0.60, f"R2 score too low: {selection_summary['r2']}"
    assert selection_summary["mae"] > 0
    assert selection_summary["rmse"] > 0

    # Check feature importance output
    assert len(fi_df) > 0
    assert "importance" in fi_df.columns
    assert math.isclose(fi_df["importance"].sum(), 1.0, rel_tol=1e-2)


def test_staffing_optimizer_ratios():
    """Verify exact staffing calculation per role based on defined productivity ratios."""
    test_covers = 160

    assert calculate_required_staff(test_covers, "Server") == math.ceil(160 / 8)      # 20
    assert calculate_required_staff(test_covers, "Cook") == math.ceil(160 / 16)        # 10
    assert calculate_required_staff(test_covers, "Host") == math.ceil(160 / 25)        # 7
    assert calculate_required_staff(test_covers, "Cashier") == math.ceil(160 / 28)     # 6
    assert calculate_required_staff(test_covers, "Dishwasher") == math.ceil(160 / 22)  # 8

    # Edge cases
    assert calculate_required_staff(0, "Server") == 1
    raised_error = False
    try:
        calculate_required_staff(100, "UnknownRole")
    except ValueError:
        raised_error = True
    assert raised_error, "calculate_required_staff should raise ValueError for unknown role"


def test_output_artifacts_exist():
    """Verify that all 10 charts, analytical reports, and predictions are generated."""
    expected_charts = [
        "daily_customer_demand_trend.png",
        "customer_demand_by_day_of_week.png",
        "lunch_vs_dinner_demand.png",
        "required_vs_actual_staff.png",
        "understaffing_rate_by_shift.png",
        "labor_cost_by_shift.png",
        "absence_rate_by_role.png",
        "staffing_gap_vs_service_score.png",
        "customer_demand_vs_labor_cost.png",
        "feature_importance.png",
    ]

    for chart_name in expected_charts:
        chart_path = CHARTS_DIR / chart_name
        assert chart_path.exists(), f"Chart missing: {chart_path}"
        assert chart_path.stat().st_size > 1000, f"Chart file is empty: {chart_path}"

    expected_reports = [
        "workforce_summary_metrics.csv",
        "performance_by_shift.csv",
        "performance_by_day.csv",
        "performance_by_role.csv",
        "model_metrics.csv",
        "feature_importance.csv",
        "business_insights.txt",
    ]
    for rep_name in expected_reports:
        rep_path = REPORTS_DIR / rep_name
        assert rep_path.exists(), f"Report missing: {rep_path}"

    rec_path = PREDICTIONS_DIR / "7_day_staffing_recommendations.csv"
    assert rec_path.exists(), f"Predictions file missing: {rec_path}"


if __name__ == "__main__":
    print("Running automated unit & integration verification tests...")
    test_synthetic_data_integrity()
    print("✓ test_synthetic_data_integrity passed.")
    test_feature_engineering_correctness()
    print("✓ test_feature_engineering_correctness passed.")
    test_demand_forecasting_pipeline()
    print("✓ test_demand_forecasting_pipeline passed.")
    test_staffing_optimizer_ratios()
    print("✓ test_staffing_optimizer_ratios passed.")
    test_output_artifacts_exist()
    print("✓ test_output_artifacts_exist passed.")
    print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
