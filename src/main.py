"""
main.py
=======
Main Orchestrator Application for Restaurant Workforce Analytics & Staff Scheduling Optimization.

Executes the complete end-to-end data analytics and machine learning pipeline:
1. Data Generation & Ingestion
2. Data Validation & Feature Engineering
3. Descriptive & Exploratory Workforce Analytics
4. Machine Learning Demand Forecasting (Linear Regression vs. Random Forest)
5. Automated Dynamic Model Evaluation & Selection
6. Operational Staff Scheduling Optimization (7-Day Schedule)
7. Visual Analytics & Publication-Quality Charts
8. Empirical Business Insights Interpretation
9. Interactive Terminal KPI Dashboard
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Python Path & Project Directory Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generate_data import main as generate_dataset_if_needed, OUTPUT_FILE as DATA_FILE
from src.data_analysis import (
    load_and_validate_data,
    engineer_features,
    compute_summary_metrics,
    generate_group_reports,
    save_reports,
    generate_all_charts,
    generate_business_insights,
)
from src.demand_forecasting import (
    prepare_demand_features,
    train_and_compare_models,
)
from src.staffing_optimizer import (
    generate_7_day_recommendations,
    preview_recommendations_by_shift,
)


def run_pipeline() -> None:
    """Run full workforce analytics and scheduling optimization pipeline."""
    # -----------------------------------------------------------------------
    # Step 1: Ensure Dataset Exists
    # -----------------------------------------------------------------------
    if not DATA_FILE.exists():
        print("[Step 1/6] Generating synthetic restaurant workforce data...")
        generate_dataset_if_needed()
    else:
        print(f"[Step 1/6] Loaded existing dataset: {DATA_FILE.name}")

    # -----------------------------------------------------------------------
    # Step 2: Data Loading, Validation & Feature Engineering
    # -----------------------------------------------------------------------
    print("[Step 2/6] Validating dataset & engineering workforce features...")
    df_raw = load_and_validate_data(DATA_FILE)
    df = engineer_features(df_raw)
    metrics = compute_summary_metrics(df)
    reports = generate_group_reports(df)
    save_reports(metrics, reports)

    # -----------------------------------------------------------------------
    # Step 3: Machine Learning Demand Forecasting
    # -----------------------------------------------------------------------
    print("[Step 3/6] Training & benchmarking ML demand forecasting models...")
    shift_demand_df = prepare_demand_features(df)
    selection_summary, metrics_df, fi_df = train_and_compare_models(shift_demand_df)

    # -----------------------------------------------------------------------
    # Step 4: Staffing Scheduling Optimization
    # -----------------------------------------------------------------------
    print("[Step 4/6] Optimizing 7-day staffing schedules by role...")
    best_pipeline = selection_summary["best_pipeline"]
    rec_df = generate_7_day_recommendations(best_pipeline, start_date="2025-01-01")

    # -----------------------------------------------------------------------
    # Step 5: Visual Analytics & Business Insights
    # -----------------------------------------------------------------------
    print("[Step 5/6] Generating visual charts and empirical business insights...")
    generate_all_charts(df, feature_importance_df=fi_df)
    insights_text = generate_business_insights(
        df=df,
        metrics=metrics,
        reports=reports,
        feature_importance_df=fi_df,
        recommendations_df=rec_df,
    )

    # -----------------------------------------------------------------------
    # Step 6: Formatted Terminal Dashboard Output
    # -----------------------------------------------------------------------
    print("[Step 6/6] Rendering executive dashboard...\n")

    # Extract sample shift recommendation (First Friday dinner or sample dinner)
    sample_preview = preview_recommendations_by_shift(rec_df, "2025-01-01", "Dinner")

    lr_row = metrics_df[metrics_df["Model"] == "Linear Regression"].iloc[0]
    rf_row = metrics_df[metrics_df["Model"] == "Random Forest"].iloc[0]

    # Display clean formatted terminal report matching specifications
    border = "=" * 50
    print(border)
    print("RESTAURANT WORKFORCE ANALYTICS")
    print(border)
    print(f"Total Records: {metrics['total_records']:,}")
    print(f"Total Labor Cost: ${metrics['total_labor_cost']:,.2f}")
    print(f"Average Customer Covers: {metrics['average_customer_covers']:.1f}")
    print(f"Average Service Score: {metrics['average_service_score']:.2f}")
    print(f"Understaffing Rate: {metrics['understaffing_percentage']:.1f}%")
    print(f"Absence Rate: {metrics['overall_absence_rate']:.1f}%")
    print()

    print(border)
    print("DEMAND FORECASTING")
    print(border)
    print("Linear Regression")
    print(f"MAE: {lr_row['MAE']:.2f}")
    print(f"RMSE: {lr_row['RMSE']:.2f}")
    print(f"R²: {lr_row['R2']:.4f}")
    print()
    print("Random Forest")
    print(f"MAE: {rf_row['MAE']:.2f}")
    print(f"RMSE: {rf_row['RMSE']:.2f}")
    print(f"R²: {rf_row['R2']:.4f}")
    print()
    print(f"Selected Model: {selection_summary['selected_model_name']}")
    print()

    print(border)
    print("STAFFING RECOMMENDATION")
    print(border)
    if sample_preview:
        print(f"Date: {sample_preview['date']}")
        print(f"Shift: {sample_preview['shift']}")
        print()
        print(f"Predicted Customers: {sample_preview['predicted_covers']}")
        print()
        for role, count in sample_preview["staffing"].items():
            print(f"{role}: {count}")
    print()

    print(border)
    print("BUSINESS INSIGHTS")
    print(border)
    print(insights_text.strip())
    print()
    print(border)
    print("PROJECT PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print(border)
    print(f"• Charts generated: outputs/charts/ (10 figures)")
    print(f"• Reports generated: outputs/reports/ (Summary metrics, models, insights)")
    print(f"• Predictions generated: outputs/predictions/ (7-day schedule)")
    print(border)


if __name__ == "__main__":
    run_pipeline()
