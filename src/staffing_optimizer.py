"""
staffing_optimizer.py
=====================
Staff Scheduling Optimization and 7-Day Forward Recommendation Engine.

Translates machine-learning predicted customer demand (covers) into optimal staffing
allocations per role based on empirical restaurant productivity ratios:
- Server: 1 per 8 customers
- Cook: 1 per 16 customers
- Host: 1 per 25 customers
- Cashier: 1 per 28 customers
- Dishwasher: 1 per 22 customers

Generates a 7-day optimized schedule across all shifts and roles, compares against
baseline scheduled levels, detects staffing gaps, and estimates labor costs.
"""

from __future__ import annotations

import math
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any
import pandas as pd

# ---------------------------------------------------------------------------
# Path Configuration & Python Path Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generate_data import ROLES, SHIFT_HOURS, is_holiday

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
RECOMMENDATIONS_FILE = PREDICTIONS_DIR / "7_day_staffing_recommendations.csv"

# Productivity thresholds (customers per staff member)
ROLE_PRODUCTIVITY = {
    "Server": 8,
    "Cook": 16,
    "Host": 25,
    "Cashier": 28,
    "Dishwasher": 22,
}


def calculate_required_staff(predicted_covers: int | float, role: str) -> int:
    """
    Calculate required employee headcount for a given role and predicted customer covers.

    Parameters
    ----------
    predicted_covers : int or float
        Expected customer covers for the shift.
    role : str
        Employee role ('Server', 'Cook', 'Host', 'Cashier', 'Dishwasher').

    Returns
    -------
    int
        Recommended staff count (minimum 1).
    """
    if role not in ROLE_PRODUCTIVITY:
        raise ValueError(f"Unknown role '{role}'. Expected one of {list(ROLE_PRODUCTIVITY.keys())}")

    ratio = ROLE_PRODUCTIVITY[role]
    if predicted_covers <= 0:
        return 1

    recommended = math.ceil(predicted_covers / ratio)
    return max(1, recommended)


def simulate_baseline_scheduled_staff(predicted_covers: float, role: str) -> int:
    """
    Simulate standard legacy/manual scheduling behavior:
    Managers without predictive analytics often schedule static headcount
    based on historical averages rather than dynamic demand, leading to:
    - Overstaffing on slow shifts
    - Understaffing on high-volume peak shifts
    """
    if role == "Server":
        base = 14 if predicted_covers > 150 else 11
    elif role == "Cook":
        base = 8 if predicted_covers > 150 else 6
    elif role == "Host":
        base = 5 if predicted_covers > 150 else 4
    elif role == "Cashier":
        base = 5 if predicted_covers > 150 else 4
    elif role == "Dishwasher":
        base = 6 if predicted_covers > 150 else 4
    else:
        base = 4

    return max(1, base)


def generate_7_day_recommendations(
    trained_pipeline: Any,
    start_date: str = "2025-01-01",
) -> pd.DataFrame:
    """
    Generate dynamic 7-day staffing recommendations across Lunch and Dinner shifts.

    Parameters
    ----------
    trained_pipeline : Pipeline
        Fitted Scikit-Learn pipeline from demand_forecasting.
    start_date : str
        First day of the 7-day forecast window.

    Returns
    -------
    pd.DataFrame
        Detailed shift-by-role staffing recommendations and cost projections.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    base_dt = date.fromisoformat(start_date)
    records = []

    # Realistic upcoming forecast features
    # (Simulated upcoming weather conditions)
    forecasted_weather = ["Sunny", "Cloudy", "Sunny", "Rainy", "Sunny", "Sunny", "Cloudy"]

    for day_offset in range(7):
        curr_dt = base_dt + timedelta(days=day_offset)
        dow = curr_dt.weekday()
        day_name = curr_dt.strftime("%A")
        weekend = 1 if dow in (4, 5, 6) else 0
        holiday_flag = is_holiday(curr_dt)
        weather = forecasted_weather[day_offset]
        special_event = 1 if (weekend and dow == 5) else 0

        for shift in ["Lunch", "Dinner"]:
            hours = SHIFT_HOURS[shift]

            # Construct input features for ML demand model
            input_row = pd.DataFrame([{
                "month": curr_dt.month,
                "day_of_month": curr_dt.day,
                "day_of_week": dow,
                "weekend": weekend,
                "holiday": holiday_flag,
                "special_event": special_event,
                "weather": weather,
                "shift": shift,
            }])

            # Predict customer demand
            predicted_covers = float(trained_pipeline.predict(input_row)[0])
            predicted_covers = max(40, round(predicted_covers))

            # Optimize staffing for each role
            for role in ["Server", "Cook", "Host", "Cashier", "Dishwasher"]:
                recommended_staff = calculate_required_staff(predicted_covers, role)
                current_scheduled = simulate_baseline_scheduled_staff(predicted_covers, role)
                staffing_gap = recommended_staff - current_scheduled

                wage = ROLES[role]["hourly_wage"]
                estimated_cost = round(recommended_staff * hours * wage, 2)
                legacy_cost = round(current_scheduled * hours * wage, 2)
                cost_variance = round(estimated_cost - legacy_cost, 2)

                status = "Balanced"
                if staffing_gap > 0:
                    status = "Understaffed (Add Staff)"
                elif staffing_gap < 0:
                    status = "Overstaffed (Reduce Staff)"

                records.append({
                    "date": curr_dt.isoformat(),
                    "day_name": day_name,
                    "shift": shift,
                    "role": role,
                    "predicted_covers": int(predicted_covers),
                    "recommended_staff": int(recommended_staff),
                    "current_scheduled_staff": int(current_scheduled),
                    "staffing_gap": int(staffing_gap),
                    "staffing_status": status,
                    "hourly_wage": wage,
                    "hours_per_employee": hours,
                    "recommended_labor_cost": estimated_cost,
                    "legacy_scheduled_labor_cost": legacy_cost,
                    "cost_variance": cost_variance,
                })

    rec_df = pd.DataFrame(records)
    rec_df.to_csv(RECOMMENDATIONS_FILE, index=False)
    return rec_df


def preview_recommendations_by_shift(rec_df: pd.DataFrame, target_date: str, target_shift: str) -> dict[str, Any]:
    """Extract a clean dictionary summary of recommendations for a specific shift."""
    sub = rec_df[(rec_df["date"] == target_date) & (rec_df["shift"] == target_shift)]
    if sub.empty:
        return {}

    covers = sub["predicted_covers"].iloc[0]
    roles_dict = dict(zip(sub["role"], sub["recommended_staff"]))
    total_cost = sub["recommended_labor_cost"].sum()

    return {
        "date": target_date,
        "shift": target_shift,
        "predicted_covers": covers,
        "staffing": roles_dict,
        "estimated_labor_cost": round(total_cost, 2),
    }


if __name__ == "__main__":
    from src.demand_forecasting import run_forecasting

    forecasting_summary, _, _ = run_forecasting()
    rec_df = generate_7_day_recommendations(forecasting_summary["best_pipeline"])
    print(f"7-Day Staffing Recommendations Generated ({len(rec_df)} shift-role rows).")
    print(f"Saved to: {RECOMMENDATIONS_FILE}\n")

    # Sample preview
    sample_preview = preview_recommendations_by_shift(rec_df, "2025-01-01", "Dinner")
    print("Sample Shift Preview:")
    print(f"Date: {sample_preview['date']} | Shift: {sample_preview['shift']} | Predicted Covers: {sample_preview['predicted_covers']}")
    for r, count in sample_preview["staffing"].items():
        print(f"  - {r}: {count}")
    print(f"Estimated Labor Cost: ${sample_preview['estimated_labor_cost']}")
