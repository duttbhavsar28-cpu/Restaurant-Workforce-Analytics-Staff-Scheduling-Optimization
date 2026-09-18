"""
generate_data.py
================
Synthetic Data Generator for Restaurant Workforce Analytics & Staff Scheduling Optimization.

Generates at least 1,500 (default: 3,650) realistic operational shift records across a full calendar
year (365 days x 2 shifts x 5 roles). Encodes realistic restaurant domain relationships:
- Shift volume variations (Dinner > Lunch)
- Day-of-week and weekend demand surges
- Holiday and local event spikes
- Adverse weather impact
- Role productivity benchmarks and labor costs
- Manual scheduling biases and understaffing penalties on customer service scores.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "restaurant_workforce.csv"

# ---------------------------------------------------------------------------
# Configuration & Domain Benchmarks
# ---------------------------------------------------------------------------
ROLES: dict[str, dict[str, float]] = {
    "Server": {"productivity": 8, "hourly_wage": 14.00, "base_absence_rate": 0.06},
    "Cook": {"productivity": 16, "hourly_wage": 20.00, "base_absence_rate": 0.04},
    "Host": {"productivity": 25, "hourly_wage": 15.00, "base_absence_rate": 0.05},
    "Cashier": {"productivity": 28, "hourly_wage": 14.00, "base_absence_rate": 0.05},
    "Dishwasher": {"productivity": 22, "hourly_wage": 13.00, "base_absence_rate": 0.08},
}

SHIFT_HOURS = {
    "Lunch": 5.0,
    "Dinner": 6.5,
}

# Major US public & food-service holidays (Month, Day)
HOLIDAYS_SET = {
    (1, 1),   # New Year's Day
    (2, 14),  # Valentine's Day
    (3, 17),  # St. Patrick's Day
    (5, 12),  # Mother's Day (approx)
    (5, 27),  # Memorial Day
    (6, 16),  # Father's Day (approx)
    (7, 4),   # Independence Day
    (9, 2),   # Labor Day
    (10, 31), # Halloween
    (11, 28), # Thanksgiving
    (12, 24), # Christmas Eve
    (12, 31), # New Year's Eve
}


def is_holiday(dt: date) -> int:
    """Return 1 if date matches a known high-volume holiday, else 0."""
    return 1 if (dt.month, dt.day) in HOLIDAYS_SET else 0


def generate_workforce_data(
    start_date: str = "2024-01-01",
    days: int = 365,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic restaurant shift records.

    Parameters
    ----------
    start_date : str
        Start date string in YYYY-MM-DD format.
    days : int
        Number of consecutive days to simulate (default 365).
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic restaurant shift records.
    """
    np.random.seed(random_seed)
    records = []
    base_date = date.fromisoformat(start_date)

    # Weather probabilities
    weather_options = ["Sunny", "Cloudy", "Rainy", "Snow"]
    weather_weights_normal = [0.55, 0.25, 0.18, 0.02]
    weather_weights_winter = [0.40, 0.25, 0.15, 0.20]

    for day_idx in range(days):
        current_date = base_date + timedelta(days=day_idx)
        dow = current_date.weekday()  # Monday is 0, Sunday is 6
        day_name = current_date.strftime("%A")
        weekend = 1 if dow in (4, 5, 6) else 0  # Fri, Sat, Sun considered peak weekend
        holiday_flag = is_holiday(current_date)

        # Weather simulation with seasonal winter snow
        if current_date.month in (12, 1, 2):
            weather = np.random.choice(weather_options, p=weather_weights_winter)
        else:
            weather = np.random.choice(weather_options, p=weather_weights_normal)

        # Special local events (concerts, games, street festivals ~ 8% of days)
        event_prob = 0.12 if weekend else 0.06
        special_event = 1 if np.random.rand() < event_prob else 0

        # Simulate customer demand for Lunch and Dinner
        for shift in ["Lunch", "Dinner"]:
            # Baseline demand
            if shift == "Lunch":
                base_covers = np.random.normal(loc=100, scale=12)
                # Weekday lunch has business crowd, weekend lunch has brunch crowd
                dow_multiplier = 1.0 + 0.05 * (dow in (4, 5, 6))
            else:  # Dinner
                base_covers = np.random.normal(loc=185, scale=18)
                # Friday and Saturday dinners see high surge
                if dow == 4:  # Friday
                    dow_multiplier = 1.28
                elif dow == 5:  # Saturday
                    dow_multiplier = 1.35
                elif dow == 6:  # Sunday
                    dow_multiplier = 1.15
                else:
                    dow_multiplier = 0.95

            # Impact modifiers
            holiday_mult = 1.30 if holiday_flag else 1.0
            event_mult = 1.20 if special_event else 1.0

            # Weather impact: Rain reduces by ~12%, Snow by ~25%
            weather_mult = 1.0
            if weather == "Rainy":
                weather_mult = 0.88
            elif weather == "Snow":
                weather_mult = 0.75
            elif weather == "Cloudy":
                weather_mult = 0.98

            # Compute final covers with minor random noise
            shift_covers = base_covers * dow_multiplier * holiday_mult * event_mult * weather_mult
            shift_covers = max(35, int(round(shift_covers + np.random.normal(0, 4))))

            # Shift duration
            hours_per_emp = SHIFT_HOURS[shift]

            # Generate records for each role
            for role, role_cfg in ROLES.items():
                prod = role_cfg["productivity"]
                wage = role_cfg["hourly_wage"]
                base_abs_rate = role_cfg["base_absence_rate"]

                # True optimal staffing required
                required = max(1, math.ceil(shift_covers / prod))

                # Simulate manual management scheduling bias:
                # Managers often schedule on static habits rather than demand data:
                # - Over-schedule slightly on slow shifts (covers < 110)
                # - Under-schedule on peak weekend dinners (underestimating surges)
                if shift_covers < 110:
                    # Propensity to over-schedule by +1 or exact
                    sched_delta = np.random.choice([0, 1], p=[0.45, 0.55])
                elif shift_covers > 210:
                    # Propensity to under-schedule by -1 or -2 due to conservative planning
                    sched_delta = np.random.choice([-2, -1, 0], p=[0.20, 0.50, 0.30])
                else:
                    sched_delta = np.random.choice([-1, 0, 1], p=[0.25, 0.55, 0.20])

                scheduled = max(1, required + sched_delta)

                # Simulate absenteeism
                # Higher likelihood on weekends and for demanding roles
                shift_abs_prob = base_abs_rate * (1.3 if weekend else 1.0)
                absent_count = np.random.binomial(n=scheduled, p=min(0.25, shift_abs_prob))

                # Actual present cannot be less than 1 if at least 1 was scheduled
                actual_present = max(1, scheduled - absent_count)

                # Total labor cost for this role shift
                labor_cost = round(actual_present * hours_per_emp * wage, 2)

                # Service Score calculation (1.0 to 5.0)
                # Base satisfaction is 4.70.
                # Understaffing (actual < required) significantly degrades customer experience
                # due to slow ticket times, delayed drinks, dirty tables, and stressed staff.
                base_service = 4.72 + np.random.normal(0, 0.12)
                staffing_shortfall = max(0, required - actual_present)

                if staffing_shortfall > 0:
                    # Penalty: ~0.55 drop per missing employee
                    service_penalty = staffing_shortfall * np.random.uniform(0.45, 0.70)
                    service_score = base_service - service_penalty
                else:
                    # Adequately or slightly overstaffed gives optimal score
                    service_score = base_service + np.random.uniform(0.02, 0.10)

                # Ensure bounded within 1.0 to 5.0
                service_score = round(float(np.clip(service_score, 1.0, 5.0)), 2)

                records.append({
                    "date": current_date.isoformat(),
                    "day_of_week": dow,
                    "day_name": day_name,
                    "weekend": weekend,
                    "holiday": holiday_flag,
                    "special_event": special_event,
                    "weather": weather,
                    "shift": shift,
                    "role": role,
                    "covers": shift_covers,
                    "employees_required": required,
                    "employees_scheduled": scheduled,
                    "absent": absent_count,
                    "actual_present": actual_present,
                    "hours_per_employee": hours_per_emp,
                    "hourly_wage": wage,
                    "labor_cost": labor_cost,
                    "service_score": service_score,
                })

    df = pd.DataFrame(records)
    return df


def main() -> Path:
    """Generate and save workforce dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating synthetic restaurant workforce records (365 days, 2 shifts, 5 roles)...")
    df = generate_workforce_data(start_date="2024-01-01", days=365, random_seed=42)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset successfully created at: {OUTPUT_FILE}")
    print(f"Total rows: {len(df):,} | Total columns: {len(df.columns)}")
    return OUTPUT_FILE


if __name__ == "__main__":
    main()
