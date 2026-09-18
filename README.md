# Restaurant Workforce Analytics & Staff Scheduling Optimization

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Scikit-Learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Analytics-Pandas-150458.svg)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Visuals-Matplotlib-11557c.svg)](https://matplotlib.org/)

An end-to-end, **Python-only** Business Analytics and Machine Learning system that replaces gut-feel restaurant employee scheduling with empirical demand forecasting and role-level staffing optimization.

---

## 📌 Executive Summary & Business Problem

In hospitality management, general managers frequently build weekly shift rosters relying on intuition, habit, or flat static headcounts. This informal approach causes systematic operational failures:

1. **Understaffing During Peak Surges**: Friday and Saturday evening surges overwhelm dining rooms and kitchens, causing long wait times, delayed food delivery, stressed employees, and depressed customer review scores.
2. **Overstaffing During Off-Peak Hours**: Slow weekday lunch shifts carry redundant staff, driving labor costs up and eroding operating profit margins.
3. **High Labor Costs**: Payroll is the single largest controllable expense for restaurants (typically 28%–35% of gross revenue); misallocations directly harm profitability.
4. **Workload Imbalance & Burnout**: Chronic understaffing spikes turnover and employee absenteeism, creating a vicious operational cycle.
5. **Lack of Predictive Foresight**: Without quantitative customer volume forecasting, managers cannot anticipate upcoming holiday surges or weather disruptions.

### Business Objective
This project builds an end-to-end, production-ready analytical engine that:
- Ingests and validates full-year restaurant operational shift records.
- Engineers workforce performance metrics (staffing gaps, absenteeism rates, cost per cover).
- Trains and benchmarks supervised Machine Learning models (Linear Regression vs. Random Forest) to forecast customer covers.
- Optimizes shift staffing headcount dynamically across five distinct operational roles (**Server, Cook, Host, Cashier, Dishwasher**).
- Produces automated business insights and a forward-looking 7-day schedule recommendation.

---

## 🛠 Technology Stack

This project is built **strictly in pure Python** without dependencies on external databases, SQL servers, Power BI, Tableau, Excel plugins, or paid cloud APIs:

| Layer | Technology | Purpose |
|---|---|---|
| **Programming Language** | Python 3.10+ | Core application logic and execution |
| **Data Manipulation** | `pandas` (3.0+), `numpy` (2.5+) | Data processing, aggregation, and vector calculations |
| **Machine Learning** | `scikit-learn` (1.9+) | Feature preprocessing, regression modeling, cross-validation, and metrics |
| **Visual Analytics** | `matplotlib` (3.11+) | Publication-quality statistical visualizations and distributions |
| **Path Handling** | `pathlib.Path` | Platform-independent path resolution (Windows, macOS, Linux) |

---

## 📂 Project Architecture

```
restaurant_workforce_python_only/
│
├── data/
│   └── restaurant_workforce.csv          # 3,650 shift records across 365 operational days
│
├── src/
│   ├── __init__.py                       # Package initializer
│   ├── generate_data.py                  # Realistic synthetic workforce data generator
│   ├── data_analysis.py                  # EDA, validation, KPI computation & chart generation
│   ├── demand_forecasting.py             # ML pipeline (Linear Regression vs. Random Forest)
│   ├── staffing_optimizer.py             # Role productivity optimization & 7-day forward scheduler
│   └── main.py                           # Application orchestrator & executive CLI dashboard
│
├── outputs/
│   ├── charts/                           # 10 publication-quality Matplotlib figures (.png)
│   │   ├── daily_customer_demand_trend.png
│   │   ├── customer_demand_by_day_of_week.png
│   │   ├── lunch_vs_dinner_demand.png
│   │   ├── required_vs_actual_staff.png
│   │   ├── understaffing_rate_by_shift.png
│   │   ├── labor_cost_by_shift.png
│   │   ├── absence_rate_by_role.png
│   │   ├── staffing_gap_vs_service_score.png
│   │   ├── customer_demand_vs_labor_cost.png
│   │   └── feature_importance.png
│   │
│   ├── reports/                          # Analytical CSV summaries & executive findings
│   │   ├── workforce_summary_metrics.csv
│   │   ├── performance_by_shift.csv
│   │   ├── performance_by_day.csv
│   │   ├── performance_by_role.csv
│   │   ├── performance_by_weekend.csv
│   │   ├── performance_by_weather.csv
│   │   ├── performance_by_holiday.csv
│   │   ├── performance_by_special_event.csv
│   │   ├── model_metrics.csv
│   │   ├── feature_importance.csv
│   │   └── business_insights.txt
│   │
│   └── predictions/                      # Model forecasts & optimization outputs
│       └── 7_day_staffing_recommendations.csv
│
├── tests/
│   └── test_pipeline.py                  # Automated unit and integration test suite
│
├── requirements.txt                      # Minimum necessary dependencies
├── README.md                             # Comprehensive technical and business documentation
├── .gitignore                            # Standard Python gitignore
└── run_project.py                        # Single-command root execution entrypoint
```

---

## 📊 Dataset Description & Domain Assumptions

> **Data Disclosure**: In compliance with professional ethics, this project utilizes a high-fidelity synthetic operational dataset (`data/restaurant_workforce.csv`) reflecting authentic restaurant operational dynamics. No proprietary corporate data was extracted.

The dataset models **365 consecutive operating days** across two daily shifts (**Lunch** and **Dinner**) and five key operational job categories, producing **3,650 records**.

### Dataset Schema
| Column | Type | Description |
|---|---|---|
| `date` | String | Operating date in ISO format (`YYYY-MM-DD`) |
| `day_of_week` | Integer | Day index (0 = Monday, 6 = Sunday) |
| `day_name` | String | Day name (Monday through Sunday) |
| `weekend` | Binary | 1 for peak weekend days (Friday, Saturday, Sunday), 0 otherwise |
| `holiday` | Binary | 1 for major public holidays / high-volume events |
| `special_event` | Binary | 1 for local concerts, sports games, or street festivals |
| `weather` | Categorical | Weather condition (`Sunny`, `Cloudy`, `Rainy`, `Snow`) |
| `shift` | Categorical | Shift type (`Lunch` [5.0 hrs], `Dinner` [6.5 hrs]) |
| `role` | Categorical | Job category (`Server`, `Cook`, `Host`, `Cashier`, `Dishwasher`) |
| `covers` | Integer | Total customer meals served during the shift |
| `employees_required` | Integer | Demand-derived optimal headcount |
| `employees_scheduled` | Integer | Historical manual scheduled headcount (incorporating human bias) |
| `absent` | Integer | Number of scheduled employees absent |
| `actual_present` | Integer | Actual staff on duty (`employees_scheduled - absent`) |
| `hours_per_employee` | Float | Scheduled shift duration (Lunch: 5.0h, Dinner: 6.5h) |
| `hourly_wage` | Float | Role-specific hourly wage ($13.00 – $20.00) |
| `labor_cost` | Float | Total labor payroll incurred for that role shift |
| `service_score` | Float | Guest review rating (1.0 to 5.0 scale) |

### Empirical Domain Benchmarks
- **Shift Dynamics**: Dinner demand averages ~197 covers/shift vs. Lunch averaging ~101 covers/shift.
- **Weekend Effect**: Friday & Saturday dinners experience a 28%–35% volume premium.
- **Weather Modifiers**: Rain dampens walk-ins by ~12%; winter snow dampens demand by ~25%.
- **Role Productivity Standards**:
  - **Server**: 1 employee per 8 covers ($\lceil \text{covers} / 8 \rceil$)
  - **Cook**: 1 employee per 16 covers ($\lceil \text{covers} / 16 \rceil$)
  - **Host**: 1 employee per 25 covers ($\lceil \text{covers} / 25 \rceil$)
  - **Cashier**: 1 employee per 28 covers ($\lceil \text{covers} / 28 \rceil$)
  - **Dishwasher**: 1 employee per 22 covers ($\lceil \text{covers} / 22 \rceil$)
- **Service Penalty**: When actual staffing falls below required capacity, guest ratings drop sharply ($\approx -0.55$ points per missing employee) due to table turnover delays and kitchen ticket bottlenecks.

---

## 🔍 Feature Engineering & Key Performance Indicators (KPIs)

The analytical module calculates key operational features:

```python
# Staffing Gap: Negative indicates shortfall (understaffing), Positive indicates surplus (overstaffing)
staffing_gap = actual_present - employees_required

understaffed = 1 if staffing_gap < 0 else 0
overstaffed  = 1 if staffing_gap > 0 else 0

# Absence Rate
absence_rate = absent / employees_scheduled

# Unit Labor Cost
labor_cost_per_cover = labor_cost / covers
```

### Core Business KPIs
- **Total Annual Labor Expenditure**: $3,000,834.50
- **Average Labor Cost per Guest**: $27.57 / cover
- **Baseline Understaffing Frequency**: 46.5% of shifts experience staffing shortfalls under manual scheduling.
- **Dinner Understaffing Rate**: Spikes to **65.4%** on evening shifts.
- **Guest Satisfaction Sensitivity**: A strong correlation ($r = +0.964$) between staffing adequacy and customer review ratings (understaffed shifts drop from 4.78 to 3.75).

---

## 🤖 Machine Learning Demand Forecasting

To move beyond static rule-of-thumb scheduling, the application trains supervised regression pipelines to predict customer demand (`covers`).

### Pipeline & Preprocessing
- **Feature Set**: `month`, `day_of_month`, `day_of_week`, `weekend`, `holiday`, `special_event`, `weather`, `shift`.
- **Preprocessors**:
  - `StandardScaler` applied to continuous temporal features (`month`, `day_of_month`, `day_of_week`).
  - `OneHotEncoder(drop='first', handle_unknown='ignore')` applied to categorical variables (`weather`, `shift`).
  - Binary pass-through for indicator flags (`weekend`, `holiday`, `special_event`).
- **Data Split**: 80% Train, 20% Out-of-Sample Test set.

### Model Benchmarking Results

| Model | MAE (Covers) | RMSE (Covers) | Out-of-Sample $R^2$ | Selection Decision |
|---|---|---|---|---|
| **Linear Regression** (Baseline) | 17.78 | 22.17 | 0.8469 | Benchmark |
| **Random Forest Regressor** (Ensemble) | **16.13** | **20.16** | **0.8735** | **Selected Winner** |

*The model selection logic is fully dynamic, evaluating RMSE and $R^2$ automatically without hardcoding.*

### Top Predictive Demand Drivers
Extracted from Random Forest Gini feature importances:
1. `shift_Lunch` (69.5% relative importance): Captures fundamental lunch vs. dinner volume differences.
2. `day_of_week` (8.3% relative importance): Captures mid-week vs. weekend ramp-up.
3. `weekend` (6.6% relative importance): Flags Saturday and Sunday volume surges.
4. `day_of_month` (4.0% relative importance): Captures monthly billing and pay-period cycles.
5. `month` (2.6% relative importance): Reflects seasonal dining patterns.

---

## 📈 Staffing Optimization Engine

The optimization module (`src/staffing_optimizer.py`) translates predicted customer demand into actionable shift schedules:

$$\text{Required Staff}_{\text{role}} = \max\left(1, \left\lceil \frac{\widehat{\text{Covers}}}{\text{Productivity}_{\text{role}}} \right\rceil\right)$$

### 7-Day Schedule Generator
For upcoming shifts, the system:
1. Simulates future calendar and forecast inputs (upcoming dates, day names, holiday flags, predicted weather).
2. Generates ML covers forecasts via `pipeline.predict()`.
3. Computes exact required headcount per role.
4. Compares against legacy manual schedules to highlight **staffing gaps** and calculate cost variances.
5. Exports the complete operational plan to `outputs/predictions/7_day_staffing_recommendations.csv`.

---

## 🖼 Visual Analytics Gallery

The system automatically generates 10 high-resolution (300 DPI) charts saved to `outputs/charts/`:

1. **`daily_customer_demand_trend.png`**: 365-day covers trend with a 7-day rolling average smoothing curve.
2. **`customer_demand_by_day_of_week.png`**: Day-of-week average covers showing the Friday–Sunday surge.
3. **`lunch_vs_dinner_demand.png`**: Box plot distribution contrasting lunch vs. dinner covers.
4. **`required_vs_actual_staff.png`**: Grouped bar chart comparing required vs. actual staff by role.
5. **`understaffing_rate_by_shift.png`**: Understaffing incidence comparing Lunch (27.5%) vs. Dinner (65.4%).
6. **`labor_cost_by_shift.png`**: Shift labor cost histograms and average cost markers.
7. **`absence_rate_by_role.png`**: Absenteeism breakdown highlighting Dishwasher (8.3%) and Server vulnerabilities.
8. **`staffing_gap_vs_service_score.png`**: Scatter plot and linear fit demonstrating service score collapse under negative staffing gaps.
9. **`customer_demand_vs_labor_cost.png`**: Customer volume vs. labor expenditure scatter plot with marginal cost slope.
10. **`feature_importance.png`**: Machine learning feature importance rankings from the Random Forest model.

---

## 🚀 Quickstart & Installation

### 1. Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed on your machine.
- Git (optional, for cloning).

### 2. Setup
Clone or navigate to the project directory:
```bash
cd "Restaurant Workforce Analytics & Staff Scheduling Optimization"
```

### 3. Install Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

*(Dependencies: `pandas`, `numpy`, `matplotlib`, `scikit-learn`)*

---

## ▶️ How to Run the Project

You can execute the entire end-to-end pipeline with a single command from the project root:

```bash
python run_project.py
```

Alternatively, you can run the main module directly:
```bash
python src/main.py
```

To run individual modular steps independently:
```bash
# 1. Generate new synthetic dataset (3,650 rows)
python src/generate_data.py

# 2. Run exploratory analytics, generate summary reports & charts
python src/data_analysis.py

# 3. Train ML demand models & export model metrics
python src/demand_forecasting.py

# 4. Generate 7-day optimized schedule recommendations
python src/staffing_optimizer.py

# 5. Run the verification test suite
python tests/test_pipeline.py
```

---

## 🖥 Sample Terminal Output

```text
==================================================
RESTAURANT WORKFORCE ANALYTICS
==================================================
Total Records: 3,650
Total Labor Cost: $3,000,834.50
Average Customer Covers: 149.1
Average Service Score: 4.30
Understaffing Rate: 46.5%
Absence Rate: 6.2%

==================================================
DEMAND FORECASTING
==================================================
Linear Regression
MAE: 17.78
RMSE: 22.17
R²: 0.8469

Random Forest
MAE: 16.13
RMSE: 20.16
R²: 0.8735

Selected Model: Random Forest

==================================================
STAFFING RECOMMENDATION
==================================================
Date: 2025-01-01
Shift: Dinner

Predicted Customers: 230

Server: 29
Cook: 15
Host: 10
Cashier: 9
Dishwasher: 11

==================================================
BUSINESS INSIGHTS
==================================================
================================================================================
EXECUTIVE BUSINESS INSIGHTS & STRATEGIC WORKFORCE RECOMMENDATIONS
================================================================================
Generated: Real-time from empirical operational dataset
Scope: Full-year operational analysis across 5 roles and 2 daily shifts

1. SHIFT DEMAND PATTERNS
--------------------------------------------------------------------------------
- Peak Demand Shift: Dinner records the highest customer demand, 
  averaging 197.1 covers per shift compared to the overall 
  restaurant baseline.
- Volume Disparity: Weekend evening shifts drive the highest guest volumes,
  warranting tiered staffing schedules rather than uniform headcount allocations.

2. WORKFORCE BOTTLENECKS & UNDERSTAFFING
--------------------------------------------------------------------------------
- Highest Understaffing Risk: Dinner shifts suffer the highest 
  understaffing frequency at 65.4%.
- Root Cause: Traditional static scheduling fails to account for surge volatility 
  on Friday and Saturday evenings, resulting in chronic employee shortages.

3. EMPLOYEE ABSENTEEISM VULNERABILITY
--------------------------------------------------------------------------------
- Most Vulnerable Role: Dishwasher exhibits the highest absence rate 
  at 8.32%.
- Operational Impact: Dishwasher and server absences directly trigger service bottlenecks.
  A proactive on-call cross-trained buffer is strongly advised.

4. CUSTOMER SATISFACTION & SERVICE SCORE IMPACT
--------------------------------------------------------------------------------
- Service Score Correlation: Staffing gap exhibits a strong positive correlation 
  (r = +0.964) with guest service ratings.
- Performance Contrast:
  * Understaffed shifts average a service score of 3.75 / 5.0.
  * Adequately staffed shifts maintain a service score of 4.78 / 5.0.
  * Conclusion: Understaffing directly harms table turnover, wait times, and brand equity.

5. LABOR COST EFFICIENCY
--------------------------------------------------------------------------------
- Total Labor Expenditure: $3,000,834.50
- Average Labor Cost per Customer: $27.57 per cover.
- Optimization Opportunity: Eliminating redundant scheduling on slow weekday lunch 
  shifts can recover up to 8-12% in unnecessary payroll expenditure.

6. PRIMARY DEMAND DRIVERS (MACHINE LEARNING FINDINGS)
--------------------------------------------------------------------------------
- Top Predictive Features: shift_Lunch, day_of_week, weekend.
- Strategic Takeaway: Shift type and calendar dynamics (day of week, weekend status)
  dominate demand forecasting, proving that rule-of-thumb scheduling is obsolete.

7. UPCOMING 7-DAY ACTIONABLE STAFFING DIRECTIVES
--------------------------------------------------------------------------------
- Schedule Realignment: Evaluated 70 role-shifts across next 7 days. Recommended 
  reinforcements on peak shifts to protect service scores > 4.7.
================================================================================

==================================================
PROJECT PIPELINE EXECUTION COMPLETED SUCCESSFULLY
==================================================
• Charts generated: outputs/charts/ (10 figures)
• Reports generated: outputs/reports/ (Summary metrics, models, insights)
• Predictions generated: outputs/predictions/ (7-day schedule)
==================================================
```

---

## 💼 Business Value & Return on Investment (ROI)

Implementing this data-driven workforce analytics system delivers tangible operational returns:

1. **Labor Payroll Optimization**:
   - Reduces overstaffing on off-peak shifts (saving an estimated 8%–12% in avoidable payroll expenses).
   - Reallocates payroll hours to high-yield weekend shifts where revenue throughput is capped by kitchen and table turnover capacity.
2. **Customer Retention & Lifetime Value**:
   - Preserves customer satisfaction ratings above 4.7 by preventing severe understaffing ($r = +0.964$ correlation with review scores).
   - Prevents table abandonment and negative online reviews associated with slow service.
3. **Managerial Time Savings**:
   - Automates the weekly schedule formulation process, reducing manager scheduling time from 4–6 hours down to minutes.
4. **Absenteeism Resilience**:
   - Quantifies role-specific absenteeism rates (identifying Dishwashers and Servers as high-risk), justifying a structured on-call float pool.

---

## ⚠️ Limitations & Future Roadmap

### Current Limitations
- **Shift Granularity**: Predictions currently operate at the shift level (Lunch vs. Dinner) rather than 15-minute or hourly time slices.
- **Delivery & Takeout Isolation**: Covers represent combined on-premise and takeout volume rather than separating kitchen ticket throughput by channel.
- **Static Wage Modeling**: Overtime premiums (>40 hrs/week) and local minimum wage changes are not dynamically modeled.

### Future Roadmap
- [ ] **Hourly POS Integration**: Integrate with Point-of-Sale (POS) event feeds to forecast hourly arrivals.
- [ ] **Employee Preference & Fair Workweek Constraints**: Incorporate mixed-integer linear programming (MILP via `scipy.optimize`) to enforce fair workweek laws, consecutive-day constraints, and worker availability.
- [ ] **Live Weather API Ingestion**: Ingest 7-day meteorological forecasts via open weather APIs.
- [ ] **Interactive Streamlit Dashboard**: Optional lightweight web dashboard for visual drag-and-drop schedule editing.

---

## 📁 GitHub Portfolio Upload Guide

When publishing this project to your GitHub profile, upload the following files:

```
├── data/
│   └── restaurant_workforce.csv
├── src/
│   ├── __init__.py
│   ├── generate_data.py
│   ├── data_analysis.py
│   ├── demand_forecasting.py
│   ├── staffing_optimizer.py
│   └── main.py
├── outputs/
│   ├── charts/
│   │   └── (All 10 generated .png files)
│   ├── reports/
│   │   └── (All generated .csv and .txt reports)
│   └── predictions/
│       └── 7_day_staffing_recommendations.csv
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── requirements.txt
├── run_project.py
└── README.md
```

Do **NOT** upload temporary IDE directories (`.vscode/`, `.idea/`), virtual environment folders (`.venv/`, `env/`), or compiled bytecode (`__pycache__/`, `*.pyc`). The included `.gitignore` handles this automatically.

---

## 👤 Author & Academic Recognition
- **Project**: Restaurant Workforce Analytics & Staff Scheduling Optimization
- **Domain**: Business Analytics, Hospitality Operations Research, Applied Machine Learning (MSBA Portfolio Project)
- **License**: MIT License
