# Day 3 Case Study: "Demand Forecasting for a Retail Chain"

## Context
A large retail chain operates 200+ stores across North America. **Inventory management** costs them **$2M/month**, and stockouts cost another **$500K/month** in lost sales. They wanted to replace their manual forecasting process with ML.

## The Business Problem

### Manual Forecasting (Baseline)
- Store managers manually predicted demand for next week
- Guesswork based on "gut feel" and past experience
- **Accuracy**: 60–65% (Mean Absolute Percentage Error = 30–40%)
- **Results**:
  - Overstock: Dead inventory, storage costs, markdowns
  - Understock: Lost sales, disappointed customers

### Business Impact of Poor Forecasts
- **$2M/month in inventory costs** (storage, obsolescence, markdowns)
- **$500K/month in stockout losses** (customers go to competitors)
- **$30M/year in total waste** (huge opportunity!)

## The ML Solution

### Approach: Time Series + Regression

Instead of complex ARIMA or neural networks, the team used **simple but effective**:
- **Lag features**: Demand from 1, 4, 13 weeks ago (capture weekly, monthly, quarterly patterns)
- **Trend**: `weeks_since_start` (capture growth/decline over time)
- **External variables**: day_of_week, is_holiday, promotions_active, competitor_activity
- **Model**: Random Forest regressor (simple, interpretable, robust)

### Why This Approach

**Time Series Breakdown**:
```
Demand(t) = Trend(t) + Seasonality(t) + External(t) + Noise(t)

Example: Week 52 demand
= 100 (baseline growth)
+ 50 (Christmas seasonality, captured by lag_52)
+ 20 (holiday promotion active)
+ random noise
= ~170 units
```

**Lag Features Capture Seasonality**:
- `lag_1`: Last week's demand (week-to-week correlation)
- `lag_4`: 4 weeks ago (monthly pattern: paycheck cycles)
- `lag_52`: 52 weeks ago (yearly seasonality: Christmas, summer, back-to-school)

**Why Random Forest?**
- Handles non-linear relationships (promotion effect varies by season)
- Robust to outliers (unexpected events)
- Interpretable (see which features matter)
- Fast to train (real-time retraining possible)

### Data & Training

**Dataset**: 2 years of historical data (104 weeks)
- Columns: week, demand, day_of_week, is_holiday, promo_active, competitor_activity
- Train: first 78 weeks
- Test: last 26 weeks (6 months holdout)

**Model Performance**:
```
Metric          Forecast Method
────────────────────────────────
MAE             Manual: $500K  →  ML: $120K (76% improvement)
RMSE            Manual: $650K  →  ML: $180K
MAPE            Manual: 35%    →  ML: 8%  (industry std: 10-15%)
Accuracy        Manual: 60%    →  ML: 92%
```

## Business Results

### Pilot Test (50 stores, 3 months)
- **Before ML**: Inventory costs $8.3M, stockouts cost $2.1M → **Total $10.4M**
- **After ML**: Inventory costs $7.3M, stockouts cost $1.6M → **Total $8.9M**
- **Savings**: **$1.5M in 3 months** (19% reduction!)

### Full Rollout (200 stores, nationwide)
**Extrapolated Impact (annualized)**:
- **Inventory cost reduction**: 12% ($240K/month saved)
- **Stockout reduction**: 25% ($125K/month saved)
- **Total benefit**: **$4.4M/year**

### Additional Benefits
1. **Inventory turns faster**: Less dead stock
2. **Customer satisfaction**: Fewer stockouts
3. **Data-driven decisions**: Regional patterns discovered (e.g., New York needs different strategy than Texas)
4. **Scalability**: Model retrains weekly with new sales data

## Why This Forecasting Approach Works

### Principle: Simple Models on Good Features Beat Complex Models

```
Naive Approach (Poor): ARIMA with default parameters
└─ Ignores external variables, inflexible to promotions

Better (Good): Lag features + Linear Regression
└─ Captures seasonality, but misses non-linear effects

Best (Excellent): Lag features + Random Forest
└─ Captures seasonality + non-linear effects + is interpretable
```

### Key Insights from Model

**Feature Importance** (which factors drive demand?):
```
lag_52 (yearly seasonality)   : 32%  ← Christmas/summer matter most!
is_holiday                     : 18%
promo_active                   : 15%
lag_4 (monthly pattern)        : 12%
lag_1 (weekly correlation)     : 10%
day_of_week                    : 8%
competitor_activity           : 5%
```

**Surprising Finding**: Competitor activity mattered less than expected (only 5%), because loyal customers don't switch easily.

---

## Key Lesson for Juniors

> **"Time series isn't magic—it's lag features + regression. Start simple, iterate."**

### Common Time Series Mistakes (Avoid These)

❌ **Using only raw data** ("I'll train a neural network on raw demand")
- Solution: Create lag features first; they're powerful!

❌ **Ignoring external variables** ("Weather/holidays don't matter")
- Solution: Collect external data; it's free signal

❌ **Forecasting too far ahead** ("Can I predict demand 1 year out?")
- Solution: Start with 1-week forecasts (high accuracy); extend gradually

❌ **No test set separation by time** ("I'll randomly split 70/30")
- Solution: Use temporal order: train on past, test on future

❌ **Forgetting about seasonality** ("I'll just use recent data")
- Solution: Lag features capture seasonal patterns automatically

---

## Implementation Steps (What You'll Do in Day 3)

1. **Create lag features**: lag_1, lag_4, lag_52
2. **Add external features**: day_of_week, is_holiday, promo_active
3. **Train 2 models**: Linear Regression (baseline) + Random Forest (better)
4. **Evaluate**: Calculate MAE, RMSE, MAPE
5. **Compare**: Which model would you deploy?
6. **Interpret**: Which features matter most? Does it match business intuition?

---

## Enterprise Context: Retraining & Monitoring

In production, the model:
- **Retrains weekly** with new sales data (stay current)
- **Monitors prediction accuracy** (if MAPE > 12%, alert humans)
- **Detects seasonality shifts** (e.g., COVID demand anomaly in 2020)
- **Rolls back** if new model performs worse than old model

**Key insight**: Deployment isn't a one-time event; it's a continuous feedback loop.

---

## Discussion Questions for Juniors

1. **Seasonality Detection**: How would you discover that demand has a yearly pattern? (Hint: lag_52 correlates strongly with target)

2. **External Variables**: What other external factors could improve forecast?
   - Weather (hot → ice cream sales up)
   - Events (concert → hotel demand up)
   - Supply disruptions (port closure → shortage)

3. **Forecast Horizon**: Why is 1-week forecasting easier than 1-year forecasting?
   - Near future: patterns are stable
   - Distant future: too many unknowns

4. **Model Choice**: Why Random Forest over ARIMA for this problem?
   - Easier to add external variables
   - Handles non-linear interactions
   - Faster to implement
