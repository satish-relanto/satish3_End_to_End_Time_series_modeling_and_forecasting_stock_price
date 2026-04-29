# Day 4 Case Study: "When a Model Broke in Production"

## Context
A fintech company deployed a **credit scoring model** to approve/reject loan applications. The model was trained on 2020–2021 data (pre-pandemic recovery period). Everything worked well for 3 weeks, then **predictions became completely unreliable**.

## The Crisis

### What Happened
- **Week 1–3**: Model performing as expected (predictions matched actual defaults)
- **Week 4**: Credit scores suddenly 20% lower than expected
- **Week 5**: Business team noticed: fewer loan approvals, customer complaints
- **Week 6**: Data science team called in to investigate

### Impact
- **$2M in lost revenue** (loan approvals dropped 40%)
- **Customer dissatisfaction**: Borderline applicants suddenly rejected
- **Regulatory risk**: If discrimination pattern detected (e.g., by zip code), compliance nightmare
- **Lost trust**: C-suite questioned the ML team: "Why didn't you catch this earlier?"

## Root Cause Analysis: Data Drift + Concept Drift

### Problem 1: Data Drift (Input Distribution Changed)

**Training Data** (2020–2021):
- Average credit score: 720
- Average debt-to-income: 0.35
- Average age: 38

**Production Data** (2022, new economic reality):
- Average credit score: 680 (↓ 40 points)
- Average debt-to-income: 0.45 (↑ 10 points)
- Average age: 35 (↓ 3 years)

**Why?**: 2022 recession + inflation → customers took on more debt, scores declined

### Problem 2: Concept Drift (Target Distribution Changed)

**Historical Default Rate** (2020–2021): 5%
**Current Default Rate** (2022): 8% (↑ 60%!)

**Why?**: Economic conditions changed. Historical patterns no longer predictive.

### Combined Effect: Catastrophic Model Failure

```
Model Logic: 
  IF (credit_score > 700) AND (debt_ratio < 0.40) 
  THEN approve_loan

2020–2021: 95% accurate (works great!)
2022: 60% accurate (worthless!)

Why? Thresholds trained on 2020 data don't apply to 2022 conditions
```

### How Did This Happen? (Lack of Monitoring)

The model was deployed, then **forgotten**:
- ❌ No monitoring dashboard
- ❌ No alerts when prediction distribution changed
- ❌ No retraining schedule
- ❌ No A/B testing with old model
- ❌ No data quality checks

## The Solution: Monitoring + Retraining + Rollback

### Step 1: Implement Monitoring

The team deployed a **monitoring dashboard** tracking:

```
Daily Monitoring Metrics:
├─ Prediction Distribution
│  ├─ Mean approval score: tracking over time
│  ├─ Std dev: detecting increased uncertainty
│  └─ Alert: if mean shifts >5%, flag for review
├─ Input Feature Distribution
│  ├─ Average credit score this week vs. historical
│  ├─ Average debt ratio
│  └─ Alert: if feature distribution shifts, retrain model
├─ Model Performance Proxy
│  ├─ If loan defaults in last 30 days, did model predict it?
│  └─ Actual default rate vs. predicted (early warning)
└─ Data Quality
   ├─ Missing values in key fields
   └─ Alert: if data completeness drops below 95%
```

### Step 2: Trigger Retraining

When monitoring detected drift:
```
Timeline:
Monday (day 20): Monitoring alert: prediction mean shifted 8% (> 5% threshold)
Tuesday: Data science team notified
Wednesday: Retrain model on last 6 months of data (captures recent patterns)
Thursday: Validate new model on holdout test set
Friday: A/B test: 20% of traffic on new model, 80% on old model
         Monitor for 1 week
Next Monday: New model passes A/B test, gradually roll out to 100%
```

**Retraining Code**:
```python
# Retrain on recent data (last 6 months)
recent_data = df[df['date'] > '2022-01-01']
model_new = RandomForestClassifier(...)
model_new.fit(X_recent, y_recent)

# Compare against old model
old_model_auc = 0.78
new_model_auc = 0.85  # ✓ Better!

# A/B test: if new model maintains accuracy, deploy
if new_model_auc > old_model_auc - 0.02:
    deploy(model_new)
else:
    rollback(model_old)
```

### Step 3: Implement Rollback Strategy

**Before rollout**:
- Keep old model in production
- Have new model staged and ready
- Version tracking: model_v1.pkl, model_v2.pkl, etc.

**During rollout**:
- Phase 1 (Day 1–3): 10% traffic on new model, 90% on old
- Phase 2 (Day 4–7): 50% traffic on new model (balanced)
- Phase 3 (Day 8+): 100% on new model

**If problems detected**:
- Instant rollback: `active_model = model_v1.pkl`
- Blameless postmortem: What did we miss?

### Results After Implementing Monitoring

**Timeline**:
```
Week 1: Manual approval process (bypass ML) → 100% accuracy (but costly)
Week 2: Retrain model on recent data
Week 3: A/B test new model
Week 4: Deploy new model
Week 5: Monitoring confirms new model working well
```

**Final Performance**:
- Retraining improved accuracy from 60% → 92% ✓
- Detection time: 2 hours (vs. 2 weeks before)
- Business impact: Minimal (early detection prevented major loss)
- Regulatory: Full audit trail (monitoring logs prove due diligence)

---

## Key Lesson for Juniors

> **"Deployment isn't the end—it's the beginning. Monitor, iterate, improve."**

### The 4 Pillars of Production ML

```
1. Model Training
   ↓
2. Model Deployment
   ↓
3. Monitoring & Drift Detection ← CRITICAL (often overlooked!)
   ↓
4. Retraining & Rollback
   ↓
   (Loop back to 1)
```

### Monitoring vs. Traditional Software

**Software Deployment**: Ship once, monitor uptime/errors
**ML Deployment**: Ship, then monitor **model degradation** (predictions getting worse)

**Why Different?**: ML models can appear to work (no crashes) but produce wrong answers (silent failures).

---

## Monitoring Checklist

### What to Monitor

✅ **Prediction Distribution**
- Mean, std dev of predictions
- Are predictions getting more confident or more uncertain?

✅ **Input Data Distribution** (Data Drift)
- Are feature values changing?
- Is the input distribution different from training?

✅ **Actual vs. Predicted** (Model Performance)
- If we have ground truth labels, how's accuracy?
- Default rate: expected vs. actual

✅ **Data Quality**
- Missing values
- Outliers in new data

❌ **NOT just monitoring**: System uptime, server memory (these don't tell you if model is breaking!)

### Alert Thresholds

```
Metric                          Green    Yellow   Red
──────────────────────────────────────────────────────
Prediction mean shift           < 2%     2-5%     > 5%
Feature distribution distance   < 0.1    0.1-0.2  > 0.2
Actual accuracy vs. baseline    > 90%    85-90%   < 85%
Data completeness               > 99%    95-99%   < 95%
```

---

## Common Deployment Mistakes (Avoid These)

❌ **No monitoring at all** ("We trained the model; we're done!")
- Solution: Monitoring is mandatory in production

❌ **Monitoring too late** ("We noticed 6 months after deployment")
- Solution: Automated alerts catch drift in days, not months

❌ **No rollback plan** ("We can't undo the bad model")
- Solution: Always keep previous model versioned and ready

❌ **Retraining without A/B test** ("New model must be better")
- Solution: Always validate new model before full rollout

❌ **One-size-fits-all retraining** ("We retrain quarterly")
- Solution: Retrain on schedule OR when drift detected (whichever comes first)

---

## Enterprise Context: MLOps

Large companies use **MLOps platforms** (MLflow, Kubeflow, SageMaker) to automate:
- Model versioning
- Monitoring & alerting
- Automated retraining
- A/B testing
- Rollback automation

**But the principles are universal**:
1. Monitor in production
2. Detect drift early
3. Retrain on recent data
4. Validate before rollout
5. Rollback if needed

---

## Implementation Steps (What You'll Do in Day 4)

1. **Serialize your best model** from Day 3
2. **Build a prediction wrapper** that loads model and makes predictions
3. **Create a deployment checklist** (document decisions)
4. **Simulate retraining scenario**: "What if new data arrived?"
5. **Think about monitoring**: "How would you detect the credit score model breaking?"

---

## Discussion Questions for Juniors

1. **Early Detection**: In the credit scoring case, how would monitoring have helped?
   - If prediction mean shifted 8%, alert within 1 day
   - vs. manual discovery after 2 weeks

2. **Retraining Frequency**: How often should we retrain a model?
   - Answer: Depends on data drift speed
   - Fast-changing domain: daily
   - Stable domain: monthly or quarterly

3. **Data Drift vs. Concept Drift**: What's the difference?
   - Data drift: input distribution changes (unemployment rate up → more debt)
   - Concept drift: target relationship changes (high income used to mean safe, now doesn't)

4. **Rollback Decision**: When would you rollback a new model?
   - If accuracy drops below threshold
   - If fairness issues detected (discrimination by protected class)
   - If business metrics decline

5. **A/B Testing**: Why not deploy new model to 100% immediately?
   - Phase rollout catches edge cases
   - Gives time to detect failures in production
   - Risk mitigation
