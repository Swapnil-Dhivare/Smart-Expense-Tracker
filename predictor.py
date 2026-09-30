"""Spending trend prediction, ML anomaly detection, and budget checks.

Next-month spending is an ensemble of two independent forecasts:
  - top-down: linear regression directly on monthly totals
  - bottom-up: linear regression per category, summed
Averaging two different model views is a standard forecasting technique
(reduces variance from either model's blind spots) and is cheap to compute
here since both are just regressions over a handful of monthly points.

Anomaly detection uses IsolationForest (unsupervised ML) over engineered
features (amount, log-amount, per-category deviation) rather than a single
fixed threshold rule.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression


@dataclass
class PredictionResult:
    predicted_amount: float
    method: str
    monthly_totals: pd.Series
    ci_lower: float
    ci_upper: float
    r2: float | None
    topdown_amount: float
    bottomup_amount: float
    category_forecast: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


def monthly_totals(df: pd.DataFrame) -> pd.Series:
    monthly = df.groupby(df["date"].dt.to_period("M"))["amount"].sum()
    monthly.index = monthly.index.to_timestamp()
    return monthly.sort_index()


def _linear_forecast(totals: pd.Series) -> tuple[float, float | None, float]:
    """Fit a line over monthly totals, return (prediction, r2, residual_std)."""
    x = np.arange(len(totals)).reshape(-1, 1)
    y = totals.values.astype(float)
    model = LinearRegression().fit(x, y)
    next_x = np.array([[len(totals)]])
    predicted = max(float(model.predict(next_x)[0]), 0.0)
    residuals = y - model.predict(x)
    resid_std = float(residuals.std(ddof=1)) if len(totals) >= 3 else float(y.std())
    r2 = float(model.score(x, y)) if len(totals) >= 3 else None
    return predicted, r2, resid_std


def _forecast_by_category(df: pd.DataFrame) -> pd.Series:
    """Bottom-up: forecast each category independently, then sum."""
    forecasts = {}
    for category, group in df.groupby("category"):
        cat_totals = monthly_totals(group)
        if len(cat_totals) >= 2:
            predicted, _, _ = _linear_forecast(cat_totals)
        else:
            predicted = float(cat_totals.iloc[-1])
        forecasts[category] = predicted
    return pd.Series(forecasts).sort_values(ascending=False)


def predict_next_month(df: pd.DataFrame) -> PredictionResult:
    totals = monthly_totals(df)
    category_forecast = _forecast_by_category(df)
    bottomup_amount = float(category_forecast.sum())

    if len(totals) >= 2:
        topdown_amount, r2, resid_std = _linear_forecast(totals)
        predicted = (topdown_amount + bottomup_amount) / 2
        ci_lower = max(predicted - 1.96 * resid_std, 0.0)
        ci_upper = predicted + 1.96 * resid_std
        method = "ensemble: top-down + bottom-up linear regression"
        return PredictionResult(
            predicted, method, totals, ci_lower, ci_upper, r2,
            topdown_amount, bottomup_amount, category_forecast,
        )

    # Only one month of data: extrapolate from daily average.
    last_period = totals.index[-1]
    days_in_month = calendar.monthrange(last_period.year, last_period.month)[1]
    days_seen = df["date"].dt.day.max()
    daily_avg = totals.iloc[-1] / max(days_seen, 1)
    predicted = float(daily_avg * days_in_month)
    spread = predicted * 0.15
    return PredictionResult(
        predicted, "daily-average extrapolation", totals,
        max(predicted - spread, 0.0), predicted + spread, None,
        predicted, bottomup_amount, category_forecast,
    )


def category_distribution(df: pd.DataFrame) -> pd.Series:
    return df.groupby("category")["amount"].sum().sort_values(ascending=False)


def detect_unusual(df: pd.DataFrame, contamination: float = 0.08) -> pd.DataFrame:
    """Flag anomalous transactions with IsolationForest over engineered features."""
    if len(df) < 8:
        return df.iloc[0:0].assign(anomaly_score=pd.Series(dtype=float), reason=pd.Series(dtype=str))

    work = df.copy()
    cat_count = work.groupby("category")["amount"].transform("count")
    cat_mean = work.groupby("category")["amount"].transform("mean")
    cat_std = work.groupby("category")["amount"].transform("std").fillna(0)
    cat_std_safe = cat_std.where(cat_std != 0, cat_mean.where(cat_mean != 0, 1.0))
    work["_deviation"] = (work["amount"] - cat_mean) / cat_std_safe
    work["_log_amount"] = np.log1p(work["amount"])
    work["_pct_rank"] = work["amount"].rank(pct=True)

    features = work[["amount", "_log_amount", "_deviation"]].values
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
    model.fit(features)
    work["anomaly_score"] = -model.decision_function(features)  # higher = more anomalous
    is_anomaly = model.predict(features) == -1

    flagged = work[is_anomaly].copy()
    if flagged.empty:
        return df.iloc[0:0].assign(anomaly_score=pd.Series(dtype=float), reason=pd.Series(dtype=str))

    def _reason(row) -> str:
        # With too few peers in a category, "vs. category average" is
        # meaningless (the average would just be itself) — fall back to
        # the transaction's rank among all spending instead.
        if cat_count[row.name] >= 3:
            pct = (row["amount"] / cat_mean[row.name] - 1) * 100 if cat_mean[row.name] else 0
            direction = "above" if pct >= 0 else "below"
            return f"₹{row['amount']:,.0f} is {abs(pct):.0f}% {direction} its {row['category']} average"
        percentile = row["_pct_rank"] * 100
        return (
            f"₹{row['amount']:,.0f} is in the top {100 - percentile:.0f}% of all transactions "
            f"({row['category']} has too few transactions for a category-relative comparison)"
        )

    flagged["reason"] = flagged.apply(_reason, axis=1)
    flagged["anomaly_score"] = flagged["anomaly_score"].round(3)
    return flagged.sort_values("anomaly_score", ascending=False).drop(
        columns=["_deviation", "_log_amount", "_pct_rank"]
    )


def budget_status(predicted_amount: float, budget: float) -> dict:
    if budget <= 0:
        return {"status": "no_budget", "over_by": 0.0, "pct_used": 0.0}
    pct_used = predicted_amount / budget * 100
    if predicted_amount > budget:
        return {"status": "over", "over_by": predicted_amount - budget, "pct_used": pct_used}
    if pct_used >= 80:
        return {"status": "warning", "over_by": 0.0, "pct_used": pct_used}
    return {"status": "ok", "over_by": 0.0, "pct_used": pct_used}
