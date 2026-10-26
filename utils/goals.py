import pandas as pd

def calculate_goal_status(df, monthly_budget, target_savings, goal_months):
    total_spend = df["Amount"].sum()

    if "Month" in df.columns:
        months = max(df["Month"].nunique(), 1)
    else:
        months = 1

    avg_monthly_spend = total_spend / months
    avg_monthly_savings = max(monthly_budget - avg_monthly_spend, 0)

    required_monthly = target_savings / goal_months

    feasible = avg_monthly_savings >= required_monthly

    progress_pct = (
        (avg_monthly_savings * goal_months / target_savings) * 100
        if target_savings > 0
        else 0
    )

    gap = max(required_monthly - avg_monthly_savings, 0)

    recommendation = (
        "You are on track to achieve your goal."
        if feasible
        else "You need to save more each month to reach this goal."
    )

    return {
        "required_monthly": round(required_monthly, 2),
        "avg_monthly_savings": round(avg_monthly_savings, 2),
        "target_savings": target_savings,
        "goal_months": goal_months,
        "progress_pct": round(progress_pct, 2),
        "feasible": feasible,
        "gap": round(gap, 2),
        "recommendation": recommendation,
    }