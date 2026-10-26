def compute_health_breakdown(df, monthly_budget):
    total_spend = float(df["Amount"].sum())

    if "Month" in df.columns:
        months = max(df["Month"].nunique(), 1)
    else:
        months = 1

    monthly_spend = total_spend / months
    savings = max(monthly_budget - monthly_spend, 0)

    savings_ratio = min((savings / monthly_budget) * 100, 100)

    expense_stability = 80

    essential_categories = ["Food", "Rent", "Utilities", "Health"]
    essential_spend = df[df["Category"].isin(essential_categories)]["Amount"].sum()

    essential_ratio = (
        (essential_spend / total_spend) * 100
        if total_spend > 0
        else 0
    )

    investment_spend = df[df["Category"] == "Investment"]["Amount"].sum()

    investment_ratio = (
        min((investment_spend / total_spend) * 100 * 5, 100)
        if total_spend > 0
        else 0
    )

    final_score = (
        savings_ratio * 0.4 +
        expense_stability * 0.2 +
        essential_ratio * 0.2 +
        investment_ratio * 0.2
    )

    if final_score >= 75:
        label = "Excellent"
        color = "#22c55e"
    elif final_score >= 50:
        label = "Good"
        color = "#eab308"
    else:
        label = "Needs Improvement"
        color = "#ef4444"

    return {
        "final_score": round(final_score, 2),
        "final_label": label,
        "final_color": color,
        "components": {
            "savings_ratio": {
                "score": round(savings_ratio, 2),
                "weight": 0.4,
                "label": "Savings",
                "description": "Monthly savings compared to budget."
            },
            "expense_stability": {
                "score": round(expense_stability, 2),
                "weight": 0.2,
                "label": "Stable",
                "description": "Consistency of spending pattern."
            },
            "essential_ratio": {
                "score": round(essential_ratio, 2),
                "weight": 0.2,
                "label": "Essential Spending",
                "description": "Percentage spent on essentials."
            },
            "investment_ratio": {
                "score": round(investment_ratio, 2),
                "weight": 0.2,
                "label": "Investment",
                "description": "Money allocated towards investments."
            }
        }
    }