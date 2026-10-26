def budget_health_score(df, monthly_budget=50000):
    monthly_spend = float(df["Amount"].sum())

    ratio = round(
        (monthly_spend / monthly_budget) * 100,
        1
    )

    score = max(
        0,
        min(
            100,
            int(100 - (ratio - 100))
        )
    )

    if ratio <= 70:
        status = "Excellent"
        color = "#22c55e"
    elif ratio <= 100:
        status = "Good"
        color = "#eab308"
    else:
        status = "Over Budget"
        color = "#ef4444"

    return {
        "score": score,
        "status": status,
        "color": color,
        "monthly_spend": monthly_spend,
        "monthly_budget": monthly_budget,
        "ratio": ratio,
    }