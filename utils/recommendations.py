def generate_recommendations(df, monthly_budget):
    recommendations = []

    cat_spend = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    total = df["Amount"].sum()

    for category, amount in cat_spend.items():
        pct = (amount / total) * 100 if total else 0

        if pct > 30:
            recommendations.append({
                "category": category,
                "message": f"Spending in {category} is relatively high.",
                "priority": "high",
                "saving_potential": round(amount * 0.10)
            })

    if not recommendations:
        recommendations.append({
            "category": "General",
            "message": "Your spending appears balanced.",
            "priority": "low",
            "saving_potential": 0
        })

    return recommendations