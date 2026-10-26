def answer_query(user_query, df, forecast_df, health, insights, monthly_budget):
    query = user_query.lower()

    # Total spending
    if "total spend" in query or "how much spent" in query:
        return f"Your total spending is ₹{df['Amount'].sum():,.0f}."

    # Highest spending category
    if "most" in query and "spend" in query:
        cat = (
            df.groupby("Category")["Amount"]
            .sum()
            .sort_values(ascending=False)
            .index[0]
        )
        amt = (
            df.groupby("Category")["Amount"]
            .sum()
            .sort_values(ascending=False)
            .iloc[0]
        )
        return f"You spent the most on **{cat}** (₹{amt:,.0f})."

    # Savings goal
    if "save" in query or "saving" in query:
        spent = df["Amount"].sum()

        if "Month" in df.columns:
            months = max(df["Month"].nunique(), 1)
        else:
            months = 1

        monthly_spend = spent / months
        savings = max(monthly_budget - monthly_spend, 0)

        return (
            f"Estimated monthly savings: ₹{savings:,.0f}. "
            f"Budget: ₹{monthly_budget:,.0f}."
        )

    # Forecast
    if "forecast" in query or "next month" in query:
        total_forecast = forecast_df["Predicted_Spend"].sum()
        return f"Projected future spending is approximately ₹{total_forecast:,.0f}."

    # Anomalies
    if "suspicious" in query or "anomaly" in query:
        count = int(df["Is_Anomaly"].sum())
        return f"I found {count} potentially unusual transactions."

    # Health score
    if "health" in query or "budget" in query:
        return (
            f"Financial Health Score: {health['score']}/100 "
            f"({health['status']})."
        )

    # Insights
    if "insight" in query or "advice" in query:
        return "<br>".join([f"• {x}" for x in insights[:5]])

    # Default
    return (
        "I can help with spending analysis, forecasts, "
        "budget health, anomalies, savings, and categories."
    )