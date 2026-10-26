def generate_insights(df, forecast_df):
    insights = []

    total = df["Amount"].sum()

    top_cat = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    if len(top_cat):
        insights.append(
            f"Highest spending category is {top_cat.index[0]}."
        )

    insights.append(
        f"Total recorded spending is ₹{total:,.0f}."
    )

    if forecast_df["Predicted_Spend"].sum() > total * 0.25:
        insights.append(
            "Forecast suggests continued spending at a similar pace."
        )

    anomalies = int(df["Is_Anomaly"].sum())

    if anomalies:
        insights.append(
            f"{anomalies} unusual transactions were detected."
        )

    return insights