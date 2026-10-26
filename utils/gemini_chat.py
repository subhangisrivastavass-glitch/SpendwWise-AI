def gemini_chat(
    user_message,
    chat_history,
    df,
    forecast_df,
    health,
    insights,
    monthly_budget,
    api_key,
    model_name="gemini-2.0-flash",
):
    """
    Simple fallback implementation.
    Returns (response, used_gemini)
    """

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(model_name)

        total_spend = float(df["Amount"].sum())
        anomaly_count = int(df["Is_Anomaly"].sum()) if "Is_Anomaly" in df.columns else 0
        forecast_total = float(forecast_df["Predicted_Spend"].sum())

        context = f"""
Financial Summary:

Total Spend: ₹{total_spend:,.2f}
Monthly Budget: ₹{monthly_budget:,.2f}
Health Score: {health.get('score', 'N/A')}
Health Status: {health.get('status', 'N/A')}
Anomalies: {anomaly_count}
Forecast Spend: ₹{forecast_total:,.2f}

Insights:
{chr(10).join(insights[:5])}
"""

        response = model.generate_content(
            context + "\n\nUser Question:\n" + user_message
        )

        return response.text, True

    except Exception as e:
        return (
            f"Gemini AI is currently unavailable.\n\nError: {str(e)}",
            False,
        )