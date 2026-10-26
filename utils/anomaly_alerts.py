import pandas as pd

def build_anomaly_alerts(df):
    alerts = []

    if "Is_Anomaly" not in df.columns:
        return alerts

    anomalies = df[df["Is_Anomaly"] == True]

    for _, row in anomalies.iterrows():
        amount = float(row["Amount"])

        if amount > 10000:
            severity = "critical"
        elif amount > 5000:
            severity = "high"
        else:
            severity = "medium"

        alerts.append({
            "date": row["Date"],
            "description": row["Description"],
            "amount": amount,
            "category": row.get("Category", "Unknown"),
            "severity": severity,
            "reason": f"Unusual transaction amount of ₹{amount:,.0f}"
        })

    return alerts


def weekly_anomaly_summary(df):
    if "Is_Anomaly" not in df.columns:
        return []

    anomalies = df[df["Is_Anomaly"] == True].copy()

    if anomalies.empty:
        return []

    anomalies["Week"] = pd.to_datetime(anomalies["Date"]).dt.isocalendar().week

    summary = []

    for week, group in anomalies.groupby("Week"):
        top_category = (
            group["Category"].mode().iloc[0]
            if "Category" in group.columns and not group["Category"].mode().empty
            else "Unknown"
        )

        summary.append({
            "week": int(week),
            "count": len(group),
            "total_amount": float(group["Amount"].sum()),
            "top_category": top_category
        })

    return summary