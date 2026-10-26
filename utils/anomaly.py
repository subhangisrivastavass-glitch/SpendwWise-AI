from sklearn.ensemble import IsolationForest


def detect_anomalies(df, contamination=0.05):
    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    values = df[["Amount"]]

    preds = model.fit_predict(values)

    df["Is_Anomaly"] = preds == -1

    scores = model.decision_function(values)

    df["Anomaly_Score"] = scores

    return df