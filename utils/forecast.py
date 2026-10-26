import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def forecast_spending(df, days_ahead=30):
    daily = (
        df.groupby("Date")["Amount"]
        .sum()
        .reset_index()
        .sort_values("Date")
    )

    daily["day_num"] = np.arange(len(daily))

    X = daily[["day_num"]]
    y = daily["Amount"]

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.arange(
        len(daily),
        len(daily) + days_ahead
    )

    future_dates = pd.date_range(
        daily["Date"].max() + pd.Timedelta(days=1),
        periods=days_ahead
    )

    preds = model.predict(future_days.reshape(-1, 1))

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Spend": np.maximum(preds, 0)
    })

    slope = float(model.coef_[0])

    return forecast_df, slope