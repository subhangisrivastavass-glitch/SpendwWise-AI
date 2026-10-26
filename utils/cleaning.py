import pandas as pd
import numpy as np


def clean_dataframe(df):
    df = df.copy()

    # Standard column names
    cols = {c.lower().strip(): c for c in df.columns}

    if "date" not in cols:
        raise ValueError("Date column not found")

    if "description" not in cols:
        raise ValueError("Description column not found")

    if "amount" not in cols:
        raise ValueError("Amount column not found")

    if "payment_mode" not in cols:
        raise ValueError("Payment_Mode column not found")

    df = df.rename(
        columns={
            cols["date"]: "Date",
            cols["description"]: "Description",
            cols["amount"]: "Amount",
            cols["payment_mode"]: "Payment_Mode",
        }
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    df["Description"] = (
        df["Description"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    df["Payment_Mode"] = (
        df["Payment_Mode"]
        .fillna("Unknown")
        .astype(str)
    )

    df = df.dropna(subset=["Date", "Amount"])

    if "Category" not in df.columns:
        df["Category"] = "Unknown"

    df["Month"] = df["Date"].dt.strftime("%Y-%m")

    return df.reset_index(drop=True)