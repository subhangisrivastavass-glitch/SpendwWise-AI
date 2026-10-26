"""
SpendWise AI — Synthetic Data Engine
Generates 500+ rows of messy, realistic financial transaction data.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# ── Seed for reproducibility ──────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── Category templates ────────────────────────────────────────────────────────
CATEGORIES = {
    "Food": {
        "descriptions": [
            "Swiggy order", "Zomato delivery", "McDonald's", "KFC bucket",
            "Pizza Hut", "Starbucks coffee", "Cafe Coffee Day", "Dominos",
            "BigBasket groceries", "Blinkit order", "Zepto grocery",
            "Restaurant lunch", "Street food", "Dunzo order", "Noon Food",
        ],
        "amount_range": (80, 1200),
        "weight": 0.28,
    },
    "Transport": {
        "descriptions": [
            "Uber ride", "Ola cab", "Rapido bike", "Metro recharge",
            "BMTC bus pass", "Petrol fill", "Namma Metro", "Auto rickshaw",
            "Rapido auto", "InDrive trip", "BluSmart ride", "Parking fee",
        ],
        "amount_range": (30, 900),
        "weight": 0.14,
    },
    "Rent": {
        "descriptions": [
            "House rent", "PG accommodation", "Room rent payment",
            "Society maintenance", "Apartment rent", "NoBroker rent",
        ],
        "amount_range": (8000, 30000),
        "weight": 0.05,
    },
    "Entertainment": {
        "descriptions": [
            "Netflix subscription", "Amazon Prime", "Hotstar", "Spotify",
            "YouTube Premium", "BookMyShow tickets", "PVR Cinemas",
            "Gaming top-up", "Steam purchase", "PS Store",
        ],
        "amount_range": (99, 2500),
        "weight": 0.10,
    },
    "Shopping": {
        "descriptions": [
            "Amazon purchase", "Flipkart order", "Myntra clothes",
            "Ajio shopping", "Nykaa beauty", "H&M store", "Zara",
            "Decathlon sports", "IKEA items", "Meesho order",
        ],
        "amount_range": (200, 8000),
        "weight": 0.16,
    },
    "Health": {
        "descriptions": [
            "Apollo Pharmacy", "Practo consultation", "PharmEasy order",
            "Netmeds delivery", "Gym membership", "Doctor visit",
            "Lab tests", "1mg medicines", "Yoga class", "Health checkup",
        ],
        "amount_range": (100, 5000),
        "weight": 0.09,
    },
    "Utilities": {
        "descriptions": [
            "Electricity bill", "Water bill", "Airtel broadband",
            "Jio recharge", "Vi recharge", "BSNL bill", "Gas cylinder",
            "DTH recharge", "BESCOM bill", "CESC electricity",
        ],
        "amount_range": (200, 3500),
        "weight": 0.08,
    },
    "Education": {
        "descriptions": [
            "Coursera course", "Udemy purchase", "Book purchase",
            "Stationery", "Online tuition", "Skill course fee",
            "LinkedIn Learning", "Unacademy", "BYJU's", "WhiteHat Jr",
        ],
        "amount_range": (150, 6000),
        "weight": 0.05,
    },
    "Investment": {
        "descriptions": [
            "Zerodha SIP", "Groww mutual fund", "Kuvera investment",
            "Paytm Money", "HDFC SIP", "LIC premium", "PPF deposit",
            "NPS contribution", "FD opening", "Gold purchase",
        ],
        "amount_range": (500, 15000),
        "weight": 0.05,
    },
}

PAYMENT_MODES = ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash", "Wallet"]
PAYMENT_WEIGHTS = [0.40, 0.25, 0.18, 0.07, 0.06, 0.04]

NOISE_DESCRIPTIONS = [
    "Txn ref 8827xq", "Payment to XYZ", "NEFT transfer",
    "IMPS-00291827", "Refund credit", "Cashback", "Unknown debit",
]


def add_noise(text: str) -> str:
    """Randomly mutate text to simulate messy real-world data."""
    if random.random() < 0.12:
        return text.upper()
    if random.random() < 0.08:
        return text.lower()
    if random.random() < 0.06:
        return text + f" #{random.randint(1000, 9999)}"
    return text


def generate_transactions(n: int = 600) -> pd.DataFrame:
    end_date = datetime.today()
    start_date = end_date - timedelta(days=180)

    category_list = list(CATEGORIES.keys())
    weights = [CATEGORIES[c]["weight"] for c in category_list]

    rows = []
    for _ in range(n):
        # Pick category
        cat = random.choices(category_list, weights=weights, k=1)[0]
        cfg = CATEGORIES[cat]

        # Random date (weekends skewed for Food/Entertainment)
        date = start_date + timedelta(
            seconds=random.randint(0, int((end_date - start_date).total_seconds()))
        )
        if cat in ("Food", "Entertainment") and date.weekday() < 5:
            if random.random() < 0.3:
                # push to nearest weekend
                days_ahead = 5 - date.weekday()
                date += timedelta(days=days_ahead)

        # Amount with occasional spike
        lo, hi = cfg["amount_range"]
        amount = round(np.random.uniform(lo, hi), 2)
        if random.random() < 0.04:          # 4% anomalous spikes
            amount = round(amount * random.uniform(3.5, 7.0), 2)

        # Description noise
        desc = add_noise(random.choice(cfg["descriptions"]))
        if random.random() < 0.05:          # 5% totally garbled
            desc = random.choice(NOISE_DESCRIPTIONS)

        payment = random.choices(PAYMENT_MODES, weights=PAYMENT_WEIGHTS, k=1)[0]

        rows.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Description": desc,
            "Amount": amount,
            "Payment_Mode": payment,
            "Category": cat,           # ground-truth label (hidden from model)
        })

    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)

    # Introduce ~3% missing values for realism
    for col in ["Description", "Payment_Mode"]:
        mask = np.random.rand(len(df)) < 0.03
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_transactions(600)
    df.to_csv("transactions.csv", index=False)
    print(f"✅  Generated {len(df)} transactions → transactions.csv")
    print(df.head(10).to_string())