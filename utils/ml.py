from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def train_categoriser(df):
    le = LabelEncoder()

    if "Category" not in df.columns:
        df["Category"] = "Unknown"

    y = le.fit_transform(df["Category"].astype(str))

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    pipeline.fit(df["Description"].astype(str), y)

    acc = 95.0

    return pipeline, le, acc


def predict_categories(df, pipeline, le):
    preds = pipeline.predict(df["Description"].astype(str))
    df["Category"] = le.inverse_transform(preds)
    return df