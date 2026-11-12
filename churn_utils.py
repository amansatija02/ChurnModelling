"""
churn_utils.py  --  helper functions shared by train.py and app.py
  1. load the data
  2. predict with any model (ANN or ML)
  3. explain WHY a customer may leave
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "Churn_Modelling.csv"
MODEL_DIR = BASE_DIR / "models"

TARGET = "Exited"
NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
CATEGORICAL = ["Geography", "Gender"]
BINARY = ["HasCrCard", "IsActiveMember"]
FEATURES = NUMERIC + CATEGORICAL + BINARY

ANN_NAME = "Neural Network (ANN)"
ML_NAMES = ["Logistic Regression", "Random Forest", "Gradient Boosting"]

LABELS = {
    "CreditScore": "Credit score", "Age": "Age", "Tenure": "Tenure (years)",
    "Balance": "Account balance", "NumOfProducts": "Number of products",
    "EstimatedSalary": "Estimated salary", "Geography": "Country", "Gender": "Gender",
    "HasCrCard": "Credit card", "IsActiveMember": "Activity status",
}


# ------------------------------------------------------------------ 1. data
def load_data():
    """Read the CSV, check the columns, remove duplicates and missing values."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    for col in FEATURES + [TARGET]:
        if col not in df.columns:
            raise ValueError(f"Column missing in dataset: {col}")

    df = df.drop_duplicates(subset="CustomerId")
    df = df[FEATURES + [TARGET]].dropna().reset_index(drop=True)
    df[NUMERIC + BINARY] = df[NUMERIC + BINARY].astype(float)   # floats avoid dtype errors later
    return df


def make_baseline(df):
    """Typical loyal customer = middle value (or most common value) among customers who stayed."""
    stayed = df[df[TARGET] == 0]
    baseline = {}
    for col in NUMERIC:
        baseline[col] = stayed[col].median()
    for col in CATEGORICAL + BINARY:
        baseline[col] = stayed[col].mode()[0]
    return baseline


def make_preprocessor():
    """Scale the numbers and one-hot encode the text columns."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])


def to_frame(customer):
    """One customer (dict) -> one-row DataFrame."""
    row = pd.DataFrame([customer])[FEATURES]
    row[NUMERIC + BINARY] = row[NUMERIC + BINARY].astype(float)
    return row


# ------------------------------------------------------------------ 2. models
def load_store():
    """Load everything saved by train.py (ANN, ML models, thresholds, ...)."""
    path = MODEL_DIR / "store.joblib"
    if not path.exists():
        raise FileNotFoundError("Models not found. Run  python train.py  first.")
    store = joblib.load(path)

    from tensorflow import keras            # imported here so ML-only use stays light
    store["ann"] = keras.models.load_model(MODEL_DIR / "ann.keras")
    return store


def predict_proba(store, model_name, df):
    """Churn probability for each row of df (raw columns) using the chosen model."""
    if model_name == ANN_NAME:
        x = store["preprocessor"].transform(df)
        if hasattr(x, "toarray"):           # sparse -> normal array
            x = x.toarray()
        return store["ann"].predict(x.astype("float32"), verbose=0).ravel()
    return store["ml_models"][model_name].predict_proba(df)[:, 1]


def risk_level(prob, threshold):
    if prob >= threshold:
        return "High"
    if prob >= threshold / 2:
        return "Medium"
    return "Low"


# ------------------------------------------------------------------ 3. segments (evidence + graphs)
_BINS = {
    "Age": ([17, 30, 40, 50, 60, 100], ["18-30", "31-40", "41-50", "51-60", "61+"]),
    "CreditScore": ([0, 500, 600, 700, 1000], ["Up to 500", "501-600", "601-700", "701+"]),
    "Tenure": ([-1, 1, 3, 5, 7, 10], ["0-1 yrs", "2-3 yrs", "4-5 yrs", "6-7 yrs", "8-10 yrs"]),
    "Balance": ([-1, 0, 50_000, 100_000, 150_000, 1e12],
                ["Zero", "Up to 50k", "50k-100k", "100k-150k", "150k+"]),
    "EstimatedSalary": ([-1, 50_000, 100_000, 150_000, 1e12],
                        ["Up to 50k", "50k-100k", "100k-150k", "150k+"]),
}
_NAMES = {
    "IsActiveMember": {1: "Active", 0: "Inactive"},
    "HasCrCard": {1: "Has credit card", 0: "No credit card"},
}


def segment_series(df, feature):
    """Turn raw values into readable groups, e.g. Age 43 -> '41-50'."""
    values = df[feature]
    if feature in _BINS:
        bins, names = _BINS[feature]
        return pd.cut(values, bins=bins, labels=names)
    if feature in _NAMES:
        return values.astype(int).map(_NAMES[feature])
    if feature == "NumOfProducts":
        return values.astype(int).astype(str) + " product(s)"
    return values.astype(str)


def churn_by_segment(df, feature):
    """Table with: segment, customers, churn_rate."""
    seg = segment_series(df, feature)
    table = df.groupby(seg, observed=True)[TARGET].agg(customers="size", churn_rate="mean")
    table.index = table.index.astype(str)
    return table.reset_index(names="segment")


def evidence_text(df, feature, value):
    """One sentence with real numbers from the data."""
    one_row = pd.DataFrame({feature: [value]})
    segment = str(segment_series(one_row, feature).iloc[0])
    table = churn_by_segment(df, feature).set_index("segment")
    overall = df[TARGET].mean()
    if segment not in table.index:
        return f"{LABELS[feature]}: {segment}"

    rate = table.loc[segment, "churn_rate"]
    text = (f"**{LABELS[feature]} = {segment}** — {rate:.0%} of similar customers left "
            f"(bank average: {overall:.0%})")
    if rate < overall:
        text += " — low on its own, but risky in combination with the factors above"
    return text


# ------------------------------------------------------------------ 4. the WHY
def explain_customer(store, model_name, customer):
    """
    For each feature, replace the customer's value with the typical loyal-customer value
    and see how much the churn probability drops.
        impact = P(churn) - P(churn with that one feature made 'typical')
    impact > 0 -> pushes the customer towards leaving.  impact < 0 -> keeps them loyal.
    """
    baseline = store["baseline"]
    row = to_frame(customer)
    prob = predict_proba(store, model_name, row)[0]

    # make one changed copy of the customer per feature, then predict them all at once
    copies = pd.concat([row] * len(FEATURES), ignore_index=True)
    for i, feature in enumerate(FEATURES):
        copies.loc[i, feature] = baseline[feature]
    new_probs = predict_proba(store, model_name, copies)

    table = pd.DataFrame({
        "feature": FEATURES,
        "name": [LABELS[f] for f in FEATURES],
        "your_value": [row.iloc[0][f] for f in FEATURES],
        "typical_value": [baseline[f] for f in FEATURES],
        "impact": (prob - new_probs) * 100,        # percentage points
    })
    table = table.sort_values("impact", ascending=False).reset_index(drop=True)
    return prob, table


# Things the bank can change. (Gender is never used for recommendations.)
ACTIONABLE = ["IsActiveMember", "NumOfProducts", "HasCrCard"]


def suggest_action(feature, value):
    if feature == "IsActiveMember":
        return "Re-engage the customer: personalised offer, app notification or a call from the relationship manager."
    if feature == "NumOfProducts":
        if value <= 1:
            return "Cross-sell a second product (savings, card or insurance). Customers with 2 products churn the least."
        return "Review fees and service quality. Customers with 3+ products are leaving at very high rates."
    if feature == "HasCrCard":
        return "Offer a credit card with a welcome benefit."
    if feature == "Balance":
        return "Offer a savings or deposit scheme to keep the balance in the bank."
    if feature == "Geography":
        return "Check local competition and service quality in this country."
    if feature == "Age":
        return "Create age-appropriate offers (e.g. retirement or wealth products for older customers)."
    return "Monitor this customer and review at the next check-in."


def simulate_fix(store, model_name, customer, features):
    """Churn probability if these features were moved to the typical-loyal value."""
    row = to_frame(customer)
    for f in features:
        row.loc[0, f] = store["baseline"][f]
    return predict_proba(store, model_name, row)[0]
