"""
train.py  --  Train and save all models.   Run:  python train.py

  * Machine Learning : Logistic Regression, Random Forest, Gradient Boosting
  * Deep Learning    : ANN built with Keras
Everything is evaluated on the same untouched test set so they can be compared fairly.
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tensorflow import keras

from churn_utils import (ANN_NAME, FEATURES, LABELS, MODEL_DIR, TARGET, load_data, make_baseline,
                         make_preprocessor, predict_proba)

SEED = 42


def best_threshold(y_true, probs):
    """Probability cut-off with the best F1 score (0.5 is poor for imbalanced data)."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return float(thresholds[np.argmax(f1[:-1])])


def get_scores(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probs),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
    }


def train_ann(X_train, y_train):
    """A simple ANN: 2 hidden layers, dropout, early stopping."""
    keras.utils.set_random_seed(SEED)
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=64,
              callbacks=[stop], verbose=0)
    return model


def feature_importance(store, name, X, y, repeats=3):
    """Shuffle one feature at a time; the bigger the drop in AUC, the more important it is."""
    rng = np.random.default_rng(SEED)
    base_auc = roc_auc_score(y, predict_proba(store, name, X))
    drops = []
    for feature in FEATURES:
        total = 0
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].values)
            total += base_auc - roc_auc_score(y, predict_proba(store, name, shuffled))
        drops.append(total / repeats)
    return pd.DataFrame({"feature": FEATURES, "name": [LABELS[f] for f in FEATURES],
                         "importance": drops}).sort_values("importance", ascending=False)


def main():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    print(f"Customers: {len(df):,} | churn rate: {y.mean():.1%}")

    # 60% train, 20% validation, 20% test (stratified keeps the churn share equal in each)
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=SEED)

    # ---- Machine Learning models (each one = preprocessing + model)
    ml_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
    }
    for name in ml_models:
        pipe = Pipeline([("prep", make_preprocessor()), ("model", ml_models[name])])
        ml_models[name] = pipe.fit(X_train, y_train)

    # ---- Deep Learning model
    preprocessor = make_preprocessor().fit(X_train)
    X_train_scaled = preprocessor.transform(X_train)
    if hasattr(X_train_scaled, "toarray"):
        X_train_scaled = X_train_scaled.toarray()
    ann = train_ann(X_train_scaled.astype("float32"), y_train.values)

    store = {"preprocessor": preprocessor, "ann": ann, "ml_models": ml_models}

    # ---- Evaluate every model the same way
    thresholds, importance, results = {}, {}, {}
    for name in [ANN_NAME] + list(ml_models):
        thresholds[name] = best_threshold(y_val, predict_proba(store, name, X_val))
        test_probs = predict_proba(store, name, X_test)
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        matrix = confusion_matrix(y_test, (test_probs >= thresholds[name]).astype(int))
        results[name] = {
            "threshold": thresholds[name],
            "scores": get_scores(y_test, test_probs, thresholds[name]),
            "confusion_matrix": matrix.tolist(),               # [[TN, FP], [FN, TP]]
            "roc": {"fpr": np.round(fpr, 3).tolist(), "tpr": np.round(tpr, 3).tolist()},
        }
        importance[name] = feature_importance(store, name, X_test, y_test)
        s = results[name]["scores"]
        print(f"{name:22s} AUC {s['roc_auc']:.3f} | recall {s['recall']:.2f} | "
              f"precision {s['precision']:.2f} | F1 {s['f1']:.2f}")

    # ---- Save
    MODEL_DIR.mkdir(exist_ok=True)
    ann.save(MODEL_DIR / "ann.keras")                    # Keras model saved separately
    joblib.dump({
        "preprocessor": preprocessor,
        "ml_models": ml_models,
        "thresholds": thresholds,
        "baseline": make_baseline(df),
        "categories": {"Geography": sorted(df["Geography"].unique()),
                       "Gender": sorted(df["Gender"].unique())},
        "importance": importance,
    }, MODEL_DIR / "store.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to the models/ folder.")


if __name__ == "__main__":
    main()
