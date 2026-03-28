import sys
import os
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.helpers import load_config
from src.data.load_data import load_selected_features
from src.data.preprocess import encode_target, encode_categoricals, scale_features, split_data
from src.models.predict import predict, predict_proba
from src.evaluation.metrics import evaluate_model, print_metrics


def main():
    config = load_config()

    # Load data
    df = load_selected_features(config["data"]["processed_path"])
    target_col = config["data"]["target_column"]

    X = df.drop(columns=[target_col])
    y = encode_target(df[target_col])

    # Preprocess
    X_encoded, _ = encode_categoricals(X)
    _, _, X_test, _, _, y_test = split_data(
        X_encoded, y,
        test_size=config["data"]["test_size"],
        val_size=config["data"]["val_size"],
        random_state=config["data"]["random_state"],
    )

    # Load model and scaler
    model = joblib.load("models/logistic_regression.joblib")
    scaler = joblib.load("models/scaler.joblib")

    # Evaluate on test set
    X_test_s = scaler.transform(X_test)
    y_pred = predict(model, X_test_s)
    y_prob = predict_proba(model, X_test_s)
    results = evaluate_model(y_test, y_pred, y_prob)

    print("Test Set Evaluation")
    print("=" * 40)
    print_metrics(results, "Test")
    print(f"\n{results['classification_report']}")


if __name__ == "__main__":
    main()
