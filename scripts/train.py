import sys
import os
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.helpers import load_config, ensure_dir
from src.data.load_data import load_selected_features
from src.data.preprocess import encode_target, encode_categoricals, scale_features, split_data
from src.models.model import create_logistic_regression
from src.models.train import train_model, cross_validate
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
    X_encoded, encoders = encode_categoricals(X)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X_encoded, y,
        test_size=config["data"]["test_size"],
        val_size=config["data"]["val_size"],
        random_state=config["data"]["random_state"],
    )
    X_train_s, X_val_s, X_test_s, scaler = scale_features(X_train, X_val, X_test)

    # Train
    lr_config = config["model"]["logistic_regression"]
    model = create_logistic_regression(**lr_config, random_state=config["data"]["random_state"])
    model = train_model(model, X_train_s, y_train)

    # Cross-validation on training data
    cv_config = config["cross_validation"]
    cv_results = cross_validate(model, X_train_s, y_train, cv=cv_config["n_folds"], scoring=cv_config["scoring"])
    print(f"Cross-validation {cv_config['scoring']}: {cv_results['mean']:.4f} (+/- {cv_results['std']:.4f})")

    # Evaluate
    for name, X_set, y_set in [("Train", X_train_s, y_train), ("Validation", X_val_s, y_val), ("Test", X_test_s, y_test)]:
        y_pred = predict(model, X_set)
        y_prob = predict_proba(model, X_set)
        results = evaluate_model(y_set, y_pred, y_prob)
        print(f"\n{'='*40}")
        print_metrics(results, name)

    # Save model
    ensure_dir("models")
    joblib.dump(model, "models/logistic_regression.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print("\nModel saved to models/")


if __name__ == "__main__":
    main()
