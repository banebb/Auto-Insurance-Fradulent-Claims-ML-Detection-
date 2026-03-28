from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(y_true, y_pred, y_proba=None):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred),
    }
    if y_proba is not None:
        results["roc_auc"] = roc_auc_score(y_true, y_proba)
    return results


def print_metrics(results, set_name=""):
    prefix = f"[{set_name}] " if set_name else ""
    print(f"{prefix}Accuracy:  {results['accuracy']:.4f}")
    print(f"{prefix}Precision: {results['precision']:.4f}")
    print(f"{prefix}Recall:    {results['recall']:.4f}")
    print(f"{prefix}F1 Score:  {results['f1']:.4f}")
    if "roc_auc" in results:
        print(f"{prefix}ROC AUC:   {results['roc_auc']:.4f}")
