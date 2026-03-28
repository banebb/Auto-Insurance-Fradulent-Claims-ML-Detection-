from sklearn.linear_model import LogisticRegression


def create_logistic_regression(C=1.0, l1_ratio=0, solver="lbfgs",
                                max_iter=1000, class_weight="balanced",
                                random_state=42):
    return LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        solver=solver,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )
