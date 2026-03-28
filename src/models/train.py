from sklearn.model_selection import cross_val_score
import numpy as np


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def cross_validate(model, X, y, cv=5, scoring="f1"):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {
        "scores": scores,
        "mean": np.mean(scores),
        "std": np.std(scores),
    }
