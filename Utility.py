import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

from sklearn.model_selection import GridSearchCV


def kmean_job(k: int, data: pd.DataFrame):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(data)
    score = silhouette_score(data, labels)
    return k, score


def tuning_job(model, param_grids, name, cv, X_train_flat, y_train):
    grid = GridSearchCV (
        estimator=model,
        param_grid=param_grids[name],
        scoring="accuracy",  # 或者换成其他你需要的指标，如 'roc_auc_ovr', 'f1_macro' 等
        cv=cv,
        n_jobs=-1,
        verbose=0,

    )
    grid.fit (X_train_flat, y_train)
    # best_estimators[name] = grid.best_estimator_
    # best_params[name] = grid.best_params_
    # best_scores[name] = grid.best_score_
    return grid