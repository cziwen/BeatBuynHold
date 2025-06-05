import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


def kmean_job(k: int, data: pd.DataFrame):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(data)
    score = silhouette_score(data, labels)
    return k, score