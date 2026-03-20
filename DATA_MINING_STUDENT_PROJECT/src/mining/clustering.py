from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def find_best_k(X, max_k):

    best_k = 2
    best_score = -1

    for k in range(2, max_k):

        model = KMeans(n_clusters=k, random_state=42)

        labels = model.fit_predict(X)

        score = silhouette_score(X, labels)

        if score > best_score:

            best_score = score
            best_k = k

    return best_k


def run_kmeans(X, max_k):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    k = find_best_k(X_scaled, max_k)

    model = KMeans(n_clusters=k, random_state=42)

    labels = model.fit_predict(X_scaled)

    return labels, model