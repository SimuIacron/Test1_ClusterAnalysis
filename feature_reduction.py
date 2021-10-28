from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection


def feature_reduction(data, algorithm="PCA", features=None):
    if algorithm == "SPARSE":
        model = SparseRandomProjection(n_components=features)
    elif algorithm == "GAUSSIAN":
        model = GaussianRandomProjection(n_components=features)
    elif algorithm == "NONE":
        return data
    else:  # algorithm == "PCA
        if features is None:
            model = PCA(n_components="mle", svd_solver="full")
        else:
            model = PCA(n_components=features)

    model.fit(data)
    return model.transform(data)
