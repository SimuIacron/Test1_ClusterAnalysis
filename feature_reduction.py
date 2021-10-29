from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, chi2
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection


def feature_reduction(data, algorithm="PCA", features=None, variance=0.8):
    if algorithm == "SPARSE":
        model = SparseRandomProjection(n_components=features)
    elif algorithm == "GAUSSIAN":
        model = GaussianRandomProjection(n_components=features)
    elif algorithm == "NONE":
        return data
    elif algorithm == "VARIANCE":
        sel = VarianceThreshold(threshold=(variance * (1 - variance)))
        return sel.fit_transform(data)
    else:  # algorithm == "PCA
        if features is None:
            model = PCA(n_components="mle", svd_solver="full")
        else:
            model = PCA(n_components=features)

    model.fit(data)
    return model.transform(data)
