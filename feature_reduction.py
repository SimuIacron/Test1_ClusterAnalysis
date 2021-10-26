from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def featureReduction(data, features=None):
    if features is None:
        model = PCA(n_components="mle", svd_solver="full")
    else:
        model = PCA(n_components=features)

    model.fit(data)
    return model.transform(data)
