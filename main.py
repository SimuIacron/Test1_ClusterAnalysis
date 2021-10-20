from numpy import where
from numpy import unique
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from matplotlib import pyplot

#X, y = make_classification(n_samples=1000, n_features=3, n_informative=2, n_redundant=0, n_clusters_per_class=1,
#                           random_state=0)
X,y = make_moons(n_samples=1000, shuffle=True, random_state=4, noise=0.05)
#X, y = make_blobs(n_samples=1000, n_features=3, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

#for class_value in range(2):
#    row_ix = where(y == class_value)
#    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
#pyplot.show()

fig = pyplot.figure()
#ax = fig.add_subplot(projection='3d')

model = DBSCAN(eps=0.3, min_samples=10)
model.fit(X)
yhat = model.labels_
clusters = unique(yhat)

for cluster in clusters:
    row_ix = where(yhat == cluster)
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])

pyplot.show()

