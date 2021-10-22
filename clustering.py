from numpy import where
from numpy import unique
import numpy as np
from matplotlib import pyplot

from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation


def cluster(instances_list, index1, index2, index3, algorithm = "DBSCAN"):
    print("Starting clustering...")

    if algorithm == "KMEANS":
        model = KMeans(n_clusters=5)
        model.fit(instances_list)
        yhat = model.labels_
    elif algorithm == "AFFINITY":
        model = AffinityPropagation()
        model.fit(instances_list)
        yhat = model.labels_
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=0.2, min_samples=10)
        model.fit(instances_list)
        yhat = model.labels_

    #print(instances_list[[355, 366]])

    clusters = unique(yhat)
    # fig = pyplot.figure()
    # ax = fig.add_subplot(projection='2d')


    for cluster in clusters:
        x_axis = []
        y_axis = []
        # z_axis = []
        for i in range(len(yhat)):
            if yhat[i] == cluster:
                x_axis.append(instances_list[i][index1])
                y_axis.append(instances_list[i][index2])
                # z_axis.append(instances_list[i][index3])

        pyplot.scatter(x_axis, y_axis) #, z_axis)

    print("Clustering finished")
    print(clusters)
    pyplot.show()
