from numpy import where
from numpy import unique
import numpy as np
from matplotlib import pyplot
import pandas as pd
import plotly.express as px
import util

from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering,\
    OPTICS, Birch


def cluster(instances_list, axis1, axis2, axis3, features, algorithm="DBSCAN"):
    print("Starting clustering...")

    if algorithm == "KMEANS":
        model = KMeans(n_clusters=5)
        model.fit(instances_list)
        yhat = model.labels_
    elif algorithm == "AFFINITY":
        model = AffinityPropagation()
        model.fit(instances_list)
        yhat = model.labels_
    elif algorithm == "MEANSHIFT":
        model = MeanShift()
        model.fit(instances_list)
        yhat = model.labels_
    elif algorithm == "SPECTRAL":
        model = SpectralClustering()
        model.fit(instances_list)
        yhat = model.labels_
    elif algorithm == "AGGLOMERATIVE":
        model = AgglomerativeClustering()
        model.fit(instances_list)
        yhat = model.labels_
    elif algorithm == "OPTICS":
        model = OPTICS()
        model.fit(instances_list)
        yhat = model.labels_
    elif algorithm == "BIRCH":
        model = Birch()
        model.fit(instances_list)
        yhat = model.labels_
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=0.2, min_samples=10)
        model.fit(instances_list)
        yhat = model.labels_

    # print(instances_list[[355, 366]])

    clusters = unique(yhat)
    # fig = pyplot.figure()
    # ax = fig.add_subplot(projection='2d')

    # x_axis = []
    # y_axis = []
    # z_axis = []
    # for cluster in clusters:
    #     for i in range(len(yhat)):
    #        if yhat[i] == cluster:
    #            x_axis.append(instances_list[i][index1])
    #            y_axis.append(instances_list[i][index2])
    #            z_axis.append(instances_list[i][index3])

    #       pyplot.scatter(x_axis, y_axis) #, z_axis)
    values = util.rotateNestedLists(instances_list)

    index1 = features.index(axis1)
    index2 = features.index(axis2)
    index3 = features.index(axis3)

    df = pd.DataFrame(
        dict(a=values[index1], b=values[index2], c=values[index3], d=yhat))
    fig = px.scatter_3d(df, x='a', y='b', z='c',
                        color='d')
    fig.show()

    print("Clustering finished")

    return clusters, yhat
    # pyplot.show()
