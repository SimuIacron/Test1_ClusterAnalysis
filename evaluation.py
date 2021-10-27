import pandas as pd
from numpy import argmin, min, median, mean
import plotly.express as px
import util
import plotly.graph_objects as go


def count_family_for_cluster(cluster_idx, yhat, family):
    familyDict = {}
    cluster_amount = 0
    for i in range(len(yhat)):
        if yhat[i] == cluster_idx:
            cluster_amount = cluster_amount + 1
            # replace timeout and failed for the set timeout_value
            if family[i] in familyDict:
                familyDict[family[i]] = familyDict[family[i]] + 1
            else:
                familyDict[family[i]] = 1

    keys = []
    values = []
    for key, value in familyDict.items():
        keys.append(key[0])
        values.append(value)

    return keys, values, cluster_amount

def clusters_scatter_plot(yhat, data, solver_time, solver_features):
    best_solver_time = [min(elem) for elem in solver_time]
    best_solver = [solver_features[argmin(elem)] for elem in solver_time]
    scatter_values = util.rotateNestedLists(data)
    df = pd.DataFrame(dict(axis1=scatter_values[0], axis2=scatter_values[1], cluster=yhat, solver_time=best_solver_time,
                           solver=best_solver))
    df["cluster"] = df["cluster"].astype(str)
    fig = px.scatter(df, x='axis1', y='axis2', color='cluster', size='solver_time', hover_data=['solver'])
    fig.show()


def clusters_statistics(cluster_idx, yhat, solver_time, solver_features):
    # stores the times of the instances in the current cluster
    timelist = []
    # counts how many elements are in the cluster
    cluster_amount = 0
    for i in range(len(yhat)):
        if yhat[i] == cluster_idx:
            cluster_amount = cluster_amount + 1
            # replace timeout and failed for the set timeout_value
            insert = solver_time[i]
            timelist.append(insert)

    # rotate list to get lists for each algorithm and calculate it's median and mean time
    timelist_s = util.rotateNestedLists(timelist)
    median_list = [median(x) for x in timelist_s]
    mean_list = [mean(x) for x in timelist_s]
    average_time = []

    # plot median and mean times for each cluster
    fig = go.Figure(data=[
        go.Bar(name='Median', x=solver_features, y=median_list),
        go.Bar(name='Mean', x=solver_features, y=mean_list)
    ])
    # Change the bar mode
    fig.update_layout(barmode='group', title=cluster_amount)
    fig.show()


def cluster_family_amount(cluster_idx, yhat, family):
    keys, values, cluster_amount = count_family_for_cluster(cluster_idx, yhat, family)
    sorted_keys = [x for _, x in sorted(zip(values, keys))]
    sorted_values = sorted(values)
    sorted_values_ratio = [(value / cluster_amount) for value in sorted_values]

    df = pd.DataFrame(dict(axis1=sorted_keys, axis2=sorted_values_ratio, values=sorted_values))
    fig = px.bar(df, x='axis1', y='axis2', hover_data=['values'])
    fig.update_layout(title=cluster_amount)
    fig.show()


def clusters_family_amount(clusters, yhat, family):
    data = []
    for cluster in clusters:
        keys, values, cluster_amount = count_family_for_cluster(cluster, yhat, family)
        data.append(go.Bar(name=cluster.astype(str), x=keys, y=values))#

    fig = go.Figure(data=data)
    # Change the bar mode
    fig.update_layout(barmode='stack')
    fig.show()





