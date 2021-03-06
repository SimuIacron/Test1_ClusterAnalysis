import pandas as pd
from numpy import argmin, min, median, mean
import plotly.express as px
import util
import plotly.graph_objects as go


# counts how often each family occurs in a cluster and returns them as a list of family names and their amounts
# also returns the amount of elements in the cluster
# cluster_idx: The index of the cluster to count
# yhat: The array that describes which instances is in which cluster
# family: List of the family of all instances
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


# draws a scatter plot of the clusters
# yhat: Describes which instance is in which cluster
# data: Features of the instances
# solver_time: The times of the solvers for each instance
# solver_features: Name of all solvers
def clusters_scatter_plot(yhat, data, solver_time, solver_features):
    best_solver_time = [min(elem) for elem in solver_time]
    best_solver = [solver_features[argmin(elem)] for elem in solver_time]
    scatter_values = util.rotateNestedLists(data)
    df = pd.DataFrame(dict(axis1=scatter_values[0], axis2=scatter_values[1], cluster=yhat, solver_time=best_solver_time,
                           solver=best_solver))
    df["cluster"] = df["cluster"].astype(str)
    fig = px.scatter(df, x='axis1', y='axis2', color='cluster', size='solver_time', hover_data=['solver'])
    fig.show()


# Shows Median and Mean of the solver times for a cluster
# cluster_idx: Index of the cluster
# yhat: Describes which instance is in which cluster
# solver_time: The times of the solvers for each instance
# solver_features: Name of all solvers
def cluster_statistics(cluster_idx, yhat, solver_time, solver_features):
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
    fig.update_layout(barmode='group', title='Cluster: ' + str(cluster_idx) + " : " + str(cluster_amount))
    fig.show()


# Shows the amount of each family in a cluster as a bar chart
# cluster_idx: Index of the cluster
# yhat: Describes which instance is in which cluster
# family: List of the family of all instances
def cluster_family_amount(cluster_idx, yhat, family):
    keys, values, cluster_amount = count_family_for_cluster(cluster_idx, yhat, family)
    sorted_keys = [x for _, x in sorted(zip(values, keys))]
    sorted_values = sorted(values)
    sorted_values_ratio = [(value / cluster_amount) for value in sorted_values]

    df = pd.DataFrame(dict(axis1=sorted_keys, axis2=sorted_values_ratio, values=sorted_values))
    fig = px.bar(df, x='axis1', y='axis2', hover_data=['values'])
    fig.update_layout(title='Cluster: ' + str(cluster_idx) + " : " + str(cluster_amount))
    fig.show()


# Shows bar chart where it is shown what amount of each family all clusters hold
# clusters: list of all clusters
# yhat: Describes which instance is in which cluster
# family: List of the family of all instances
def clusters_family_amount(clusters, yhat, family):
    data = []
    for cluster in clusters:
        keys, values, cluster_amount = count_family_for_cluster(cluster, yhat, family)
        data.append(go.Bar(name=cluster.astype(str), x=keys, y=values))#

    fig = go.Figure(data=data)
    # Change the bar mode
    fig.update_layout(barmode='stack')
    fig.show()


# shows bar chart of how many instances with timeout each cluster has
# a instance is considered to have a timeout when all solvers timeout
# clusters: list of all clusters
# yhat: Describes which instance is in which cluster
# timeout_value: The value at which the time is considered a timeout
# solver_time: List of times each solver needs for each instance
def clusters_timeout_amount(clusters, yhat, timeout_value, solver_time):
    not_timeout_list = []
    timeout_list = []
    for cluster in clusters:
        timeout = 0
        not_timeout = 0
        for i in range(len(yhat)):
            if yhat[i] == cluster:
                if min(solver_time[i]) >= timeout_value:
                    timeout = timeout + 1
                else:
                    not_timeout = not_timeout + 1

        timeout_list.append(timeout)
        not_timeout_list.append(not_timeout)

    fig = go.Figure(data=[
        go.Bar(name='Timeout', x=clusters, y=timeout_list),
        go.Bar(name='No Timeout', x=clusters, y=not_timeout_list)
    ])
    # Change the bar mode
    fig.update_layout(barmode='stack')
    fig.show()








