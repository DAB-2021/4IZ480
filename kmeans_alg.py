from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

cluster_names = ["Cluster 0","Cluster 1","Cluster 2"]

number_of_clusters = range(2,10)

def kmeans_build(kmeans_data, number_of_components=None, number_of_clusters=3):
    if number_of_components != None:
        pca = PCA(n_components=number_of_components, random_state = 453)
        kmeans_data = pca.fit(kmeans_data).transform(kmeans_data)

    cluster_model = KMeans(n_clusters=number_of_clusters, random_state=2)
    cluster_model.fit(kmeans_data)
    
    print(f"The clusters are: {cluster_model.labels_}")

    print(f"The inertia is: {cluster_model.inertia_}")

    predictions = cluster_model.predict(kmeans_data)

    unique, counts = np.unique(predictions, return_counts=True)
    counts = counts.reshape(1,3)

    countscldf = pd.DataFrame(counts, columns = cluster_names)
    print(countscldf)  

def cluster_optimization(kmeans_data, number_of_components=None):
    print("Initializing check for optimal number of clusters")
    
    if number_of_components != None:
        pca = PCA(n_components=number_of_components, random_state = 453)
        kmeans_data = pca.fit(kmeans_data).transform(kmeans_data)
    inertia = list()
    inertia_difference = dict()

    for number in number_of_clusters:
        cluster_test_model = KMeans(n_clusters=number, random_state=4)
        cluster_test_model.fit(kmeans_data)
        models_inertia = cluster_test_model.inertia_
        inertia.append(models_inertia)
        if len(inertia) >= 2:
            difference = inertia[-2]-models_inertia
            print(f"Difference in inertia between clusters: {difference}")
            inertia_difference[number] = difference
        print("The innertia for :", number, "Clusters is:", models_inertia)

    fig, (ax1) = plt.subplots(1, figsize=(16,6))
    xx = np.arange(len(number_of_clusters))
    ax1.plot(xx, inertia)
    ax1.set_xticks(xx)
    ax1.set_xticklabels(number_of_clusters, rotation="vertical")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia score")
    plt.title("Inertia Plot per k")
    plt.show()

    print(inertia_difference)

def components_optimization(kmeans_data):
    X = kmeans_data

    n_components = X.shape[1]
    print(f"Number of components: {n_components}")

    # Running PCA with all components
    pca = PCA(n_components=n_components, random_state = 453)
    pca.fit(kmeans_data).transform(kmeans_data)

    # Calculating the 95% Variance
    total_variance = sum(pca.explained_variance_)
    print("Total Variance in our dataset is: ", total_variance)
    var_95 = total_variance * 0.95
    print("The 95% variance we want to have is: ", var_95)
    print("")

    # Creating a df with the components and explained variance
    a = zip(range(0,n_components), pca.explained_variance_)
    a = pd.DataFrame(a, columns=["PCA Comp", "Explained Variance"])

    # Trying to hit 95%
    print("Variance explain with 1 n_compononets: ", sum(a["Explained Variance"][0:1]))
    print("Variance explain with 2 n_compononets: ", sum(a["Explained Variance"][0:2]))
    print("Variance explain with 3 n_compononets: ", sum(a["Explained Variance"][0:3]))
    print("Variance explain with 4 n_compononets: ", sum(a["Explained Variance"][0:4]))
    print("Variance explain with 5 n_compononets: ", sum(a["Explained Variance"][0:5]))
    print("Variance explain with 10 n_compononets: ", sum(a["Explained Variance"][0:10]))
    print("Variance explain with 15 n_compononets: ", sum(a["Explained Variance"][0:15]))
    print("Variance explain with 20 n_compononets: ", sum(a["Explained Variance"][0:20]))
    print("Variance explain with 25 n_compononets: ", sum(a["Explained Variance"][0:25]))
    print("Variance explain with 30 n_compononets: ", sum(a["Explained Variance"][0:30]))
    print("Variance explain with 32 n_compononets: ", sum(a["Explained Variance"][0:32]))

    # Plotting the Data
    plt.figure(1, figsize=(14, 8))
    plt.plot(pca.explained_variance_ratio_, linewidth=2, c="r")
    plt.xlabel('n_components')
    plt.ylabel('explained_ratio_')

    # Plotting line with 95% e.v.
    plt.axvline(2,linestyle=':', label='n_components - 95% explained', c ="blue")
    plt.legend(prop=dict(size=12))

    # adding arrow
    plt.annotate('2 eigenvectors used to explain 95% variance', xy=(2, pca.explained_variance_ratio_[2]), 
                xytext=(58, pca.explained_variance_ratio_[10]),
                arrowprops=dict(facecolor='blue', shrink=0.05))

    plt.show()

# def clusters_optimization_components(kmeans_data, number_of_components):
#     X = kmeans_data

#     pca = PCA(n_components=number_of_components, random_state = 453)
#     X_r = pca.fit(X).transform(X)

#     inertia = list()
#     inertia_difference = dict()

#     #running Kmeans

#     for f in number_of_clusters:
#         kmeans = KMeans(n_clusters=f, random_state=2)
#         kmeans = kmeans.fit(X_r)
#         u = kmeans.inertia_
#         inertia.append(u)
#         print("The innertia for :", f, "Clusters is:", u)
#         if len(inertia) >= 2:
#             difference = inertia[-2]-u
#             print(f"Difference in inertia between clusters: {difference}")
#             inertia_difference[f] = difference

#     print(inertia_difference)
#     # Creating the scree plot for Intertia - elbow method
#     fig, (ax1) = plt.subplots(1, figsize=(16,6))
#     xx = np.arange(len(number_of_clusters))
#     ax1.plot(xx, inertia)
#     ax1.set_xticks(xx)
#     ax1.set_xticklabels(number_of_clusters, rotation='vertical')
#     plt.xlabel('n_components Value')
#     plt.ylabel('Inertia Score')
#     plt.title("Inertia Plot per k")
#     plt.show()