from functools import partial
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from k_means import KMeans
import constants
import os
import multiprocessing as mp
from multiprocessing import Pool
from sklearn.decomposition import PCA
import constants
import pandas as pd

global_kmeans = KMeans()

class Test:
    def __init__(self):
        pass

    def test_elbow_method(self, k_values, dataset, cpu_count=os.cpu_count()):
        k_avgs = []
        partial_application = partial(self.get_mean_parallel,dataset, constants.STOP_EPSILON)
        with mp.Pool(cpu_count) as pool:
            k_avgs = pool.map(partial_application, k_values)
        self.__graph_elbow_method(k_values, k_avgs)
        return

    def test_silhouette(self, k_values, dataset, cpu_count=os.cpu_count()):
        k_avgs = []
        partial_application = partial(self.get_silhouette_parallel, dataset, constants.STOP_EPSILON)
        with mp.Pool(cpu_count) as pool:
            k_avgs = pool.map(partial_application, k_values)
        self.__graph_silhouette(k_values, k_avgs)    
    
    def get_mean_parallel(self, df, stop_epsilon, k):
        kmeans = KMeans()
        avg = 0 
        for i in range(constants.AMOUNT_SEED_TESTS):
            k_centroid = kmeans.k_means(df,k,stop_epsilon,seed = i)
            avg += self.__loss_function(df, k_centroid)
        return avg/constants.AMOUNT_SEED_TESTS

    def get_silhouette_parallel(self, df, stop_epsilon, k):
        kmeans = KMeans()
        avg = 0 
        for i in range(constants.AMOUNT_SEED_TESTS):
            k_centroids = kmeans.k_means(df, k, stop_epsilon, seed = i)
            clustered_dataset = kmeans.augment_dataset(df, clustered_dataset, k_centroids)
            labels = kmeans.assign_to_cluster(df, clustered_dataset, k_centroids)
            avg += silhouette_score(df, labels, metric="euclidean") #TODO: labels
        return avg/constants.AMOUNT_SEED_TESTS

    def __graph_elbow_method(self, values_k, values_y):
        TITLE = "Elbow Method"
        K_LABEL = "k"
        LOSS_FUNCTION_LABEL = "Loss Function (Euclidean)"
        self.__continuous_function(TITLE, values_k, values_y, K_LABEL, LOSS_FUNCTION_LABEL)

    def __graph_silhouette(self, values_k, values_y):
        TITLE = "Silhouette"
        K_LABEL = "k"
        SILHOUETTE_LABEL = "Silhouette (Euclidean)"
        self.__continuous_function(TITLE, values_k, values_y, K_LABEL, SILHOUETTE_LABEL)

    def __continuous_function(self, title, values_x, values_y, x_label, y_label):
        fig, axs = plt.subplots()
        axs.plot(values_x, values_y)
        axs.set(xlabel=x_label, ylabel=y_label,title=title)

        axs.grid(True)
        plt.show()

    def __loss_function(self, dataset, centroids):
        clustered_ds = global_kmeans.augment_dataset(dataset)
        global_kmeans.assign_to_cluster(dataset, clustered_ds, centroids)

        sse = 0
        for idx, row in clustered_ds.iterrows():
            centroid_idx = int(row[constants.CLUSTER_COLUMN])
            centroid = centroids[centroid_idx]
            point = dataset.iloc[idx]
            sse += global_kmeans.calculate_distance(centroid, point)

        return sse

    def PCA_2D(self, df, k = constants.DEFAULT_CLUSTERS, epsilon=constants.STOP_EPSILON):
        pca = PCA(n_components=2,random_state=constants.DEFAULT_SEED)
        pca_df = pca.fit_transform(df)
        pca_df = pd.DataFrame(pca_df, columns=["pca1", "pca2"])

        kmeans = KMeans()
        centroids = kmeans.k_means(pca_df, k, epsilon, constants.DEFAULT_SEED)

        clustered_df = pca_df.copy()
        kmeans.assign_to_cluster(pca_df, clustered_df, centroids)

        plt.scatter(clustered_df["pca1"], clustered_df["pca2"], c=clustered_df[constants.CLUSTER_COLUMN])
        for centroid in centroids:
            plt.scatter(centroid[0], centroid[1], c="black")
        plt.show()
    
    def PCA_3D(self, df, k = constants.DEFAULT_CLUSTERS, epsilon=constants.STOP_EPSILON):
        pca = PCA(n_components=3,random_state=constants.DEFAULT_SEED)
        pca_df = pca.fit_transform(df)
        pca_df = pd.DataFrame(pca_df, columns=["pca1", "pca2", "pca3"])
    
        kmeans = KMeans()
        centroids = kmeans.k_means(pca_df,k,epsilon,constants.DEFAULT_SEED)
    
        clustered_df = pca_df.copy()
        kmeans.assign_to_cluster(pca_df, clustered_df, centroids)
            
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        ax.scatter(clustered_df["pca1"], clustered_df["pca2"], clustered_df["pca3"], c=clustered_df[constants.CLUSTER_COLUMN])
        for centroid in centroids:
            ax.scatter(centroid[0], centroid[1], c="black")
        plt.show()