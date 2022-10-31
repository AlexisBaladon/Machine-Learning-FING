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
import numpy as np

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

    def PCA_graph(self, df, dimensions = 2, number_of_clusters = constants.DEFAULT_CLUSTERS, epsilon=constants.STOP_EPSILON, random_init_method = 'from_cluster', seed = constants.DEFAULT_SEED):
        pca = PCA(n_components=dimensions,random_state=constants.DEFAULT_SEED)
        pca.fit(df)
        kmeans = KMeans(random_init_method=random_init_method, seed=seed)
        
        centroids = kmeans.k_means(df, number_of_clusters, epsilon, seed)
        clustered_dataset = kmeans.augment_dataset(df)
        kmeans.assign_to_cluster(df, clustered_dataset, centroids)
        pca_component = pca.components_
        
        transformed_matrix = pca.transform(df).transpose()
        means = df.mean(axis=0)

        if dimensions == 2:
            plt.scatter(transformed_matrix[0], transformed_matrix[1], c=clustered_dataset[constants.CLUSTER_COLUMN])
        
            for centroid in centroids:
                centroid = np.dot(pca_component, centroid - means)
                plt.scatter(centroid[0], centroid[1], c="red")
            plt.show()

        elif dimensions == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        
            ax.scatter(transformed_matrix[0], transformed_matrix[1], transformed_matrix[2], c=clustered_dataset[constants.CLUSTER_COLUMN])
            for centroid in centroids:
                centroid = np.dot(pca_component, centroid - means)
                ax.scatter(centroid[0], centroid[1], centroid[2], c="black")
            plt.show()

 
    def PCA_eigen_values(self, df: pd.DataFrame):
        pca = PCA(n_components=len(df.columns),random_state=constants.DEFAULT_SEED)
        pca.fit(df)
        eigen_values = pca.explained_variance_

        #Logarithmic scale
        _, ax = plt.subplots()
        ax.set_yscale('log')

        plt.bar(range(len(eigen_values)), eigen_values)
        return

    def correlation_matrix(self, df: pd.DataFrame):
        figure_number = plt.figure(figsize=(6, 10)).number
        plt.matshow(df.corr().abs(), fignum=figure_number)
        plt.xticks(range(df.shape[1]), df.columns, fontsize=11, rotation=90)
        plt.yticks(range(df.shape[1]), df.columns, fontsize=11)
        plt.colorbar()
        plt.show()
        return

    def covariance_matrix(self, df: pd.DataFrame):
        figure_number = plt.figure(figsize=(6, 10)).number
        plt.matshow(df.cov().abs(), fignum=figure_number)
        plt.xticks(range(df.shape[1]), df.columns, fontsize=11, rotation=90)
        plt.yticks(range(df.shape[1]), df.columns, fontsize=11)
        plt.colorbar()
        plt.show()
        return

    def find_city_idx(self, original_df, df, city = "Montevideo", country = "Uruguay"):
        query_df = original_df.query("city == @city and country == @country")
        idx = query_df.index[0]
        return idx

    def find_city_df(self, original_df, df, city = "Montevideo", country = "Uruguay"):
        idx = self.find_city_idx(original_df, df, city, country)
        return df.iloc[idx]
    
