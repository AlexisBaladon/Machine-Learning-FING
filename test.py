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

class Test:
    def __init__(self):
        pass

    def test_elbow_method(self, k_values, dataset):
        k_avgs = []
        partial_application = partial(self.get_mean,dataset, constants.STOP_EPSILON)
        for k in k_values:
            k_avgs.append(partial_application(k))
        self.__graph_elbow_method(k_values, k_avgs)
        return

    def test_silhouette(self, k_values, dataset):
        k_avgs = []
        partial_application = partial(self.get_silhouette, dataset, constants.STOP_EPSILON)
        for k in k_values:
            k_avgs.append(partial_application(k))
        self.__graph_silhouette(k_values, k_avgs)    
    
    def get_mean(self, df, stop_epsilon, k):
        kmeans = KMeans()
        avg = 0 
        for i in range(constants.AMOUNT_SEED_TESTS):
            k_centroid = kmeans.k_means(df,k,stop_epsilon,seed = i)
            avg += self.__loss_function(df, k_centroid)
        return avg/constants.AMOUNT_SEED_TESTS

    def get_silhouette(self, df, stop_epsilon, k):
        avg = 0 
        for i in range(constants.AMOUNT_SEED_TESTS):
            kmeans = KMeans()
            k_centroids = kmeans.k_means(df, k, stop_epsilon, seed = i)
            clustered_dataset = kmeans.augment_dataset(df)
            _, labels = kmeans.assign_to_cluster(df, clustered_dataset, k_centroids)
            avg += silhouette_score(df, labels, metric="euclidean")
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
        kmeans = KMeans()
        clustered_ds = kmeans.augment_dataset(dataset)
        kmeans.assign_to_cluster(dataset, clustered_ds, centroids)

        sse = 0
        for idx, row in clustered_ds.iterrows():
            centroid_idx = int(row[constants.CLUSTER_COLUMN])
            centroid = centroids[centroid_idx]
            point = dataset.iloc[idx]
            sse += kmeans.calculate_distance(centroid, point)

        return sse

    def PCA_graph(self, df, original_df, dimensions = 2, number_of_clusters = constants.DEFAULT_CLUSTERS, epsilon=constants.STOP_EPSILON, random_init_method = 'from_cluster', seed = constants.DEFAULT_SEED):
        pca = PCA(n_components=dimensions,random_state=constants.DEFAULT_SEED)
        pca.fit(df)
        kmeans = KMeans(random_init_method=random_init_method, seed=seed)
        
        centroids = kmeans.k_means(df, number_of_clusters, epsilon, seed)
        clustered_dataset = kmeans.augment_dataset(df)
        kmeans.assign_to_cluster(df, clustered_dataset, centroids)
        pca_component = pca.components_
        
        transformed_matrix = pca.transform(df)
        transposed_matrix = transformed_matrix.transpose()
        means = df.mean(axis=0)

        montevideo_idx = self.find_city_idx(original_df)
        montevideo_x = transformed_matrix[montevideo_idx][0]
        montevideo_y = transformed_matrix[montevideo_idx][1]

        bern_idx = self.find_city_idx(original_df, city="Bern", country="Switzerland")
        bern_x = transformed_matrix[bern_idx][0]
        bern_y = transformed_matrix[bern_idx][1]

        oslo_idx = self.find_city_idx(original_df, city="Oslo", country="Norway")
        oslo_x = transformed_matrix[oslo_idx][0]
        oslo_y = transformed_matrix[oslo_idx][1]

        if dimensions == 2:
            plt.scatter(transposed_matrix[0], transposed_matrix[1], c=clustered_dataset[constants.CLUSTER_COLUMN])
        
            for centroid in centroids:
                centroid = np.dot(pca_component, centroid - means)
                plt.scatter(centroid[0], centroid[1], c="red")

            plt.scatter(montevideo_x, montevideo_y, marker="s", c="black")
            plt.scatter(bern_x, bern_y, marker="s", c="black")
            plt.scatter(oslo_x, oslo_y, marker="s", c="black")
            plt.text(bern_x, bern_y, 'Bern')
            plt.text(montevideo_x, montevideo_y, 'Montevideo')
            plt.text(oslo_x, oslo_y, 'Oslo')

        elif dimensions == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
        
            ax.scatter(transposed_matrix[0], transposed_matrix[1], transposed_matrix[2], c=clustered_dataset[constants.CLUSTER_COLUMN])
            for centroid in centroids:
                centroid = np.dot(pca_component, centroid - means)
                ax.scatter(centroid[0], centroid[1], centroid[2], c="red")
            
            montevideo_z = transformed_matrix[montevideo_idx][2]
            bern_z = transformed_matrix[bern_idx][2]
            oslo_z = transformed_matrix[oslo_idx][2]
            ax.scatter(montevideo_x, montevideo_y, montevideo_z, marker="s", c="black")
            ax.text(montevideo_x, montevideo_y, montevideo_z, 'Montevideo')
            ax.scatter(bern_x, bern_y, bern_z, marker="s", c="black")
            ax.text(bern_x, bern_y, bern_z, 'Bern')
            ax.scatter(oslo_x, oslo_y, oslo_z, marker="s", c="black")
            ax.text(oslo_x, oslo_y, oslo_z, 'Oslo')

        plt.show()
 
    def PCA_eigen_values(self, df: pd.DataFrame, scale='log', drop_first = False):
        pca = PCA(n_components=len(df.columns),random_state=constants.DEFAULT_SEED)
        pca.fit(df)
        eigen_values = pca.explained_variance_
        if drop_first:
            eigen_values = eigen_values[1:]

        #Logarithmic scale
        _, ax = plt.subplots()
        ax.set_yscale(scale)

        plt.bar(range(len(eigen_values)), eigen_values)
        return

    def abs_correlation_matrix(self, df: pd.DataFrame):
        figure_number = plt.figure(figsize=(6, 10)).number
        plt.matshow(df.corr().abs(), fignum=figure_number)
        plt.xticks(range(df.shape[1]), df.columns, fontsize=11, rotation=90)
        plt.yticks(range(df.shape[1]), df.columns, fontsize=11)
        plt.colorbar()
        plt.show()
        return

    def cutpoint_abs_correlation_matrix(self, df: pd.DataFrame, cutpoint = 0.5):
        figure_number = plt.figure(figsize=(6, 10)).number
        plt.matshow(df.corr().abs().applymap(lambda x: 0 if x < cutpoint else 1), fignum=figure_number)
        plt.xticks(range(df.shape[1]), df.columns, fontsize=11, rotation=90)
        plt.yticks(range(df.shape[1]), df.columns, fontsize=11)
        plt.colorbar()
        plt.show()
        return

    def abs_covariance_matrix(self, df: pd.DataFrame):
        figure_number = plt.figure(figsize=(6, 10)).number
        plt.matshow(df.cov().abs(), fignum=figure_number)
        plt.xticks(range(df.shape[1]), df.columns, fontsize=11, rotation=90)
        plt.yticks(range(df.shape[1]), df.columns, fontsize=11)
        plt.colorbar()
        plt.show()
        return

    def find_city_idx(self, original_df, city = "Montevideo", country = "Uruguay"):
        query_df = original_df.query("city == @city and country == @country")
        idx = query_df.index[0]
        return idx

    def find_city_df(self, original_df, df, city = "Montevideo", country = "Uruguay"):
        idx = self.find_city_idx(original_df, city, country)
        return df.iloc[idx]
    
    def get_cities_in_the_same_cluster(self, original_df, df, city = "Montevideo", country = "Uruguay", clusters=2):
        kmeans = KMeans()
        clustered_df = kmeans.augment_dataset(df)
        centroids = kmeans.k_means(df, clusters, constants.STOP_EPSILON)
        kmeans.assign_to_cluster(df, clustered_df, centroids)
        
        city_idx = self.find_city_idx(original_df, city, country)
        city_cluster = clustered_df[constants.CLUSTER_COLUMN][city_idx]
        query_df_idxs = clustered_df[clustered_df[constants.CLUSTER_COLUMN] == city_cluster].index
        return original_df["city"][query_df_idxs]
