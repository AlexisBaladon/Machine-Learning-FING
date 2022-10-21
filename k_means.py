from multiprocessing.resource_sharer import stop
import pandas as pd
import numpy as np
from constants import DEFAULT_SEED, CLUSTER_COLUMN
import random

class KMeans:
    def __init__(self, random_init_method = 'from_cluster', seed = DEFAULT_SEED, scale_factor = 1.1):
        self.scale_factor = scale_factor
        self.seed = seed
        if random_init_method == 'from_cluster':
            self.random_assignment = self.__random_assignment_from_cluster
        elif random_init_method == 'from_space':
            self.random_assignment = self.__random_assignment_from_space
        else:
            raise Exception("Unrecognized initialization method")
        self.centroids = []
        self.random_generator = random
        self.random_generator.seed(seed)

    def k_means(self, dataset: pd.DataFrame, number_of_clusters: int, stop_epsilon: float):
        #  1: Inicializar K centroides aleatoriamente
        #  2: Mientras no se de la condición de fin:
        #  3: Asignar cada instancia al centroide más cercano
        #  4: Recalcular los centroides
        #  5: Retornar los clusters

        self.centroids = [np.zeros(len(dataset.columns)) for _ in range(number_of_clusters)] # chequear esto luego, deberia andar ok
        clustered_dataset = self.augment_dataset(dataset)
        self.new_cetroids = self.random_assignment(dataset, number_of_clusters)
        while not self.__stop_condition(stop_epsilon):
            self.centroids = self.new_cetroids
            self.assign_to_cluster(clustered_dataset, self.centroids)
            self.new_cetroids = self.__calculate_centroids(clustered_dataset)
        self.centroids = self.new_cetroids
        return np.array(self.centroids)

    def __calculate_distances(self, centroids, new_centroids):
        #np.sum([np.linalg.norm(x - y) for x in x, y in zip(centroids, new_centroids)])        
        return np.sum(list(map(self.__calculate_distance, centroids, new_centroids)))

    def __calculate_distance(self, centroid, point):
        return np.linalg.norm(centroid - point)

    def __calculate_centroids(self, dataset: pd.DataFrame):
        new_centroids = []

        for idx, _ in enumerate(self.centroids):
            centroid_dataset = dataset[dataset[CLUSTER_COLUMN] == idx].drop(columns=[CLUSTER_COLUMN])
            centroid_mean = centroid_dataset.mean(axis=0)
            new_centroids.append(np.array(centroid_mean))

        return new_centroids

    def assign_to_cluster(self, dataset: pd.DataFrame, centroids):   
        labels = []     
        for row, idx in dataset.iterrows():
            point = np.array(row)
            labels.append(np.argmin(self.__calculate_distance(centroid, point) for centroid in centroids)) # quizas agregar [] si no anda
            
        dataset[CLUSTER_COLUMN] = labels
        return dataset

    def __stop_condition(self, stop_epsilon: float) -> bool:
        distance = self.__calculate_distances(self.centroids, self.new_cetroids)
        print(distance)
        return distance < stop_epsilon

    # Precondition: run agument_dataset once on the dataset
    def __random_assignment_from_space(self, dataset: pd.DataFrame, number_of_clusters):
        df_len = len(dataset.columns)
        centroids = []
        scale_vector = []

        for i in df_len:
            scale_vector.append(dataset.iloc[i].max())
        scale_vector = np.array(scale_vector) * self.scale_factor
        
        for _ in range(number_of_clusters):
            centroid = []
            for i in range(df_len):
                centroid.append(self.random_generator.random() * scale_vector[i])
            centroids.append(np.array(centroid))
        
        return centroids

    # Precondition: run agument_dataset once on the dataset
    def __random_assignment_from_cluster(self, dataset: pd.DataFrame, number_of_clusters):
        centroids = []
        bag = [x for x in range(len(dataset.index))]

        sample_idxs = self.random_generator.sample(bag, number_of_clusters)
        for idx in sample_idxs:
            centroids.append(np.array(dataset.iloc[idx,:]))
            
        return centroids

    # Returns a dataset such that each point can now have a cluster
    # must be called before assing_to_cluster
    def augment_dataset(self, dataset: pd.DataFrame):
        copy = dataset.copy()
        copy.insert(0, CLUSTER_COLUMN, pd.NA)
        return copy
        