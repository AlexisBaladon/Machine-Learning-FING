from multiprocessing.resource_sharer import stop
import pandas as pd
import numpy as np
from constants import DEFAULT_SEED, CLUSTER_COLUMN, ITERATION_THRESHOLD
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
        self.random_generator = random.Random()
        self.random_generator.seed(seed)
        self.new_centroids = []

    def k_means(self, dataset: pd.DataFrame, number_of_clusters: int, stop_epsilon: float, seed = None):
        if seed != None:
            self.random_generator.seed(seed)
        self.centroids = [np.zeros(len(dataset.columns)) for _ in range(number_of_clusters)]
        clustered_dataset = self.augment_dataset(dataset)
        self.new_centroids = self.random_assignment(dataset, number_of_clusters)
        i = 0
        while not self.__stop_condition(stop_epsilon, i):
            self.centroids = self.new_centroids
            self.assign_to_cluster(dataset, clustered_dataset, self.centroids)
            self.__check_empty_cluster(dataset, clustered_dataset, self.centroids)
            self.new_centroids = self.__calculate_centroids(clustered_dataset)
            i += 1
        self.centroids = self.new_centroids
        return np.array(self.centroids)
    
    def __stop_condition(self, stop_epsilon: float, i) -> bool:
        distance = self.__calculate_distances(self.centroids, self.new_centroids)
        return i > ITERATION_THRESHOLD or distance < stop_epsilon

    def __calculate_distances(self, centroids, new_centroids):
        return np.sum(list(map(self.calculate_distance, centroids, new_centroids)))

    def calculate_distance(self, centroid, point):
        return np.linalg.norm(centroid - point)

    def __calculate_centroids(self, dataset: pd.DataFrame):
        new_centroids = []

        for idx, _ in enumerate(self.centroids):
            centroid_dataset = dataset[dataset[CLUSTER_COLUMN] == idx].drop(columns=[CLUSTER_COLUMN])
            centroid_mean = centroid_dataset.mean(axis=0)
            new_centroids.append(np.array(centroid_mean))

        return new_centroids

    def assign_to_cluster(self, dataset: pd.DataFrame, clustered_dataset: pd.DataFrame, centroids):   
        labels = []     
        for _, row  in dataset.iterrows():
            point = np.array(row)
            labels.append(int(np.argmin([self.calculate_distance(centroid, point) for centroid in centroids])))
            
        clustered_dataset[CLUSTER_COLUMN] = labels
        return

    def ____check_empty_cluster(self, dataset: pd.DataFrame, clustered_dataset: pd.DataFrame, centroids):
        clustered_dataset = clustered_dataset[CLUSTER_COLUMN]
        for label in labels:
            if label not in clustered_dataset:
                clustered[CLUSTER_COLUMN] = np.argmin([self.calculate_distance(centroid, point) for centroid in centroids])

        return

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
        copy.insert(0, CLUSTER_COLUMN, -1)
        return copy
        