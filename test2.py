import itertools
import os
import multiprocessing as mp
from test import Test
CLUSTERS = 4
STOP_EPSILON = 0.1

kmeans = k_means.KMeans()
PROCESSORS = os.cpu_count()
K_MEANS_TOTAL = 120
AMOUNT_SEED_TESTS = 30


# def get_mean_parallel(df, k_ranges, stop_epsilon):
#     start, stop = k_ranges
#     avgs = []
#     for k in range(start, stop):
#         avg = 0
#         for i in range(AMOUNT_SEED_TESTS):
#             k_centroid = kmeans.k_means(df,k,stop_epsilon)
#             avg += test.__loss_function(df, k_centroid)
#         avgs.append(avg/AMOUNT_SEED_TESTS)
#     return avgs


def get_mean_parallel(df, k, stop_epsilon):
    avg = 0 
    for i in range(AMOUNT_SEED_TESTS):
        k_centroid = kmeans.k_means(df,k,stop_epsilon)
        avg += test.__loss_function(df, k_centroid)
    return avg/AMOUNT_SEED_TESTS



if __name__ == "__main__":
    df = load_csv()
    df = preprocess(df)

    # size = int(K_MEANS_TOTAL/PROCESSORS)
    # k_ranges = [[(size) * i, (size) * (i+1)] for i in range(PROCESSORS)]
    # k_ranges[0][0] = 1
    # k_ranges[-1][-1] = K_MEANS_TOTAL
    k_avgs = []
    k_range = [range(1,K_MEANS_TOTAL+1)]
    with mp.Pool(PROCESSORS) as pool:
        k_avgs = pool.map(get_mean_parallel, df, k_range, STOP_EPSILON)
    test = Test()
    test.__graph_elbow_method(k_avgs, k_avgs.values())