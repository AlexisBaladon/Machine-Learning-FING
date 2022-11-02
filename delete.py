import preprocessing
import constants
import os
from test import Test
from k_means import KMeans

original_df = preprocessing.load_csv()
minimal_preprocess_df = preprocessing.minimal_preprocess(original_df)
df = preprocessing.preprocess(original_df, shouldNormalize=False)

test = Test()
k_values = list(range(1, constants.MAX_K_PLUS_1))

a = test.get_cities_in_the_same_cluster(original_df, df, city = "Montevideo", country = "Uruguay", clusters=2)
print(a)