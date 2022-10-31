import k_means
import pandas as pd
import constants
from preprocessing import preprocess
from test import Test

def load_csv():
    return pd.read_csv(constants.DATASET_PATH)

if __name__ == "__main__":
    original_df = load_csv()
    df = preprocess(original_df)

    kmeans = k_means.KMeans()
    k_values = [x for x in range(2,constants.MAX_K_PLUS_1)]

    #TODO: tendriamos que hacer silouette con varias SEED
    test = Test()
    #print(test.find_city_df(original_df, "Montevideo", "Uruguay"))
    test.PCA_graph(df,original_df,dimensions=2,number_of_clusters=2)
    test.test_elbow_method(k_values,df)
    test.test_silhouette(k_values, df)