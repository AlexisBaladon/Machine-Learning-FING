import k_means
import pandas as pd
import constants
from preprocessing import remove_columns

def load_csv():
    return pd.read_csv(constants.DATASET_PATH)


if __name__ == "__main__":
    df = load_csv()
    DROP_COLUMNS = ["city", "country"]
    df = remove_columns(df, DROP_COLUMNS)
    kmeans = k_means.KMeans()
    kmeans.k_means(df,2,0.1)