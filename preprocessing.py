from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import constants

def remove_columns(dataset: pd.DataFrame, columns: list[str]):
    dataset = dataset.drop(columns=columns)
    return dataset

def normalize(dataset):
    scaler = MinMaxScaler()
    df_columns = dataset.columns.copy()
    df_scaled = scaler.fit_transform(dataset.to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=df_columns)
    return df_scaled

def load_csv():
    return pd.read_csv(constants.DATASET_PATH)

def preprocess(dataset, shouldNormalize = False):
    columns_to_remove = ["total_score", "city", "country", "pos_2022", "pos_2021", "access_to_mental_healthcare", "minimum_vacations_offered", "covid_impact", "inclusivity_and_tolerance"]
    dataset = remove_columns(dataset, columns_to_remove)
    if shouldNormalize:
        dataset = normalize(dataset)
    return dataset

def minimal_preprocess(dataset, shouldNormalize = False):
    columns_to_remove = ["total_score", "city", "country", "pos_2022", "pos_2021"]
    dataset = remove_columns(dataset, columns_to_remove)
    if shouldNormalize:
        dataset = normalize(dataset)
    return dataset
