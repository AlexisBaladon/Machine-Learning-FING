from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def remove_columns(dataset: pd.DataFrame, columns: list[str]):
    dataset = dataset.drop(columns=columns)
    return dataset

def one_hot_encoding():
    return

def normalize(dataset):
    scaler = MinMaxScaler()
    df_columns = dataset.columns.copy()
    df_scaled = scaler.fit_transform(dataset.to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=df_columns)
    return df_scaled

def preprocess(dataset):
    #columns_to_remove = ["affordability","inclusivity_and_tolerance","pos_2021","total_score","city","country"]
    #columns_to_remove = ["paid_parental_leave","covid_impact","covid_support","healthcare","access_to_mental_healthcare","inclusivity_and_tolerance","affordability","happiness_culture_and_leisure","city_safety","outdoor_spaces","pos_2022","pos_2021","city","country","remote_jobs","overworked_population","minimum_vacations_offered","vacations_taken","unemployment","multiple_jobholders","inflation"]
    columns_to_remove = ["total_score", "city", "country", "pos_2022", "pos_2021", "covid_impact", "covid_support", "inclusivity_and_tolerance", "air_quality"]
    dataset = remove_columns(dataset, columns_to_remove)
    dataset = normalize(dataset)
    return dataset