import pandas as pd

def remove_columns(dataset: pd.DataFrame, columns: list[str]):
    dataset = dataset.drop(columns=columns)
    return dataset

def one_hot_encoding():
    return

def normalize(): #chiruzzo doesnt approve
    return

