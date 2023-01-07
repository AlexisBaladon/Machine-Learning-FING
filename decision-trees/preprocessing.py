import pandas as pd
import ID3
import json
import jsonpickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTENC
from constants import *
from tree import TreeWrapper

def get_calssified_attributes():
    discrete_attributes = []
    continuous_attributes = []
    idx_categorical = []
    i = 0
    for attr in other_attributes:
        if is_discrete[attr]:
            discrete_attributes.append(attr)
            idx_categorical.append(i)
        else:
            continuous_attributes.append(attr)
        i += 1 
    return discrete_attributes, continuous_attributes, idx_categorical

over_sampler = RandomOverSampler(random_state=SEED, sampling_strategy="minority")
under_sampler = RandomUnderSampler(random_state=SEED, sampling_strategy="majority")
_, _, categorical_features = get_calssified_attributes()
SMOTENC_oversampler = SMOTENC(categorical_features=categorical_features, random_state=SEED)

def toEnum(ds: pd.DataFrame, mappings) -> pd.DataFrame:
    for attribute in mappings.keys():
        ds[attribute] = ds[attribute].map(mappings[attribute])
    return ds

def get_most_common_values(ds: pd.DataFrame, discrete_attributes) -> dict:
    res = {}
    for attr in discrete_attributes:
        table = ds.groupby([attr])[attr].count()
        idx_max = table.idxmax(skipna=True)
        res[attr] = idx_max
    return res

def clean_NaN_discrete(corrected_dataset: pd.DataFrame, discrete_attributes, most_common_values) -> None:
    for attr in discrete_attributes:
        corrected_dataset.loc[:][attr].fillna(value=most_common_values[attr],inplace=True,axis="rows",method=None)
    return


def clean_NaN_continuous(corrected_dataset: pd.DataFrame, continuous_attributes, target_attribute, medians) -> None:
    
    def clean_single_continuous(median, corrected_dataset: pd.DataFrame, continuous_attribute, target_attribute) -> None:
        corrected_dataset.loc[:][continuous_attribute].fillna(value=median,inplace=True,axis="rows",method=None)
        return
        
    for attr in continuous_attributes:
        clean_single_continuous(medians[attr], corrected_dataset, attr, target_attribute)
    pass

def get_medians(ds: pd.DataFrame, continuous_attributes: list[str], target_attribute: str) -> dict:
    medians = {}
    for attr in continuous_attributes:
        medians[attr] = ds[:][attr].agg("median")
    return medians

def remove_columns(dataset: pd.DataFrame, columns: list[str]):
    dataset = dataset.drop(columns=columns)
    return dataset

def preprocess_dataset(ds: pd.DataFrame):
    data = toEnum(ds, mappings)
    data = remove_columns(data,["id"])

    discrete_attributes, continuous_attributes, _ = get_calssified_attributes()

    train, test = train_test_split(ds, test_size=0.2, random_state=SEED)
    most_common_values = get_most_common_values(train,discrete_attributes)
    medians = get_medians(train, continuous_attributes, target_attribute)

    clean_NaN_discrete(train, discrete_attributes, most_common_values)
    clean_NaN_continuous(train, continuous_attributes, target_attribute, medians)

    clean_NaN_discrete(test, discrete_attributes, most_common_values)
    clean_NaN_continuous(test, continuous_attributes, target_attribute, medians)
    return train, test, most_common_values, medians

def get_smote_oversampled_dataset(ds: pd.DataFrame):
    SMOTENC_ds, _ = SMOTENC_oversampler.fit_resample(ds,ds.loc[:][target_attribute])
    return SMOTENC_ds

def get_oversampled_dataset(ds: pd.DataFrame):
    oversampled_train, _ = over_sampler.fit_resample(ds,ds.loc[:][target_attribute])
    return oversampled_train

def get_undersampled__dataset(ds: pd.DataFrame):
    undersampled_train, _ = under_sampler.fit_resample(ds,ds.loc[:][target_attribute])
    return undersampled_train

def predict(tree: TreeWrapper, test: pd.DataFrame):
    y_pred = []
    for idx, row in test.iterrows():
        y_pred.append(tree.predict(row))
    return y_pred

def get_report(values, predictions):
    return classification_report(values, predictions)

def get_confusion_matrix(values, predictions):
    return confusion_matrix(values, predictions)

if __name__ == "__main__":
    ds = pd.read_csv("./dataset/healthcare-dataset-stroke-data.csv")

    train, test, most_common_values, medians = preprocess_dataset(ds)

    oversampled_train = get_oversampled_dataset(train)
    undersampled_train = get_undersampled__dataset(train)

    tree = ID3.ID3(train,target_attribute,other_attributes,domain_values_numeric,is_discrete, medians, most_common_values)
    parsed = json.loads(jsonpickle.encode(tree, unpicklable=False))
    oversampled_tree = ID3.ID3(oversampled_train, target_attribute, other_attributes, domain_values_numeric, is_discrete, medians, most_common_values)
    undersampled_tree = ID3.ID3(undersampled_train, target_attribute, other_attributes, domain_values_numeric, is_discrete, medians, most_common_values)

    y_pred = predict(tree, test)
    y_pred_oversampled = predict(oversampled_tree, test)
    y_pred_undersampled = predict(undersampled_tree, test)

    report = get_report(test[:][target_attribute], y_pred)
    res_confusion_matrix = get_confusion_matrix(test[:][target_attribute], y_pred)
    print("Original")
    print(report)
    print(res_confusion_matrix) 
    print("\n")

    oversampled_report = get_report(test[:][target_attribute], y_pred_oversampled)
    oversampled_res_confusion_matrix = get_confusion_matrix(test[:][target_attribute], y_pred_oversampled)
    print("Oversampled")
    print(oversampled_report)
    print(oversampled_res_confusion_matrix)
    print("\n")

    undersampled_report = get_report(test[:][target_attribute], y_pred_undersampled)
    undersampled_res_confusion_matrix = get_confusion_matrix(test[:][target_attribute], y_pred_undersampled)
    print("Undersampled")
    print(undersampled_report)
    print(undersampled_res_confusion_matrix)
    print("\n")
    
    
    clf = DecisionTreeClassifier(criterion="entropy",random_state=SEED)
    clf = clf.fit(train[:][other_attributes], train[:][target_attribute])
    y_pred_tree = clf.predict(test[:][other_attributes])
    report_tree = get_report(test[:][target_attribute], y_pred_tree)
    res_confusion_matrix_tree = get_confusion_matrix(test[:][target_attribute], y_pred_tree)

    print("sklearn Tree report:")
    print(report_tree)
    print("sklearn Tree confusion_matrix: ")
    print(res_confusion_matrix_tree)



    #SMOTENC_train, _ = SMOTENC_oversampler.fit_resample(train,train.loc[:][target_attribute])

    #SMOTENC_tree = ID3.ID3(SMOTENC_train, target_attribute, other_attributes, domain_values_numeric, is_discrete, medians, most_common_values)

    #y_pred_SMOTENC = []
    #for idx, row in test.iterrows():
    #    y_pred_SMOTENC.append(SMOTENC_tree.predict(row))

    #SMOTENC_report = classification_report(test[:][target_attribute], y_pred_SMOTENC)
    #SMOTENC_res_confusion_matrix = confusion_matrix(test[:][target_attribute], y_pred_SMOTENC)
    #
    #print("SMOTENC")
    #print(SMOTENC_report)
    #print(SMOTENC_res_confusion_matrix)
    #print("\n")