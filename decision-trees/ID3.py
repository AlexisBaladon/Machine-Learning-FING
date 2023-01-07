from tree import Tree, TreeWrapper
import pandas as pd
import math

information = lambda x: -x*math.log2(x)

# Assumes that vector contains boolean values
def binary_entropy(entry) -> float:
    dataset_size = len(entry)
    if dataset_size > 0 :
        vector = list(map(bool, entry))
        positive_examples = vector.count(True)
        if positive_examples == dataset_size or positive_examples == 0:
            return 0
        else:
            positive_probability = positive_examples / float(dataset_size)
            negative_probability = 1 - positive_probability
            entropy = information(positive_probability) + information(negative_probability)
            if math.isnan(entropy):
                return 0
            else:
                return entropy
    else:
        raise Exception("entropia vector nulo")

def information_gain_discrete(dataset: pd.DataFrame, target_attribute, split_attribute):
    dataset_entropy = binary_entropy(dataset[:][target_attribute])
    all_vals = dataset[:][split_attribute].unique()
    entropies = []
    dataset_len = float(len(dataset.index))
    for val in all_vals:
        vector = dataset.loc[dataset[split_attribute] == val][target_attribute]
        entropy = binary_entropy(vector)
        entropies.append([len(vector), entropy])
    for e in entropies:
        dataset_entropy -= (e[0]/dataset_len)*e[1] 
    return dataset_entropy

def information_gain_splitted(total_dataset_length, total_dataset_entropy, first_dataset: pd.DataFrame, second_dataset: pd.DataFrame, target_attribute:str):
    if not first_dataset.empty:
        first_entropy = binary_entropy(first_dataset.loc[:][target_attribute])
    else:
        first_entropy = 0
    if not second_dataset.empty:
        second_entropy = binary_entropy(second_dataset.loc[:][target_attribute])
    else:
        second_entropy = 0
    normalized_first_entropy = len(first_dataset.index) * first_entropy / float(total_dataset_length)
    normalized_second_entropy = len(second_dataset.index) * second_entropy / float(total_dataset_length)
    return total_dataset_entropy - (normalized_first_entropy + normalized_second_entropy)

def same_value(dataset: pd.DataFrame, target_attribute) -> bool:
    vector = dataset[:][target_attribute]
    first_value = 0
    first_iteration = True
    for value in vector:
        if(first_iteration):
            first_iteration = False
            first_value = value
        else:   
            if(value != first_value):
                return False
    return True

def most_frequent(dataset, target_value):
    target_column = list(dataset.loc[:][target_value])
    return True if max(target_column, key = target_column.count) == 1 else False

def best_discrete_attribute(dataset, target_attribute, other_attributes) -> str:
    best_att = other_attributes[0]
    best_val = information_gain_discrete(dataset, target_attribute, best_att)
    for att in other_attributes[1:]:
        val = information_gain_discrete(dataset, target_attribute, att)
        if val > best_val:
            best_att = att
            best_val = val
    return best_att, best_val

def best_attribute_continuous(dataset, target_attribute, other_attributes):
    best_att = other_attributes[0]
    best_minor_dataset, best_major_dataset, best_threshold, best_info_gain = get_best_datasets(dataset, split_attribute=best_att, target_attribute=target_attribute)
    for i in other_attributes[1:]:
        minor_dataset, major_dataset, threshold, info_gain = get_best_datasets(dataset, split_attribute=i, target_attribute=target_attribute)
        if info_gain > best_info_gain:
            best_minor_dataset = minor_dataset
            best_major_dataset = major_dataset
            best_threshold = threshold
            best_info_gain = info_gain
            best_att = i
    return best_att, best_minor_dataset, best_major_dataset, best_threshold, best_info_gain 

def get_thresholds(dataset: pd.DataFrame, target_attribute: str, threshold_attribute: str):
    sorted_dataset = dataset.sort_values(threshold_attribute, ascending=True)
    column = sorted_dataset[threshold_attribute]
    max_val = column.max()
    min_val = column.min()

    thresholds = []
    first_target_value = min_val
    last_val = min_val
    for idx, row in sorted_dataset.iterrows():
        new_val = row[threshold_attribute]
        if row[target_attribute] != first_target_value:
            first_target_value = row[target_attribute]
            new_threshold = (last_val + new_val) / float(2)
            if new_threshold not in thresholds:
                thresholds.append(new_threshold) # There may be repeated thresholds in this implementation, hence we decide to remove them.
        last_val = new_val
    if max_val in thresholds:
        thresholds.remove(max_val)
    if min_val in thresholds:
        thresholds.remove(min_val)
    return thresholds

def split_dataset(dataset: pd.DataFrame, split_attribute: str, threshold):
    lesser_than  = dataset[dataset[split_attribute] <= threshold]
    greater_than = dataset[dataset[split_attribute] > threshold]
    return lesser_than, greater_than

def get_best_datasets(dataset: pd.DataFrame, split_attribute: str, target_attribute: str):
    thresholds = get_thresholds(dataset, target_attribute, split_attribute)
    dataset_len = len(dataset.index)
    dataset_entropy = binary_entropy(dataset.loc[:][target_attribute])
    
    best_info_gain = -1
    best_threshold = None
    best_minor_dataset = None
    best_major_dataset = None
    for threshold in thresholds:
        minor_dataset, major_dataset = split_dataset(dataset, split_attribute ,threshold)
        current_info_gain = information_gain_splitted(dataset_len, dataset_entropy, minor_dataset, major_dataset, target_attribute)
        if current_info_gain > best_info_gain:
            best_info_gain = current_info_gain
            best_threshold = threshold
            best_minor_dataset = minor_dataset
            best_major_dataset = major_dataset
    return best_minor_dataset, best_major_dataset, best_threshold, best_info_gain


    
equality_predicate           = lambda y : (lambda x: x == y)
greater_inequality_predicate = lambda y : (lambda x: x >= y)
less_inequality_predicate    = lambda y : (lambda x: x < y )

def __ID3(dataset: pd.DataFrame, target_attribute: str, other_attributes: list[str], domain_values: dict, is_discrete: dict[bool]):
    discrete_attributes = []
    continuous_attributes = []
    for attr in other_attributes:
        if is_discrete[attr]:
            discrete_attributes.append(attr)
        else:
            continuous_attributes.append(attr)
    
    if len(dataset) == 0:
        raise "Empty dataset!"

    if same_value(dataset, target_attribute):
        child_value = list(dataset[:][target_attribute])[0]
        return Tree(value = child_value)

    if other_attributes == None or len(other_attributes) == 0:
        child_value = most_frequent(dataset, target_attribute)
        return Tree(value = child_value)
        
    best_discrete_attr = None
    discrete_information_gain = None
    best_continuous_attr = None
    minor_dataset = None
    major_dataset = None
    best_threshold = None
    continuous_information_gain = None
    best_attr = None
    best_attr_is_discrete = None

    if len(discrete_attributes) > 0:
        best_discrete_attr, discrete_information_gain = best_discrete_attribute(dataset, target_attribute, discrete_attributes)
    if len(continuous_attributes) > 0:
        best_continuous_attr, minor_dataset, major_dataset, best_threshold, continuous_information_gain = best_attribute_continuous(dataset, target_attribute, continuous_attributes)

    if (len(discrete_attributes) > 0 and len(continuous_attributes) > 0):
        if (continuous_information_gain > discrete_information_gain):
            best_attr = best_continuous_attr
            best_attr_is_discrete = False
        else:
            best_attr = best_discrete_attr
            best_attr_is_discrete = True
    elif len(discrete_attributes) > 0:
            best_attr = best_discrete_attr
            best_attr_is_discrete = True
    else:
            best_attr = best_continuous_attr
            best_attr_is_discrete = False
            if best_attr == None:
                child_value = most_frequent(dataset, target_attribute)
                return Tree(value = child_value)

    if best_attr_is_discrete:
        node = Tree(label = best_attr, children = [])
        
        for v_i in domain_values[best_attr]:
            examples_vi = dataset.loc[dataset[best_attr] == v_i]
            new_child = None
            if examples_vi.empty:
                most_frequent_value = most_frequent(dataset, target_attribute)
                new_child = Tree(value= most_frequent_value)
                new_child.filter_predicate = equality_predicate(v_i)
                new_child.description = best_attr + "==" + str(v_i)
                node.add_child(new_child)
            else:
                attributes_without_A = other_attributes.copy()
                attributes_without_A.remove(best_attr)
                new_dataset = examples_vi
                new_child = __ID3(new_dataset, target_attribute, attributes_without_A, domain_values, is_discrete)
                new_child.filter_predicate = equality_predicate(v_i)
                new_child.description = best_attr + "==" + str(v_i)
                node.add_child(new_child)
    else:
        node = Tree(label = best_attr, children = [])

        new_child1 = __ID3(minor_dataset, target_attribute, other_attributes, domain_values, is_discrete)
        new_child1.filter_predicate = less_inequality_predicate(best_threshold)
        new_child1.description = best_attr + "<" + str(best_threshold)
        node.add_child(new_child1)

        new_child2 = __ID3(major_dataset, target_attribute, other_attributes, domain_values, is_discrete)
        new_child2.filter_predicate = greater_inequality_predicate(best_threshold)
        new_child2.description = best_attr + ">=" + str(best_threshold)
        node.add_child(new_child2)

    return node

def ID3(dataset: pd.DataFrame, target_attribute: str, other_attributes: list[str], domain_values: dict, is_discrete: dict[bool], medians, most_common_values) -> TreeWrapper:
    tree = __ID3(dataset, target_attribute, other_attributes, domain_values, is_discrete)
    return TreeWrapper(tree,medians,most_common_values)