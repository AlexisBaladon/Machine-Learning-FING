import pickle
import pandas as pd
import argparse
import ID3 as ID3
from preprocessing import *
from constants import *
 
parser = argparse.ArgumentParser()
#Dataset balance
parser.add_argument("--unbalanced_dataset", type=int, default=1, required=False, help=f"If this flag is 0, the unbalanced dataset won't be tested. Otherwise, it will")
parser.add_argument("--random_oversampling", type=int, default=0, required=False, help=f"Set this flag to 1 to test with an oversampled training dataset using imblean's RandomOverSampler")
parser.add_argument("--smotenc_oversampling", type=int, default=0, required=False, help=f"Set this flag to 1 to test with an oversampled training dataset using imblean's SMOTENCOverSampler")
parser.add_argument("--undersampling", type=int, default=0, required=False, help=f"If this flag is 0, the undersampled dataset won't be tested. Otherwise, it will")
#Trees
parser.add_argument("--ID3", type=int, default=1, required=False, help=f"If this flag is 0, our ID3 implementation won't be tested. Otherwise, it will")
parser.add_argument("--sklearn_tree", type=int, default=0, required=False, help=f"If this flag is 0, the sklearn implementation won't be tested. Otherwise, it will")
parser.add_argument("--sklearn_tree_type", type=str, default="entropy", required=False, help=f"This flag sets the sklearn's tree algorithm variation ('entropy' or 'gini'")
#Output
parser.add_argument("--output_dir", type=str, default="./", required=False, help=f"If this flag is set the report's output will be stored inside this directory. Otherwise, it will be stored in the same as this file")

args = parser.parse_args()
args = {"unbalanced_dataset": args.unbalanced_dataset,
        "random_oversampling": args.random_oversampling,
        "smotenc_oversampling": args.smotenc_oversampling,
        "undersampling": args.undersampling,
        "ID3": args.ID3,
        "sklearn_tree": args.sklearn_tree,
        "output_dir": args.output_dir,
        "sklearn_tree_type": args.sklearn_tree_type
        }

TREE_FLAGS = ["ID3", "sklearn_tree"]
DATASET_FLAGS = {"unbalanced_dataset": lambda x:x,
                "random_oversampling": get_oversampled_dataset,
                "smotenc_oversampling": get_smote_oversampled_dataset,
                "undersampling": get_undersampled__dataset}
ds = pd.read_csv("./dataset/healthcare-dataset-stroke-data.csv")

train, test, most_common_values, medians = preprocess_dataset(ds)
try:
    with open(f"{args['output_dir']}/output.txt", "w") as external_file:
        for tf in TREE_FLAGS:
            if args[tf]:
                external_file.write(f"\n{tf.upper()}\n")
                for balance_method in DATASET_FLAGS.keys():
                    if args[balance_method]:
                        external_file.write(f"{balance_method.upper()}\n")
                        train_dataset = DATASET_FLAGS[balance_method](train)
                        predictions = None
                        if tf == "sklearn_tree":
                            tree = DecisionTreeClassifier(criterion=args["sklearn_tree_type"],random_state=SEED)
                            tree = tree.fit(train_dataset[:][other_attributes], train_dataset[:][target_attribute])
                            predictions = tree.predict(test[:][other_attributes])
                        else:
                            tree = ID3.ID3(train_dataset, target_attribute, other_attributes, domain_values_numeric, is_discrete, medians, most_common_values)
                            predictions = predict(tree, test)
                        test_values = test[:][target_attribute]
                        report = get_report(test_values, predictions)
                        conf_matrix = get_confusion_matrix(test_values, predictions)
                        external_file.write(f"The report for {tf} is: \n{report}\n")
                        external_file.write(f"The confusion matrix for {tf} is: \n{conf_matrix}\n\n")
except Exception as inst:
    print(inst.args)