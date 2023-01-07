import pandas as pd
SEED = 27
target_attribute = "stroke"
other_attributes = ["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]
domain_values_numeric = {"gender": [0, 1], "hypertension": [0, 1], "heart_disease": [0, 1], "ever_married": [0, 1], "work_type": [0,1,2,3,4], "Residence_type": [0, 1], "smoking_status": [0, 1, 2]}
is_discrete = {"gender": True, "age": False, "hypertension": True, "heart_disease": True, "ever_married": True, "work_type": True, "Residence_type": True, "avg_glucose_level": False, "bmi": False, "smoking_status": True}
mappings = {"gender":{"Male": 0, "Female": 1, "Other": pd.NA}, "ever_married":{"No": 0,"Yes": 1}, "work_type":{"Private" : 0, "Self-employed": 1, "Govt_job": 2, "children": 3, "Never_worked": 4}, "Residence_type":{"Urban": 0,"Rural": 1}, "smoking_status":{"formerly smoked": 0,"never smoked": 1, "smokes": 2, "Unknown": pd.NA}, "stroke":{1: True, 0: False}}