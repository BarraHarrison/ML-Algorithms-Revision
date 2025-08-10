import pandas as pd 
import numpy as np 
import matplotlib as plt 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score

np.random.seed(42)
data_size = 500

data = pd.DataFrame({
    "age": np.random.randint(30, 80, data_size),
    "cholesterol": np.random.randint(150, 300, data_size),
    "blood_pressure": np.random.randint(90, 180, data_size),
    "smoker": np.random.choice([0, 1], size=data_size),
    "diabetes": np.random.choice([0, 1], size=data_size)
})

data["risk"] = ((data["age"] > 55) &
                (data["cholesterol"] > 240) &
                (data["blood_pressure"] > 140) |
                (data["smoker"] == 1) & (data["diabetes"] == 1)).astype(int)

