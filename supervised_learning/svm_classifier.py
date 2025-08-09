import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
X = X[y != 2]
y = y[y != 2]