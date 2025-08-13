import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
from ucimlrepo import fetch_ucirepo

heart = fetch_ucirepo(id=45)
X = heart.data.features
y = heart.data.targets.squeeze()