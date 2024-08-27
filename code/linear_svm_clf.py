import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load dataset
df = pd.read_csv('data\\linear_separable_bin_clf.csv',
                 names=['label', 'features_1', 'features_2'])

print(df)