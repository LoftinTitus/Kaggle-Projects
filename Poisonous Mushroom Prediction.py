import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import lightgbm as lgb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename)) # Prints 3 files, test, train, submission

df = pd.read_csv('/kaggle/input/playground-series-s4e8/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e8/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s4e8/sample_submission.csv')

original_column_names = df.columns.tolist()
test_columns = test.columns.tolist()
# Creating a list of column names for later

df.head(10)
df.isna().sum()
df.dtypes

encode = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = encode.fit_transform(df[column])
        
for column in test.columns:
    if test[column].dtype == 'object':
        test[column] = encode.fit_transform(test[column])
# For loop so that each column that contains an object will be encoded
df.dtypes

imputer = KNNImputer(n_neighbors = 1776)

df = imputer.fit_transform(df)
test = imputer.fit_transform(test)
mushroom = pd.DataFrame(df, columns=original_column_names)
test_set = pd.DataFrame(test, columns=test_columns)
# Imputing all the missing values in the dataset
# After using the imputer, I had to rename the columns because they were encoded, that's what the list was for

mushroom.isna().sum()
mushroom.head(5)

X_sub = mushroom.drop(columns=['class'])
Y_sub = mushroom['class']

X_test_sub = test_set.copy()

pipeline = make_pipeline(
StandardScaler(),
lgb.LGBMClassifier(n_estimators=1000, num_threads=-1)
) # I created a pipeline to make the code run smoother

pipeline.fit(X_sub, Y_sub)
sub_prediction = pipeline.predict(X_test_sub)

submission = sample.copy()
submission['class'] = sub_prediction

submission.head(10)

submission.to_csv('submission.csv', index=False) # Output my predictions
