import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from zlib import crc32
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename)) # Prints out the file name

heart_failure_data = pd.read_csv('/kaggle/input/heart-failure-prediction/heart.csv')

print(heart_failure_data.shape)
print(heart_failure_data.head(5))

heart_failure_data.isna().sum()
heart_failure_data.dtypes.sample(12)

heart_failure_encoded = pd.get_dummies(heart_failure_data) # One-Hot Encodes objects in the Dataset
heart_failure_encoded.dtypes.sample(12)

X = heart_failure_encoded.drop(columns=['HeartDisease'])
y = heart_failure_encoded['HeartDisease'] # Setting target column to Heart Disease


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2529)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print(classification_report(y_test,prediction)) # Report for precision, recall, f1 etc.
print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != prediction).sum()))
