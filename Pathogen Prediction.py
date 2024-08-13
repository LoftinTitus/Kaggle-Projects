import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename)) # Will print the name of all associated files

df = pd.read_csv('/kaggle/input/bacdive-bacterial-data/bacteria.csv')
df2 = pd.read_csv('/kaggle/input/bacdive-bacterial-data/bacteria2.csv')
df3 = pd.read_csv('/kaggle/input/bacdive-bacterial-data/bacteria3.csv') 
# Three seperate datasets to determine which features correlate strongest

df.head(10)
df.isna().sum()

df.rename(columns={'Anitbiotic Resistance':'Antibiotic Resistance'}) # Typo in one of the columns

df.dtypes

encode = LabelEncoder()
df['Species'] = encode.fit_transform(df['Species'])
df['Genus'] = encode.fit_transform(df['Genus'])
df['pH Range'] = encode.fit_transform(df['pH Range'])
df.dtypes

X = df.drop(columns=['Biosafety Level'])
Y = df['Biosafety Level']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

model_one = RandomForestClassifier(random_state=42)
model_one.fit(X_train, Y_train)

prediction_one = model_one.predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (Y_test != prediction_one).sum()))
precision = precision_score(Y_test, prediction_one, zero_division=1)
print(precision)

df2.head()
df2.isna().sum()

cleaned = df2.dropna(axis=0) # Removing rows with NaN values
cleaned.isna().sum()

cleaned.head()
cleaned.dtypes

cleaned = cleaned.copy()

def averaged_gc(value):
    if '-' in value:
        lower, upper = value.split('-')
        return (float(lower) + float(upper)) / 2
    else:
        return float(value)
#There were ranges in some of the data, this block will take the two numbers in the range and get their average  

cleaned['Gc Content'] = cleaned['Gc Content'].astype(str).apply(averaged_gc)
cleaned.dtypes

cleaned['Species'] = encode.fit_transform(cleaned['Species'])
cleaned['Biosafety Level'] = encode.fit_transform(cleaned['Biosafety Level'])
cleaned['Cell Shape'] = encode.fit_transform(cleaned['Cell Shape'])
cleaned.dtypes

level_counts = cleaned['Biosafety Level'].value_counts()
plt.figure(figsize=(6, 4))
plt.pie(level_counts, labels=level_counts.index, autopct='%1.1f%%', colors=['purple', 'green'])
plt.show() 
# I knew there was a big difference in one of the columns so I wanted to visualize it

X_square = bacteria2.drop(columns=['Biosafety Level'])
Y_square = bacteria2['Biosafety Level']

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X_square, Y_square, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X2_resampled, Y2_resampled = smote.fit_resample(X2_train, Y2_train)
# I wanted to correct the oversampling of the data type previously mentioned

model_two = RandomForestClassifier(random_state=2000)
model_two.fit(X2_resampled, Y2_resampled)

prediction_two = model_two.predict(X2_test)
print(classification_report(Y2_test,prediction_two))

df3.isna().sum()
df3 = df3.dropna(axis=0)
df3.head(5)

df3 = df3.copy()

def averaged(value):
    value = value.strip()
    if '-' in value:
        parts = value.split('-')
        if len(parts) == 2:
            try:
                lower, upper = parts
                return (float(lower) + float(upper)) / 2
            except ValueError:
                return None
    elif value.startswith('>'):
        try:
            return float(value[1:]) 
        except ValueError:
            return None
    elif value.startswith('<'):
        try:
            return float(value[1:])
        except ValueError:
            return None
    else:
        try:
            return float(value)
        except ValueError:
            return None
# Similar to the previous data, this column had ranges and greater/less than statements

df3['Salt Conc.'] = df3['Salt Conc.'].astype(str).apply(averaged)
df3['Temp'] = df3['Temp'].astype(str).apply(averaged)
df3.dtypes

df3['Species'] = encode.fit_transform(df3['Species'])
df3['Biosafety Level'] = encode.fit_transform(df3['Biosafety Level'])
df3['Gram Stain'] = encode.fit_transform(df3['Gram Stain'])
df3.dtypes

X_cubed = df3.drop(columns=['Biosafety Level'])
Y_cubed = df3['Biosafety Level']

X3_train, X3_test, Y3_train, Y3_test = train_test_split(X_cubed, Y_cubed, test_size=0.2, random_state=42)
X3_resampled, Y3_resampled = smote.fit_resample(X3_train, Y3_train)

model_three = RandomForestClassifier(random_state=42)
model_three.fit(X3_resampled, Y3_resampled)

prediction_three = model_three.predict(X3_test)
print(classification_report(Y3_test,prediction_three))
