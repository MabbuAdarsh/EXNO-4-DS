# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from scipy import stats
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/7f1938c1-bb0d-4d74-9e12-ec5c569d7905)

```
df_null_sum=df.isnull().sum()
df_null_sum
```

![image](https://github.com/user-attachments/assets/c9a4e6cb-4d38-4cb1-9939-a4630c3de124)

```
df.dropna()
```

![image](https://github.com/user-attachments/assets/016e2134-5aca-4da3-98b5-a4b702b2bee5)

```
max_vals=np.max(np.abs(df[['Height','Weight']]),axis=0)
max_vals
```

![image](https://github.com/user-attachments/assets/c0d257f1-dfd8-46ee-b625-42c0147e9f5b)

```
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("/content/bmi.csv")
df1.head()
```

![image](https://github.com/user-attachments/assets/31c46fe3-bcb0-4572-b35b-7deb407ba43b)

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```

![image](https://github.com/user-attachments/assets/0f4f3ab6-06d9-4daf-a033-fd2bc194fe8d)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

![image](https://github.com/user-attachments/assets/28fd283c-242d-4d82-b201-b42f8f56387f)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/a7845ea8-da33-404a-8b66-6514e6af3a6a)

```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/d0a9ec6a-24e4-4343-83a5-6fd39de7d931)

```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/409c62c3-fee0-41ff-bd7f-d11d64c69ffc)

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![image](https://github.com/user-attachments/assets/c7ff48b0-3d4a-49a7-ae78-32a82bc4fe76)

```
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![image](https://github.com/user-attachments/assets/3620c3e0-6428-4739-a624-ca622211a2cc)

```
import pandas as pd
import numpy as np
df=pd.read_csv("/content/income(1) (1).csv")
df
```
![image](https://github.com/user-attachments/assets/2f66e97a-b86d-44c7-9ddb-0be4b925721d)

```
df.info()
```
![image](https://github.com/user-attachments/assets/9ab7074d-69e4-431b-af8c-51c1ffad89f1)

```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/9e57c0da-d877-4d32-9466-276f055fdb2a)

```
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/4c4cb8c4-5e3e-4891-b2ba-fd0bdacb2a86)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/171ba190-2173-4f43-877c-821ede706f6d)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![image](https://github.com/user-attachments/assets/295fd4bf-f1a0-4a34-bba3-a0443a417f2b)

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/3c61ec31-b3da-43b8-b80d-30c296dc1f9d)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/9bd5904e-cf80-4b1d-812c-77cb5a5fa8a1)

```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/d0ab0679-1d23-4fc5-82b7-ff005945e7f7)

```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
k_chi2 = 6
selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
X_chi2 = selector_chi2.fit_transform(X, y)
selected_features_chi2 = X.columns[selector_chi2.get_support()]
print("Selected features using chi-square test:")
print(selected_features_chi2)

```
![image](https://github.com/user-attachments/assets/1374e24f-8aec-4427-8cd6-0ee1d030a758)

```
selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
'hoursperweek']
X = df[selected_features]
y = df['SalStat']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
![Screenshot 2025-04-25 133917](https://github.com/user-attachments/assets/7641bad8-9686-456b-9fd0-26054d5435a5)

```
y_pred = rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/2e3049c2-ab86-4055-9de9-d8863cfd30ed)

```
# @title
!pip install skfeature-chappers
```
![image](https://github.com/user-attachments/assets/7195b6fd-aa2e-481d-9943-36181733fcf5)

```
# @title
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# @title
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
# @title
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/631e240c-3d92-459b-a8af-d2e3442cf1a5)

```
# @title
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/b305433a-a736-4f40-8605-5e58385addc2)

```
# @title
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataframe. Assuming your file is named 'income(1) (1).csv' and is in the current directory.
df = pd.read_csv("/content/income(1) (1).csv")

# @title
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
# @title
df[categorical_columns]
# %%
# @title
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
# %%
# @title
X = df.drop(columns=['SalStat'])
y = df['SalStat']
# @title
scores = fisher_score.fisher_score(X.to_numpy(), y.to_numpy())
# @title
ranked_features = np.argsort(scores)[::-1]
# @title
num_top_features = 5
top_features = X.columns[ranked_features[:num_top_features]]
print(f"Top features selected by Fisher Score: {list(top_features)}")
# @title
X_selected = X[top_features]
# @title
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
# @title
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# @title
rf.fit(X_train, y_train)
# @title
y_pred = rf.predict(X_test)
```
![image](https://github.com/user-attachments/assets/76785cce-b342-4bbb-a7a5-aec0ea998d77)

```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
df = pd.read_csv("/content/income(1) (1).csv")
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
X = df.drop(columns=['SalStat'])
y = df['SalStat']


k_anova = 5
selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
X_anova = selector_anova.fit_transform(X, y) # Use X instead of x
selected_features_anova = X.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
![image](https://github.com/user-attachments/assets/1862c5c9-5e52-49a0-9c79-4374990a7091)

```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("/content/income(1) (1).csv") # Assuming this is the path to your data

categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/2d16e005-8e20-4ee5-b0d9-367bb3e98928)

```
import pandas as pd
df = pd.read_csv("/content/income(1) (1).csv")  

categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')

df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)

df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/ec142bfc-5822-48a7-9872-cd556ce07e80)

```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression # Import LogisticRegression

df = pd.read_csv("/content/income(1) (1).csv")  

categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')

df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)

df[categorical_columns]

X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select = 6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
![image](https://github.com/user-attachments/assets/8c610a13-325b-4a19-9eb5-2f7f0cd996c5)

```
selected_features = X.columns[rfe.support_]
print("Selected features using RFE:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/9fa3efc5-487d-4c29-b2f9-090ffbeb0af1)

```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using Fisher Score selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/aa9967f8-fa54-41b2-97eb-c7203dccbc2d)

# RESULT:
  Thus feature scaling and selection is performed.
