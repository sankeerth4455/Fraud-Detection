# Importing Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

#### Data Formation
# Loading Data
df = pd.read_csv("C:/Users/sanke/Desktop/Personal Projects/Credit Card Fraud/fraudTest.csv")
df.head(10)

# Data Description
print(f"The columns in this dataset are: {df.columns}")
print(f"The shape of this dataset is: {df.shape}")
print("Number of unique values in each column")
df.nunique()

# Filtering Columns
df_filter = df[['cc_num','amt', 'zip', 'city_pop', 'category','gender', 'is_fraud']]
print('Data Preview:')
print(df_filter.head())
print(f"The columns in this dataset are: {df_filter.columns}")
print(f"The shape of this dataset is: {df_filter.shape}")
print("Number of unique values in each column")
df_filter.nunique()

# Encoding Columns
df_encoded = df_filter.copy()
df_encoded['category'] = LabelEncoder().fit_transform(df_encoded['category'])
df_encoded = pd.get_dummies(df_encoded, columns=['gender'])
df_encoded.rename(columns={'gender_F':'female', 'gender_M':'male'}, inplace=True)
df_encoded['male'] = df_encoded['male'].astype(int)
df_encoded['female'] = df_encoded['female'].astype(int)
df_encoded = df_encoded[['cc_num', 'amt','zip','city_pop','category','male','female','is_fraud']]
print('Data Preview:')
print(df_encoded.head())
print(f"The columns in this dataset are: {df_encoded.columns}")
print(f"The shape of this dataset is: {df_encoded.shape}")
print("Number of unique values in each column")
df_encoded.nunique()

#### Scaling Data
final_sl_data = df_encoded.copy()
final_sl_data = final_sl_data.drop('is_fraud', axis=1)
print('Data Preview:')
print(final_sl_data.head())
scaled_final_sl_data = StandardScaler().fit_transform(final_sl_data)
scaled_final_sl_data = pd.DataFrame(scaled_final_sl_data, columns = final_sl_data.columns, index=df_encoded.index)
scaled_final_sl_data['is_fraud'] = df_encoded['is_fraud']
print('Data Preview:')
print(scaled_final_sl_data.head())

# Splitting Data
train_df,test_df = train_test_split(scaled_final_sl_data, test_size = 0.2, stratify = scaled_final_sl_data['is_fraud'], random_state = 310)
# Training Data
print('Data Preview:')
print(train_df.head())
print(f"The columns in this dataset are: {train_df.columns}")
print(f"The shape of this dataset is: {train_df.shape}")
print("Number of unique values in each column")
print(train_df.nunique())
print('Distribution of target:')
print(train_df['is_fraud'].value_counts())
# Testing Data
print('Data Preview:')
print(test_df.head())
print(f"The columns in this dataset are: {test_df.columns}")
print(f"The shape of this dataset is: {test_df.shape}")
print("Number of unique values in each column")
print(test_df.nunique())
print('Distribution of target:')
print(test_df['is_fraud'].value_counts())

# Implementing SMOTE
train_smote_df = train_df.copy()
smote = SMOTE(sampling_strategy='auto', random_state=310)
X_smote, y_smote = smote.fit_resample(train_smote_df.drop('is_fraud', axis=1), train_smote_df['is_fraud'])
train_smote_df_resampled = pd.DataFrame(X_smote, columns=train_smote_df.drop('is_fraud', axis=1).columns)
train_smote_df_resampled['is_fraud'] = y_smote
print('Data Preview:')
print(train_smote_df_resampled.head())
print(f"The columns in this dataset are: {train_smote_df_resampled.columns}")
print(f"The shape of this dataset is: {train_smote_df_resampled.shape}")
print("Number of unique values in each column")
print(train_smote_df_resampled.nunique())
print('Distribution of target:')
print(train_smote_df_resampled['is_fraud'].value_counts())

# Training and Testing Data for model
x_train_lr = train_smote_df_resampled.drop('is_fraud', axis=1)
y_train_lr = train_smote_df_resampled['is_fraud']
x_test_lr = test_df.drop('is_fraud', axis=1)
y_test_lr = test_df['is_fraud']

# LR Model
lr_model = LogisticRegression()
param_grid = {
    'C': [0.01, 0.1, 1],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 250, 500],
    'random_state': [310]
    }
gridsearch = GridSearchCV(lr_model, param_grid, cv=3, scoring='f1_weighted')
gridsearch.fit(x_train_lr, y_train_lr)
best_model = gridsearch.best_estimator_
print(best_model)
pred_lr = best_model.predict(x_test_lr)

# Metrics
report = classification_report(y_test_lr, pred_lr)
print(report)
cm = confusion_matrix(y_test_lr, pred_lr)
print(cm)
plt.figure(figsize=(6,6))
sns.heatmap(cm, fmt='d', cmap='inferno', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()