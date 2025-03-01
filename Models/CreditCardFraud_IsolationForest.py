#%%Importing Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, accuracy_score
from sklearn.ensemble import IsolationForest

#%% Data Formation
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
print(df_encoded.nunique())

#%% Isolation Forest
# Data Formation
training_forest_df = df_encoded.copy()
training_forest_df = training_forest_df.drop('is_fraud', axis=1)
print('Data Preview:')
print(training_forest_df.head())
scaled_forest_data = StandardScaler().fit_transform(training_forest_df)
scaled_forest_df = pd.DataFrame(scaled_forest_data, columns = training_forest_df.columns)
print('Data Preview:')
print(scaled_forest_df.head())

# Parameter Identification
def anomaly_score_spread(model, X):
    scores = model.decision_function(X)
    return np.mean(scores) / np.std(scores)
custom_scorer = make_scorer(anomaly_score_spread, greater_is_better=True)
iso = IsolationForest()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_samples': ['auto', 0.5, 1.0],
    'contamination': [0.001, 0.005],
    'random_state': [310]
}
gridsearch_forest = GridSearchCV(iso, param_grid, cv=5, scoring=custom_scorer)
gridsearch_forest.fit(scaled_forest_df)
best_params_forest = gridsearch_forest.best_params_
print(f"Best Parameters: {best_params_forest}")

# Final Model
iso_model = IsolationForest(**best_params_forest)
iso_model.fit(scaled_forest_df)
predictions = iso_model.predict(scaled_forest_df)
scaled_forest_df['is_fraud'] = df_encoded['is_fraud']
for i in range(len(predictions)):
  if predictions[i] == -1:
    predictions[i] = 1
  else:
    predictions[i] = 0
scaled_forest_df['predicted_fraud'] = predictions
print('Data Preview:')
print(scaled_forest_df.head())

# Metrics
report_forest = classification_report(scaled_forest_df['is_fraud'], scaled_forest_df['predicted_fraud'])
print(report_forest)
cm_forest = confusion_matrix(scaled_forest_df['is_fraud'], scaled_forest_df['predicted_fraud'])
plt.figure(figsize=(6,6))
sns.heatmap(cm_forest, fmt='d', cmap='inferno', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show(block=False)

# Plotting Results
TP_forest = (scaled_forest_df['is_fraud'] == 1) & (scaled_forest_df['predicted_fraud'] == 1)
TN_forest = (scaled_forest_df['is_fraud'] == 0) & (scaled_forest_df['predicted_fraud'] == 0)
FP_forest = (scaled_forest_df['is_fraud'] == 0) & (scaled_forest_df['predicted_fraud'] == 1)
FN_forest = (scaled_forest_df['is_fraud'] == 1) & (scaled_forest_df['predicted_fraud'] == 0)
plt.figure(figsize=(10,10))
plt.scatter(scaled_forest_df.loc[TP_forest, scaled_forest_df.columns[0]], scaled_forest_df.loc[TP_forest, scaled_forest_df.columns[1]], c='green', label='True Positives')
plt.scatter(scaled_forest_df.loc[TN_forest , scaled_forest_df.columns[0]], scaled_forest_df.loc[TN_forest , scaled_forest_df.columns[1]], c='red', label='True Negatives')
plt.scatter(scaled_forest_df.loc[FP_forest , scaled_forest_df.columns[0]], scaled_forest_df.loc[FP_forest , scaled_forest_df.columns[1]], c='yellow', label='False Positives')
plt.scatter(scaled_forest_df.loc[FN_forest , scaled_forest_df.columns[0]], scaled_forest_df.loc[FN_forest , scaled_forest_df.columns[1]], c='orange', label='False Negatives')
plt.title('DBSCAN Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show(block=False)