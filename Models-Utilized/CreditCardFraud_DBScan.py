#%%Importing Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.cluster import DBSCAN

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

#%% DBScan
# Data Formation
training_df = df_encoded.copy()
training_df = training_df.drop('is_fraud', axis=1)
print('Data Preview:')
print(training_df.head())
scaled_data = StandardScaler().fit_transform(training_df)
scaled_df = pd.DataFrame(scaled_data, columns = training_df.columns)
print('Data Preview:')
print(scaled_df.head())

# Parameter Identification
best_score = -1
best_parameters = {}

sample_size = 50000
if len(scaled_df) > sample_size:
    sample_indices = np.random.choice(scaled_df.shape[0], sample_size, replace=False)
    scaled_df_sample = scaled_df.iloc[sample_indices]
else:
    scaled_df_sample = scaled_df

sub_sample_size = 10000
if len(scaled_df_sample) > sub_sample_size:
    sub_sample_indices = np.random.choice(scaled_df_sample.shape[0], sub_sample_size, replace=False)
    sub_scaled_df_sample = scaled_df_sample.iloc[sub_sample_indices]
else:
    sub_scaled_df_sample = scaled_df_sample

for eps in np.arange(0.5, 1.1, 0.1):
  for min_samples in range(3,8):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(scaled_df_sample)
    labels = dbscan.labels_
    if len(set(labels)) > 1:
        sub_sample_labels = labels[sub_sample_indices]
        score = silhouette_score(sub_scaled_df_sample, sub_sample_labels)
        if score > best_score:
            best_score = score
            best_parameters = {'eps': eps, 'min_samples': min_samples}
    print(f"Completed run for eps: {eps}, min_samples: {min_samples}, Silhouette Score: {score:.4f}")

print(f"Best Score: {best_score}")
print(f"Best Parameters: {best_parameters}")

# Model
chunk_size=200000
num_chunks=len(scaled_df)//chunk_size+(len(scaled_df)%chunk_size>0)
all_labels=np.zeros(len(scaled_df),dtype=int)
for i in range(num_chunks):
    start=i*chunk_size
    end=min((i+1)*chunk_size,len(scaled_df))
    batch_data=scaled_df.iloc[start:end]
    dbscan=DBSCAN(eps=best_parameters['eps'],min_samples=best_parameters['min_samples'])
    batch_labels=dbscan.fit_predict(batch_data)
    all_labels[start:end]=batch_labels
scaled_df['cluster_labels']=all_labels
scaled_df['is_fraud']=df_encoded['is_fraud']
scaled_df['predicted_fraud']=np.where(scaled_df['cluster_labels']==-1,1,0)
print('Data Preview:')
print(scaled_df.head())

# Metrics
report = classification_report(scaled_df['is_fraud'], scaled_df['predicted_fraud'])
print(report)
cm = confusion_matrix(scaled_df['is_fraud'], scaled_df['predicted_fraud'])
plt.figure(figsize=(6,6))
sns.heatmap(cm, fmt='d', cmap='inferno', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show(block=False)

# Plotting Results
TP = (scaled_df['is_fraud'] == 1) & (scaled_df['predicted_fraud'] == 1)
TN = (scaled_df['is_fraud'] == 0) & (scaled_df['predicted_fraud'] == 0)
FP = (scaled_df['is_fraud'] == 0) & (scaled_df['predicted_fraud'] == 1)
FN = (scaled_df['is_fraud'] == 1) & (scaled_df['predicted_fraud'] == 0)
plt.figure(figsize=(10,10))
plt.scatter(scaled_df.loc[TP, scaled_df.columns[0]], scaled_df.loc[TP, scaled_df.columns[1]], c='green', label='True Positives')
plt.scatter(scaled_df.loc[TN, scaled_df.columns[0]], scaled_df.loc[TN, scaled_df.columns[1]], c='red', label='True Negatives')
plt.scatter(scaled_df.loc[FP, scaled_df.columns[0]], scaled_df.loc[FP, scaled_df.columns[1]], c='yellow', label='False Positives')
plt.scatter(scaled_df.loc[FN, scaled_df.columns[0]], scaled_df.loc[FN, scaled_df.columns[1]], c='orange', label='False Negatives')
plt.title('DBSCAN Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show(block=False)