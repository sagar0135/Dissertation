

# Importing required libraries
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Loading the dataset
url = "http://kdd.ics.uci.edu/databases/kddcup99/kddcup.data_10_percent.gz"
columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
           "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
           "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
           "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
           "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
           "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
           "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

# Reading the dataset
data = pd.read_csv(url, names=columns)

# Printing the information of the dataset
print(data.info())

# Checking the first few rows of the dataset
print(data.head())

# Printing the Data Shape
print("Data Shape:", data.shape)

# Printing Basic statistics of numerical features
print(data.describe())

# Checking for missing values
print(data.isnull().sum())

# Exploring the distribution of the label column
print("Label Value Counts:")
print(data['label'].value_counts())

# Visualizing the distribution of the label column
plt.figure(figsize=(12, 6))
sns.countplot(x='label', data=data)
plt.title('Distribution of Label')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.show()

# Exploring the distribution of categorical features
CategoricalFeatures = data.select_dtypes(include=['object'])
for column in CategoricalFeatures.columns:
  plt.figure(figsize=(8, 4))
  sns.countplot(x=column, data=data)
  plt.title(f'Distribution of {column}')
  plt.xlabel(column)
  plt.ylabel('Count')
  plt.xticks(rotation=90)
  plt.show()

"""## User Activity Analysis"""

# Calculating total connections initiated by each source IP address.
UserActivity = data.groupby('label')['duration'].count().reset_index()
print(UserActivity)

"""## Service Activity Analysis"""

# Analysing the distribution of connections to different services.
ServiceActivity = data.groupby('service')['label'].count().reset_index()
print(ServiceActivity)

"""## Network Traffic Analysis"""

# Analyzing the distribution of network traffic by protocol type.
ProtocolTraffic = data.groupby('protocol_type')['label'].count().reset_index()
print(ProtocolTraffic)

# Plotting the distribution of network traffic by protocol type.
plt.figure(figsize=(8, 4))
sns.barplot(x='protocol_type', y='label', data=ProtocolTraffic)
plt.title('Distribution of Network Traffic by Protocol Type')
plt.xlabel('Protocol Type')
plt.ylabel('Count')
plt.show()

"""## Correlations Matrix"""

# Exploring the correlation between numerical features
CorrelationMatrix = data.corr(numeric_only=True)
plt.figure(figsize=(16, 12))
sns.heatmap(CorrelationMatrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

"""## Data Processing for modelling"""

# Encoding categorical columns
Labencoder = LabelEncoder()
CategoricalColumns = ['protocol_type', 'service', 'flag', 'label']
for col in CategoricalColumns:
    data[col] = Labencoder.fit_transform(data[col])

# Separating features and labels
X = data.drop('label', axis=1)
y = data['label']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Printing the Training-Testing Shape
print(f"X Training set size: {X_train.shape}")
print(f"X Testing set size: {X_test.shape}")
print(f"y Training target size: {y_train.shape}")
print(f"y Test target size: {y_test.shape}")

"""## Modelling Implementation"""

# Implementing the Random Forest classifier
RandomForestModel = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the model
RandomForestModel.fit(X_train, y_train)

# Predicting on the testing data
RandomForestModely_pred = RandomForestModel.predict(X_test)

# Printing the performance of the model
print(f"Accuracy of the Random Forest Model: {accuracy_score(y_test, RandomForestModely_pred) * 100:.2f} %")
print(f"\nClassification Report: \n{classification_report(y_test, RandomForestModely_pred)}")

# Plotting confusion matrix
RandomForestModelCM = confusion_matrix(y_test, RandomForestModely_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(RandomForestModelCM, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Implementing the Decision Tree Classifier
DecisionTreeModel = DecisionTreeClassifier(random_state=42)

# Fitting a Decision Tree Classifier
DecisionTreeModel.fit(X_train, y_train)

# Making Predictions for the Decision Tree Model
DecisionTreey_pred = DecisionTreeModel.predict(X_test)

# Calculating the peformance of the Decision Tree Model
print(f"Accuracy of the Decision Tree Model: {accuracy_score(y_test, DecisionTreey_pred) * 100:.2f} %")

# Printing the classfication report of the Decision Tree Model
print(f"\nClassification Report: \n{classification_report(y_test, DecisionTreey_pred)}")

# Plotting confusion matrix for Decision Tree Model
DecisionTreeModelCM = confusion_matrix(y_test, DecisionTreey_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(DecisionTreeModelCM, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Decision Tree Model")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Implementing Logistic Regression
LogisticRegressionModel = LogisticRegression(random_state=42)

# Fitting the Logistic Regression Model
LogisticRegressionModel.fit(X_train, y_train)

# Making Predictions
LogisticRegressiony_pred = LogisticRegressionModel.predict(X_test)

# Evaluating the Performance
print(f"Accuracy of the Logistic Regression Model: {accuracy_score(y_test, LogisticRegressiony_pred) * 100:.2f} %")
print(f"\nClassification Report: \n{classification_report(y_test, LogisticRegressiony_pred)}")

# Plotting the confusion Matrix
LogisticRegressionModelCM = confusion_matrix(y_test, LogisticRegressiony_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(LogisticRegressionModelCM, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression Model")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

"""## Models Comparison"""

# Plotting the performance of the models
Models = ['Random Forest', 'Decision Tree', 'Logistic Regression']
Accuracies = [accuracy_score(y_test, RandomForestModely_pred) * 100,
              accuracy_score(y_test, DecisionTreey_pred) * 100,
              accuracy_score(y_test, LogisticRegressiony_pred) * 100
              ]

plt.figure(figsize=(10, 6))
plt.bar(Models, Accuracies, color=['blue', 'green', 'Red'])
plt.title('Model Performance Comparison')
plt.xlabel('Models')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.show()

"""## Anomaly Detection Analysis"""

# Training an Isolation Forest model on the training data
IsolationModel = IsolationForest(contamination='auto', random_state=42)
IsolationModel.fit(X_train)

# Predicting anomalies in the test data
y_predAnomaly = IsolationModel.predict(X_test)

# Identifying anomalous data points
anomalies = X_test[y_predAnomaly == -1]

# Printing Total Anomalies detected
print(f"Number of anomalies detected: {len(anomalies)}")

# Plotting the distribution of anomalies by protocol type (if available)
if 'protocol_type' in anomalies.columns:
  plt.figure(figsize=(8, 4))
  sns.countplot(x='protocol_type', data=anomalies)
  plt.title('Distribution of Anomalies by Protocol Type')
  plt.xlabel('Protocol Type')
  plt.ylabel('Count')
  plt.show()
else:
  print("No 'protocol_type' column found in the anomalies DataFrame.")

"""## Feature Importances Analysis"""

# Getting feature importances from the Random Forest model
FeatureImportances = RandomForestModel.feature_importances_

# Getting feature importances from the Decision Tree model
FeatureImportancesDecisionTree = DecisionTreeModel.feature_importances_

# Creating a DataFrame for Feature Importances
FeatureImportances = pd.DataFrame({'Feature': X.columns, 'Importance': FeatureImportances})

# Creating a DataFrame for Feature Importances
FeatureImportancesDecisionTree = pd.DataFrame({'Feature': X.columns, 'Importance': FeatureImportancesDecisionTree})

# Sorting the features by importance in descending order
FeatureImportances = FeatureImportances.sort_values('Importance', ascending=False)

# Sorting the features by importance in descending order
FeatureImportancesDecisionTree = FeatureImportancesDecisionTree.sort_values('Importance', ascending=False)

# Plotting feature importances
plt.figure(figsize=(10, 10))
sns.barplot(x='Importance', y='Feature', data=FeatureImportances)
plt.title('Feature Importance of Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Plotting feature importances of Decision Tree
plt.figure(figsize=(10, 10))
sns.barplot(x='Importance', y='Feature', data=FeatureImportancesDecisionTree)
plt.title('Feature Importance of Decision Tree')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

"""## Version Control"""

# Performing version control
#!git config --global user.email "sample@git.com"
#!git config --global user.name "Sample"
#!git init
#!git add .
#!git commit -m "Initial commit"