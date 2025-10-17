import pandas as pd
pd.set_option("display.max_columns", None)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np

df = pd.read_csv('dns_data2.csv')
class_counts = df['anomaly_label'].value_counts()
print(class_counts)
plt.figure(figsize=(6, 4))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='Set2')
plt.xlabel('Anomaly Label')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()

df['anomaly_label'] = df['anomaly_label'].map({'Normal': 0, 'Anomalous': 1})

categorical_cols = df.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, cbar=True)
plt.title("Feature Correlation with Anomaly Label")
plt.show()

df = df.drop(columns=['src_ip', 'dst_ip', 'query_name', 'src_port', 'dst_port', 'geo_region', 'response_code_ratio'])
x = df.drop(columns=['anomaly_label'])
y = df['anomaly_label']

scaler = MinMaxScaler()
scaler.fit(x.values)
x_scaled = scaler.transform(x.values)
new_x = pd.DataFrame(data=x_scaled, columns=x.columns)
X_train, X_test, y_train, y_test = train_test_split(new_x, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=20, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Anomalous"], yticklabels=["Normal", "Anomalous"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

def introduce_label_noise(y, noise_level):
    # 4% of the labels will be flipped(calculates the number of labels to flip. By multiplying the total number of labels by noise_level)
    n_flips = int(len(y) * noise_level) # This gets the total number of elements (or instances) in the target labels y *  percentage of the labels that will be flipped
    # It randomly selects n_flips (the number of labels to flip) indices from the range 0 to len(y)-1 (i.e., the valid indices of the y array).
    # replace=False ensures that the same index isn't chosen more than once.
    # The result is an array called flip_indices, which contains the indices (positions) in y where the labels should be flipped.
    flip_indices = np.random.choice(len(y), n_flips, replace=False)
    y_noisy = y.copy() # The copy is made to avoid modifying the original y array
    y_noisy[flip_indices] = 1 - y_noisy[flip_indices]
    return y_noisy

new_y = introduce_label_noise(y, noise_level=0.04)
X_trainAfter, X_testAfter, y_trainAfter, y_testAfter = train_test_split(new_x, new_y, test_size=0.2, random_state=42, shuffle=True, stratify=new_y)

param_grid = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 10, 20],
}
# search over a specified parameter grid
# estimator=rf  The model to be tuned. In this case, it is the RandomForestClassifier (rf), which was defined earlier in the code.
# cv The dataset is split into 5 folds, and for each fold, the model is trained on 4 parts and tested on the remaining 1 part. 
# This ensures that the model is validated on different subsets of the data.
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_trainAfter, y_trainAfter)

best_rf = grid_search.best_estimator_

y_pred_best = best_rf.predict(X_testAfter)
accuracy_best = accuracy_score(y_testAfter, y_pred_best)
print(f"Best Model Accuracy: {accuracy_best:.4f}")
print("Best Hyperparameters: ", grid_search.best_params_)
print("Classification Report (Best Model):\n", classification_report(y_testAfter, y_pred_best))

conf_matrix_best = confusion_matrix(y_testAfter, y_pred_best)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix_best, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Anomalous"], yticklabels=["Normal", "Anomalous"])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Best Model)')
plt.show()
