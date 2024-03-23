from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
X_train_all = pd.read_csv('./data/X_train.csv', index_col = 0, header=[0, 1, 2])
y_train_all = pd.read_csv('./data/y_train.csv', index_col = 0).to_numpy()
X_test = pd.read_csv('./data/X_test.csv', index_col = 0, header=[0, 1, 2])

# Scale and format the data
X_train_all = X_train_all.apply(lambda x: (np.mean(x)-x)/np.std(x),axis=0)
y_train_all = np.unique(y_train_all, return_inverse=True)[1]

# Split it into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_all,
                                                    test_size=0.2, random_state=42, stratify=y_train_all)


# Convert y_train to numerical values
label_encoder = LabelEncoder()
y_train_numerical = label_encoder.fit_transform(y_train)

pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_train)

# Plotting the first two PCA dimensions
plt.figure(figsize=(8, 6))
# Use y_train_numerical for coloring
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train_numerical, cmap='Spectral')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('PCA of Features')
plt.colorbar()
plt.savefig('PCA-features.png', format='png', dpi=300)
plt.show()


# Create a pipeline that standardizes, then applies PCA, and finally fits a logistic regression model
pipeline = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),  # keep 95% of variance
    LogisticRegression()
)

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the validation set
predictions = pipeline.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Predict on the validation set
predictions = pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Generate and print the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Optionally, print a classification report for a detailed performance analysis
class_report = classification_report(y_test, predictions)
print("Classification Report:")
print(class_report)
