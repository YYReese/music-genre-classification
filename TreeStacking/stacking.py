# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold



# Load data
X_train_all = pd.read_csv('../data/X_train.csv', index_col = 0, header=[0, 1, 2]) # inputs of the training set
y_train_all = pd.read_csv('../data/y_train.csv', index_col = 0).to_numpy() # outputs of the training set
X_test = pd.read_csv('../data/X_test.csv', index_col = 0, header=[0, 1, 2]) # inputs of the test set

# Scale and format the data
X_train_all = X_train_all.apply(lambda x: (np.mean(x)-x)/np.std(x),axis=0)
y_train_all = np.unique(y_train_all, return_inverse=True)[1]
X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_all, stratify=y_train_all,
                                                    test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,stratify=y_train,
                                                      test_size=0.2, random_state=42)

# Base models
model1 = RandomForestClassifier(n_estimators =300, random_state=42)
model2 = AdaBoostClassifier(n_estimators =300,random_state=42)
model3 = GradientBoostingClassifier(n_estimators =300,random_state=42)

# Train base models on the training data
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)
model3.fit(X_train, y_train)

# Generate predictions from the base models
pred1 = model1.predict(X_valid)
pred2 = model2.predict(X_valid)
pred3 = model3.predict(X_valid)

# Create the stacked dataset (concatenate base model predictions)
stacked_X = np.column_stack((pred1, pred2,pred3))

# Meta model
meta_model = RandomForestClassifier(random_state=42)
# Train the meta model on the stacked dataset
meta_model.fit(stacked_X, y_valid)

# Generate predictions using the meta model
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)
stacked_X = np.column_stack((pred1, pred2,pred3))

stacked_pred = meta_model.predict(stacked_X)

# Evaluate the performance of the stacked model
accuracy = accuracy_score(y_test, stacked_pred)
print("accuray = ", accuracy)
