# Import libraries
import joblib
import pandas as pd
import numpy as np

from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


# Load data
X_train_all = pd.read_csv('./data/X_train.csv', index_col = 0, header=[0, 1, 2]) # inputs of the training set
y_train_all = pd.read_csv('./data/y_train.csv', index_col = 0).to_numpy() # outputs of the training set
X_test_all = pd.read_csv('./data/X_test.csv', index_col = 0, header=[0, 1, 2]) # inputs of the test set

# Scale and format the data
X_train_all = X_train_all.apply(lambda x: (np.mean(x)-x)/np.std(x),axis=0)
y_train_all = np.unique(y_train_all, return_inverse=True)[1]
X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_all,
                                                    test_size=0.2, random_state=42,
                                                    stratify=y_train_all)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train,
                                                    test_size=0.2, random_state=42,
                                                    stratify=y_train)




# Base models
model_ann = keras.models.load_model("./models/Music_Genre_ANN_on_tensor")
model_rf = joblib.load('models/Music_Genre_RF')
model_cnn = keras.models.load_model("./models/Music_Genre_CNN_on_tensor")


# Generate predictions from the base models
pred_ann = model_ann.predict(X_valid)
pred_rf = model_rf.predict(X_valid)
pred_cnn = model_cnn.predict(X_valid)

# Create the stacked dataset (concatenate base model predictions)
stacked_X = np.column_stack((pred_ann, pred_rf, pred_cnn))

# Meta model
meta_model = RandomForestClassifier(random_state=42)

# Train the meta model on the stacked dataset
meta_model.fit(stacked_X, y_valid)

# Generate predictions using the meta model
stacked_pred = meta_model.predict(stacked_X)

# Evaluate the performance of the stacked model
accuracy = accuracy_score(y_valid, stacked_pred)


