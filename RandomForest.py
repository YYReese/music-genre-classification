import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

# Load data
X_train_all = pd.read_csv('./data/X_train.csv', index_col = 0, header=[0, 1, 2]) # inputs of the training set
y_train_all = pd.read_csv('./data/y_train.csv', index_col = 0).to_numpy() # outputs of the training set
X_test = pd.read_csv('./data/X_test.csv', index_col = 0, header=[0, 1, 2]) # inputs of the test set

# Scale and format the data
X_train_all = X_train_all.apply(lambda x: (np.mean(x)-x)/np.std(x),axis=0)
y_train_all = np.unique(y_train_all, return_inverse=True)[1]

# Split it into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_all,
                                                    test_size=0.2, random_state=42, stratify=y_train_all)

model_rf = RandomForestClassifier(random_state=42, n_estimators=300)
model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print('\nTest accuracy:', accuracy)

# Save the trained model
joblib.dump(model_rf, '/Music_Genre_RF')

# Load the saved model
loaded_model = joblib.load('models/Music_Genre_RF')

