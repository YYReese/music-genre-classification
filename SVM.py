import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

# Load data
X_train_all = pd.read_csv('./data/X_train.csv', index_col = 0, header=[0, 1, 2]) # inputs of the training set
y_train_all = pd.read_csv('./data/y_train.csv', index_col = 0).to_numpy() # outputs of the training set
X_test_all = pd.read_csv('./data/X_test.csv', index_col = 0, header=[0, 1, 2]) # inputs of the test set

# Scale and format the data
X_train_all = X_train_all.apply(lambda x: (np.mean(x)-x)/np.std(x),axis=0)
y_train_all = np.unique(y_train_all, return_inverse=True)[1]

# Split it into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_all,
                                                    test_size=0.2, random_state=42, stratify=y_train_all)


from statistics import mode
from tqdm import tqdm

# Parameter grid
params_grid = [{'kernel': ['rbf'], 'gamma': [0.001, 0.002],'C': [9.3,9.4,9.5]}]

# # Model
svm_model = GridSearchCV(SVC(), params_grid, cv=5)

# Setting the number of the bootstrapped samples
M = 10

n = y_train.shape[0]
m = y_test.shape[0]

y_b_pred = np.zeros((m , M))

for i in tqdm(range(M)):
    # Sampling n rows with replacement from X_train
    index = np.random.choice(len(X_train), size = n, replace = True)

    # Bootstrapped X
    X_bootstrapped = X_train[index]
    y_bootstrapped = y_train[index]

    # Fitting the optimal model with bootstrapped data
    # svm = svm_model.best_estimator_
    svm_model.fit(X_bootstrapped, y_bootstrapped)

    y_b_pred[:, i] = svm_model.predict(X_test)

y_fin = np.zeros(m)

for i in range(m):
    y_fin[i] = mode(y_b_pred[i, :])

count = 0

for i in range(m):
    if y_fin[i] == y_test[i]:
        count += 1
    else:
        count = count

print("bagging svm accuracy: ", count/m)
