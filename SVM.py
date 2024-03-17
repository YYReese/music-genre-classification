from statistics import mode

import joblib
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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


# Setting the number of the bootstrapped samples
M = 200

n = y_train.shape[0]
m = y_test.shape[0]

y_b_pred = np.zeros((m , M))


from sklearn.svm import SVC

# Define the SVM model with the specified hyperparameters
svm_model = SVC(C=9.3, kernel='rbf', gamma=0.001)

# Train the SVM model on your training data
svm_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(svm_model, './models/Music_Genre_SVM')

# Load the saved model
#loaded_model = joblib.load('models/Music_Genre_SVM')

y_pred = svm_model.predict(X_test)

print("Training set score for SVM: %f" % svm_model.score(X_train , y_train))
print("Testing  set score for SVM: %f" % svm_model.score(X_test , y_test ))

#for i in tqdm(range(M)):
    # Sampling n rows with replacement from X_train
#    index = np.random.choice(X_train.shape[0], size = n, replace = True)
    # Bootstrapped X
#    X_bootstrapped = X_train.sample(X_train.shape[0],replace=True)
#    y_bootstrapped = y_train[index]

    # Fitting the optimal model with bootstrapped data
#    svm = SVC(C=9.1, kernel='rbf', gamma=0.001)
#    svm.fit(X_bootstrapped, y_bootstrapped)

#    y_b_pred[:, i] = svm.predict(X_test)

#y_fin = np.zeros(m)

#for i in range(m):
#    y_fin[i] = mode(y_b_pred[i, :])

#count = 0

#for i in range(m):
#    if y_fin[i] == y_test[i]:
#        count += 1
#    else:
#        count = count

#print(count/m)
