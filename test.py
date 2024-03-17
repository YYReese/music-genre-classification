from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import accuracy_score

# Load data
X_train_all = pd.read_csv('./data/X_train.csv', index_col = 0, header=[0, 1, 2]) # inputs of the training set
y_train_all = pd.read_csv('./data/y_train.csv', index_col = 0).to_numpy() # outputs of the training set
X_test = pd.read_csv('./data/X_test.csv', index_col = 0, header=[0, 1, 2]) # inputs of the test set

# Scale and format the data
X_train_all = X_train_all.apply(lambda x: (np.mean(x)-x)/np.std(x),axis=0)
y_train_all = np.unique(y_train_all, return_inverse=True)[1]

# make tensors
features_info = {
    'chroma_cens': {'chromas': 12, 'stats': 7},
    'chroma_cqt': {'chromas': 12, 'stats': 7},
    'chroma_stft': {'chromas': 12, 'stats': 7},
    'mfcc': {'coefficients': 20, 'stats': 7},
    'rmse': {'stats': 7},
    'spectral_bandwidth': {'stats': 7},
    'spectral_centroid': {'stats': 7},
    'spectral_contrast': {'bands': 7, 'stats': 7},
    'spectral_rolloff': {'stats': 7},
    'tonnetz': {'features': 6, 'stats': 7},
    'zcr': {'stats': 7}
}

features_matrices = {}

for feature, info in features_info.items():
    # Special handling for features with multiple chromas, bands, or coefficients
    if 'chromas' in info:
        for chroma_num in range(1, info['chromas'] + 1):
            features_matrices[f'{feature}_{chroma_num:02}'] = X_train_all.xs(key=(feature, f'{chroma_num:02}'), axis=1, level=('feature', 'number'))
    elif 'coefficients' in info:
        for coeff_num in range(1, info['coefficients'] + 1):
            features_matrices[f'{feature}_{coeff_num:02}'] = X_train_all.xs(key=(feature, f'{coeff_num:02}'), axis=1, level=('feature', 'number'))
    elif 'bands' in info or 'features' in info:
        # Assuming bands or features follow a similar structure to chromas
        count = info.get('bands', info.get('features', 0))
        for num in range(1, count + 1):
            features_matrices[f'{feature}_{num:02}'] = X_train_all.xs(key=(feature, f'{num:02}'), axis=1, level=('feature', 'number'))
    else:
        # For features without chromas/bands, directly extract all stats
        features_matrices[feature] = X_train_all.filter(like=feature, axis=1)

X_train_tensor = np.zeros((6000,74,7))
i = 0
for key in features_matrices.keys():
  X_train_tensor[:,i,:] = (features_matrices[key]).values
  i += 1
X_train_tensor = (np.array(X_train_tensor))

# split training set
X_train_nn, X_test_nn, y_train, y_test = train_test_split(X_train_tensor, y_train_all,
                                                          test_size=0.20, random_state=42,stratify=y_train_all)

X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_all,
                                                    test_size=0.20, random_state=42,stratify=y_train_all)

import numpy as np


model1 = keras.models.load_model("./models/Music_Genre_CNN_on_tensor")
model2 = keras.models.load_model("./models/Music_Genre_ANN_on_original")
model3 = joblib.load('models/Music_Genre_SVM')
model4 = joblib.load('models/Music_Genre_RF')



test_loss, test_acc = model1.evaluate(X_test_nn, y_test, verbose=2)
print('\nTest accuracy(cnn):', test_acc)

test_loss, test_acc = model2.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy (ann):', test_acc)

print("\n Test accuracy (svm): ", accuracy_score(y_test, model3.predict(X_test)))

# stacking
pred1 = model1.predict(X_train_nn)
pred2 = model2.predict(X_train)
pred3 = model3.predict(X_train)
pred4 = model4.predict(X_train)

pred1_test = model1.predict(X_test_nn)
pred2_test = model2.predict(X_test)
pred3_test = model3.predict(X_test)
pred4_test = model4.predict(X_test)


stacked_X_train = np.column_stack((pred1, pred2,pred3))
stacked_X_test = np.column_stack((pred1_test, pred2_test,pred3_test))


from sklearn.linear_model import LogisticRegression

# Create a Logistic Regression classifier
meta_model = LogisticRegression(random_state=42,max_iter=1000)

# Train the classifier
#meta_model.fit(stacked_X_train, y_train)


#meta_model = RandomForestClassifier(random_state=42,max_depth=42,n_estimators=128)
#meta_model.fit(stacked_X_train, y_train)
#meta_model = KNeighborsClassifier( n_neighbors=4)

#meta_model = SVC(C=6.323280699687165,gamma='auto',kernel='linear')


# Train the classifier
meta_model.fit(stacked_X_train, y_train)

stacked_pred = meta_model.predict(stacked_X_test)
accuracy = accuracy_score(y_test, stacked_pred)

print("stacking accuracy: ", accuracy)


from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import uniform

# Define the parameter grid for random search
param_grid = {
    'C': uniform(loc=0, scale=10),
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf']
}


rf = RandomForestClassifier()

from scipy.stats import randint
# Define the parameter grid for random search
param_grid = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['sqrt', 'log2']
}

# Perform random search with cross-validation
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=10, cv=5)
random_search.fit(stacked_X_train, y_train)

# Get the best Random Forest model
best_rf = random_search.best_estimator_
print(best_rf)

# Train the classifier
meta_model = best_rf
meta_model.fit(stacked_X_train, y_train)

stacked_pred = meta_model.predict(stacked_X_test)
accuracy = accuracy_score(y_test, stacked_pred)

print("stacking accuracy: ", accuracy)
