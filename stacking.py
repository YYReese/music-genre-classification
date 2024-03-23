# Import libraries
import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
import joblib



# Data preprocessing

# Load the training data and the test inputs
X_train_all = pd.read_csv('./data/X_train.csv', index_col = 0, header=[0, 1, 2]) # inputs of the training set
y_train_all = pd.read_csv('./data/y_train.csv', index_col = 0).to_numpy() # outputs of the training set
X_test_all = pd.read_csv('./data/X_test.csv', index_col = 0, header=[0, 1, 2]) # inputs of the test set

# scale the features and convert label strings to numeric
X_train_all = X_train_all.apply(lambda x: (x-np.mean(x))/np.std(x), axis=0)
y_train_all = np.unique(y_train_all, return_inverse=True)[1]

# create train and test split for ANN, RF and SVM
X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_all, test_size=0.20, random_state=42,stratify=y_train_all)


# create feature dictionary
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

# Make tensor for cnn
X_train_tensor = np.zeros((6000,74,7))
i = 0
for key in features_matrices.keys():
  X_train_tensor[:,i,:] = (features_matrices[key]).values
  i += 1
X_train_tensor = (np.array(X_train_tensor))

# make training validation and testing data for nn
X_train_nn, X_test_nn, y_train, y_test = train_test_split(X_train_tensor, y_train_all, test_size=0.20, random_state=42,stratify=y_train_all)
X_train_nn, X_valid_nn, y_train, y_valid = train_test_split(X_train_nn, y_train, random_state=42,test_size=0.15,stratify=y_train)


# Base model 1 -- ANN
## Do NOT run if a new training process is not expected!!!
nn_clf = MLPClassifier(hidden_layer_sizes=(800),
                    batch_size=32,alpha = 1,
                    max_iter = 300, random_state=3,
                    solver='lbfgs', activation='relu',
                    validation_fraction=0.15)

nn_clf.fit(X_train,y_train)

print("training accuracy: ", nn_clf.score(X_train, y_train))
print("test accuracy: ", nn_clf.score(X_test, y_test))
#save the model
joblib.dump(nn_clf, "/content/shared/output/trained_models/nn_clf.joblib")

# Base model -- CNN
########! DO NOT RUN IF NOT EXPECTING TRAINING AGAIN !####

# build the CNN
model_cnn = keras.Sequential()

# 1st conv layer
model_cnn.add(keras.layers.Conv1D(128, 7, activation='relu',
                                  input_shape=(X_train_tensor.shape[1], X_train_tensor.shape[2])))
model_cnn.add(keras.layers.BatchNormalization())

# flatten output and feed it into dense layer
model_cnn.add(keras.layers.Flatten())
model_cnn.add(keras.layers.Dense(64, activation='relu',kernel_regularizer=keras.regularizers.l2(0.1)))
model_cnn.add(keras.layers.Dropout(0.3))

# output layer
model_cnn.add(keras.layers.Dense(10, activation='softmax',
                                 kernel_regularizer=keras.regularizers.l2(0.1)))


# compile model
optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model_cnn.compile(optimizer=optimiser,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model_cnn.summary())

history = model_cnn.fit(X_train_nn, y_train, validation_data=(X_valid_nn, y_valid), batch_size=32, epochs=50)

test_loss, test_acc = model_cnn.evaluate(X_test_nn, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# uncomment to save the current model (but a trained model has been already saved to drive)
# model_cnn.save("/content/shared/output/trained_models/Music_Genre_CNN_on_tensor")

# Base model -- SVM


# Create an instance of the SVM classifier
svm_classifier = SVC(kernel='rbf', C=9.3, gamma = 0.003 ,random_state=42)

# Train the SVM classifier
svm_classifier.fit(X_train, y_train)

# Predict on the validation set
y_pred = svm_classifier.predict(X_test)
print(accuracy_score(y_test, svm_classifier.predict(X_test)))

#save the model
joblib.dump(svm_classifier, "./models/svm_clf.joblib")

# Base model -- Random forest
rf_clf = RandomForestClassifier(random_state=42, n_estimators=300 )
rf_clf.fit(X_train, y_train)
# save the model
joblib.dump(rf_clf, "./models/rf_clf.joblib")



# Load saved base models
model1 = keras.models.load_model("/content/shared/input/trained_models/Music_Genre_CNN_on_tensor")
model2 = keras.models.load_model("./models/Music_Genre_ANN_on_tensor")
model3 = joblib.load("./models/rf_clf.joblib")
model4 = joblib.load("./models/rf_clf.joblib")


# Print the accuracy for the base models
test_loss, test_acc = model1.evaluate(X_test_nn, y_test, verbose=2)
print('\nTest accuracy(cnn):', test_acc)
print('\nTest accuracy (ann):', accuracy_score(y_test, model2.predict(X_test)))
print("\n Test accuracy (svm): ", accuracy_score(y_test, model3.predict(X_test)))
print("\n Test accuracy (rf): ", accuracy_score(y_test, model4.predict(X_test)))

# Stacking
pred1 = model1.predict(X_train_nn)
pred2 = model2.predict(X_train)
pred3 = model3.predict(X_train)
pred4 = model4.predict(X_train)

pred1_test = model1.predict(X_test_nn)
pred2_test = model2.predict(X_test)
pred3_test = model3.predict(X_test)
pred4_test = model4.predict(X_test)


stacked_X_train = np.column_stack((pred1, pred2,pred3,pred4))
stacked_X_test = np.column_stack((pred1_test, pred2_test,pred3_test,pred4_test))

meta_model = RandomForestClassifier(random_state=4)
meta_model.fit(stacked_X_train, y_train)

stacked_pred = meta_model.predict(stacked_X_test)
accuracy = accuracy_score(y_test, stacked_pred)

print("\n Stacking accuracy: ", accuracy)
