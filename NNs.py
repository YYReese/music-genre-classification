from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



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

# Initialize a dictionary to hold the matrices for each feature category
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
X_train, X_test, y_train, y_test = train_test_split(X_train_tensor, y_train_all, test_size=0.20, random_state=42,stratify=y_train_all)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42,test_size=0.15,stratify=y_train)

#model_ann = keras.models.load_model("./models/Music_Genre_ANN_on_tensor")
#test_loss, test_acc = model_ann.evaluate(X_test, y_test, verbose=2)
#print('\nTest accuracy:', test_acc)


# build network topology
model_ann = keras.Sequential([
    keras.layers.Dense(800, activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.1),
                       input_shape=(X_train_tensor.shape[1], X_train_tensor.shape[2])),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    # output layer
    keras.layers.Dense(16, activation='sigmoid')
])


# compile model
# adam_optimiser = keras.optimizers.Adam(learning_rate=0.001)
# sgd_optimiser = keras.optimizers.SGD(learning_rate=0.001)
model_ann.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model_ann.summary())

history = model_ann.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=16, epochs=80)

test_loss, test_acc = model_ann.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

#odel_ann.save("./models/Music_Genre_ANN_on_tensor")
