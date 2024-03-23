from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# Load data
X_train_all = pd.read_csv('./data/X_train.csv', index_col = 0, header=[0, 1, 2]) # inputs of the training set
y_train_all = pd.read_csv('./data/y_train.csv', index_col = 0).to_numpy() # outputs of the training set
X_test = pd.read_csv('./data/X_test.csv', index_col = 0, header=[0, 1, 2]) # inputs of the test set

# Scale and format the data
X_train_all = X_train_all.apply(lambda x: (np.mean(x)-x)/np.std(x),axis=0)
y_train_all = np.unique(y_train_all, return_inverse=True)[1]


# split training set
X_train, X_test, y_train, y_test = train_test_split(X_train_all, y_train_all, test_size=0.20, random_state=42,stratify=y_train_all)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=42,test_size=0.15,stratify=y_train)


# build network topology
model_ann = keras.Sequential([
    keras.layers.Dense(800,activation='relu',
                       kernel_regularizer=keras.regularizers.l2(0.1)
                       ),
    keras.layers.BatchNormalization(),
    # output layer
    keras.layers.Dense(64, activation='softmax',kernel_regularizer=keras.regularizers.l2(0.05))
])

# compile model
# adam_optimiser = keras.optimizers.Adam(learning_rate=0.001)
# sgd_optimiser = keras.optimizers.SGD(learning_rate=0.001)
model_ann.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model_ann.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=32, epochs=100)

test_loss, test_acc = model_ann.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

#save the model
#model_ann.save("./models/Music_Genre_ANN_on_original")

# load the saved model
model_ann = keras.models.load_model("./models/Music_Genre_ANN_on_original")
test_loss, test_acc = model_ann.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
