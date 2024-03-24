from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for handling the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file selected')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')

    try:
        size = float(request.form['size'])  # Get the input size from the form
        price = predict_price(size, file)
        return render_template('result.html', size=size, price=price)
    except ValueError:
        return render_template('index.html', error='Invalid size value')

def predict_price(file):

    model = keras.models.load_model("./models/Music_Genre_CNN_on_tensor")

    # Process the uploaded file
    data = pd.read_csv(file, index_col = 0, header=[0, 1, 2])
    X_test = data.apply(lambda x: (x - np.mean(x)) / np.std(x), axis=0)

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
                features_matrices[f'{feature}_{chroma_num:02}'] = X_test.xs(key=(feature, f'{chroma_num:02}'),
                                                                                 axis=1, level=('feature', 'number'))
        elif 'coefficients' in info:
            for coeff_num in range(1, info['coefficients'] + 1):
                features_matrices[f'{feature}_{coeff_num:02}'] = X_test.xs(key=(feature, f'{coeff_num:02}'),
                                                                                axis=1, level=('feature', 'number'))
        elif 'bands' in info or 'features' in info:
            # Assuming bands or features follow a similar structure to chromas
            count = info.get('bands', info.get('features', 0))
            for num in range(1, count + 1):
                features_matrices[f'{feature}_{num:02}'] = X_test.xs(key=(feature, f'{num:02}'), axis=1,
                                                                          level=('feature', 'number'))
        else:
            # For features without chromas/bands, directly extract all stats
            features_matrices[feature] = X_test.filter(like=feature, axis=1)

    # Make tensor
    X_train_tensor = np.zeros((6000, 74, 7))
    i = 0
    for key in features_matrices.keys():
        X_train_tensor[:, i, :] = (features_matrices[key]).values
        i += 1
    X_test_tensor = (np.array(X_train_tensor))

    predicted_labels = model.predict(X_test_tensor)

    # Return the predicted price for the given size
    return predicted_labels

if __name__ == '__main__':
    app.run()
