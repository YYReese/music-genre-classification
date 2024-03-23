
## Description
This project aims at predicting the musical genre of a song, based on some pre-computed features. The song genres to be classified are Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop or Rock. The dataset consists of p = 518 pre-computed features extracted from 8000 audio tracks. Each song i is represented by an input vector $x_i = (x_{i1}, . . . , x_{ip})$ where $x_{ij} \in \mathbb{R}$ represents the j’th feature of song i. The features are real-valued and correspond to summary statistics (mean, standard deviation, skewness, kurtosis, median, minimum and maximum) of time series of various music features, such as the chromagram or the Mel-frequency cepstrum. Each song may be of eight different classes: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop or Rock. The objective is to (i) construct a classifier which, based on the features of a song, predicts its genre, and to (ii) estimate its generalisation error under the 0–1 loss.
The dataset is split into a training set of 6000 audio tracks, and a test set of 2000 audio tracks. For the training observations, we have access to both the inputs (X train) and outputs (y train). For the test set, we have only access to the inputs (X test), and the objective is to predict their genre, and to estimates its generalisation error under the 0–1 loss. 

Various baseline classification techniques are considered, including Logistic Regression on Principle Components, Random Forests, Neural Networks and Support Vector Machines for this purpose. Ensemble stacking of several base models is studied. As an extension, we propose to apply bootstrapping to SVM which turns out to improve the testing accuracy. 

## Features
The data consists of a total of 518 inputs. In our scenario, each music time series has been summarized with seven summary statistics in the dataset. A description can be found in the [corresponding table](#summary-of-audio-features).

Moreover, the chart below indicates that each class has a relatively balanced distribution among the different genres of music. This means that there is not a significant imbalance where one class dominates the dataset, which is important for model training as it helps prevent biases towards certain classes during model training. Therefore, there is no need to use techniques like oversampling or undersampling.

### Summary of Audio Features
The following table provides a summary of the audio features used in the dataset:

| Name                | Description                                     | Number of Features |
|---------------------|-------------------------------------------------|--------------------|
| chroma cens         | Chroma Energy Normalized (CENS 12 chroma)       | 7*12               |
| chroma cqt          | Constant-Q chromagram (12 chroma)               | 7*12               |
| chroma stft         | Chromagram (12 chroma)                          | 7*12               |
| mfcc                | Mel-frequency cepstrum (20 coefficients)        | 7*20               |
| rmse                | Root-mean-square                                | 7                  |
| spectral bandwidth | Spectral bandwidth                             | 7                  |
| spectral centroid  | Spectral centroid                              | 7                  |
| spectral contrast  | Spectral contrast (7 frequency bands)           | 7*7                |
| spectral rolloff   | Roll-off frequency                              | 7                  |
| tonnetz             | Tonal centroid features (6 features)            | 7*6                |
| zcr                 | Zero-crossing rate                              | 7                  |

![Training Data](Categories.png)
*Figure: Counts of individual music genres in the training data*




