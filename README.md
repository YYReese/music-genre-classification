
## Description
This project aims at predicting the musical genre of a song, based on some pre-computed features. The song genres to be classified are Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop or Rock. The dataset consists of p = 518 pre-computed features extracted from 8000 audio tracks. Each song i is represented by an input vector $x_i = (x_{i1}, . . . , x_{ip})$ where $x_{ij} \in \mathbb{R}$ represents the j’th feature of song i. The features are real-valued and correspond to summary statistics (mean, standard deviation, skewness, kurtosis, median, minimum and maximum) of time series of various music features, such as the chromagram or the Mel-frequency cepstrum. Each song may be of eight different classes: Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop or Rock. The objective is to (i) construct a classifier which, based on the features of a song, predicts its genre, and to (ii) estimate its generalisation error under the 0–1 loss.
The dataset is split into a training set of 6000 audio tracks, and a test set of 2000 audio tracks. For the training observations, we have access to both the inputs (X train) and outputs (y train). For the test set, we have only access to the inputs (X test), and the objective is to predict their genre, and to estimates its generalisation error under the 0–1 loss. 

Various baseline classification techniques are considered, including Logistic Regression on Principle Components, Random Forests, Neural Networks and Support Vector Machines for this purpose. Ensemble stacking of several base models is studied. As an extension, we propose to apply bootstrapping to SVM which turns out to improve the testing accuracy. 

## Features
The data consists of a total of 518 inputs. In our scenario, each music time series has been summarized with seven summary statistics in the dataset. A description can be found in the [corresponding table](#summary-of-audio-features).

Moreover, the chart below indicates that each class has a relatively balanced distribution among the different genres of music. This means that there is not a significant imbalance where one class dominates the dataset, which is important for model training as it helps prevent biases towards certain classes during model training. Therefore, there is no need to use techniques like oversampling or undersampling.

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

![Training Data](/figures/Categories.png)
*Figure: Counts of individual music genres in the training data*

## Testing accuracy report
The performances of models considered for a 20% hold-out set are summarized in **Table 1**. We will use this hold-out set as an estimate of the accuracy for the given test set. Among the models, SVM with bagging was selected as the final model for music genre classification. The predicted test accuracy for SVM with bagging is 0.6192, and the estimate of its generalization error under the 0-1 loss is 0.3808.

Despite having the highest accuracy, bagging SVM tends to be computationally intensive. This is due to two rounds of grid search and fitting an SVM for every bootstrapped sample. LDA, QDA, and logistic regression on PCs took little time to fit but produced poor results. Random forest adaboosting and standard sSVM required slightly more time and had moderate performances. CNNs are the most expensive model among those, with an acceptable accuracy. It is worth noting that although SVM itself is not expensive, applying bagging makes it costly.

From the prediction results **Table 3**, for the validation data, we found that pop, experimental, and instrumental music are the hardest to predict, with estimated accuracies of 0.4177, 0.5145, and 0.5364, respectively. These genres are relatively less ambiguously defined, as some rock music is also considered pop (known as pop-rock nowadays). Furthermore, from **Table 2**, we observe that instrumental music has a precision slightly higher than recall, indicating more ambiguity or overlap with other genres. For experimental music, the precision for this class is lower than recall, suggesting that experimental music has less distinct features or overlaps with other classes. However, there is still potential for improvement. Notably, the instrumental genre has an accuracy of only around 40%, significantly lower than other genres. Developing a model capable of addressing this discrepancy could significantly improve the overall accuracy and reduce the generalization error under the 0-1 loss metric.

**Table 1: Testing accuracy summary**

| Model                         | Testing accuracy |
|-------------------------------|------------------|
| Logistic regression on PCs | 55.1%            |
| Random forest                 | 56.2%            |
| AdaBoosting                   | 52.4%            |
| Feedforward NN                | 59.42%           |
| Stacking (RF, Feedforward, CNN, and SVM) | 61.09% |
| CNN                           | 58.25%           |
| SVM                           | 60.6%            |
| Bagging SVM                   | 61.9%            |

**Table 2: Classification Report**

|        | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| Electronic  | 0.59      | 0.61   | 0.60     | 160     |
| Experimental  | 0.47      | 0.51   | 0.49     | 138     |
| Folk | 0.66      | 0.65   | 0.65     | 152     |
| Hip-Hop | 0.69      | 0.75   | 0.72     | 158     |
|Instrumental | 0.62      | 0.56   | 0.59     | 151     |
| International | 0.72      | 0.70   | 0.71     | 146     |
|  Pop | 0.46      | 0.42   | 0.44     | 158     |
|Rock | 0.64      | 0.66   | 0.65     | 137     |****


**Table 3: Validation Accuracy by Genre**

|    Genre        | Validation accuracy |
|-----------------|---------------------|
| Electronic      | 0.6025              |
| Experimental    | 0.5145              |
| Folk            | 0.6447              |
| Hip-Hop         | 0.7468              |
| Instrumental    | 0.5364              |
| International   | 0.6986              |
| Pop             | 0.4177              |
| Rock            | 0.6569              |
