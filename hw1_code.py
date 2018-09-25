import sklearn
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, vstack, csr_matrix
from sklearn import tree



def _preprocess_data():
    """
    Reads clean and fake data, processes it using a vectorizer, and combines the 2 datasets.
    :return: fake data vectorizer, real data vectorizer, combined data
    """
    with open('clean_fake.txt', 'r') as f:
        fake_data = f.readlines()
        fake_data = [x1.strip() for x1 in fake_data]

    with open('clean_real.txt', 'r') as r:
        real_data = r.readlines()
        real_data = [x2.strip() for x2 in real_data]

    fake_data_vectorizer = TfidfVectorizer(analyzer='word', input='content')
    fake_data_vectorized = fake_data_vectorizer.fit_transform(fake_data)
    print(fake_data_vectorized.shape)
    
    real_data_vectorizer = TfidfVectorizer(analyzer='word', input='content')
    real_data_vectorized = real_data_vectorizer.fit_transform(real_data)
    print(real_data_vectorized.shape)

    padding_matrix = csr_matrix((real_data_vectorized.shape[0] - fake_data_vectorized.shape[0], fake_data_vectorized.shape[1]))
    padded_fake_data = vstack((fake_data_vectorized, padding_matrix))
    combined_features = hstack([padded_fake_data, real_data_vectorized])
    print(combined_features.shape)

    real_labels = np.ones((real_data_vectorized.shape[1], 1), dtype=int)
    fake_labels = np.zeros((fake_data_vectorized.shape[1], 1), dtype=int)
    labels = np.vstack((real_labels, fake_labels))
    print(labels.shape)

    #v, k = max((v, k) for k, v in real_data_vectorizer.vocabulary_.items())
    #print(real_data_vectorizer.vocabulary_)
    #print(v,k)
    
    return fake_data_vectorizer, real_data_vectorizer, combined_features, labels


def _split_data(features):
    """
    Splits combined feature matrix into training, validation and test sets by the ratio of 70:15:15 respectively.
    :type features: csr_matrix
    :return training matrix, validation matrix, test matrix
    """
    features_mtx = features.data
    random.shuffle(features_mtx)
    print(features_mtx.shape)
    train_data = features.data[:int((len(features.data) + 1) * .70)]  # Splits 70% data to training set
    validation_data = features.data[int(len(features.data) * .70 + 1):
                                    int(len(features.data) * .85)]  # Splits 15% data to validation set
    test_data = features.data[int(len(features.data) * .85 + 1):]  # Splits 15% data to test set

    return train_data, validation_data, test_data


def load_data():
    """
    Preprocesses data and splits dataset randomly into training, validation, and test sets.
    :return: fake data vectorizer, real data vectorizer, training matrix, validation matrix, test matrix
    """
    (fake_vectorizer, real_vectorizer, combined_features, labels) = _preprocess_data()
    (training, validation, test) = _split_data(combined_features)
    return fake_vectorizer, real_vectorizer, training, validation, test, labels


def _classify_data(data, labels):
    """


    :param data:
    :param labels:
    :return:
    """
    # Fit decision tree classifier to x_train, y_train with 2 split criteria and 5 max_depth
    IG_classifications = []
    Gini_classifications = []
    for i in range(2, 7):
        Gini_clf = tree.DecisionTreeClassifier(max_depth=i)
        Gini_classifications.append(Gini_clf.fit(data.toarray(), labels))

        #IG_clf = tree.DecisionTreeClassifier(max_depth=i, criterion="entropy")
        #IG_classifications.append(IG_clf.fit(data, labels))

    return IG_classifications, Gini_classifications


def _predict(classifiers_set1, classifiers_set2, data):
    """
    Predict values of validation of each model (10 of them)
    :param classifiers_set1:
    :param classifiers_set2:
    :param data:
    :return:
    """
    predictions1 = []
    predictions2 = []

    for i in range(0, len(classifiers_set1)):
        predictions1.append(classifiers_set1[i].predict(data))
        predictions2.append(classifiers_set2[i].predict(data))

    return predictions1, predictions2


def _calculate_accuracy(predictions1, predictions2):
    """
    Calculate accuracy using prediction of validation data
    :param pedictions1:
    :param predictions2:
    :return:
    """


def select_model():
    """
    Prints accuracies of all the prediction made using DecisionTreeClassifier models
    """
    (fake_vectorizer, real_vectorizer, training, validation, test, labels) = load_data()

    (IG_classifiers, Gini_classifiers) = _classify_data(training, labels)
    #(IG_predictions, Gini_predictions) = _predict(IG_classifiers, Gini_classifiers, validation)
    #(IG_accuracy, Gini_accuracy) = _calculate_accuracy(IG_predictions, Gini_predictions)


if __name__ == "__main__":
    select_model()
