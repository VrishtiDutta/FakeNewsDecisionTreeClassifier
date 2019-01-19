import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
from subprocess import check_call
import math


def _preprocess_data():
    """
    Reads clean and fake data, processes it using a vectorizer, and combines the 2 datasets.
    :return: data vectorizer, data matrix, label vector
    """
    with open('clean_fake.txt', 'r') as f:
        fake_data = f.readlines()
        fake_data = [x1.strip() for x1 in fake_data]

    with open('clean_real.txt', 'r') as r:
        real_data = r.readlines()
        real_data = [x2.strip() for x2 in real_data]

    combined_data = real_data + fake_data

    data_vectorizer = TfidfVectorizer(analyzer='word', input='content')
    data_vectorized = data_vectorizer.fit_transform(combined_data).toarray()


    real_labels = np.ones((len(real_data), 1), dtype=int)
    fake_labels = np.zeros((len(fake_data), 1), dtype=int)
    labels = np.vstack((real_labels, fake_labels))

    labeled_data = np.hstack((data_vectorized.data, labels))

    random.shuffle(labeled_data)

    data = labeled_data[:, : data_vectorized.shape[1]].copy()
    labels = labeled_data[:, labeled_data.shape[1] - 1 : ].copy().astype(int)

    return data_vectorizer, data, labels


def _split_data(features, labels):
    """
    Splits combined feature matrix into training, validation and test sets by the ratio of 70:15:15 respectively.
    :param features: data matrix
    :param labels: label vector
    :return training, validation, test data matrices and corresponding labels
    """

    (train_x, train_y) = features[:int((len(features.data) + 1) * .70)].copy(), \
                         labels[:int((len(features.data) + 1) * .70)].copy()
    (validation_x, validation_y) = features[int(len(features.data) * .70 + 1) : int(len(features.data) * .85)].copy(), \
                                   labels[int(len(features.data) * .70 + 1): int(len(features.data) * .85)].copy()
    (test_x, test_y) = features[int(len(features.data) * .85 + 1):].copy(), \
                       labels[int(len(features.data) * .85 + 1):].copy()

    return (train_x, train_y), (validation_x, validation_y), (test_x, test_y)


def load_data():
    """
    Preprocesses data and splits dataset randomly into training, validation, and test sets.
    :return: data vectorizer, training matrix, validation matrix, test matrix with corresponding labels
    """
    (vectorizer, data, labels) = _preprocess_data()
    (train_x, train_y), (validation_x, validation_y), (test_x, test_y) = _split_data(data, labels)
    return vectorizer, (train_x, train_y), (validation_x, validation_y), (test_x, test_y)


def _classify_data(data, labels, max_depths):
    """
    Classifies data based on data, labels, and max_depth hyperparameter
    :param data: training matrix X
    :param labels: training labels y
    :param max_depths: maximum depths' list for training
    :return: list of classifiers trained with max_depth hyperparameters and Information Gain & Gini training criterion
    """
    IG_classifications = []
    Gini_classifications = []

    for i in range(len(max_depths)):
        Gini_clf = tree.DecisionTreeClassifier(max_depth=max_depths[i], random_state=0)
        Gini_classifications.append(Gini_clf.fit(data, labels))

        IG_clf = tree.DecisionTreeClassifier(max_depth=max_depths[i], criterion="entropy", random_state=0)
        IG_classifications.append(IG_clf.fit(data, labels))

    return IG_classifications, Gini_classifications


def _predict(IG_clf, Gini_clf, pred_data):
    """
    Predict values of validation of each model (10 of them)
    :param IG_clf: list of Information Gain based classifiers
    :param Gini_clf: list of Gini criterion based classifiers
    :param pred_data: validation data matrix
    :return: predictions based on validation matrix and provided classifiers
    """
    IG_predictions = []
    Gini_predictions = []

    for i in range(0, len(IG_clf)):
        IG_predictions.append(IG_clf[i].predict(pred_data))
        Gini_predictions.append(Gini_clf[i].predict(pred_data))

    return IG_predictions, Gini_predictions


def _calculate_accuracy(IG_predictions, Gini_predictions, validation, max_depths):
    """
    Calculate accuracy using prediction of validation data
    :param IG_predictions: Information Gain classifiers' predictions
    :param Gini_predictions: Gini trained classifiers' prediction
    :param validation: validation label vector
    :param max_depths: max_depths parameters used for training
    :return: returns list of accuracies based on provided predictions
    """
    IG_accuracies = []
    Gini_accuracies = []
    for i in range(0, len(max_depths)):

        IG_prediction = np.logical_not(np.logical_xor(IG_predictions[i], validation))
        IG_accuracies.append(IG_prediction[:,0].sum()/IG_predictions[i].shape[0])
        print("Information Gain accuracy for max_depth " + str(max_depths[i]) + ": " + str(IG_accuracies[i]))

        Gini_prediction = np.logical_not(np.logical_xor(Gini_predictions[i], validation))
        Gini_accuracies.append(Gini_prediction[:,0].sum() / Gini_predictions[i].shape[0])
        print("Gini accuracy for max_depth " + str(max_depths[i]) + ": " + str(Gini_accuracies[i]) + "\n")

    return IG_accuracies, Gini_accuracies

def select_model():
    """
    Prints accuracies of all the prediction made using DecisionTreeClassifier models
    """
    vectorizer, (train_x, train_y), (validation_x, validation_y), (test_x, test_y) = load_data()

    max_depths = [2, 3, 5, 11, 17]
    (IG_classifiers, Gini_classifiers) = _classify_data(train_x, train_y, max_depths)
    (IG_predictions, Gini_predictions) = _predict(IG_classifiers, Gini_classifiers, validation_x)
    (IG_accuracies, Gini_accuracies) = _calculate_accuracy(IG_predictions, Gini_predictions, validation_y, max_depths)

    with open("tree1.dot", 'w') as f:
        f = tree.export_graphviz(IG_classifiers[2],
                                 out_file=f,
                                 max_depth=2,
                                 impurity=False,
                                 class_names=['fake', 'real'],
                                 rounded=True,
                                 filled=True)

    check_call(['dot', '-Tpng', 'tree1.dot', '-o', 'tree1.png'])


def compute_information_gain(word):
    """
    :param word: feature word which is given to calculate Information Gained value
    :return: Information Gain value of the word chosen based on validation data
    """
    vectorizer, (train_x, train_y), (validation_x, validation_y), (test_x, test_y) = load_data()
    total_count = train_y.shape[0]
    real_count = int(train_y[:].sum())
    fake_count = int(total_count - real_count)

    entropy_Y = -1*(real_count / total_count)*(math.log2(real_count / total_count)) + \
                -1*(fake_count / total_count)*(math.log2(fake_count / total_count))


    word_idx = vectorizer.vocabulary_[word]
    word_column = train_x[:,word_idx].reshape(train_y.shape[0],1)
    word_in_headline = np.count_nonzero(word_column)
    word_in_real_headline = np.count_nonzero(word_column.transpose().dot(train_y))
    idx_headlines_without_word = np.where(word_column == 0.0)
    vec_headlines_without_word = np.zeros(train_y.shape)
    vec_headlines_without_word[idx_headlines_without_word] = 1
    real_headlines_without_word = np.count_nonzero(vec_headlines_without_word.transpose().dot(train_y))


    p_fake_given_word = (word_in_headline - word_in_real_headline)/word_in_headline
    p_real_given_word = word_in_real_headline/word_in_headline
    entropy_Y_given_word_in_headline = -1*p_fake_given_word*math.log2(p_fake_given_word) + \
                                       -1*p_real_given_word*math.log2(p_real_given_word)


    p_fake_given_not_word = (len(idx_headlines_without_word) - real_headlines_without_word) / \
                                                                                        (total_count - word_in_headline)
    p_real_given_not_word = real_headlines_without_word/(total_count - word_in_headline)
    entropy_Y_given_word_not_in_headline = -1*p_fake_given_not_word*math.log2(p_fake_given_not_word) + \
                                           -1*p_real_given_not_word*math.log2(p_real_given_not_word)

    p_word_in_headline = word_in_headline / total_count
    p_word_not_in_headline = (total_count - p_word_in_headline) / total_count

    entropy_Y_given_word = p_word_in_headline * entropy_Y_given_word_in_headline + \
                           p_word_not_in_headline * entropy_Y_given_word_not_in_headline

    information_gain = entropy_Y - entropy_Y_given_word

    print("Information Gain for word \"" + word + "\" is: " + str(information_gain))

    return information_gain


if __name__ == "__main__":

    compute_information_gain("trump")
    # compute_information_gain("hillary")
    # compute_information_gain("debate")
    # compute_information_gain("the")
