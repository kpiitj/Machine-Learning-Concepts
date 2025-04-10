import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

file_path = 'ion_binary_classification.csv'
data = pd.read_csv(file_path)

data_cleaned = data.drop(columns=["Unnamed: 0"])

X = data_cleaned.drop(columns=["Class"])
y = data_cleaned["Class"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


class NaiveBayesFromScratch:
    def __init__(self):
        self.class_priors = {}
        self.feature_stats = defaultdict(dict)
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)

        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_priors[cls] = len(X_cls) / len(X)
            self.feature_stats[cls]['mean'] = np.mean(X_cls, axis=0)
            self.feature_stats[cls]['var'] = np.var(X_cls, axis=0)

    def calculate_gaussian_probability(self, x, mean, var):
        eps = 1e-9
        coeff = 1 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var + eps))
        return coeff * exponent

    def calculate_class_probabilities(self, x):
        probabilities = {}
        for cls in self.classes:
            prior = np.log(self.class_priors[cls])
            likelihood = np.sum(np.log(self.calculate_gaussian_probability(x, self.feature_stats[cls]['mean'],
                                                                           self.feature_stats[cls]['var'])))
            probabilities[cls] = prior + likelihood
        return probabilities

    def predict_single(self, x):
        probabilities = self.calculate_class_probabilities(x)
        return max(probabilities, key=probabilities.get)

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])


def naive_bayes_from_scratch():
    import time
    start_time_scratch = time.time()
    nb_scratch = NaiveBayesFromScratch()
    nb_scratch.fit(X_train.values, y_train)
    y_pred_scratch = nb_scratch.predict(X_test.values)
    end_time_scratch = time.time()
    time_taken_by_custom_naive = end_time_scratch - start_time_scratch

    accuracy_scratch = accuracy_score(y_test, y_pred_scratch)
    precision_scratch = precision_score(y_test, y_pred_scratch)
    recall_scratch = recall_score(y_test, y_pred_scratch)
    f1_scratch = f1_score(y_test, y_pred_scratch)
    conf_matrix_scratch = confusion_matrix(y_test, y_pred_scratch)

    print("Custom Naive Bayes Results:")
    print(f"Accuracy: {accuracy_scratch}")
    print(f"Precision: {precision_scratch}")
    print(f"Recall: {recall_scratch}")
    print(f"F1 Score: {f1_scratch}")
    print(f"Confusion Matrix:\n{conf_matrix_scratch}\n")
    print("Time taken by custom naive bayes algorithm:", time_taken_by_custom_naive)

    # plot the graph
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix_scratch, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix for Custom Naive Bayes")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


def naive_bayes_from_sklearn():
    import time
    start_time_scratch = time.time()
    nb_sklearn = GaussianNB()
    nb_sklearn.fit(X_train, y_train)
    y_pred_sklearn = nb_sklearn.predict(X_test)
    end_time_scratch = time.time()
    time_taken_sklearn = end_time_scratch - start_time_scratch
    accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
    precision_sklearn = precision_score(y_test, y_pred_sklearn)
    recall_sklearn = recall_score(y_test, y_pred_sklearn)
    f1_sklearn = f1_score(y_test, y_pred_sklearn)
    conf_matrix_sklearn = confusion_matrix(y_test, y_pred_sklearn)
    print("Sklearn Naive Bayes Result:")
    print(f"Accuracy: {accuracy_sklearn}")
    print(f"Precision: {precision_sklearn}")
    print(f"Recall: {recall_sklearn}")
    print(f"F1 Score: {f1_sklearn}")
    print(f"Confusion Matrix:\n{conf_matrix_sklearn}")
    print("Time taken by sklearn Algorithm:", time_taken_sklearn)
    # plot the graph

    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix_sklearn, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix for Sklearn")
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()


naive_bayes_from_scratch()
naive_bayes_from_sklearn()
