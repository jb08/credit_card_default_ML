import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import BernoulliRBM
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# build KNeighborsClassifier
def build_KNN_classifier(data_sets, labels):
    classifier = KNeighborsClassifier()
    classifier.fit(data_sets, labels)
    return classifier


# build LogisticRegression Classifier
def build_LR_classifier(data_sets, labels):
    classifier = linear_model.LogisticRegression()
    classifier.fit(data_sets, labels)
    return classifier


# build LinearDiscriminantAnalysis 
def build_DA_classifier(data_sets, labels):
    classifier = discriminant_analysis.LinearDiscriminantAnalysis()
    classifier.fit(data_sets, labels)
    return classifier


# build Neural Network Classifier
def build_NN_classifier(data_sets, labels):
    classifier = BernoulliRBM()
    classifier.fit(data_sets, labels)
    return classifier


# build Decision Tree Classifier
def build_DT_classifier(data_sets, labels):
    classifier = DecisionTreeClassifier()
    classifier.fit(data_sets, labels)
    return classifier


# build Gaussian Naive Bayes Classifier
def build_NB_classifier(data_sets, labels):
    classifier = GaussianNB()
    classifier.fit(data_sets, labels)
    return classifier


#runs the classifier on test data sets 
def predict_test_data(data_sets, classifier):
    return classifier.predict(data_sets)


# measure the performace between predicted and actual
def preformance_measure(predicted, actual):
    preformance = 
    return preformance

