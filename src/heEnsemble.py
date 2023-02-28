import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

seed = 292
[0.9375, 0.95625, 0.93125, 0.9375, 0.93125, 0.9625, 0.94375, 0.9375, 0.94375, 0.94375, 0.95625, 0.95, 0.94375, 0.95, 0.9625, 0.9375, 0.9625, 0.9625, 0.9625, 0.94375]
def loadData(traintest):
    # load in the cleaned CSV file
    df = pd.read_csv(f'../data/cleaned{traintest}Data.csv').drop('Unnamed: 0', axis=1)
    data = df.values
    X = data[:,1:]
    Y = data[:,0]
    return X, Y

def predictIndividual(ensemble, X, proba=False):
    n_estimators = len(ensemble) 
    n_samples = X.shape[0]
    y = np.zeros((n_samples, n_estimators))
    for i, (model, estimator) in enumerate(ensemble):
        if proba:
            y[:, i] = estimator.predict_proba(X)[:, 1] 
        else:
            y[:, i] = estimator.predict(X) 
    return y

def combine_majority_vote(ensemble, X):
    y_individual = predictIndividual(ensemble, X, proba=False)
    y_final = mode(y_individual, axis=1)
    return y_final[0].reshape(-1, ) 


def combine_using_accuracy_weighting(estimators, X, Xval, yval): 
    n_estimators = len(estimators)
    yval_individual = predictIndividual(estimators, Xval, proba=False) 
    wts = [accuracy_score(yval, yval_individual[:, i]) for i in range(n_estimators)] 
    wts /= np.sum(wts)
    ypred_individual = predictIndividual(estimators, X, proba=False) 
    y_final = np.dot(ypred_individual, wts) 
    return np.round(y_final)

def predict(ensemble, X, y):
    for name, model in ensemble:
        print(f'prediction {name}: {model.score(X, y)}')

    return ensemble

def fit(ensemble, X, y):
    for name, model in ensemble:
        model.fit(X, y)
    return ensemble

if __name__ == "__main__":

    finalconfmat = np.zeros((2,2))
    finalaccs = []
    for trial in range(20):
        seed = int(np.random.rand() * 100000)
        estimators = [
                        ('dt', DecisionTreeClassifier(max_depth=3, criterion='gini', splitter='best', random_state=seed)),
                        ('svm', SVC(C= 100, gamma=0.001, kernel='sigmoid', shrinking=True, random_state=seed)),
                        ('3nn', KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3, p=1, weights='uniform')),
                        ('rf', RandomForestClassifier(bootstrap= True, criterion='entropy', max_depth=10, max_features='sqrt', min_samples_split=6, random_state=seed)),
                        ('gnb', GaussianNB())
                    ]

        X, y = loadData('Train')
        trainX, validateX, trainy, validatey = train_test_split(X, y, test_size=0.33, random_state=seed)
        fit(estimators, trainX, trainy)
        testX, testy = loadData('Test')
        # ypred = combine_majority_vote(estimators, testX)
        ypred = combine_using_accuracy_weighting(estimators, testX, validateX, validatey)
        confmat = confusion_matrix(testy, ypred)
        finalconfmat += confmat
        acc = accuracy_score(testy, ypred)
        finalaccs.append(acc)

    print(finalaccs)
    finalconfmat /= 20
    finalconfmat = pd.DataFrame(finalconfmat, columns=['Benign', 'Malignant'])
    finalconfmat = finalconfmat.transpose()
    finalconfmat.columns
    finalconfmat.columns = ['Benign', 'Malignant']
    finalconfmat = finalconfmat.transpose()
    sns.heatmap(finalconfmat, annot=True, xticklabels=True, fmt='.2f')
    plt.xlabel("Model Classification")
    plt.ylabel("True Classification")
    plt.title("Confusion Matrix Of Heterogeneous Ensemble Method")
    plt.show()
