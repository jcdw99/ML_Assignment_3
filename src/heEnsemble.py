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

seed = 292
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


    # {'C': 100, 'gamma': 0.001, 'kernel': 'sigmoid', 'shrinking': True} for SVM
    # {'algorithm': 'ball_tree', 'n_neighbors': 3, 'p': 1, 'weights': 'uniform'} for KNN
    # {'bootstrap': True, 'criterion': 'entropy', 'max_depth': 10, 'max_features': 'sqrt', 'min_samples_split': 6}
    # X, Y = loadData('Train')
    # param_grid = {
    #     'max_depth' : [1, 3, 5, 10, None],
    #     'criterion': ['gini', 'entropy', 'log_loss'],
    #     'max_features': ['sqrt', 'log2'],
    #     'min_samples_split' :[2,4,6],
    #     'bootstrap':[True, False]
    # }
    # kfold = model_selection.KFold(n_splits=10, shuffle=True)

    # clf = model_selection.GridSearchCV(
    #         RandomForestClassifier(), param_grid, cv=kfold, verbose=2)                
    # clf.fit(X, Y)
    # print(clf.cv_results_)
    # print(clf.best_params_)

    # exit()
    estimators = [('dt', DecisionTreeClassifier(max_depth=3, criterion='gini', splitter='best', random_state=seed)),
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
    acc = accuracy_score(testy, ypred)

    print(accuracy_score(testy, ypred))
