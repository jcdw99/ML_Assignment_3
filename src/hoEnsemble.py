import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def loadData(traintest):
    # load in the cleaned CSV file
    df = pd.read_csv(f'../data/cleaned{traintest}Data.csv').drop('Unnamed: 0', axis=1)
    data = df.values
    X = data[:,1:]
    Y = data[:,0]
    return X, Y

def do_basic_classification():
    X, Y = loadData('Train')
    kfold = model_selection.KFold(n_splits=10, shuffle=True)
    cart = DecisionTreeClassifier(criterion='gini', max_depth=None, splitter='random')
    model = BaggingClassifier(base_estimator=cart, n_estimators=50, bootstrap=True, max_samples=0.5, max_features=0.25)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    model.fit(X, Y)
    testX, testY = loadData('Test')
    return model.predict(testX), model.score(testX, testY)

def do_grid_search():
    """
    {'base_estimator__criterion': 'log_loss', 'base_estimator__max_depth': 5, 'base_estimator__splitter': 'random', 'bootstrap': True, 'max_samples': 0.5, 'n_estimators': 50}
    """
    X, Y = loadData('Train')
    param_grid = {
        'base_estimator__max_depth' : [1, 3, 5, 10, None],
        'base_estimator__criterion': ['gini', 'entropy', 'log_loss'],
        'base_estimator__splitter': ['best', 'random'],
        'max_samples' : [0.1, 0.25, 0.5, 0.75, 1],
        'max_features': [0.1, 0.25, 0.5, 0.75, 1],
        'n_estimators': [10, 50, 100],
        'bootstrap': [True, False]
    }
    kfold = model_selection.KFold(n_splits=10, shuffle=True)

    clf = model_selection.GridSearchCV(
        BaggingClassifier(DecisionTreeClassifier()), 
                    param_grid, cv=kfold, verbose=3)                
    clf.fit(X, Y)
    print(clf.cv_results_)
    print(clf.best_params_)
    
if __name__ == "__main__":
    finalconfmat = np.zeros((2,2))
    finalaccs = []
    testX, testY = loadData('Test')

    for i in range(20):
        vec, score = do_basic_classification()
        confmat = confusion_matrix(testY, vec)
        finalconfmat += confmat
        finalaccs.append(score)
    
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
    plt.title("Confusion Matrix Of Homogeneous Ensemble Method")
    plt.show()

