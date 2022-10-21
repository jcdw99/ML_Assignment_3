import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer


def clean_train_data():
    """ Cleaning the training data involves:
          1)  removing redundant / irrelevant fields
          2)  identifying all missing values 
          3)  imputing missing values
          4)  Normalizing the data set
    """
    # drop Id and Gender fields, irrelevant
    df = pd.read_csv("data/breastCancerTrainRaw.csv", sep=";").drop(['id', 'Gender'], axis=1)

    # replace '?' with NAN
    df = df.replace(['?', 0], np.nan)

    # set malignant/begnign to 1/0
    df['diagnosis'] =  df['diagnosis'] == 'M'
    df['diagnosis'] = df['diagnosis'].astype(int)

    # set type to float64
    df = df.astype(np.float64)

    # grab column fields, the KNN imputer of later drops them..
    cols = df.columns

    imputer = KNNImputer()
    # fit on the dataset
    imputer.fit(df)
    # transform the dataset
    df = imputer.transform(df)

    df = pd.DataFrame(df, columns=cols)
    # Now we will noramlize, but save old diagnosis, we dont want to normalize the labels
    cols = df.columns[1:]
    for col in cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # training descriptions
    df[df.columns].describe().to_csv("data/training_description.csv")

    # write cleaned data to csv
    df.to_csv("data/cleanedTrainData.csv")

def clean_test_data():
    """ Cleaning the teset data involves:
        1) removing redundant / irrelevant fields
        2) identifying all missing values
        3) dropping missing values
        4) Normalize the data set
    """
    # drop Id and Gender fields, irrelevant
    df = pd.read_csv("data/breastCancerTestRaw.csv", sep=";").drop(['id', 'Gender'], axis=1)

    # replace '?' with NAN, and drop
    df = df.replace(['?', 0], np.nan).dropna()

    # set malignant/begnign to 1/0
    df['diagnosis'] =  df['diagnosis'] == 'M'
    df['diagnosis'] = df['diagnosis'].astype(int)

    # set type to float64
    df = df.astype(np.float64)

    # Now we will noramlize, but save old diagnosis, we dont want to normalize the labels
    cols = df.columns[1:]
    for col in cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    # testing descriptions
    df[df.columns].describe().to_csv("data/testing_descriptions.csv")

    # write cleaned csv
    df.to_csv("data/cleanTestData.csv")

if __name__ == "__main__":
    clean_train_data()



