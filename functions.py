import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree


def one_hot_encode(dataframe, column):
    vectorizer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    vectorizer.fit_transform(dataframe[[column]].values)
    #assigning the value obtained above to a new variable
    one_hot_array = vectorizer.fit_transform(dataframe[[column]].values)
    #obtaining the column names
    column_names = [f'{column}_{cat}' for cat in vectorizer.categories_[0]]
    #converting it to a dataframe
    one_hot_data = pd.DataFrame(one_hot_array, columns=column_names)
    #concatting this new dataframe to the existing dataframe
    dataframe = pd.concat([dataframe, one_hot_data], axis = 1)
    dataframe = dataframe.drop([column], axis = 1)
    return dataframe

def preprocess(path):
    # read csv
    dataframe = pd.read_csv(path)
    # subset dataframe
    #subset = dataframe.loc[((dataframe['impervious_1'] > 0) & (dataframe['impervious_1'] <= 100))]
    # reset index
    subset = dataframe
    subset = subset.reset_index(drop=True)
    # fill invalid values
    subset['dem_1'] = subset['dem_1'].replace(-9999, subset['dem_1'].mean())
    subset['aspect_1'] = subset['aspect_1'].replace(255, 18)
    subset['wetlands_1'] = subset['wetlands_1'].replace(-1, 0)
    subset['posidex_1'] = subset['posidex_1'].replace(-1, 0)
    # one hot encode categorical features
    subset = one_hot_encode(subset, "aspect_1")
    subset = one_hot_encode(subset, "wetlands_1")
    # return transformed dataframe
    return subset

def binary_classification_preprocess(dataframe):
    # create target 'class' column
    dataframe['class'] = dataframe.apply(lambda x: 0 if x['impervious_1'] == 0 else 1, axis=1)
    # reset index
    dataframe = dataframe.reset_index(drop=True)
    # fill invalid values
    dataframe['dem_1'] = dataframe['dem_1'].replace(-9999, dataframe['dem_1'].mean())
    dataframe['aspect_1'] = dataframe['aspect_1'].replace(255, 18)
    dataframe['wetlands_1'] = dataframe['wetlands_1'].replace(-1, 0)
    # one hot encode categorical features
    dataframe = one_hot_encode(dataframe, "aspect_1")
    dataframe = one_hot_encode(dataframe, "wetlands_1")
    return dataframe

def linear_regression(path):
    dataset = preprocess(path)
    # define target and feature variables
    target = 'impervious_1'
    features = [
        'landsat_1', 'landsat_2', 'landsat_3', 'landsat_4', 'landsat_5', 'landsat_6',  
        'aspect_1_0', 'aspect_1_1', 'aspect_1_2', 'aspect_1_3', 'aspect_1_4',
        'aspect_1_5', 'aspect_1_6', 'aspect_1_7', 'aspect_1_8', 'aspect_1_9', 
        'aspect_1_10', 'aspect_1_11', 'aspect_1_12', 'aspect_1_13', 'aspect_1_14',
        'aspect_1_15', 'aspect_1_16', 'aspect_1_17','aspect_1_18', 
        'wetlands_1_0', 'wetlands_1_2', 'wetlands_1_3', 'wetlands_1_4', 
        'wetlands_1_5', 'wetlands_1_6', 'wetlands_1_7', 'wetlands_1_8',
        'dem_1', 'posidex_1'
    ]
    X = dataset[features]
    y = dataset[target]
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def logistic_regression(path):
    dataframe = pd.read_csv(path)
    dataset = binary_classification_preprocess(dataframe)
    target = 'class'
    features = [
        'landsat_1', 'landsat_2', 'landsat_3', 'landsat_4', 'landsat_5', 'landsat_6']
    X = dataset[features]
    y = dataset[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    model = LogisticRegression(max_iter = 1000)
    model.fit(X_train, y_train)
    return model

def multiclass_classification_preprocess(path):
    # read csv
    dataframe = pd.read_csv(path)
    # subset dataframe
    subset = dataframe.loc[((dataframe['landcover_1'] > 0) & (dataframe['impervious_1'] <= 100))]
    # reset index
    subset = dataframe
    subset = subset.reset_index(drop=True)
    # fill invalid values
    subset['dem_1'] = subset['dem_1'].replace(-9999, subset['dem_1'].mean())
    subset['aspect_1'] = subset['aspect_1'].replace(255, 18)
    subset['wetlands_1'] = subset['wetlands_1'].replace(-1, 0)
    subset['posidex_1'] = subset['posidex_1'].replace(-1, 0)
    # one hot encode categorical features
    subset = one_hot_encode(subset, "aspect_1")
    subset = one_hot_encode(subset, "wetlands_1")
    # return transformed dataframe
    return subset