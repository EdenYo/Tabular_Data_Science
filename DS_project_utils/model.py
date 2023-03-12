import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score


def split(df, target):
    # drop_first=True means to get k-1 dummies out of k categorical levels by removing the first level.
    df = pd.get_dummies(df, drop_first=True)
    
    df_train, df_test = train_test_split(df, test_size=0.25, random_state=42)
    
    #separate X from y
    X_train = df_train.drop(target,axis=1)
    X_test = df_test.drop(target,axis=1)

    y_train = df_train[target]
    y_test = df_test[target]
    
    return X_train, X_test, y_train, y_test

def predict(X_train, X_test, y_train, dataset):
    # using LinearRegression for the regression dataset
    if dataset == "houses":
        model = LinearRegression()
    # using LogisticRegression for the classification datasets
    else:
        model = LogisticRegression(random_state=0)
    prediction = model.fit(X_train,y_train).predict(X_test)
    return model, prediction

def evaluate(dataset, X_test, y_test, prediction):
    if dataset == "houses":
        score = r2_score(y_test,prediction)
        # calculate adjusted R-squared
        score = 1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
    else:
        score = accuracy_score(y_test,prediction.round())
    return score


def explainability_score(model, X_test):
    # compute SHAP values
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    scores_list = []
    # going over all the test samples
    for i  in range(shap_values.shape[0]):   
        arr = shap_values[i].values
        extraction_shap = 0
        for j,x in enumerate(arr):
            # we donn't want to count the extracted features in the final score because they can't be explained
            if "col_PCA" in X_test.columns[j] or "col_ICA" in X_test.columns[j]:
                arr[j] = 0
                extraction_shap = extraction_shap + abs(x)
        arr = np.absolute(arr)
        sum_all = np.sum(arr)
        sorted_array = arr[np.argsort(arr)]
        # get top 5 features
        top_5 = sorted_array[-5 : ]
        top_5_sum = np.sum(top_5)
        # calculate score for this specific sample
        score = top_5_sum / (sum_all + extraction_shap)
        scores_list.append(score)
    # calculate the avg score
    avg = sum(scores_list) / len(scores_list)
    if avg !=  avg:
        avg = 0
    return avg


def explainability_score_multiclass(model, X_test):
    # compute SHAP values
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    scores_list = []
    # going over all the test samples
    for i  in range(shap_values.shape[0]):   
        arr = shap_values[i].values
        extraction_shap = 0
        new_arr = []
        # sum each class scores
        for classes in arr:
            classes = np.absolute(classes)
            feature = np.sum(classes)
            new_arr.append(feature)
        for j,x in enumerate(new_arr):
            # we donn't want to count the extracted features in the final score because they can't be explained
            if "col_PCA" in X_test.columns[j] or "col_ICA" in X_test.columns[j]:
                new_arr[j] = 0
                extraction_shap = extraction_shap + abs(x)
        new_arr = np.absolute(new_arr)
        sum_all = np.sum(new_arr)
        sorted_array = new_arr[np.argsort(new_arr)]
        # get top 5 features
        top_5 = sorted_array[-5 : ]
        top_5_sum = np.sum(top_5)
        # calculate score for this specific sample
        score = top_5_sum / (sum_all + extraction_shap)
        scores_list.append(score)
    # calculate the avg score
    avg = sum(scores_list) / len(scores_list)
    if avg !=  avg:
        avg = 0
    return avg