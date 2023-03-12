import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from DS_project_utils.load_data import *
from DS_project_utils.feature_selection import *


def feature_extraction_3D_visual(df,dataset, target):
    *_ , df = preprocess_data(df)
    df = pd.get_dummies(df, drop_first=True)
    my_dpi=96
    plt.figure(figsize=(480/my_dpi, 480/my_dpi), dpi=my_dpi)

    # Keep the target column appart + make it numeric for coloring
    if dataset == "houses":
        high_price_threshod = df.SalePrice.mean() + df.SalePrice.std()
        df[target] = df.SalePrice.apply(lambda x: 1 if x>high_price_threshod else 0)

    df[target]=pd.Categorical(df[target])
    my_color=df[target].cat.codes
    df = df.drop(target, 1)

    # Run The PCA
    pca = PCA(n_components=3)
    pca.fit(df)
        
    # Store results of PCA in a data frame
    result=pd.DataFrame(pca.transform(df), columns=['PCA%i' % i for i in range(3)], index=df.index)
    # Plot initialisation
    angle = 140
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], c=my_color, cmap="Set2_r", s=60)

    # make simple, bare axis lines through space:
    xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0,0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
    yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0,0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
    zAxisLine = ((0, 0), (0,0), (min(result['PCA2']), max(result['PCA2'])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

    ax.view_init(30,angle)

    # label the axes
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

def feature_extraction(df, target,method,n, is_selection):
    if method == "PCA":
        # extract n features using PCA method
        extraction = PCA(n_components=n, random_state=0)
        X = df.drop([target], axis = 1)
        X_extraction = extraction.fit_transform(X)
        extraction_df = pd.DataFrame(data = X_extraction, columns=['col_PCA%i' % i for i in range(n)])
    elif method == "ICA":
        # extract n features using ICA method
        extraction = FastICA(n_components=n, random_state=0)
        X = df.drop([target], axis = 1)
        X_extraction = extraction.fit_transform(X)
        extraction_df = pd.DataFrame(data = X_extraction, columns=['col_ICA%i' % i for i in range(n)])   
    elif method == "both":
        # extract n features using PCA + n features using ICA
        extraction_ICA = FastICA(n_components=n, random_state=0)
        extraction_PCA = PCA(n_components=n, random_state=0)
        X = df.drop([target], axis = 1)
        ICA_extraction = extraction_ICA.fit_transform(X)
        ICA_df = pd.DataFrame(data = ICA_extraction, columns=['col_ICA%i' % i for i in range(n)])
        PCA_extraction = extraction_PCA.fit_transform(X)
        PCA_df = pd.DataFrame(data = ICA_extraction, columns=['col_ICA%i' % i for i in range(n)]) 
        # merge both features
        extraction_df = pd.merge(PCA_df, ICA_df, left_index=True, right_index=True)
        selected_extraction_df = pd.merge(df[target], extraction_df, left_index=True, right_index=True)
        # using feature selection in order to chooese the best extracted features
        corr_selected_featurs, corr_droped_featurs = correlation_selection(selected_extraction_df,list(extraction_df.columns),target,"spearman",0.2)
        selected_extraction_df = selected_extraction_df.drop(columns=corr_droped_featurs)
        selected_extraction_df = selected_extraction_df.drop([target], axis = 1)
        # merge with the regular features
        df = pd.merge(df, selected_extraction_df, left_index=True, right_index=True)
    if is_selection:
        df = pd.merge(df, extraction_df, left_index=True, right_index=True)
    # if only feature extraction method, return only the extracted features
    else:
        df = pd.merge(df[target], extraction_df, left_index=True, right_index=True)
    return df