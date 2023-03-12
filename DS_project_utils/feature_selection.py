import numpy as np
import pandas as pd
from scipy import stats

def correlation_selection(df,fetures, target_feature, method, feture_selection_per):
    fetures = fetures + [target_feature]
    df_corr = df[fetures].corr(method=method).loc[[target_feature]]
    cor_list = df_corr.values.tolist()[0]
    feature_name = df[fetures].columns.tolist()
    # calculate the number of features to keep
    num_feats = round(len(fetures) * feture_selection_per)
    selected_featurs = df[fetures].iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    droped_featurs = [i for i in fetures if i not in selected_featurs]
    if target_feature in droped_featurs:
        droped_featurs.remove(target_feature)
    return selected_featurs, droped_featurs

def chi_squared_selection(df,dataset,fetures, target_feature, feture_selection_per):
    feature_name = df[fetures].columns.tolist()
    chi2_list = []
    # in the case of numeric target feature, create a binary feature to compare to
    if dataset == "houses":
        target_feature_threshold = df[target_feature].mean() + df[target_feature].std()
        target_feature_binary = df[target_feature].apply(lambda x: 'high' if x>target_feature_threshold else 'low')
    else: 
        target_feature_binary = df[target_feature]
    for feature in fetures:
        contingency_table = pd.crosstab(df[feature], target_feature_binary)
        c, p, dof, expected = stats.chi2_contingency(contingency_table)
        chi2_list.append(c)
    # calculate the number of features to keep
    num_feats = round(len(fetures) * feture_selection_per)
    selected_featurs = df[fetures].iloc[:,np.argsort(np.abs(chi2_list))[-num_feats:]].columns.tolist()
    droped_featurs = [i for i in fetures if i not in selected_featurs]
    if target_feature in droped_featurs:
        droped_featurs.remove(target_feature)
    return selected_featurs, droped_featurs
