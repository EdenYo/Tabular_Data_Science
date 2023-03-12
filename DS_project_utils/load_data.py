import pandas as pd

def load_data(path, dataset):
    if dataset == 'breast_cancer':
        df = pd.read_csv(path)
        # replacing to numeric classes
        df["diagnosis"]=df["diagnosis"].replace(['M', 'B'],[1, 0])
        # dealing with the imbalanced data by deleting instances from the over-represented class
        df_1 = df[df["diagnosis"] == 1]
        df_0 = df[df["diagnosis"] == 0]
        df_0 = df_0.sample(n = df_1["diagnosis"].count(), random_state=2)
        df = pd.concat([df_1, df_0], ignore_index=True)  
    elif dataset == "MobilePrice":
        df = pd.read_csv(path)
        # replacing to numeric classes
        df["price_range"]=df["price_range"].replace(['Low Cost','Medium Cost','High Cost','Very High Cost'],[0,1,2,3])
    elif dataset == "mushrooms":
        df = pd.read_csv(path)
        # replacing to numeric classes
        df["class"]=df["class"].replace(['e', 'p'],[1, 0])
    else:
        df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Defining numeric and categorical columns
    numeric_columns = df.dtypes[(df.dtypes=="float64") | (df.dtypes=="int64")].index.tolist()
    very_numerical = [nc for nc in numeric_columns if df[nc].nunique()>20]
    categorical_columns = [c for c in df.columns if c not in very_numerical]

    # Filling Null Values with the column's mean
    na_columns = df[very_numerical].isna().sum()
    na_columns = na_columns[na_columns>0]
    for nc in na_columns.index:
        df[nc].fillna(df[nc].mean(),inplace=True)

    # Dropping and filling NA values for categorical columns:
    nul_cols = df[categorical_columns].isna().sum()/len(df)
    drop_us = nul_cols[nul_cols>0.7]
    df = df.drop(drop_us.index,axis=1)
    categorical_columns = list(set(categorical_columns)-set(drop_us.index))
    # Fill with a new 'na' category:
    df[categorical_columns].fillna('na',inplace=True)
    
    return numeric_columns, very_numerical, categorical_columns, df