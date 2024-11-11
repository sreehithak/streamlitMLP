import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def prepare_data(data):
    target = 'G3'

    #separate features and target var
    
    #dropped = data[target].values()
    X = data.drop(columns=[target])
    y = data[target]
    
    X = pd.get_dummies(X, drop_first=True)

    #print(X.shape)
    #print(y.shape)
    
    return X, y

def feature_selection(X, y):
    selector = SelectKBest(f_regression, k=10)
    X_selected = selector.fit_transform(X, y)
    return X_selected

def normalization(X):
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized

def pca(data):
    pca = PCA(n_components=0.95)
    principal_components = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.show()

    column_names = []


    for i in range(principal_components.shape[1]):
        column_name = f'PC{i+1}'
        column_names.append(column_name)


    principal_df = pd.DataFrame(
        principal_components,
        columns=column_names
    )

    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    column_names = []

    for i in range(len(pca.explained_variance_)):
        column_name = f'PC{i+1}'
        column_names.append(column_name)


    #loading_df = pd.DataFrame(
        #loadings,
        #columns=column_names,
        #index=data.columns
    #)

    return principal_df


