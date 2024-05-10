import pandas as pd
import numpy as np

def remove(X, y, columns=None):
    if columns is None:
        X_ = X.dropna()
        indices_linhas_removidas = X.index.difference(X_.index)
        y_ = y.drop(indices_linhas_removidas)
        return X_, y_
    else:
        X_ = X.dropna(subset=columns)
        indices_linhas_removidas = X.index.difference(X_.index)
        y_ = y.drop(indices_linhas_removidas)
        return X_, y_

def median(dataset, columns=None):
    if columns is None:
        dataset = dataset.fillna(dataset.median())
    else:
        for column in columns:
            if column in dataset.columns:
                dataset.loc[:, column] = dataset[column].fillna(dataset[column].median())
            else:
                print("Mensagem de erro")
    
    return dataset

def mean(dataset, columns=None):
    if columns is None:
        dataset = dataset.fillna(dataset.mean())
    else:
      for column in columns:
        if column in dataset.columns:
            dataset.loc[:, column] = dataset[column].fillna(dataset[column].mean())
        else:
            print("Mensagem de erro")
    
    return dataset

def mode(dataset, columns=None):
    if columns is None:
        dataset = dataset.fillna(dataset.mode().iloc[0])
    else:
      for column in columns:
        if column in dataset.columns:
            dataset.loc[:, column] = dataset[column].fillna(dataset[column].mode().iloc[0])
        else:
            print("Mensagem de erro")
    
    return dataset

def fixed_value(dataset, value=-1, columns=None):
    if columns is None:
        dataset = dataset.fillna(value)
    else:
      for column in columns:
        if column in dataset.columns:
            dataset.loc[:, column] = dataset[column].fillna(value)
        else:
            print("Mensagem de erro")
    
    return dataset