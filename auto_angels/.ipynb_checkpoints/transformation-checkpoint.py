import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def codificar(dataset, columns, label_encoder_dict=None):

    dataset_codificado = dataset.copy()

    le_dict = label_encoder_dict if label_encoder_dict is not None else {}

    for colunm in columns:
        if colunm not in le_dict:
            le = LabelEncoder()
            dataset_codificado[colunm] = le.fit_transform(dataset[colunm])
            le_dict[colunm] = le
        else:
            le = le_dict[colunm]
            dataset_codificado[colunm] = le.transform(dataset[colunm])

    return dataset_codificado, le_dict

def categorizar(dataset, column, intervalos, rotulos):

    dataset_categorizado = dataset.copy()

    dataset_categorizado[column + '_categoria'] = pd.cut(dataset[column], bins=intervalos, labels=rotulos, include_lowest=True)
    dataset_categorizado = dataset_categorizado.drop(column, axis=1)

    return dataset_categorizado

def one_hot_encoding(dataset, columns, categories=None):
    dataset_encoding = dataset.copy()
    
    if categories is None:
        dataset_encoding = pd.get_dummies(dataset_encoding, columns=columns)
        categories = {col: dataset_encoding.filter(like=col).columns.tolist() for col in columns}
    else:
        for col in columns:
            for category in categories[col]:
                if category not in dataset_encoding.columns:
                    dataset_encoding[category] = 0
        dataset_encoding = pd.get_dummies(dataset_encoding, columns=columns)
    
    return dataset_encoding, categories
