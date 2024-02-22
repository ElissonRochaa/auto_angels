import sys
sys.path.append('../auto_angels')

import pandas as pd 
import numpy as np
from auto_angels import auto_angels

# data = {'Nome': ['Alice', 'Bob', 'Charlie', None, 'David'],
#         'Altura': [158, 175, 184, None, 184],
#         'Idade': [25, 30, None, 30, 40],
#         'Codigo': ['A', 'B', 'C', 'A', 'B'],
#         'Categoria': [1, 2, 3, 1, 3],
#         'Salario': [55000, 60000, 70000, None, 60000],
#         'Target': [0, 1, 0, 1, 0]
#         }

data = {
    'Nome': ['Alice', 'Bob', 'Charlie', None, 'David'] * 10,
    'Altura': [158, 175, 184, None, 184] * 10,
    'Idade': [25, 30, None, 30, 40] * 10,
    'Codigo': ['A', 'B', 'C', 'A', 'B'] * 10,
    'Categoria': [1, 2, 3, 1, 3] * 10,
    'Salario': [55000, 60000, 70000, None, 60000] * 10,
    'Target': [0, 1, 0, 1, 0] * 10
}

df = pd.DataFrame(data)

#missing = {'mode':['Altura'], 'mean':['Idade', 'Salario']}
missing = {'remove':['Altura'], 'mean':['Idade', 'Salario']}
#missing = 'mode'

transformation = {
        'codificar':['Codigo'],
        'one-hot-encoding':['Categoria'], 
        'categorizar':{'Idade': [0, 18, 35, 50, 100]}     
        }

features = ['Altura', 'Idade', 'Codigo', 'Categoria', 'Salario']
#target = 'Target'
target = 'Target' 
models = ['RandomForest', 'AdaBoost', 'GradientBoost', 'XGBoost', 'DecisionTree']
#models = ['RandomForest', 'GradientBoost']
#balancing={0:1.5, 1:1}
balancing=False
levels = {
        'RandomForest': [ # aqui colocamos os valores dos hiperparametros que o grid vai percorrer
                {
                'n_estimators': [50, 100, 150, 200],
                'criterion' : ['entropy', 'gini'],
                'max_depth': [None, 1, 3, 5, 7, 9, 11]
                }
        ],
        'GradientBoost': [ # aqui colocamos os valores dos hiperparametros que o grid vai percorrer
                {
                'n_estimators': [50,100,150,200],
                'learning_rate' : [0.01, 0.1, 0.5, 1],
                'loss': ['deviance', 'exponential'],
                'max_depth': [None, 1, 3, 5, 7, 9, 11]
                }
        ]
}

results = auto_angels(df, features, target, missing=missing, transformation=transformation, balancing=balancing,
                                                feature_selection='SFS', feature_selection_models=1, models=models, opt_hyperparam=None,
                                                levels=levels)

print(results)