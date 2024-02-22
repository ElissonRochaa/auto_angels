import sys
sys.path.append('../auto_angels')

import pandas as pd 
import numpy as np
from auto_angels import auto_angels

df = pd.read_csv('dataset_full_morte_fetal.csv')

features = ['mc_get_peso_anterior', 'mc_get_risco_gestacional', 'mc_dae_escolaridade', 'has_arterial_hypertension', 'has_diabetes', 
            'has_cirurgia_pelvica', 'has_infeccao_urinaria', 'has_malformacao_familiar', 'has_gemelaridade_familiar', 'sd_quant_gest',
            'sd_quant_aborto', 'sd_quant_partos', 'sd_quant_partos_cesarios', 'idade', 'primeiro_pre_natal', 'time_between_pregnancies']

target = 'target'


missing = {'mean':['idade', 'mc_get_peso_anterior'], 'fixed-value':{'mc_get_risco_gestacional':-1, 'mc_dae_escolaridade':-1}}

models = ['RandomForest', 'AdaBoost', 'GradientBoost', 'lightGBM']


results = auto_angels(df, features, target, 
                        test_size=0.2, 
                        missing=missing, 
                        balancing=True, 
                        models=models, 
                        #feature_selection='SFS', 
                        #feature_selection_models=2, 
                        #opt_hyperparam='Grid-search', 
                        ensemble=["stacking", "mean", "major"], 
                        opt_metric='accuracy'
                        )


# #missing = {'mode':['Altura'], 'mean':['Idade', 'Salario']}
# missing = {'remove':['Altura'], 'mean':['Idade', 'Salario']}
# #missing = 'mode'

# transformation = {
#         'codificar':['Codigo'],
#         'one-hot-encoding':['Categoria'], 
#         'categorizar':{'Idade': [0, 18, 35, 50, 100]}     
#         }

# features = ['Altura', 'Idade', 'Codigo', 'Categoria', 'Salario']
# #target = 'Target'
# target = 'Target' 
# models = ['RandomForest', 'AdaBoost', 'GradientBoost', 'XGBoost', 'DecisionTree']
# #models = ['RandomForest', 'GradientBoost']
# #balancing={0:1.5, 1:1}
# balancing=False
# levels = {
#         'RandomForest': [ # aqui colocamos os valores dos hiperparametros que o grid vai percorrer
#                 {
#                 'n_estimators': [50, 100, 150, 200],
#                 'criterion' : ['entropy', 'gini'],
#                 'max_depth': [None, 1, 3, 5, 7, 9, 11]
#                 }
#         ],
#         'GradientBoost': [ # aqui colocamos os valores dos hiperparametros que o grid vai percorrer
#                 {
#                 'n_estimators': [50,100,150,200],
#                 'learning_rate' : [0.01, 0.1, 0.5, 1],
#                 'loss': ['deviance', 'exponential'],
#                 'max_depth': [None, 1, 3, 5, 7, 9, 11]
#                 }
#         ]
# }

# results = auto_angels(df, features, target, missing=missing, transformation=transformation, balancing=balancing,
#                                                 feature_selection='SFS', feature_selection_models=1, models=models, opt_hyperparam=None,
#                                                 levels=levels)

print(results)