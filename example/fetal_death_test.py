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


#missing = {'mean':['idade', 'mc_get_peso_anterior'], 'fixed-value':{'mc_get_risco_gestacional':-1, 'mc_dae_escolaridade':-1}}
missing = 'median'

models = ['RandomForest', 'AdaBoost', 'DecisionTree']
#models = ['DecisionTree', 'AdaBoost']


results = auto_angels(df, features, target, 
                        test_size=0.2, 
                        missing=missing, 
                        balancing='Hybrid',
                        #hybrid_size=2.0, 
                        models=models, 
                        #feature_selection='SFS', 
                        #feature_selection_models=2, 
                        opt_hyperparam='Grid-search', 
                        ensemble=["stacking", "mean", "major"], 
                        opt_metric='ROC-AUC',
                        n_jobs=-1,
                        #use_threading=True,
                        save_model=True
                        )


print(results)