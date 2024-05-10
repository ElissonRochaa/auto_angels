import sys
sys.path.append('../auto_angels')

import pandas as pd 
import numpy as np
from auto_angels import auto_angels

df = pd.read_csv('dataset_full_morte_fetal_complementary.csv', low_memory=False)

#print(df['resultados'].value_counts(dropna=False))

df_filtro = df[df['resultados'].isin([0, 1])]

#print(df_filtro['resultados'].value_counts(dropna=False))

features = ['mc_get_peso_anterior', 'mc_get_risco_gestacional', 'mc_dae_escolaridade', 'has_arterial_hypertension', 'has_diabetes', 
            'has_cirurgia_pelvica', 'has_infeccao_urinaria', 'has_malformacao_familiar', 'has_gemelaridade_familiar', 'sd_quant_gest',
            'sd_quant_aborto', 'sd_quant_partos', 'sd_quant_partos_cesarios', 'idade', 'primeiro_pre_natal', 'time_between_pregnancies', 'resultados']

target = 'target'

missing = {'mean':['idade', 'mc_get_peso_anterior'], 'fixed-value':{'mc_get_risco_gestacional':-1, 'mc_dae_escolaridade':-1}}

#models = ['RandomForest', 'AdaBoost', 'GradientBoost', 'lightGBM']
models = ['RandomForest', 'AdaBoost', 'GradientBoost']


results = auto_angels(df_filtro, features, target, 
                        test_size=0.2, 
                        missing=missing, 
                        #balancing='Hybrid', 
                        models=models, 
                        #feature_selection='SFS', 
                        #feature_selection_models=2, 
                        opt_hyperparam='Grid-search', 
                        ensemble=["stacking", "mean", "major"], 
                        opt_metric='accuracy'
                        )

print(results)