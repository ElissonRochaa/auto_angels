import sys
sys.path.append('../auto_angels')

import pandas as pd 
import numpy as np
import json
from auto_angels import auto_angels

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def main():
    df = pd.read_csv('data_set_premature.csv', low_memory=False)  
    
    df = df[(df['mc_par_prematuro'].notnull()) | (df['mc_par_idade_gestacional_semana'].notnull())]
    
    df.loc[df['mc_par_idade_gestacional_semana']<37, 'mc_par_prematuro'] = True
    df.loc[((df['mc_par_idade_gestacional_semana']>=37) & (df['mc_par_prematuro'].isnull())), 'mc_par_prematuro'] = False
    #print(df[['mc_par_prematuro', 'mc_par_idade_gestacional_semana']].sample(50))
    
    df.loc[df['mc_par_prematuro']==True, 'mc_par_prematuro'] = 1
    df.loc[df['mc_par_prematuro']==False, 'mc_par_prematuro'] = 0
    
    df['mc_par_prematuro'] = df['mc_par_prematuro'].astype(int)
    
    # features = [
    #     'mc_get_alcool',
    #     'mc_get_gravidez_planejada',
    #     'mc_get_peso_anterior',
    #     'mc_get_risco_gestacional',
    #     'mc_mul_est_civil',
    #     'mc_pes_raca_etnia',
    #     'mc_dae_diagnostico_desnutricao',
    #     'mc_dae_escolaridade',
    #     'has_arterial_hypertension',
    #     'has_diabetes',
    #     'has_cirurgia_pelvica',
    #     'has_cardiopatia',
    #     'has_infeccao_urinaria',
    #     'has_malformacao',
    #     'has_malformacao_familiar',
    #     'has_gemelaridade',
    #     'has_gemelaridade_familiar',
    #     'has_les',
    #     'sd_quant_gest',
    #     'sd_quant_aborto',
    #     'sd_quant_partos',
    #     'sd_quant_partos_cesarios',
    #     'time_between_pregnancies',
    #     'idade'
    # ]
    
    columns = {
        'mc_get_gravidez_planejada':'plan_pregnancy',
        'mc_get_peso_anterior':'previous_weight',
        'mc_get_risco_gestacional':'gestational_risk',
        'mc_mul_est_civil':'marital_status',
        'mc_pes_raca_etnia':'mother_race',
        'mc_dae_diagnostico_desnutricao': 'food_insecurity',
        'mc_dae_escolaridade':'schooling',
        'has_arterial_hypertension':'has_hypertension',
        'has_diabetes':'has_diabetes',
        'has_cirurgia_pelvica':'has_pelvic_surgery',
        'has_infeccao_urinaria':'has_urinary_infection',
        'has_malformacao':'has_congenital_malformation',
        'has_gemelaridade':'has_family_twinship',
        'sd_quant_gest':'amount_gestation',
        'sd_quant_aborto':'amount_abortion',
        'sd_quant_partos':'amount_deliveries',
        'sd_quant_partos_cesarios': 'amount_cesarean',
        'time_between_pregnancies': 'time_between_pregnancies',
        'idade':'age'
    }

    target = 'mc_par_prematuro'
    
    df.rename(columns=columns)

    features = list(columns.values())

    missing = {
        'mean':['age', 'previous_weight'], 
        'fixed-value':{ 'plan_pregnancy': -1,'gestational_risk':-1,
                       'marital_status':-1,'mother_race':-1,
                       'food_insecurity':-1,'schooling':-1}
        }
    
    models = ['RandomForest', 'AdaBoost', 'GradientBoost', 'XGBoost', 'DecisionTree']
    metrics = ['f1', 'accuracy', 'precision', 'recall', 'specificity', 'ROC-AUC']

    balancing = {'Under':[1], 'Hybrid': [1.5, 2, 3]}
    
    resultados = []
    resultados_2 = {}

    df_copy = df.copy()
    features_copy = features.copy()

    for tecn, size in balancing.items():
        resultados_2[tecn] = {}
        for s in size:
            resultados_2[tecn][str(s)] = {}
            df = df_copy.copy()
            features = features_copy.copy()
            print('df: ', df.shape)
            print('features: ', len(features))
            print('target: ', target)
            results = auto_angels(df, features, target, 
                                    test_size=0.3, 
                                    missing=missing,
                                    # transformation=transformation,
                                    balancing=tecn,
                                    hybrid_size=s,
                                    models=models,
                                    metrics=metrics,
                                    feature_selection='SFS', 
                                    feature_selection_models=2, 
                                    opt_hyperparam='Grid-search', 
                                    #ensemble=["stacking", "mean", "major"], 
                                    opt_metric='ROC-AUC'
                                    )
        
        print(results)
        resultados.append(results)
        resultados_2[tecn][str(s)] = results
    
    print("\n" * 20)
    print(resultados_2)
    with open('resultado_prematuro_com_sfs_roc_auc.json', 'w') as file:
        json.dump(resultados_2, file, indent = 4, default=convert_numpy)

if __name__ == '__main__':
    main()