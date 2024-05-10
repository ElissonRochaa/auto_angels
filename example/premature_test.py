import sys
sys.path.append('../auto_angels')

import pandas as pd 
import numpy as np
from auto_angels import auto_angels

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
#     #'mc_get_frequencia_uso_alcool',
#     #'mc_get_fator_rh',
#     #'mc_get_frequencia_uso_drogas',
#     'mc_get_fumo',
#     'mc_get_gravidez_planejada',
#     #'mc_get_grupo_sanguineo',
#     'mc_get_peso_anterior',
#     #'mc_get_qtd_cigarro_dia',
#     'mc_get_reforco',
#     'mc_get_risco_gestacional',
#     #'mc_get_sensibilizada',
#     'mc_get_ini_env_kit_bebe',
#     #'mc_mul_amamentacao',
#     'mc_mul_chefe_familia',
#     'mc_mul_em_risco',
#     'mc_mul_est_civil',
#     #'mc_mul_nivel_inseguranca',
#     'mc_mul_renda_familiar',
#     'mc_pes_raca_etnia',
#     'mc_dae_diagnostico_desnutricao',
#     'mc_dae_ener_elet_dom',
#     'mc_dae_escolaridade',
#     'mc_dae_loc_moradia',
#     'mc_dae_mrd_lgd_red_esg',
#     'mc_dae_rfa',
#     'mc_dae_sit_moradia',
#     'mc_dae_trat_agua_uso',
#     'has_arterial_hypertension',
#     'has_arterial_hypertension_familiar',
#     'has_diabetes',
#     'has_diabetes_familiar',
#     'has_cirurgia_pelvica',
#     'has_cirurgia_pelvica_familiar',
#     'has_cardiopatia',
#     'has_cardiopatia_familiar',
#     'has_infeccao_urinaria',
#     'has_infeccao_urinaria_familiar',
#     'has_malformacao',
#     'has_malformacao_familiar',
#     'has_gemelaridade',
#     'has_gemelaridade_familiar',
#     'has_les',
#     'has_les_familiar',
#     'sd_quant_gest',
#     'sd_quant_aborto',
#     'sd_quant_partos',
#     'sd_quant_partos_cesarios',
#     'time_between_pregnancies',
#     'idade'
# ]

features = [
    'mc_get_alcool',
    'mc_get_gravidez_planejada',
    'mc_get_peso_anterior',
    'mc_get_risco_gestacional',
    'mc_get_ini_env_kit_bebe',
    'mc_mul_est_civil',
    'mc_pes_raca_etnia',
    'mc_dae_diagnostico_desnutricao',
    'mc_dae_escolaridade',
    'has_arterial_hypertension',
    'has_diabetes',
    'has_cirurgia_pelvica',
    'has_cardiopatia',
    'has_infeccao_urinaria',
    'has_malformacao',
    'has_malformacao_familiar',
    'has_gemelaridade',
    'has_gemelaridade_familiar',
    'has_les',
    'sd_quant_gest',
    'sd_quant_aborto',
    'sd_quant_partos',
    'sd_quant_partos_cesarios',
    'time_between_pregnancies',
    'idade'
]

target = 'mc_par_prematuro'


#missing = {'mean':['idade', 'mc_get_peso_anterior'], 'fixed-value':{'mc_get_risco_gestacional':-1, 'mc_dae_escolaridade':-1}}
missing = {'mean':['idade', 'mc_get_peso_anterior'], 'fixed-value':{
                                            'mc_get_alcool':-1, 
                                            #'mc_get_fumo':-1, 
                                            'mc_get_gravidez_planejada': -1,
                                            #'mc_get_reforco':-1,
                                            'mc_get_risco_gestacional':-1,
                                            #'mc_mul_chefe_familia':-1,
                                            'mc_mul_est_civil':-1,
                                            #'mc_mul_renda_familiar':-1,
                                            'mc_pes_raca_etnia':-1,
                                            'mc_dae_diagnostico_desnutricao':-1,
                                            #'mc_dae_ener_elet_dom':-1,
                                            'mc_dae_escolaridade':-1,
                                            #'mc_dae_loc_moradia':-1,
                                            #'mc_dae_mrd_lgd_red_esg':-1,
                                            #'mc_dae_rfa':-1,
                                            #'mc_dae_sit_moradia':-1,
                                            #'mc_dae_trat_agua_uso':-1,
                                            }
    }

models = ['RandomForest', 'AdaBoost', 'GradientBoost', 'XGBoost', 'DecisionTree']


results = auto_angels(df, features, target, 
                        test_size=0.2, 
                        missing=missing, 
                        #balancing='Hybrid', 
                        models=models, 
                        feature_selection='SFS', 
                        feature_selection_models=1, 
                        opt_hyperparam='Grid-search', 
                        #ensemble=["stacking", "mean", "major"], 
                        opt_metric='accuracy'
                        )

print(results)