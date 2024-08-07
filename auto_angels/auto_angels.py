import pandas as pd
import numpy as np
from util import preprocessing, check_feature_selection, check_balancing, train, test, verificar_valores_vazios, exec_ensemble, save_models, save_log, save4angels
import warnings
import json
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

import time

# Ignorar os FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)


def auto_angels(dataset, features, target, test_size=0.3, feature_selection=None, 
    feature_selection_models=1, missing='remove', transformation=None, balancing='Under', hybrid_size=2.0, 
    models='RandomForest', metrics=['f1', 'accuracy', 'precision', 'recall', 'specificity'], 
    opt_metric='accuracy', opt_hyperparam=None, levels=None, n_jobs=-1, cv=5, save_model=False, save_angels=False,
    path_save="../runs", ensemble=None, seed=42, use_threading=False, log_file='log.txt'):

    """
    Função que executa um pipeline de pré-processamento, treinamento, e teste de modelos de machine learning.

    Parâmetros:
    ----------
    dataset : DataFrame
        O conjunto de dados a ser utilizado.
    features : list
        Lista de strings contendo os nomes das features a serem utilizadas no modelo ou no feature_selection.
    target : str
        O nome da coluna alvo.
    test_size : float, opcional (padrão=0.3)
        A proporção do conjunto de dados a ser utilizado como conjunto de teste. Deve ser um valor entre 0.1 e 0.9.
    feature_selection : str ou None, opcional (padrão=None)
        Indica se o autoangels buscará as próprias features ou se serão passadas as features escolhidas.
    feature_selection_models : int, opcional (padrão=1)
        Define o número de modelos a serem utilizados para selecionar as features.
    missing : str ou dict, opcional (padrão='remove')
        Indica como tratar os valores vazios encontrados no conjunto de dados. Valores permitidos são 'remove', 'mean', 'mode', 'median' or 'fixed-value'
        ou um dicionário com tratamentos específicos para cada coluna.
    transformation : dict, opcional (padrão=None)
        Indica se será realizada alguma transformação no conjunto de dados, e quais colunas serão tratadas.
    balancing : bool ou dict, opcional (padrão=True)
        Indica se o conjunto de dados preprocessado será balanceado para o treinamento dos modelos.
    hybrid_size: float (padrão 2.0)
        Utilizado somente quando o param 'balancing' estiver a opção escolhida de 'Hybrid'
        Esse valor representa o tamanho que será aumentado da classe minoritaria
    models : str ou list, opcional (padrão='RandomForest')
        Indica os modelos a serem treinados pelo autoangels.
    metrics : list, opcional (padrão=['f1', 'acc', 'precision', 'recall', 'especificity'])
        Métricas a serem utilizadas para avaliar os modelos na fase de teste.
    opt_metric : str, opcional (padrão='f1')
        Métrica a ser usada para avaliar os modelos na otimização.
    opt_hyperparam : str, opcional (padrão='Grid-search')
        Método de otimização de hiperparâmetros a ser utilizado.
    levels : dict, opcional (padrão=None)
        Níveis de hiperparâmetros a serem explorados para cada modelo.
    cv : int, opcional (padrão=5)
        Número de folds para a validação cruzada.

    Retorna:
    -------
    results : dict
        Um dicionário contendo os resultados do pipeline, incluindo métricas de desempenho, informações sobre o pré-processamento,
        e configurações dos modelos treinados.
    """

    ##### SALVAR OS MODELOS OTIMIZADOS
    ##### Ter a opção de rodar 30x (ou selecionar a quantidade de vezes)
    ##### Fazer teste estatistico?
    ##### Configurar um bot de notificação no telegram?
    ##### PARALELIZAR O TREINAMENTO (POR MODELO, POR EXEMPLO)

    results = {}
    results['status'] = 0

    print("- O AutoAngels está sendo iniciado -")
    save_log("- O AutoAngels está sendo iniciado -", log_file)
    print()

    ### Tratamento de erros dos parametros
    # Verificar os valores permitidos para os parâmetros
    if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
        raise ValueError("O parâmetro 'features' deve ser uma lista de strings.")

    if not isinstance(target, str):
        raise ValueError("O parâmetro 'target' deve ser uma string.")

    # Verificar se a string fornecida em 'target' não está presente nas 'features'
    if target in features:
        raise ValueError("A variável alvo ('target') não deve estar presente nas 'features'.")

    # Verificar se a string fornecida em 'target' está presente nas colunas do dataset
    if target not in dataset.columns:
        raise ValueError("A variável alvo ('target') não está presente nas colunas do dataset.")

    # Verificar se todas as strings fornecidas em 'features' estão presentes nas colunas do dataset
    missing_features = [feature for feature in features if feature not in dataset.columns]
    if missing_features:
        raise ValueError(f"As seguintes 'features' não estão presentes nas colunas do dataset: {', '.join(missing_features)}")

    if not (isinstance(test_size, float) and 0.1 <= test_size <= 0.9):
        raise ValueError("O parâmetro 'test_size' deve ser um float entre 0.1 e 0.9.")

    if feature_selection is not None and feature_selection not in [None, 'SFS']:
        raise ValueError("O parâmetro 'feature_selection' deve ser None ou 'SFS'.")

    if not isinstance(feature_selection_models, int):
        raise ValueError("O parâmetro 'feature_selection_models' deve ser um inteiro.")

    valid_missing_values = ['remove', 'mean', 'mode', 'median']

    if isinstance(missing, str):
        if missing not in valid_missing_values:
            raise ValueError("Se 'missing' for uma string, deve ser uma das opções válidas: 'remove', 'mean', 'mode', 'median'")

    elif isinstance(missing, dict):
        valid_missing_values.append('fixed-value')
        if not all(key in valid_missing_values for key in missing.keys()):
            raise ValueError("As chaves do dicionário 'missing' devem ser uma das opções válidas: 'remove', 'mean', 'mode', 'median', 'fixed-value'.")
        
        for treatment, columns in missing.items():
            if not isinstance(columns, list) and treatment != 'fixed-value':
                raise ValueError(f"Os valores do dicionário 'missing' devem ser listas. Encontrado: {type(columns)}")

            missing_features = [column for column in columns if column not in dataset.columns]
            if missing_features:
                raise ValueError(f"Os seguintes atributos para '{treatment}' não estão presentes nas colunas do dataset: {', '.join(missing_features)}")


    else:
        raise ValueError("O parâmetro 'missing' deve ser uma string ou um dicionário.")

    
    if transformation is not None:
        if not isinstance(transformation, dict):
            raise ValueError("O parâmetro 'transformation' deve ser None ou um dicionário.")

        elif not all(key in ['codificar', 'categorizar', 'one-hot-encoding'] for key in transformation.keys()):
            raise ValueError("As chaves dentro do dicionário 'transformation' devem ser uma das opções válidas: 'codificar', 'categorizar', ou 'one-hot-encoding'.")

        # Verificar os valores do dicionário 'transformation'
        for treatment, columns_or_ranges in transformation.items():
            if treatment == 'categorizar':
                if not isinstance(columns_or_ranges, dict):
                    raise ValueError(f"Os valores para '{treatment}' devem ser um dicionário. Encontrado: {type(columns_or_ranges)}")

                for column, ranges in columns_or_ranges.items():
                    if not isinstance(ranges, list) or len(ranges) < 2:
                        raise ValueError(f"Os valores para '{column}' em 'categorizar' devem ser uma lista com pelo menos dois valores.")

                    # Verificar se as colunas mencionadas em 'categorizar' estão presentes no dataset
                    if column not in dataset.columns:
                        raise ValueError(f"A coluna '{column}' mencionada em 'categorizar' não está presente nas colunas do dataset.")

            else:
                if not isinstance(columns_or_ranges, list):
                    raise ValueError(f"Os valores para '{treatment}' devem ser uma lista. Encontrado: {type(columns_or_ranges)}")

                # Verificar se as colunas mencionadas em 'codificar' ou 'one-hot-encoding' estão presentes no dataset
                missing_features = [column for column in columns_or_ranges if column not in dataset.columns]
                if missing_features:
                    raise ValueError(f"As seguintes colunas para '{treatment}' não estão presentes nas colunas do dataset: {', '.join(missing_features)}")

    if not isinstance(balancing, (str, dict)):
        raise ValueError("O parâmetro 'balancing' deve ser uma string ou um dicionário.")
    
    if isinstance(balancing, str) and balancing not in ['Under', 'Over', 'Hybrid', 'Imba']:
        raise ValueError("Se 'balancing' for uma string, deve ser uma das opções: 'Under', 'Over', 'Hybrid' ou 'Imba'.")

    if isinstance(models, str) and models not in ['RandomForest', 'AdaBoost', 'GradientBoost', 'XGBoost', 'DecisionTree', 'lightGBM']:
        raise ValueError("Se 'models' for uma string, deve ser uma das opções válidas: 'RandomForest', 'AdaBoost', 'GradientBoost', 'XGBoost', 'DecisionTree', 'lightGBM'.")

    elif isinstance(models, list) and not all(model in ['RandomForest', 'AdaBoost', 'GradientBoost', 'XGBoost', 'DecisionTree', 'lightGBM'] for model in models):
        raise ValueError("Se 'models' for uma lista, todos os elementos devem ser uma das opções válidas: 'RandomForest', 'AdaBoost', 'GradientBoost', 'XGBoost', 'DecisionTree', 'lightGBM'.")

    valid_metrics = ['f1', 'accuracy', 'precision', 'recall', 'specificity', 'ROC-AUC', 'mcc']

    if not isinstance(metrics, list) or not all(isinstance(metric, str) for metric in metrics):
        raise ValueError("O parâmetro 'metrics' deve ser uma lista de strings.")

    if not all(metric in valid_metrics for metric in metrics):
        raise ValueError("As métricas em 'metrics' devem ser uma das opções válidas: 'f1', 'accuracy', 'precision', 'recall', 'specificity', 'ROC-AUC', 'mcc'.")

    if not isinstance(opt_metric, str):
        raise ValueError("O parâmetro 'opt_metric' deve ser uma string.")

    if opt_metric not in valid_metrics:
        raise ValueError("O valor do parâmetro 'opt_metric' deve ser uma das opções válidas: 'f1', 'accuracy', 'precision', 'recall', 'specificity', 'ROC-AUC', 'mcc'.")

    if opt_hyperparam not in ['Grid-search', 'Random-search', 'optuna', None]:
        raise ValueError("O parâmetro 'opt_hyperparam' deve ser 'Grid-search', 'Random-search', 'optuna', ou None.")

    if not isinstance(levels, (dict, type(None))):
        raise ValueError("O parâmetro 'levels' deve ser um inteiro, uma lista, um dicionário ou None.")

    model_classes = {
        'RandomForest': RandomForestClassifier,
        'AdaBoost': AdaBoostClassifier,
        'GradientBoost': GradientBoostingClassifier,
        'XGBoost': XGBClassifier,
        'DecisionTree': DecisionTreeClassifier,
        'lightGBM': LGBMClassifier
    }

    if isinstance(levels, dict):

        # Verificar os hiperparâmetros para cada modelo em levels
        for model, params in levels.items():
            if model not in model_classes:
                raise ValueError(f"O modelo '{model}' não é suportado.")

            # Obter os hiperparâmetros suportados pelo modelo correspondente
            valid_hyperparameters = model_classes[model]().get_params().keys()

            # Verificar se os hiperparâmetros definidos em levels para o modelo estão presentes
            for param_set in params:
                for param, values in param_set.items():
                    if param not in valid_hyperparameters:
                        raise ValueError(f"O hiperparâmetro '{param}' não é suportado pelo modelo '{model}'.")

    else:
        if opt_hyperparam in ['Grid-search', 'Random-search', 'optuna']:
            levels = {
                'RandomForest': [
                    {
                        'n_estimators': [50, 100, 150, 200],
                        'criterion' : ['entropy', 'gini'],
                        'max_depth': [None, 1, 3, 5, 7, 9, 11]
                    }
                ],
                'AdaBoost': [
                    {
                        'n_estimators': [50, 100, 150, 200],
                        'learning_rate': [0.01, 0.1, 0.5, 1],
                    }
                ],
                'GradientBoost': [
                    {
                        'n_estimators': [50, 100, 150, 200],
                        'learning_rate': [0.01, 0.1, 0.5, 1],
                        'loss': ['log_loss', 'exponential'],
                        'max_depth': [None, 1, 3, 5, 7, 9, 11]
                    }
                ],
                'XGBoost': [
                    {
                        'n_estimators': [50, 100, 150, 200],
                        'learning_rate': [0.01, 0.1, 0.5, 1],
                        'max_depth': [1, 3, 5, 7, 9]
                    }
                ],
                'DecisionTree': [
                    {
                        'criterion': ['gini', 'entropy'],
                        'splitter': ['best', 'random'],
                        'max_depth': [None, 1, 3, 5, 7, 9, 11]
                    }
                ],
                'lightGBM': [
                    {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.1, 0.01],
                        'max_depth': [3, 5, 7],
                        'min_child_samples': [10, 20, 50],
                    }
                ]
            }
    if not isinstance(cv, int) or cv < 2:
        raise ValueError("O parâmetro 'cv' deve ser um inteiro maior ou igual a 2.")

    ### ---------------------------

    inicio = time.time()

    ###Pre-processamento
    status_missing, status_transformation, X_train, X_test, y_train, y_test = preprocessing(dataset, features, target, test_size, missing, transformation, seed, log_file)

    datasets = [X_train, X_test]
    for data in datasets:
        colunas_com_valores_vazios = verificar_valores_vazios(data)
        if isinstance(colunas_com_valores_vazios, dict):
            print("As seguintes colunas contêm valores vazios:")
            for key, value in colunas_com_valores_vazios.items():
                print(f"{key}: {value}")
            results['status'] = 1
            return results


    ###Verificar o balanceamento
    status_balancing, X_train, X_test, y_train, y_test = check_balancing(X_train, X_test, y_train, y_test, balancing, hybrid_size=hybrid_size, seed=seed, log_file=log_file)
    
    ### Preciso verificar se irá realizar feature selection ou não
    
    opt_metric_temp = opt_metric
    if opt_metric == 'mcc':
        mcc_scorer = make_scorer(matthews_corrcoef)
        opt_metric=mcc_scorer
    
    if opt_metric == 'ROC-AUC':
        opt_metric='roc_auc'

    inicio_feature = time.time()
    status_feature_selection, lista_de_features = check_feature_selection(X_train, y_train, feature_selection, feature_selection_models, 
                                                                            models, scoring=opt_metric, cv=cv, n_jobs= n_jobs, use_threading=use_threading, log_file= log_file)
    fim_feature = time.time()
    
    feature_time_exex = fim_feature - inicio_feature
    ###--------------------------

    print(lista_de_features)


    if opt_metric_temp == 'ROC-AUC':
        roc_auc_scorer = {'AUC': 'roc_auc'}
        opt_metric=roc_auc_scorer

    ###Treinar o modelo
    inicio_train = time.time()
    trained_models = train(X_train, y_train, lista_de_features, models, opt_metric, opt_hyperparam, levels, n_jobs, cv, log_file)
    fim_train = time.time()
    
    train_time_exex = fim_train - inicio_train

    #### SALVAR OS MODELOS TREINADOS\
    if save_model:
        save_models(trained_models, base_dir=path_save)


    ###Testar o modelo
    results['results'], predictions = test(X_test, y_test, lista_de_features, trained_models, metrics, ensemble, log_file)

    if ensemble is not None and len(trained_models)>1:
        if isinstance(ensemble, str):
            results['results'] = exec_ensemble(ensemble, trained_models, results['results'], predictions, X_train, y_train, X_test, y_test, metrics)
        elif isinstance(ensemble, list):
            for type_ensemble in ensemble:
                results['results'] = exec_ensemble(type_ensemble, trained_models, results['results'], predictions, X_train, y_train, X_test, y_test, metrics)


    fim = time.time()
    
    time_exec = fim - inicio
    print("- O AutoAngels está encerrando -")
    save_log("- O AutoAngels está encerrando -", log_file)
    print()
    
    ##AJUSTES PARA RETORNAR AO USUARIO
    features_names = {}
    for model_name, selected_indexes in lista_de_features.items():
        features_names[model_name] = [X_train.columns[idx] for idx in selected_indexes]

    results['feature_selection'] = status_feature_selection
    if status_feature_selection:
        results['features'] = features_names
    else:
        results['features'] = features

    results['preprocessing'] = {'missing': status_missing, 'transformation':status_transformation}
    results['balancing'] = {'done': status_balancing, 'value_counts_train': [y_train.value_counts()[0], y_train.value_counts()[1]], 'value_counts_test': [y_test.value_counts()[0], y_test.value_counts()[1]]}
    results['optimization'] = {}
    for model_name, infos in trained_models.items():
        if infos['best_param'] is not None:
            results['optimization'][model_name] = (infos['best_param'], infos['best_score'])

    results['time_exec'] = time_exec
    results['feature_time_exex'] = feature_time_exex
    results['train_time_exec'] = train_time_exex


    if save_angels:
        save4angels(results, X_train, trained_models)
    #results['optimization']
    # def convert_numpy_to_list(obj):
    #     if isinstance(obj, np.ndarray):
    #         return obj.tolist()
    #     elif isinstance(obj, np.integer):
    #         return int(obj)
    #     elif isinstance(obj, np.floating):
    #         return float(obj)
    #     elif isinstance(obj, np.bool_):
    #         return bool(obj)
    #     elif isinstance(obj, np.str_):
    #         return str(obj)
    #     elif isinstance(obj, np.unicode_):
    #         return str(obj)
    #     else:
    #         return str(obj)

    # # Converter o dicionário para uma string JSON formatada
    # results_json = json.dumps(results, default=convert_numpy_to_list, indent=4)

    # return results_json
    
    return results

