from missing import remove, mean, mode, fixed_value, median
from transformation import codificar, categorizar, one_hot_encoding
from optimization import opt_grid_search, opt_random_search, opt_optuna
from feature_selection import selecao_caracteristicas_sfs
from balancing import random_undersampling, over_sampling_SMOTE, hybrid_sampling
from ensemble import voto_marjoritario, media_proba, stacking

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import warnings
from collections import Counter
import numpy as np
import threading

import os
import pickle

# Ignorar os FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

model_classes = {
        'RandomForest': RandomForestClassifier,
        'AdaBoost': AdaBoostClassifier,
        'GradientBoost': GradientBoostingClassifier,
        'XGBoost': XGBClassifier,
        'DecisionTree': DecisionTreeClassifier,
        'lightGBM': LGBMClassifier
    }


def print_target(y_train, y_test):
    print(y_train.value_counts())
    print(y_test.value_counts())

def preprocessing(dataset, features, target, test_size, missing, transformation, seed):
    print("----- Iniciando o pre-processamento -----")
    ### Pre-processing
    features_ = features
    features_.append(target)
    dataset_ = dataset[features_]

    #Remover os registros que tem o target vazio
    dataset_ = dataset_.dropna(subset=[target])

    #Dividir o dataset em treino e teste
    # X = dataset_.drop(target, axis=1)
    # y = dataset_[target]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    dataset_['num_colunas_vazias'] = dataset_.isnull().sum(axis=1) 

    dataset_minoritario = dataset_[dataset_[target] == 1]
    dataset_majoritario = dataset_[dataset_[target] == 0]

    dataset_minoritario = dataset_minoritario.sort_values(by='num_colunas_vazias')
    dataset_majoritario = dataset_majoritario.sort_values(by='num_colunas_vazias')

    quantidade_test = int(dataset_minoritario.shape[0]*test_size)


    ########## Aqui estou separando o dataset de teste, dando prioridade para os melhores registros (registro sem dados faltantes ou que tem menos dados faltantes)
    ########## Verificar se isso é uma boa estratégia!
    X_test_min = dataset_minoritario.head(quantidade_test)
    X_train_min = dataset_minoritario.drop(X_test_min.index)

    X_test_maj = dataset_majoritario.head(quantidade_test)
    X_train_maj = dataset_majoritario.drop(X_test_maj.index)

    X_train = pd.concat([X_train_min, X_train_maj])
    X_test = pd.concat([X_test_min, X_test_maj])

    y_train = X_train[target]
    X_train = X_train.drop([target, 'num_colunas_vazias'], axis=1)

    y_test = X_test[target]
    X_test = X_test.drop([target, 'num_colunas_vazias'], axis=1)

    print_target(y_train, y_test)

    status_missing = False
    ### Tratamento de dados faltantes
    if isinstance(missing, str):
        print("------- Tratando os dados faltantes -------")
        if missing == 'remove':
            status_missing = 'remove'
            X_train, y_train = remove(X_train, y_train)
            X_test, y_test = remove(X_test, y_test)
        
        elif missing == 'mean':
            status_missing = 'mean'
            X_train = mean(X_train)
            X_test = mean(X_test)
            
        elif missing == 'median':
            status_missing = 'median'
            X_train = median(X_train)
            X_test = median(X_test)
        
        elif missing == 'mode':
            status_missing = 'mode'
            X_train = mode(X_train)
            X_test = mode(X_test)
        
        elif missing == 'fixed-value':
            status_missing = 'fixed-value'
            X_train = fixed_value(X_train, value=-1)
            X_test = fixed_value(X_test, value=-1)
        
        else:
            print("Parametro missing com valor não permitido, será utilizado o defaulf.")
            status_missing = 'remove'
            X_train, y_train = remove(X_train, y_train)
            X_test, y_test = remove(X_test, y_test)
    elif isinstance(missing, dict):
        print("------- Tratando os dados faltantes -------")
        status_missing = {}
        if 'remove' in missing:
            status_missing['remove'] = missing['remove']
            X_train, y_train = remove(X_train, y_train, missing['remove'])
            X_test, y_test = remove(X_test, y_test, missing['remove'])
        
        if 'mean' in missing:
            status_missing['mean'] = missing['mean']
            X_train = mean(X_train, missing['mean'])
            X_test = mean(X_test, missing['mean'])
        
        if 'median' in missing:
            status_missing['median'] = missing['median']
            X_train = median(X_train, missing['median'])
            X_test = median(X_test, missing['median'])
        
        if 'mode' in missing:
            status_missing['mode'] = missing['mode']
            X_train = mode(X_train, missing['mode'])
            X_test = mode(X_test, missing['mode'])

        if 'fixed-value' in missing:
            status_missing['fixed-value'] = {}
            for key, value in missing['fixed-value'].items():
                status_missing['fixed-value'][key] = value 
                X_train = fixed_value(X_train, value=value, columns=[key])
                X_test = fixed_value(X_test, value=value, columns=[key])
        
    # else:
    #     print("Parametro missing com valor não permitido, será utilizado o defaulf.")
    #     X_train = remove(X_train, missing['remove'])
    #     X_test = remove(X_test, missing['remove'])
    ###---

    ### Aplicar transformação nos dados
    status_transformation = False
    if transformation is not None:
        print("------- Aplicando a transformação dos dados -------")
        if isinstance(transformation, dict):
            status_transformation = {}
            if 'codificar' in transformation:
                status_transformation['codificar'] = transformation['codificar']
                X_train, le_dict = codificar(X_train, transformation['codificar'])
                X_test, le_dict = codificar(X_test, transformation['codificar'], le_dict)
            
            if 'categorizar' in transformation:
                status_transformation['categorizar'] = transformation['categorizar']
                #for dicionario in transformation['categorizar']:
                for column, intervalos in transformation['categorizar'].items():
                    rotulos = list(range(len(intervalos) - 1))
                    X_train = categorizar(X_train, column, intervalos, rotulos)
                    X_test = categorizar(X_test, column, intervalos, rotulos)
            
            if 'one-hot-encoding' in transformation:
                status_transformation['one-hot-encoding'] = transformation['one-hot-encoding']
                X_train = one_hot_encoding(X_train, transformation['one-hot-encoding'])
                X_test = one_hot_encoding(X_test, transformation['one-hot-encoding'])
        else:
            print("Parametro transformation precisa ser um dicionario, por esse motivo, não foi realizado nenhuma transformaçao nos dados")


    print("----- Finalizando o pre-processamento -----")
    print()

    return status_missing, status_transformation, X_train, X_test, y_train, y_test

def check_balancing(X_train, X_test, y_train, y_test, balancing, hybrid_size=2, seed=42):

    print("----- Iniciando o Balanceamento -----")
    if isinstance(balancing, str):
        if balancing == 'Under':
            X_train, X_test, y_train, y_test = random_undersampling(X_train, X_test, y_train, y_test, seed=seed)
        elif balancing == 'Over':
            X_train, X_test, y_train, y_test = over_sampling_SMOTE(X_train, X_test, y_train, y_test, seed=seed)
        elif balancing == 'Hybrid':
            X_train, X_test, y_train, y_test = hybrid_sampling(X_train, X_test, y_train, y_test, hybrid_size=hybrid_size, seed=seed)
        else:
            return False, X_train, X_test, y_train, y_test
    elif isinstance(balancing, dict):
        X_train, X_test, y_train, y_test = random_undersampling(X_train, X_test, y_train, y_test, weight=balancing, seed=seed)
    else:
        print("O parametro precisa ser uma string ou dicionario, será utilizado o default")
        X_train, X_test, y_train, y_test = random_undersampling(X_train, X_test, y_train, y_test, seed=seed)
    
    print("----- Finalizando o Balanceamento -----")
    print()
    print_target(y_train, y_test)
    return True, X_train, X_test, y_train, y_test


def thread_func(model_name, X_train, y_train, model_classes, lista_de_features, scoring, cv):
    print(f"Iniciando o {model_name}")
    model = model_classes[model_name]()
    lista_de_features[model_name] = selecao_caracteristicas_sfs(X_train, y_train, model, scoring=scoring, cv=cv)
    print(f"Finalizando o {model_name}")  

def check_feature_selection(X_train, y_train, feature_selection, feature_selection_models, models, scoring, cv, n_jobs=-1, use_threading=False):
    
    status_feature_selection = False
    lista_de_features = {}
    if feature_selection is not None:
        print("----- Iniciando a seleção de características -----")
        
        if feature_selection == 'SFS':
            status_feature_selection={}
            status_feature_selection['Done'] = True
            if feature_selection_models > 1:
                #'RandomForest', 'AdaBoost', 'GradientBoost', 'XGBoost', 'DecisionTree'
                if isinstance(models, list):
                    if use_threading:
                        threads = []
                        for model_name in models:
                            thread = threading.Thread(target=thread_func, args=(model_name, X_train, y_train, model_classes, lista_de_features, scoring, cv))
                            thread.start()
                            threads.append(thread)

                        for thread in threads:
                            thread.join()
                    else:
                        for model_name in models:
                            print(model_name)
                            model = model_classes[model_name]()
                        
                            lista_de_features[model_name] = selecao_caracteristicas_sfs(X_train, y_train, model, scoring=scoring, cv=cv, n_jobs=n_jobs)
                elif isinstance(models, str):
                    model_name = models
                    print(model_name)
                    model = model_classes[model_name]()
                    lista_de_features[model_name] = selecao_caracteristicas_sfs(X_train, y_train, model, scoring=scoring, cv=cv, n_jobs=n_jobs)
                    

            else:
                print("Random_forest")
                model = RandomForestClassifier()
                features = selecao_caracteristicas_sfs(X_train, y_train, model, cv)
                if isinstance(models, list):
                    for model_name in models: 
                        lista_de_features[model_name] = features
                elif isinstance(models, str):
                    lista_de_features[models] = features
        
        print("----- Finalizando a seleção de características -----")
        print()
            
    else:
        if isinstance(models, list):
            for model_name in models: 
                lista_de_features[model_name] = list(range(len(X_train.columns)))
        elif isinstance(models, str):
            lista_de_features[models] = list(range(len(X_train.columns)))

    return status_feature_selection, lista_de_features

def train(X_train, y_train, lista_de_features, models, opt_metric, opt_hyperparam, levels, n_jobs, cv):

    trained_models = {}
    X_train_org = X_train.copy()

    if isinstance(models, str):
        models = [models]
    
    for model_name in models:

        model = model_classes[model_name]()

        X_train = X_train_org.iloc[:, lista_de_features[model_name]]

        #'Grid-search', 'Random-search', 'optuna' or None
        if opt_hyperparam is not None:
            print("----- Iniciando a Optimização -----")
            if 'Grid-search' == opt_hyperparam:
                param_grid = levels[model_name]
                print(f'Iniciando o Grid-search {model_name}')
                status, best_param, best_score = opt_grid_search(X_train, y_train, model, param_grid, scoring=opt_metric, n_jobs=n_jobs, cv=cv)
                print(f'Finalizando o Grid-search {model_name}')
            elif 'Random-search' == opt_hyperparam:
                param_distributions = levels[model_name]
                print(f'Iniciando o Random-search {model_name}')
                status, best_param, best_score = opt_random_search(X_train, y_train, model, param_distributions, scoring=opt_metric, n_iter=-1, n_jobs=n_jobs, cv=cv)
                print(f'Finalizando o Random-search {model_name}')
            elif 'optuna' == opt_hyperparam:
                param_distributions = levels[model_name]
                print(f'Iniciando o Optuna {model_name}')
                status, best_param, best_score = opt_optuna(X_train, y_train, model, param_distributions, scoring=opt_metric, n_iter=100, n_jobs=n_jobs, cv=cv)
                print(f'Finalizando o Optuna {model_name}')
            else:
                best_param = None
                best_score = None
            
            print("----- Finalizando a Optimização -----")
            print()

        else:
            best_param = None
            best_score = None

        if best_param is not None:
            model.set_params(**best_param)
        
        print("----- Iniciando o treinamento -----")

        model.fit(X_train, y_train)
        predicts = model.predict_proba(X_train)

        print("----- Finalizando o treinamento -----")
        print()

        trained_models[model_name] = {'model':model, 'best_param':best_param, 'best_score': best_score, 'train_predict':predicts}
        
    return trained_models

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity

def test(X_test, y_test, lista_de_features, trained_models, metrics, ensemble):
    results = {}

    X_test_org = X_test.copy()

    print("----- Iniciando os testes -----")
    predictions = []
    # Itera sobre o dicionário de modelos treinados
    for model_name, infos in trained_models.items():
        model_results = {}
        model = infos['model']

        X_test = X_test_org.iloc[:, lista_de_features[model_name]]

        # Faz previsões usando o modelo
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        #print(type(y_pred_proba))
        #print(y_pred_proba)
        #input()
        predictions.append(y_pred_proba)

        # Calcula as métricas especificadas
        if 'accuracy' in metrics:
            acc = accuracy_score(y_test, y_pred)
            model_results['accuracy'] = acc
        if 'precision' in metrics:
            precision = precision_score(y_test, y_pred)
            model_results['precision'] = precision
        if 'recall' in metrics:
            recall = recall_score(y_test, y_pred)
            model_results['recall'] = recall
        if 'f1' in metrics:
            f1 = f1_score(y_test, y_pred)
            model_results['f1'] = f1
        if 'ROC-AUC' in metrics:
            roc_auc = roc_auc_score(y_test, y_pred)
            model_results['ROC-AUC'] = roc_auc
        if 'specificity' in metrics:
            specificity = calculate_specificity(y_test, y_pred)
            model_results['specificity'] = specificity
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        model_results['confusion_matrix'] = conf_matrix

        # Adiciona os resultados para o modelo ao dicionário de resultados
        results[model_name] = model_results

    print("----- Finalizando os testes -----")
    print()

    return results, predictions

def exec_ensemble(ensemble, trained_models, results, predictions, X_train, y_train, X_test, y_test, metrics):
    model_results = {}
    name_model = 'ensemble'  

    if ensemble == "major":
        y_pred = voto_marjoritario(predictions)
        name_model = 'ensemble-voto-majoritario'

    elif ensemble == "mean":
        y_pred = media_proba(predictions)
        name_model = 'ensemble-mean'
    
    elif ensemble == "stacking":
        y_pred = stacking(X_train, y_train, predictions, trained_models)
        name_model = 'ensemble-stacking'

    else:
        y_pred = media_proba(predictions)
        name_model = 'ensemble-mean'

    # Calcula as métricas especificadas
    if 'accuracy' in metrics:
        acc = accuracy_score(y_test, y_pred)
        model_results['accuracy'] = acc
    if 'precision' in metrics:
        precision = precision_score(y_test, y_pred)
        model_results['precision'] = precision
    if 'recall' in metrics:
        recall = recall_score(y_test, y_pred)
        model_results['recall'] = recall
    if 'f1' in metrics:
        f1 = f1_score(y_test, y_pred)
        model_results['f1'] = f1
    if 'ROC-AUC' in metrics:
        roc_auc = roc_auc_score(y_test, y_pred)
        model_results['ROC-AUC'] = roc_auc
    if 'specificity' in metrics:
        specificity = calculate_specificity(y_test, y_pred)
        model_results['specificity'] = specificity
    
    conf_matrix = confusion_matrix(y_test, y_pred)
    model_results['confusion_matrix'] = conf_matrix

    # Adiciona os resultados para o modelo ao dicionário de resultados
    results[name_model] = model_results

    return results


def save_models(trained_models, base_dir='../runs'):
    # Verifica se o diretório base existe, se não, cria-o
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Encontra o próximo número de execução
    exec_number = 0
    while os.path.exists(os.path.join(base_dir, f'exec{exec_number}')):
        exec_number += 1

    # Cria o diretório para esta execução
    exec_dir = os.path.join(base_dir, f'exec{exec_number}')
    os.makedirs(exec_dir)

    # Salva cada modelo treinado em arquivos pickle dentro do diretório da execução
    for model_name, model in trained_models.items():
        model_filename = os.path.join(exec_dir, f'{model_name}.pkl')
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)

    print(f'Modelos salvos em: {exec_dir}')


def verificar_valores_vazios(dataset):
    """
    Verifica se o dataset contém valores vazios e retorna as colunas com valores vazios e a porcentagem de valores vazios, se houver.

    Args:
    - dataset: DataFrame do pandas contendo os dados a serem verificados.

    Returns:
    - Dicionário onde as chaves são os nomes das colunas com valores vazios e os valores são as porcentagens de valores vazios.
    """
    colunas_com_valores_vazios = dataset.columns[dataset.isnull().any()]
    resultado = {}

    for coluna in colunas_com_valores_vazios:
        total_valores = len(dataset[coluna])
        valores_vazios = dataset[coluna].isnull().sum()
        porcentagem_vazios = (valores_vazios / total_valores) * 100
        resultado[coluna] = porcentagem_vazios

    return resultado if resultado else None