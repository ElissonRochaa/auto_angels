from missing import remove, mean, mode, fixed_value
from transformation import codificar, categorizar, one_hot_encoding
from optimization import opt_grid_search, opt_random_search, opt_optuna
from feature_selection import selecao_caracteristicas_sfs
from balancing import random_undersampling

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import warnings
from collections import Counter
import numpy as np

# Ignorar os FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

model_classes = {
        'RandomForest': RandomForestClassifier,
        'AdaBoost': AdaBoostClassifier,
        'GradientBoost': GradientBoostingClassifier,
        'XGBoost': XGBClassifier,
        'DecisionTree': DecisionTreeClassifier
    }

def preprocessing(dataset, features, target, test_size, missing, transformation, seed):
    print("----- Iniciando o pre-processamento -----")
    ### Pre-processing
    features_ = features
    features_.append(target)
    dataset_ = dataset[features_]

    #Remover os registros que tem o target vazio
    dataset_ = dataset_.dropna(subset=[target])

    #Dividir o dataset em treino e teste
    X = dataset_.drop(target, axis=1)
    y = dataset_[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

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

def check_balancing(X_train, X_test, y_train, y_test, balancing, seed=42):

    if balancing is not False:
        print("----- Iniciando o Balanceamento -----")
        if isinstance(balancing, bool):
            X_train, X_test, y_train, y_test = random_undersampling(X_train, X_test, y_train, y_test, seed=seed)
        elif isinstance(balancing, dict):
            X_train, X_test, y_train, y_test = random_undersampling(X_train, X_test, y_train, y_test, weight=balancing, seed=seed)
        else:
            print("O parametro precisa ser booleano ou dicionario, será utilizado o default")
            X_train, X_test, y_train, y_test = random_undersampling(X_train, X_test, y_train, y_test, seed=seed)
        
        print("----- Finalizando o Balanceamento -----")
        print()
        return True, X_train, X_test, y_train, y_test

    return False, X_train, X_test, y_train, y_test

def check_feature_selection(X_train, y_train, feature_selection, feature_selection_models, models, scoring, cv):
    
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
                    for model_name in models:
                        print(model_name)
                        model = model_classes[model_name]()
                        
                        lista_de_features[model_name] = selecao_caracteristicas_sfs(X_train, y_train, model, scoring=scoring, cv=cv)
                elif isinstance(models, str):
                    model_name = models
                    print(model_name)
                    model = model_classes[model_name]()
                    lista_de_features[model_name] = selecao_caracteristicas_sfs(X_train, y_train, model, scoring=scoring, cv=cv)
                    

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

        print("----- Finalizando o treinamento -----")
        print()

        trained_models[model_name] = {'model':model, 'best_param':best_param, 'best_score': best_score}
        
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
    
    if ensemble and len(trained_models)>1:


        ##VOTO MAJORITARIO
        # model_results = {}
        # transposta = zip(*predictions)

        # y_pred = [Counter(coluna).most_common(1)[0][0] for coluna in transposta]

        ##MEDIA DAS PROBABILIDADES

        soma_prob = []
        for i, prob in enumerate(predictions):
            print(prob)
            if i == 0:
                soma_prob = prob
            else:
                soma_prob = soma_prob + prob
            
            print(soma_prob)
        
        media_prob = soma_prob/len(predictions)

        y_pred = np.argmax(media_prob, axis=1)

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
        results['ensemble'] = model_results

    print("----- Finalizando os testes -----")
    print()

    return results

def verificar_valores_vazios(dataset):
    """
    Verifica se o dataset contém valores vazios e retorna as colunas com valores vazios, se houver.

    Args:
    - dataset: DataFrame do pandas contendo os dados a serem verificados.

    Returns:
    - Lista contendo os nomes das colunas com valores vazios, ou uma mensagem indicando que não há valores vazios.
    """
    colunas_com_valores_vazios = dataset.columns[dataset.isnull().any()].tolist()

    if colunas_com_valores_vazios:
        return colunas_com_valores_vazios
    else:
        return None