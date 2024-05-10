from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings

# Ignorar os FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

def selecao_caracteristicas_sfs(X, y, model, forward=True, scoring='accuracy', cv=5, n_jobs=1):

    # Divisão em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Escolha do modelo - exemplo: RandomForestClassifier
    #modelo = RandomForestClassifier(n_estimators=100, random_state=42)

    # Inicialização do SequentialFeatureSelector
    # sfs = SequentialFeatureSelector(modelo,
    #                                 k_features=num_caracteristicas,
    #                                 forward=forward,
    #                                 scoring=scoring,
    #                                 cv=5)

    sfs = SequentialFeatureSelector(model,
                                    forward=forward, 
                                    k_features=(4, X_train.shape[1]), 
                                    floating=False, 
                                    verbose=1, 
                                    scoring=scoring, 
                                    cv=cv,
                                    n_jobs=n_jobs)

    # Realiza a seleção de características
    
    sfs.fit(X_train, y_train)


    # Retorna a lista de índices das características selecionadas
    return list(sfs.k_feature_idx_)