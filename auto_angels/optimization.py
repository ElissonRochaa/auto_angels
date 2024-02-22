from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
import warnings

# Ignorar os FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=FutureWarning)

def opt_grid_search(X_train, y_train, model, param_grid, scoring='f1', n_jobs=-1, cv=5):
    model_grid = GridSearchCV(
        estimator=model, # modelo a ser aplicado o grid
        param_grid=param_grid, # os hiperparametros que o grid vai percorrer
        cv=cv, # valor de k para o kfold, usado no cross-validation
        scoring=scoring, # métrica de avaliação usada para avaliar a performance dos hiperparametros
        n_jobs=n_jobs,
        verbose=1
    )

    try:
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train.ravel()
      
        model_grid.fit(X_train, y_train)
        return 0, model_grid.best_params_, model_grid.best_score_

    except Exception as e:
        print(f'Algo deu errado: {e}')
        return 1, None, None


def opt_random_search(X_train, y_train, model, param_distributions, scoring='f1', n_iter=-1, n_jobs=-1, cv=5):
    
    if n_iter == -1:
        quant = 1
        for value in param_distributions[0].values():
            quant = quant * len(value) 
        quant = quant/2
        n_iter = int(quant) 
    
    model_random = RandomizedSearchCV(
        estimator=model,  # modelo a ser aplicado o grid
        param_distributions=param_distributions,  # os hiperparâmetros que o grid vai percorrer
        cv=cv,  # valor de k para o kfold, usado no cross-validation
        scoring=scoring,  # métrica de avaliação usada para avaliar a performance dos hiperparâmetros
        n_iter=n_iter,  # número de iterações
        n_jobs=n_jobs,
        verbose=1
    )

    try:
        if y_train.ndim > 1 and y_train.shape[1] == 1:
            y_train = y_train.ravel()
      
        model_random.fit(X_train, y_train)
        return 0, model_random.best_params_, model_random.best_score_

    except Exception as e:
        print(f'Algo deu errado: {e}')
        return 1, None, None


def opt_optuna(X_train, y_train, model, param_distributions, scoring='f1', n_iter=100, n_jobs=-1, cv=5):
    print("Em desenvolvimento...")
    return 1, None, None
