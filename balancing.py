from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

def random_undersampling(X_train, X_test, y_train, y_test, weight=None, seed=42):
    
    if weight is None:
        rus = RandomUnderSampler(random_state=seed)
        X_train, y_train = rus.fit_resample(X_train, y_train)
        X_test, y_test = rus.fit_resample(X_test, y_test)
    else:
        if isinstance(weight, dict):
            sampling_strategy_train = {0: int(sum(y_train == 1) * weight[0]), 1: int(sum(y_train == 1)* weight[1])}
            sampling_strategy_test = {0: int(sum(y_test == 1) * weight[0]), 1: int(sum(y_test == 1)* weight[1])}
            rus_train = RandomUnderSampler(sampling_strategy=sampling_strategy_train, random_state=seed)
            rus_test = RandomUnderSampler(sampling_strategy=sampling_strategy_test, random_state=seed)
            X_train, y_train = rus_train.fit_resample(X_train, y_train)
            X_test, y_test = rus_test.fit_resample(X_test, y_test)
        else:
            print("O peso para o balanceamento precisa ser um dicionario, por causa disso, foi realizado o undersample 1 pra 1")
            rus = RandomUnderSampler(random_state=seed)
            X_train, y_train = rus.fit_resample(X_train, y_train)
            X_test, y_test = rus.fit_resample(X_test, y_test)

    return X_train, X_test, y_train, y_test

def over_sampling_SMOTE(X_train, X_test, y_train, y_test, seed=42):
    smote = SMOTE(random_state=seed)

    X_train, y_train = smote .fit_resample(X_train, y_train)
    X_test, y_test = smote .fit_resample(X_test, y_test)

    return X_train, X_test, y_train, y_test