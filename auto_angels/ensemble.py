import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier

def voto_marjoritario(predictions):

    predictions_convertidas = [np.argmax(probabilidades, axis=1) for probabilidades in predictions]
    transposta = zip(*predictions_convertidas)
    y_pred = [Counter(coluna).most_common(1)[0][0] for coluna in transposta]
    return y_pred

def media_proba(predictions):
    soma_prob = []
    for i, prob in enumerate(predictions):
        if i == 0:
            soma_prob = prob
        else:
            soma_prob = soma_prob + prob
        
    media_prob = soma_prob/len(predictions)

    y_pred = np.argmax(media_prob, axis=1)

    return y_pred

# def stacking(X_train, y_train, X_test, trained_models):

#     X_stacking_train = np.empty((len(X_train), 0))
    
#     for model_name, model_info in trained_models.items():
#         train_predict = model_info['train_predict']
#         X_stacking_train = np.hstack((X_stacking_train, train_predict.reshape(-1, 1)))

#     ensemble_model = RandomForestClassifier(n_estimators=100, random_state=42)
#     ensemble_model.fit(X_stacking_train, y_train)

#     stacking_pred = ensemble_model.predict(X_test)
    
#     return stacking_pred

def stacking(X_train, y_train, predictions, trained_models):

    combined_predictions = []
    
    for i, model_info in enumerate(trained_models.values()):
        predicts = model_info['train_predict']
        if i == 0:
            combined_predictions = predicts
        else:
            combined_predictions = np.concatenate((combined_predictions, predicts), axis=1)

    ensemble_model = RandomForestClassifier(n_estimators=100, random_state=42)
    ensemble_model.fit(combined_predictions, y_train)

    X_predictions = []
    for i, pred in enumerate(predictions):
        if i == 0:
            X_predictions = pred
        else:
            X_predictions = np.concatenate((X_predictions, pred), axis=1)

    print(X_predictions.shape)

    stacking_pred = ensemble_model.predict(X_predictions)
    
    return stacking_pred