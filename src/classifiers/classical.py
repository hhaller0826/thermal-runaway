import pandas as pd
from src.config import config
from src.classifiers.keras import MyKerasClassifier

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc

from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam

#Deep Learning Models
def build_lstm(units=64, learning_rate=0.001, features=2):
    model = Sequential([
        LSTM(units, input_shape=(config.WINDOW_SIZE, features)),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_cnn(filters=32, kernel_size=3, learning_rate=0.001, features=2):
    model = Sequential([
        Conv1D(filters, kernel_size, activation='relu', input_shape=(config.WINDOW_SIZE, features)),
        MaxPooling1D(2),
        Flatten(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Models with GridSearchCV
CLASSICAL_MODELS = ['RandomForest', 'SVM', 'GB']
DL_MODELS = ['LSTM', 'CNN']
def model_defs(name):
    assert name in CLASSICAL_MODELS+DL_MODELS, f'{name} not a valid model'
    if name == "RandomForest": 
        return (RandomForestClassifier(), {"n_estimators": [100, 200]})
    if name == "SVM": 
        return (SVC(probability=True), {
            "C": [0.1, 1, 10], 
            "gamma": ["scale"], 
            "kernel": ["rbf"]
        })
    if name == "GB": 
        return (GradientBoostingClassifier(), {
            "n_estimators": [100, 200], 
            "learning_rate": [0.05, 0.1]
        })
    if name == "LSTM": 
        return (MyKerasClassifier(build_fn=build_lstm), {
            "units": [32, 64],
            "learning_rate": [0.001, 0.0005],
            "epochs": [10],
            "batch_size": [32]
        })
    if name == "CNN": 
        return (MyKerasClassifier(build_fn=build_cnn), {
            "filters": [16, 32],
            "kernel_size": [3],
            "learning_rate": [0.001],
            "epochs": [10],
            "batch_size": [32]
        })

def get_best(name, X, y):
    (model, grid) = model_defs(name)
    gs = GridSearchCV(model, grid, cv=3, scoring="roc_auc", n_jobs=-1)
    gs.fit(X, y)
    return gs.best_params_, gs.best_estimator_

def classify(name, train, test, confusion_matrices={}, roc_curves={}, train_auc={}, test_auc={}, best_params_all={}):
    """
    Parameters:
    - name (String)
    """
    is_classic = (name in CLASSICAL_MODELS)
    (train_X, test_X) = (train.X_scaled, test.X_scaled) if is_classic else (train.X, test.X)

    best_params, best_est = get_best(name, train_X, train.y)
    print(f"🔍 Best {name} params:", best_params)
    best_params_all[name] = best_params

    def get_y_prob(X):
        prob = best_est.predict_proba(X)
        return prob[:,1] if is_classic else prob

    y_pred = best_est.predict(test_X)
    y_prob = get_y_prob(test_X)

    confusion_matrices[name] = confusion_matrix(test.y, y_pred)
    roc_curves[name] = (test.y, y_prob)
    train_auc[name] = roc_auc_score(train.y, get_y_prob(train_X))
    test_auc[name] = roc_auc_score(test.y, y_prob)

    df = pd.DataFrame(classification_report(test.y, y_pred, output_dict=True)).transpose()
    df['model'] = name
    return df 


    