import pandas as pd

from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, auc

from src.classifiers.classical import model_defs, CLASSICAL_MODELS

ENSEMBLE_MODELS = ["Voting", "Stacking"]
def _model_defs(name, estimators):
    if name == "Voting":
        return VotingClassifier(estimators=estimators, voting='soft')
    if name == "Stacking":
        return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    return

def classify(models_ran, name, train, test, confusion_matrices={}, roc_curves={}, train_auc={}, test_auc={}, best_params_all={}):
    """
    Parameters:
    - name (String)
    """
    estimators = []
    for m in models_ran:
        if m in CLASSICAL_MODELS:
            (v, _) = model_defs(m)
            estimators.append((m, v))
    model = _model_defs(name, estimators)
    model.fit(train.X_scaled, train.y)
    y_prob = model.predict_proba(test.X_scaled)[:,1]
    y_pred = model.predict(test.X_scaled)
    
    confusion_matrices[name] = confusion_matrix(test.y, y_pred)
    roc_curves[name] = (test.y, y_prob)
    train_auc[name] = roc_auc_score(train.y, model.predict_proba(train.X_scaled)[:,1])
    test_auc[name] = roc_auc_score(test.y, y_prob)
    df = pd.DataFrame(classification_report(test.y, y_pred, output_dict=True)).transpose()
    df['model'] = name
    return df