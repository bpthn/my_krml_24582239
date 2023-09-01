# Solution:
def print_regressor_scores(y_preds, y_actuals, set_name=None):
    """Print the RMSE and MAE for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae

    print(f"RMSE {set_name}: {mse(y_actuals, y_preds, squared=False)}")
    print(f"MAE {set_name}: {mae(y_actuals, y_preds)}")


    # Solution:
def assess_regressor_set(model, features, target, set_name=''):
    """Save the predictions from a trained model on a given set and print its RMSE and MAE scores

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Trained Sklearn model with set hyperparameters
    features : Numpy Array
        Features
    target : Numpy Array
        Target variable
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    preds = model.predict(features)
    print_regressor_scores(y_preds=preds, y_actuals=target, set_name=set_name)



# Solution:
def fit_assess_regressor(model, X_train, y_train, X_val, y_val):
    """Train a regressor model, print its RMSE and MAE scores on the training and validation set and return the trained model

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Instantiated Sklearn model with set hyperparameters
    X_train : Numpy Array
        Features for the training set
    y_train : Numpy Array
        Target for the training set
    X_train : Numpy Array
        Features for the validation set
    y_train : Numpy Array
        Target for the validation set

    Returns
    sklearn.base.BaseEstimator
        Trained model
    -------
    """
    model.fit(X_train, y_train)
    assess_regressor_set(model, X_train, y_train, set_name='Training')
    assess_regressor_set(model, X_val, y_val, set_name='Validation')
    return model



# Performance for classification

def print_classify_scores(y_preds, y_actuals, set_name=None):
    """Print the f1, recall, precision and accuracy for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    print(f"f1_score {set_name}: {f1_score(y_actuals, y_preds)}")
    print(f"precision_score {set_name}: {precision_score(y_actuals, y_preds)}")
    print(f"recall_score {set_name}: {recall_score(y_actuals, y_preds)}")
    print(f"accuracy_score {set_name}: {accuracy_score(y_actuals, y_preds)}")


    # Solution:
def assess_classify_set(model, features, target, set_name=''):
    """Save the predictions from a trained model on a given set and print its f1, recall, precision and accuracy scores

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Trained Sklearn model with set hyperparameters
    features : Numpy Array
        Features
    target : Numpy Array
        Target variable
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """
    preds = model.predict(features)
    print_classify_scores(y_preds=preds, y_actuals=target, set_name=set_name)



# Solution:
def fit_assess_classify(model, X_train, y_train, X_val, y_val):
    """Train a classifier model, print its RMSE and MAE scores on the training and validation set and return the trained model

    Parameters
    ----------
    model: sklearn.base.BaseEstimator
        Instantiated Sklearn model with set hyperparameters
    X_train : Numpy Array
        Features for the training set
    y_train : Numpy Array
        Target for the training set
    X_train : Numpy Array
        Features for the validation set
    y_train : Numpy Array
        Target for the validation set

    Returns
    sklearn.base.BaseEstimator
        Trained model
    -------
    """
    model.fit(X_train, y_train)
    assess_classify_set(model, X_train, y_train, set_name='Training')
    assess_classify_set(model, X_val, y_val, set_name='Validation')
    return model

def print_classify_scores(model, X_actuals, y_actuals, set_name=''):
    from sklearn.metrics import roc_auc_score
    pred_probs_val = model.predict_proba(X_actuals)[:, 1]
    print("AUROC score: ", roc_auc_score(y_actuals, pred_probs_val))
    return pred_probs_val