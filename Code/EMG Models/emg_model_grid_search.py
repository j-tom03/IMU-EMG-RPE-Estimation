import pandas as pd
import numpy as np
from itertools import product
import random
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import make_scorer, f1_score, root_mean_squared_error
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso, Ridge
from sklearn.base import *
from skopt import BayesSearchCV
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.svm import SVR


def best_to_file(best_params, model, score, target_col, filename="best_emg_parameters.txt"):
    """Stores best parameters and score for each model to a file"""
    f = open(filename, "a")
    f.write(f"{model}: {best_params} -- Score: {score} -- Target Column: {target_col} \n")
    f.close()

def remove_unnecesary_cols(rep_df):
    """Removes the unusable columns"""
    return rep_df.drop(['ID', 'rep_num', 'RPE', 'emg_mav', 'emg_rms', 'emg_iemg', 'emg_var', 'emg_zc', 'emg_wl', 'emg_ssc', 'emg_mean_amp', 'emg_peak_amp', 'emg_umap_1', 'emg_umap_2', 'emg_tsne_1', 'emg_tsne_2', "emg_umap_db_class", "emg_tsne_db_class",], axis=1)

def preprocess(rep_df, target_col):
    """Prepares the DF for processing by the models"""
    rep_df = remove_unnecesary_cols(rep_df)
    tar_cols_to_drop = [x for x in ["emg_umap_km_class", "emg_tsne_km_class", "emg_pca_1", "emg_pca_2"] if x!=target_col]
    rep_df = rep_df.drop(tar_cols_to_drop, axis=1)

    X = rep_df.drop([target_col], axis=1)
    y = rep_df[target_col]

    return X, y

def set_seed(seed=19):
    """Sets the random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    
def rf_search(rep_df, target_col, search="random"):
    """Performs Hyperparameter searching on the RandomForest Model"""
    X, y = preprocess(rep_df, target_col)

    # Applying SMOTE to prevent poor minority class prediction
    smote = SMOTE(random_state=19)
    X, y = smote.fit_resample(X, y)

    # defining hyperparameter grid
    param_grid = {
        'n_estimators' : [50, 100, 250, 500, 750, 1000, 1250, 1500, 2000],
        'criterion' : ["gini", "entropy", "log_loss"],
        'max_depth' : [10, 20, 30, None],
        'max_features' : ["sqrt", "log2", None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    rf = RandomForestClassifier(random_state=19)
    f1_scorer = make_scorer(f1_score, average="weighted")

    # selection of hyperparameter search method
    if search == "grid":
        search = GridSearchCV(rf, param_grid, cv=4, scoring=f1_scorer, n_jobs=-1, verbose=3)
    elif search == "random":
        search = RandomizedSearchCV(rf, param_grid, cv=4, n_iter=30, random_state=19, scoring=f1_scorer, n_jobs=-1, verbose=3)
    elif search == "bayes":
        search = BayesSearchCV(rf, param_grid, cv=4, n_iter=30, scoring=f1_scorer, n_jobs=-1, verbose=3)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best F1-score:", search.best_score_)

    best_to_file(best_params=search.best_params_, model="RF Classifier", target_col=target_col, score=search.best_score_)

def lr_search(rep_df, target_col, search="random"):
    """Performs Hyperparameter searching on the Logistic Regression Model"""
    X, y = preprocess(rep_df, target_col)

    smote = SMOTE(random_state=19)
    X, y = smote.fit_resample(X, y)

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [100, 500, 1000]
    }

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    logr = LogisticRegression(random_state=19)
    f1_scorer = make_scorer(f1_score, average="weighted")

    if search == "grid":
        search = GridSearchCV(logr, param_grid, cv=4, scoring=f1_scorer, n_jobs=-1, verbose=3)
    elif search == "random":
        search = RandomizedSearchCV(logr, param_grid, cv=4, n_iter=30, random_state=19, scoring=f1_scorer, n_jobs=-1, verbose=3)
    elif search == "bayes":
        search = BayesSearchCV(logr, param_grid, cv=4, n_iter=30, random_state=19, scoring=f1_scorer, n_jobs=-1, verbose=3)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best F1-score:", search.best_score_)

    best_to_file(best_params=search.best_params_, model="Logistic Regression Classifier", target_col=target_col, score=search.best_score_)

def svm_search(rep_df, target_col, search="random"):
    """Performs Hyperparameter searching on the SVM Model"""
    X, y = preprocess(rep_df, target_col)

    smote = SMOTE(random_state=19)
    X, y = smote.fit_resample(X, y)

    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'decision_function_shape' : ['ovo', 'ovr'],
        'kernel': ['poly', 'rbf'],
        'degree': [2,3,4,5]
    }

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    svm_model = SVC()
    f1_scorer = make_scorer(f1_score, average="weighted")

    if search == "grid":
        search = GridSearchCV(svm_model, param_grid, cv=4, scoring=f1_scorer, n_jobs=-1, verbose=3)
    elif search == "random":
        search = RandomizedSearchCV(svm_model, param_grid, cv=4, n_iter=30, random_state=19, scoring=f1_scorer, n_jobs=-1, verbose=3)
    elif search == "bayes":
        search = BayesSearchCV(svm_model, param_grid, cv=4, n_iter=30, scoring=f1_scorer, n_jobs=-1, verbose=3)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best F1-score:", search.best_score_)

    best_to_file(best_params=search.best_params_, model="SVM Classifier", target_col=target_col, score=search.best_score_)

def build_ann_classification(units, output_shape=4, learning_rate=0.001, input_dim=None):
    """Defining an ANN using Keras for the given architecture -- with classification goal"""
    ann = Sequential()
    ann.add(Input(shape=(input_dim,)))
    ann.add(Dense(units=units[0], activation='relu'))
    for unit in units[1:]: #dynamically adding layers
        ann.add(Dense(units=unit, activation='relu'))
        ann.add(Dropout(0.2)) #dropout for overfitting prevention

    ann.add(Dense(units=output_shape, activation='softmax')) #output layer
    optimiser = Adam(learning_rate=learning_rate)
    ann.compile(optimizer = optimiser, loss = 'categorical_crossentropy', metrics = ["accuracy"]) 
    return ann

def build_ann_regression(units, learning_rate=0.001, input_dim=None):
    """Defining an ANN using Keras for the given architecture -- with regression goal"""
    ann = Sequential()
    ann.add(Input(shape=(input_dim,)))
    ann.add(Dense(units=units[0], activation='relu'))
    for unit in units[1:]: #dynamically adding layers
        ann.add(Dense(units=unit, activation='relu'))
        ann.add(Dropout(0.2)) #dropout for overfitting prevention

    ann.add(Dense(units=1, activation='linear')) #output layer
    optimiser = Adam(learning_rate=learning_rate)
    ann.compile(optimizer = optimiser, loss = 'mse', metrics = ["mae", RootMeanSquaredError()]) 
    return ann

def ann_search(rep_df, target_col, search="random"):
    """Performs Hyperparameter searching on the ANN Model"""
    X, y = preprocess(rep_df, target_col)

    param_grid = {
        'batch_size': [32, 64],  
        'epochs': [50, 100, 200, 300, 400, 500],
        'learning_rate': [0.001, 0.01, 0.0001],
        'units': [
            (16,16),
            (32,32),
            (64,64),
            (16,16,16),
            (32,32,32),
            (64,64,64),
            (32,64,32),
            (128,128,128),
            (64,128,128),
            (64,128,64),
            (64,64,64,64),
            (64,128,128,64),
        ],
    }

    param_combinations = list(product(*param_grid.values()))
    param_dicts = [dict(zip(param_grid.keys(), values)) for values in param_combinations]

    if search == "random":
        # selects random sample from the combinations of the grid
        param_dicts = random.sample(param_dicts, min(len(param_dicts), 50))

    X = np.array(X)
    y = np.array(y)

    if target_col in ['emg_pca_1', 'emg_pca_2']:
        best_score = float('inf') # best score set to infinity if regression
    else:
        smote = SMOTE(random_state=19)
        X, y = smote.fit_resample(X, y)

        num_classes = len(np.unique(y))
        y = to_categorical(y, num_classes=num_classes)
        best_score = 0 # best score set to 0 if classification

    best_params = None

    # iterating the selected parameters
    for param_dict in tqdm(param_dicts, colour='red'):
        # set all totals to zero for KFoldCV
        total_acc_score = 0
        total_mae_score = 0
        total_rmse_score = 0
        total_f1_score = 0

        kf = KFold(n_splits=4)

        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            if target_col in ['emg_pca_1', 'emg_pca_2']:
                # scaling features for reasonable comparison with other models 
                y_scaler = StandardScaler()
                y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
                y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

                model = build_ann_regression(
                    units=param_dict['units'],
                    learning_rate=param_dict['learning_rate'],
                    input_dim=X_train.shape[1],
                )
            else:
                model = build_ann_classification(
                    units=param_dict['units'],
                    output_shape=num_classes,
                    learning_rate=param_dict['learning_rate'],
                    input_dim=X_train.shape[1],
                )

            # early stopping used to prevent overfitting 
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=param_dict['epochs'], batch_size=param_dict['batch_size'], verbose=0, callbacks=[early_stopping])

            if target_col in ['emg_pca_1', 'emg_pca_2']:
                # updating the regression metrics
                loss, mae, rmse = model.evaluate(X_test, y_test)
                total_mae_score += mae
                total_rmse_score += rmse
            else:
                # updating the classification metrics
                loss, accuracy = model.evaluate(X_test, y_test)
                total_acc_score += accuracy

                y_pred_probs = model.predict(X_test)
                y_true = np.argmax(y_test, axis=1)
                y_pred = np.argmax(y_pred_probs, axis=1)

                f1 = f1_score(y_true, y_pred, average='macro')
                total_f1_score += f1
        

        if target_col in ['emg_pca_1', 'emg_pca_2']:
            avg_rmse = total_rmse_score/kf.get_n_splits(X)
            # makes ammendments if the new model is the best
            if avg_rmse < best_score:
                best_params = {
                    'units': param_dict['units'],
                    'batch_size': param_dict['batch_size'],
                    'epochs' :param_dict['epochs'],
                    'learning_rate': param_dict['learning_rate'],
                }

                best_score = avg_rmse
            else:
                pass
        else:
            avg_f1 = total_f1_score/kf.get_n_splits(X)
            # makes ammendments if the new model is the best
            if avg_f1 > best_score:
                best_params = {
                    'units': param_dict['units'],
                    'batch_size': param_dict['batch_size'],
                    'epochs' :param_dict['epochs'],
                    'learning_rate': param_dict['learning_rate'],
                }

                best_score = avg_f1
            else:
                pass

    print("Best Parameters:", best_params)
    print("Best F1-score:", best_score)

    best_to_file(best_params=best_params, model="ANN", target_col=target_col, score=best_score)

def xgb_search(rep_df, target_col, search="random"):
    """Performs Hyperparameter searching on the XGBoost Model"""
    X, y = preprocess(rep_df, target_col)

    smote = SMOTE(random_state=19)
    X, y = smote.fit_resample(X, y)

    param_grid = {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 6, 9],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0, 0.1, 0.3],
        "reg_lambda": [0.1, 1, 10],
    }

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = xgb.XGBClassifier()
    f1_scorer = make_scorer(f1_score, average="weighted")

    if search == "grid":
        search = GridSearchCV(model, param_grid, cv=4, scoring=f1_scorer, n_jobs=-1, verbose=3)
    elif search == "random":
        search = RandomizedSearchCV(model, param_grid, cv=4, n_iter=30, random_state=19, scoring=f1_scorer, n_jobs=-1, verbose=3)
    elif search == "bayes":
        search = BayesSearchCV(model, param_grid, cv=4, n_iter=30, scoring=f1_scorer, n_jobs=-1, verbose=3)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best F1-score:", search.best_score_)

    best_to_file(best_params=search.best_params_, model="XGBoost", target_col=target_col, score=search.best_score_)

def lasso_search(rep_df, target_col, search="random"):
    """Performs Hyperparameter searching on the Lasso Regression Model"""
    X, y = preprocess(rep_df, target_col)

    param_grid = {
        'alpha': np.logspace(-4, 1, 50),
        'max_iter': [1000, 2000, 3000],
        'tol': [1e-4, 1e-3, 1e-2]
    }

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    model = Lasso(random_state=19)
    rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

    if search == "grid":
        search = GridSearchCV(model, param_grid, cv=4, scoring=rmse_scorer, n_jobs=-1, verbose=3)
    elif search == "random":
        search = RandomizedSearchCV(model, param_grid, cv=4, n_iter=30, random_state=19, scoring=rmse_scorer, n_jobs=-1, verbose=3)
    elif search == "bayes":
        search = BayesSearchCV(model, param_grid, cv=4, n_iter=30, scoring=rmse_scorer, n_jobs=-1, verbose=3)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best RMSE:", -search.best_score_)

    best_to_file(best_params=search.best_params_, model="Lasso Regression", target_col=target_col, score=-search.best_score_)

def ridge_search(rep_df, target_col, search="random"):
    """Performs Hyperparameter searching on the Ridge Regression Model"""
    X, y = preprocess(rep_df, target_col)

    param_grid = {
        'alpha': np.logspace(-4, 1, 50),
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        'max_iter': [1000, 2000, 3000],
        'tol': [1e-4, 1e-3, 1e-2]
    }

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    model = Ridge(random_state=19)
    rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

    if search == "grid":
        search = GridSearchCV(model, param_grid, cv=4, scoring=rmse_scorer, n_jobs=-1, verbose=3)
    elif search == "random":
        search = RandomizedSearchCV(model, param_grid, cv=4, n_iter=30, random_state=19, scoring=rmse_scorer, n_jobs=-1, verbose=3)
    elif search == "bayes":
        search = BayesSearchCV(model, param_grid, cv=4, n_iter=30, scoring=rmse_scorer, n_jobs=-1, verbose=3)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best RMSE:", -search.best_score_)

    best_to_file(best_params=search.best_params_, model="Ridge Regression", target_col=target_col, score=-search.best_score_)

def elastic_net_search(rep_df, target_col, search="random"):
    """Performs Hyperparameter searching on the Elastic Net Regression Model"""
    X, y = preprocess(rep_df, target_col)

    param_grid = {
        'alpha': np.logspace(-4, 1, 30),
        'l1_ratio': np.linspace(0.1, 0.9, 9),  # 0 is Ridge, 1 is Lasso
        'max_iter': [1000, 3000],
        'tol': [1e-4, 1e-3]
    }

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    model = ElasticNet(random_state=19)
    rmse_scorer = make_scorer(root_mean_squared_error, greater_is_better=False)

    if search == "grid":
        search = GridSearchCV(model, param_grid, cv=4, scoring=rmse_scorer, n_jobs=-1, verbose=3)
    elif search == "random":
        search = RandomizedSearchCV(model, param_grid, cv=4, n_iter=30, random_state=19, scoring=rmse_scorer, n_jobs=-1, verbose=3)
    elif search == "bayes":
        search = BayesSearchCV(model, param_grid, cv=4, n_iter=30, scoring=rmse_scorer, n_jobs=-1, verbose=3)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best RMSE:", -search.best_score_)

    best_to_file(best_params=search.best_params_, model="ElasticNet Regression", target_col=target_col, score=-search.best_score_)

def rf_reg_search(rep_df, target_col, search="random"):
    """Performs Hyperparameter searching on the RandomForest Regression Model"""
    X, y = preprocess(rep_df, target_col)

    # dynamically changing hyperparameter grid based on y values
    has_negative_values = (y < 0).any()
    if has_negative_values:
        criterion_options = ["squared_error", "absolute_error", "friedman_mse"]
    else:
        criterion_options = ["squared_error", "absolute_error", "friedman_mse", "poisson"]
    
    param_grid = {
        'n_estimators': [50, 100, 250, 500, 750, 1000, 1250, 1500, 2000],
        'criterion': criterion_options,
        'max_depth': [10, 20, 30, None],
        'max_features': ["sqrt", "log2", None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    rf = RandomForestRegressor(random_state=19)

    if search == "grid":
        search = GridSearchCV(rf, param_grid, cv=4, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    elif search == "random":
        search = RandomizedSearchCV(rf, param_grid, cv=4, n_iter=30, random_state=19, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    elif search == "bayes":
        search = BayesSearchCV(rf, param_grid, cv=4, n_iter=30, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best RMSE:", -search.best_score_)

    best_to_file(best_params=search.best_params_, model="RF Regression", target_col=target_col, score=-search.best_score_)

def svr_search(rep_df, target_col, search="random"):
    """Performs Hyperparameter searching on the SVR Model"""
    X, y = preprocess(rep_df, target_col)

    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2,3,4,5]
    }

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    svr = SVR()

    if search == "grid":
        search = GridSearchCV(svr, param_grid, cv=4, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    elif search == "random":
        search = RandomizedSearchCV(svr, param_grid, cv=4, n_iter=30, random_state=19, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    elif search == "bayes":
        search = BayesSearchCV(svr, param_grid, cv=4, n_iter=30, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best RMSE:", -search.best_score_)

    best_to_file(best_params=search.best_params_, model="SVR", target_col=target_col, score=-search.best_score_)

def xgb_reg_search(rep_df, target_col, search="random"):
    """Performs Hyperparameter searching on the XGBoost Regression Model"""
    X, y = preprocess(rep_df, target_col)

    param_grid = {
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 6, 9],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "gamma": [0, 0.1, 0.3],
        "reg_lambda": [0.1, 1, 10],
    }

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    model = xgb.XGBRegressor()

    if search == "grid":
        search = GridSearchCV(model, param_grid, cv=4, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    elif search == "random":
        search = RandomizedSearchCV(model, param_grid, cv=4, n_iter=30, random_state=19, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    elif search == "bayes":
        search = BayesSearchCV(model, param_grid, cv=4, n_iter=30, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best RMSE:", -search.best_score_)

    best_to_file(best_params=search.best_params_, model="XGB Regressor", target_col=target_col, score=-search.best_score_)

if __name__=="__main__":
    set_seed()
    rep_df = pd.read_csv("rep_dataset.csv")

    options = [
        ('rf', 'emg_umap_km_class'),
        ('rf', 'emg_tsne_km_class'),
        ('lr', 'emg_umap_km_class'),
        ('lr', 'emg_tsne_km_class'),
        ('svm', 'emg_umap_km_class'),
        ('svm', 'emg_tsne_km_class'),
        ('xgb', 'emg_umap_km_class'),
        ('xgb', 'emg_tsne_km_class'),
        ('ann', 'emg_umap_km_class'),
        ('ann', 'emg_tsne_km_class'),

        ('ann', 'emg_pca_1'),
        ('ann', 'emg_pca_2'),
        ('enet', 'emg_pca_1'),
        ('enet', 'emg_pca_2'),
        ('las', 'emg_pca_1'),
        ('las', 'emg_pca_2'),
        ('rid', 'emg_pca_1'),
        ('rid', 'emg_pca_2'),
        ('rf', 'emg_pca_1'),
        ('rf', 'emg_pca_2'),
        ('xgb', 'emg_pca_1'),
        ('xgb', 'emg_pca_2'),
        ('svr', 'emg_pca_1'),
        ('svr', 'emg_pca_2'),
    ]

    for m, t in tqdm(options, colour="blue"):
        if m=="rf":
            if t in ['emg_pca_1', 'emg_pca_2']:
                rf_reg_search(rep_df, target_col=t, search="bayes")
            else:
                rf_search(rep_df, target_col=t, search="bayes")
        elif m=="lr":
            lr_search(rep_df, target_col=t, search="bayes")
        elif m=="svm":
            svm_search(rep_df, target_col=t, search="bayes")
        elif m=="ann":
            ann_search(rep_df, target_col=t)
        elif m=="xgb":
            if t in ['emg_pca_1', 'emg_pca_2']:
                xgb_reg_search(rep_df, target_col=t, search="bayes")
            else:
                xgb_search(rep_df, target_col=t, search="bayes")
        elif m=="las":
            lasso_search(rep_df, target_col=t, search="bayes")
        elif m=="rid":
            ridge_search(rep_df, target_col=t, search="bayes")
        elif m=="enet":
            elastic_net_search(rep_df, target_col=t, search="bayes")
        elif m=="svr":
            svr_search(rep_df, target_col=t, search="bayes")
