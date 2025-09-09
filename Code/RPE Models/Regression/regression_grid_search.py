import pandas as pd
import numpy as np
from itertools import product
import random
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, root_mean_squared_error
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from skopt import BayesSearchCV

import tensorflow as tf
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import xgboost as xgb
from sklearn.svm import SVR

def best_to_file(best_params, model, score, emg, filename="best_parameters.txt"):
    """Stores the best model for each variation to the file"""
    f = open(filename, "a")
    f.write(f"{model}: {best_params} -- Score: {score} -- EMG: {emg} \n")
    f.close()

def set_seed(seed=19):
    """Sets the random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    
def preprocess(rep_df, emg=True):
    """Preprocesses the dataset by removing columns and OHE categorical data"""
    rep_df = rep_df.drop(['ID', 'rep_num'], axis=1)

    if not emg:
        rep_df = rep_df.drop(['emg_tsne_km_class', 'emg_umap_km_class', 'emg_pca_1', 'emg_pca_2'], axis=1)
    else:
        cat_cols = ['emg_tsne_km_class', 'emg_umap_km_class']
        rep_df = pd.get_dummies(rep_df, columns=cat_cols, drop_first=True)

    X = rep_df.drop(["RPE"], axis=1)
    y = rep_df["RPE"]

    return X, y

def rf_search(rep_df, emg=True, search="random"):
    """Performs Hyperparameter searching on the RF Model"""
    X, y = preprocess(rep_df, emg=emg)

    param_grid = {
        'n_estimators' : [50, 100, 250, 500, 750, 1000, 1250, 1500, 2000],
        'criterion' : ["squared_error", "absolute_error", "friedman_mse", "poisson"],
        'max_depth' : [10, 20, 30, None],
        'max_features' : ["sqrt", "log2", None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    rf = RandomForestRegressor(random_state=19)

    if search == "grid":
        search = GridSearchCV(rf, param_grid, cv=4, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    elif search == "random":
        search = RandomizedSearchCV(rf, param_grid, cv=4, n_iter=50, random_state=19, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    elif search == "bayes":
        search = BayesSearchCV(rf, param_grid, cv=4, n_iter=50, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best Score:", -search.best_score_)

    best_to_file(best_params=search.best_params_, model="RF Regressor", score=-search.best_score_, emg=emg)

def svr_search(rep_df, emg=True, search="random"):
    """Performs Hyperparameter searching on the SVR Model"""
    X, y = preprocess(rep_df, emg=emg)

    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2,3,4,5]
    }

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    svr_model = SVR()

    if search == "grid":
        search = GridSearchCV(svr_model, param_grid, cv=4, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    elif search == "random":
        search = RandomizedSearchCV(svr_model, param_grid, cv=4, n_iter=50, random_state=19, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    elif search == "bayes":
        search = BayesSearchCV(svr_model, param_grid, cv=4, n_iter=50, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best Score:", -search.best_score_)

    best_to_file(best_params=search.best_params_, model="SVR", score=-search.best_score_, emg=emg)

def build_ann(units, learning_rate=0.001, input_dim=None): 
    """Defining an ANN using Keras for the given architecture -- with regression goal"""
    ann = Sequential()
    ann.add(Input(shape=(input_dim,)))
    ann.add(Dense(units=units[0], activation='relu'))
    for unit in units[1:]: #dynamically assigning layers
        ann.add(Dense(units=unit, activation='relu'))
        ann.add(Dropout(0.2)) #dropout to prevent overfitting

    ann.add(Dense(units=1, activation='linear')) #output layer
    optimiser = Adam(learning_rate=learning_rate)
    ann.compile(optimizer = optimiser, loss = 'mse', metrics=[RootMeanSquaredError()]) 
    return ann

def ann_search(rep_df, emg=True, search="random"):
    """Performs Hyperparameter searching on the ANN Model"""
    X, y = preprocess(rep_df, emg=emg)

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
        param_dicts = random.sample(param_dicts, min(len(param_dicts), 50))

    X = np.array(X)
    y = np.array(y)

    best_params = None
    best_score = float('inf') #assign to infinity

    for param_dict in tqdm(param_dicts, colour='red'):
        # manual kfold CV implementation
        total_rsme_score = 0
        kf = KFold(n_splits=4)

        for i, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = build_ann(
                units=param_dict['units'],
                learning_rate=param_dict['learning_rate'],
                input_dim=X_train.shape[1],
            )

            # early stopping to stop overfittting if validation loss increases
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=param_dict['epochs'], batch_size=param_dict['batch_size'], verbose=0, callbacks=[early_stopping])

            loss, rsme = model.evaluate(X_test, y_test)

            total_rsme_score += rsme

        avg_rsme = total_rsme_score/kf.get_n_splits(X)

        if avg_rsme < best_score:
            best_params = {
                'units': param_dict['units'],
                'batch_size': param_dict['batch_size'],
                'epochs' :param_dict['epochs'],
                'learning_rate': param_dict['learning_rate'],
            }

            best_score = avg_rsme
        else:
            pass

    print("Best Parameters:", best_params)
    print("Best Score:", best_score)

    best_to_file(best_params=best_params, model="ANN Regression", score=best_score, emg=emg)

def xgb_search(rep_df, emg=True, search="random"):
    """Performs Hyperparameter searching on the XGBoost Model"""
    X, y = preprocess(rep_df, emg=emg)

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

    model = xgb.XGBRegressor()

    if search == "grid":
        search = GridSearchCV(model, param_grid, cv=4, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    elif search == "random":
        search = RandomizedSearchCV(model, param_grid, cv=4, n_iter=50, random_state=19, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
    elif search == "bayes":
        search = BayesSearchCV(model, param_grid, cv=4, n_iter=50, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best Score:", -search.best_score_)

    best_to_file(best_params=search.best_params_, model="XGBoost", score=-search.best_score_, emg=emg)


def lasso_search(rep_df, emg=True, search="random"):
    """Performs Hyperparameter searching on the Lasso Regression Model"""
    X, y = preprocess(rep_df, emg=emg)

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
        search = RandomizedSearchCV(model, param_grid, cv=4, n_iter=50, random_state=19, scoring=rmse_scorer, n_jobs=-1, verbose=3)
    elif search == "bayes":
        search = BayesSearchCV(model, param_grid, cv=4, n_iter=50, scoring=rmse_scorer, n_jobs=-1, verbose=3)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best Score:", -search.best_score_)

    best_to_file(best_params=search.best_params_, model="Lasso Regression", score=-search.best_score_, emg=emg)

def ridge_search(rep_df, emg=True, search="random"):
    """Performs Hyperparameter searching on the Ridge Regression Model"""
    X, y = preprocess(rep_df, emg=emg)

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
        search = RandomizedSearchCV(model, param_grid, cv=4, n_iter=50, random_state=19, scoring=rmse_scorer, n_jobs=-1, verbose=3)
    elif search == "bayes":
        search = BayesSearchCV(model, param_grid, cv=4, n_iter=50, scoring=rmse_scorer, n_jobs=-1, verbose=3)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best Score:", -search.best_score_)

    best_to_file(best_params=search.best_params_, model="Ridge Regression", score=-search.best_score_, emg=emg)

def elastic_net_search(rep_df, emg=True, search="random"):
    """Performs Hyperparameter searching on the Elastic Net Regression Model"""
    X, y = preprocess(rep_df, emg=emg)

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
        search = RandomizedSearchCV(model, param_grid, cv=4, n_iter=50, random_state=19, scoring=rmse_scorer, n_jobs=-1, verbose=3)
    elif search == "bayes":
        search = BayesSearchCV(model, param_grid, cv=4, n_iter=50, scoring=rmse_scorer, n_jobs=-1, verbose=3)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best Score:", -search.best_score_)

    best_to_file(best_params=search.best_params_, model="ElasticNet Regression", score=-search.best_score_, emg=emg)


if __name__=="__main__":
    rep_df = pd.read_csv("imu_plus_pred_emg.csv")

    options = [
        ("rf", True),
        ("rf", False),
        ("svr", True),
        ("svr", False),
        ("xgb", True),
        ("xgb", False),
        ("lasso", True),
        ("lasso", False),
        ("ridge", True),
        ("ridge", False),
        ("elastic", True),
        ("elastic", False),
        ("ann", True),
        ("ann", False),
    ]

    for m, b in tqdm(options, colour="blue"):
        if m=="rf":
            rf_search(rep_df, emg=b, search="bayes")
        elif m=="svr":
            svr_search(rep_df, emg=b, search="bayes")
        elif m=="ann":
            ann_search(rep_df, emg=b)
        elif m=="xgb":
            xgb_search(rep_df, emg=b, search="bayes")
        elif m=="lasso":
            lasso_search(rep_df, emg=b, search="bayes")
        elif m=="ridge":
            ridge_search(rep_df, emg=b, search="bayes")
        elif m=="elastic":
            elastic_net_search(rep_df, emg=b, search="bayes")
