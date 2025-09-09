import pandas as pd
import numpy as np
from itertools import product
import random
from tqdm import tqdm

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.base import *
from sklearn.svm import SVC
from skopt import BayesSearchCV
import xgboost as xgb
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping


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

    smote = SMOTE(random_state=19)
    X, y = smote.fit_resample(X, y)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    rf = RandomForestClassifier(random_state=19)
    f1_scorer = make_scorer(f1_score, average="weighted")

    param_grid = {
        'n_estimators' : [50, 100, 250, 500, 750, 1000, 1250, 1500, 2000],
        'criterion' : ["gini", "entropy", "log_loss"],
        'max_depth' : [10, 20, 30, None],
        'max_features' : ["sqrt", "log2", None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    if search == "grid":
        search = GridSearchCV(rf, param_grid, cv=4, scoring=f1_scorer, n_jobs=-1, verbose=2)
    elif search == "random":
        search = RandomizedSearchCV(rf, param_grid, cv=4, n_iter=50, random_state=19, scoring=f1_scorer, n_jobs=-1, verbose=2)
    elif search == "bayes":
        search = BayesSearchCV(rf, param_grid, cv=4, n_iter=50, scoring=f1_scorer, n_jobs=-1, verbose=2)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best F1-score:", search.best_score_)

    best_to_file(best_params=search.best_params_, model="RF Classifier", score=search.best_score_, emg=emg)

def lr_search(rep_df, emg=True, search="random"):
    """Performs Hyperparameter searching on the LR Model"""
    X, y = preprocess(rep_df, emg=emg)

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
        search = GridSearchCV(logr, param_grid, cv=4, scoring=f1_scorer, n_jobs=-1, verbose=2)
    elif search == "random":
        search = RandomizedSearchCV(logr, param_grid, cv=4, n_iter=50, random_state=19, scoring=f1_scorer, n_jobs=-1, verbose=2)
    elif search == "bayes":
        search = BayesSearchCV(logr, param_grid, cv=4, n_iter=50, random_state=19, scoring=f1_scorer, n_jobs=-1, verbose=2)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best F1-score:", search.best_score_)

    best_to_file(best_params=search.best_params_, model="Logistic Regression Classifier", score=search.best_score_, emg=emg)

def svm_search(rep_df, emg=True, search="random"):
    """Performs Hyperparameter searching on the SVM Model"""
    X, y = preprocess(rep_df, emg=emg)

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
        search = GridSearchCV(svm_model, param_grid, cv=4, scoring=f1_scorer, n_jobs=-1, verbose=2)
    elif search == "random":
        search = RandomizedSearchCV(svm_model, param_grid, cv=4, n_iter=50, random_state=19, scoring=f1_scorer, n_jobs=-1, verbose=2)
    elif search == "bayes":
        search = BayesSearchCV(svm_model, param_grid, cv=4, n_iter=50, scoring=f1_scorer, n_jobs=-1, verbose=2)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best F1-score:", search.best_score_)

    best_to_file(best_params=search.best_params_, model="SVM Classifier", score=search.best_score_, emg=emg)

def build_ann(units, learning_rate=0.001, input_dim=None):
    """Defining an ANN using Keras for the given architecture -- with classification goal"""
    ann = Sequential()
    ann.add(Input(shape=(input_dim,)))
    ann.add(Dense(units=units[0], activation='relu'))
    for unit in units[1:]: #dynamically adding layers
        ann.add(Dense(units=unit, activation='relu'))
        ann.add(Dropout(0.2)) #dropout for overfitting prevention

    ann.add(Dense(units=11, activation='softmax')) #output layer
    optimiser = Adam(learning_rate=learning_rate)
    ann.compile(optimizer = optimiser, loss = 'categorical_crossentropy', metrics = ["accuracy"]) 
    return ann

def ann_search(rep_df, emg=True, search="random"):
    """Performs Hyperparameter searching on the ANN Model"""
    X, y = preprocess(rep_df, emg=emg)

    smote = SMOTE(random_state=19)
    X, y = smote.fit_resample(X, y)

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
        # selecting a subset of the parameter grid combinations
        param_dicts = random.sample(param_dicts, min(len(param_dicts), 50))

    y = to_categorical(y, num_classes=11)

    best_params = None
    best_score = 0

    X = np.array(X)
    y = np.array(y)

    for param_dict in tqdm(param_dicts, colour='red'):
        # setting totals to 0 for KFold CV
        total_acc_score = 0
        total_f1_score = 0
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

            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=param_dict['epochs'], batch_size=param_dict['batch_size'], verbose=0, callbacks=[early_stopping])

            loss, accuracy = model.evaluate(X_test, y_test)

            total_acc_score += accuracy

            y_pred_probs = model.predict(X_test)
            y_true = np.argmax(y_test, axis=1)
            y_pred = np.argmax(y_pred_probs, axis=1)

            f1 = f1_score(y_true, y_pred, average='macro')
            total_f1_score += f1

        avg_acc = total_acc_score/kf.get_n_splits(X)
        avg_f1 = total_f1_score/kf.get_n_splits(X)

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

    best_to_file(best_params=best_params, model="ANN Classifier", score=best_score, emg=emg)

def xgb_search(rep_df, emg=True, search="random"):
    """Performs Hyperparameter searching on the XGBoost Model"""
    X, y = preprocess(rep_df, emg=emg)

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
        search = GridSearchCV(model, param_grid, cv=4, scoring=f1_scorer, n_jobs=-1, verbose=2)
    elif search == "random":
        search = RandomizedSearchCV(model, param_grid, cv=4, n_iter=50, random_state=19, scoring=f1_scorer, n_jobs=-1, verbose=2)
    elif search == "bayes":
        search = BayesSearchCV(model, param_grid, cv=4, n_iter=50, scoring=f1_scorer, n_jobs=-1, verbose=2)

    search.fit(X, y)

    print("Best Parameters:", search.best_params_)
    print("Best F1-score:", search.best_score_)

    best_to_file(best_params=search.best_params_, model="XGBoost", score=search.best_score_, emg=emg)

if __name__=="__main__":
    rep_df = pd.read_csv("imu_plus_pred_emg.csv")
    set_seed()
    
    options = [
        ("rf", True),
        ("rf", False),
        ("lr", True),
        ("lr", False),
        ("svm", True),
        ("svm", False),
        ("ann", True),
        ("ann", False),
        ("xgb", True),
        ("xgb", False),
    ]

    for m, b in tqdm(options, colour="magenta"):
        if m=="rf":
            rf_search(rep_df, emg=b, search="bayes")
        elif m=="lr":
            lr_search(rep_df, emg=b, search="bayes")
        elif m=="svm":
            svm_search(rep_df, emg=b, search="bayes")
        elif m=="ann":
            ann_search(rep_df, emg=b)
        elif m=="xgb":
            xgb_search(rep_df, emg=b, search="bayes")
