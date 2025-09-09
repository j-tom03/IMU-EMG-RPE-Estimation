import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import re
import ast

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, classification_report, accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso, Ridge
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

def extract_model_params(model_name, target_column, file_path="./best_emg_parameters.txt"):
    """Reads best model file and finds parameter dictionary"""
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            if model_name in line and f"Target Column: {target_column}" in line:
                # Extract the dictionary using regex and ast for safe parsing
                match = re.search(r'{.*}', line)
                if match:
                    try:
                        param_dict = ast.literal_eval(match.group())
                        results.append(param_dict)
                    except Exception as e:
                        print(f"Error parsing line: {line}\n{e}")
    
    return results[0] # returning first match as should be only match

def set_up_files():
    """Resets the result output files"""
    targets = ['emg_umap_km_class', 'emg_tsne_km_class', 'emg_pca_1', 'emg_pca_2']

    for target in targets:
        f = open(f"./results/{target}_model_eval.csv", mode="w")
        if target in ['emg_pca_1', 'emg_pca_2']:
            f.write("mae,mse,rmse,r2,model,target\n")
        else:
            f.write("accuracy,precision_macro,recall_macro,f1_macro,precision_weighted,recall_weighted,f1_weighted,model,target\n")

        f.close()

def results_to_file(results_dict):
    """Stores metrics for each model to a file"""

    target = results_dict['target'][0]
    df = pd.DataFrame(results_dict)
    df.to_csv(f"./results/{target}_model_eval.csv", index=False, mode='a', header=False)

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

    X = np.array(X)
    y = np.array(y)

    return X, y

def set_seed(seed=19):
    """Sets the random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def regression_results(y_test, y_pred):
    """Generate results dictionary for regression models"""
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def rf_model(rep_df, target_col, k=4):
    """Fits and evaluates an RF Model using KFold cross-validation"""
    X, y = preprocess(rep_df, target_col)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    smote = SMOTE(random_state=19)
    X, y = smote.fit_resample(X, y)

    # extracting best parameters from file
    parameters = extract_model_params("RF Classifier", target_col)

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    metrics = {
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'precision_weighted': [],
        'recall_weighted': [],
        'f1_weighted': []
    }

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        rf = RandomForestClassifier(**parameters, random_state=19)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision_macro'].append(report['macro avg']['precision'])
        metrics['recall_macro'].append(report['macro avg']['recall'])
        metrics['f1_macro'].append(report['macro avg']['f1-score'])
        metrics['precision_weighted'].append(report['weighted avg']['precision'])
        metrics['recall_weighted'].append(report['weighted avg']['recall'])
        metrics['f1_weighted'].append(report['weighted avg']['f1-score'])

    avg_results = {key: [np.mean(vals)] for key, vals in metrics.items()}
    avg_results.update({
        'model': ["RF Classifier"],
        'target': [target_col]
    })

    results_to_file(avg_results)


def lr_model(rep_df, target_col, k=4):
    """Fits and evaluates the best LR Model for the target"""
    X, y = preprocess(rep_df, target_col)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Applying SMOTE to prevent overfitting
    smote = SMOTE(random_state=19)
    X, y = smote.fit_resample(X, y)

    # extracting best parameters from file
    parameters = extract_model_params("Logistic Regression Classifier", target_col)

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    metrics = {
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'precision_weighted': [],
        'recall_weighted': [],
        'f1_weighted': []
    }

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        logr = LogisticRegression(**parameters, random_state=19)
        logr.fit(X_train, y_train)
        y_pred = logr.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision_macro'].append(report['macro avg']['precision'])
        metrics['recall_macro'].append(report['macro avg']['recall'])
        metrics['f1_macro'].append(report['macro avg']['f1-score'])
        metrics['precision_weighted'].append(report['weighted avg']['precision'])
        metrics['recall_weighted'].append(report['weighted avg']['recall'])
        metrics['f1_weighted'].append(report['weighted avg']['f1-score'])

    avg_results = {key: [np.mean(vals)] for key, vals in metrics.items()}
    avg_results.update({
        'model': ["Logistic Regression Classifier"],
        'target': [target_col]
    })

    results_to_file(avg_results)

def svm_model(rep_df, target_col, k=4):
    """Fits and evaluates the best SVM Model for the target"""
    X, y = preprocess(rep_df, target_col)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Applying SMOTE to prevent overfitting
    smote = SMOTE(random_state=19)
    X, y = smote.fit_resample(X, y)

    # extracting best parameters from file
    parameters = extract_model_params("SVM Classifier", target_col)

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    metrics = {
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'precision_weighted': [],
        'recall_weighted': [],
        'f1_weighted': []
    }

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        svm_model = SVC(**parameters)
        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision_macro'].append(report['macro avg']['precision'])
        metrics['recall_macro'].append(report['macro avg']['recall'])
        metrics['f1_macro'].append(report['macro avg']['f1-score'])
        metrics['precision_weighted'].append(report['weighted avg']['precision'])
        metrics['recall_weighted'].append(report['weighted avg']['recall'])
        metrics['f1_weighted'].append(report['weighted avg']['f1-score'])

    avg_results = {key: [np.mean(vals)] for key, vals in metrics.items()}
    avg_results.update({
        'model': ["SVM Classifier"],
        'target': [target_col]
    })

    results_to_file(avg_results)

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

def ann_model(rep_df, target_col, k=4):
    """Fits and evaluates the best ANN Model for the target"""
    X, y = preprocess(rep_df, target_col)

    parameters = extract_model_params("ANN", target_col)

    batch = parameters['batch_size']
    epochs = parameters['epochs']
    learning_rate = parameters['learning_rate']
    units = parameters['units']

    X = np.array(X)
    y = np.array(y)

    is_regression = target_col in ['emg_pca_1', 'emg_pca_2']

    if not is_regression:
        smote = SMOTE(random_state=19)
        X, y = smote.fit_resample(X, y)

        num_classes = len(np.unique(y))
        y = to_categorical(y, num_classes=num_classes)
        metrics = {
            'accuracy': [],
            'precision_macro': [],
            'recall_macro': [],
            'f1_macro': [],
            'precision_weighted': [],
            'recall_weighted': [],
            'f1_weighted': []
        }
    else:
        metrics = {
            'mae': [],
            'mse': [],
            'rmse': [],
            'r2': []
        }

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if is_regression:
            y_scaler = StandardScaler()
            y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

            model = build_ann_regression(
                units=units,
                learning_rate=learning_rate,
                input_dim=X_train.shape[1],
            )
        else:
            model = build_ann_classification(
                units=units,
                output_shape=num_classes,
                learning_rate=learning_rate,
                input_dim=X_train.shape[1],
            )

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs,
                  batch_size=batch, verbose=0, callbacks=[early_stopping])

        y_pred = model.predict(X_test)

        if is_regression:
            report = regression_results(y_test, y_pred)
            for key in metrics:
                metrics[key].append(report[key])
        else:
            y_pred = np.argmax(y_pred, axis=1)
            y_test = np.argmax(y_test, axis=1)
            report = classification_report(y_test, y_pred, output_dict=True)

            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['precision_macro'].append(report['macro avg']['precision'])
            metrics['recall_macro'].append(report['macro avg']['recall'])
            metrics['f1_macro'].append(report['macro avg']['f1-score'])
            metrics['precision_weighted'].append(report['weighted avg']['precision'])
            metrics['recall_weighted'].append(report['weighted avg']['recall'])
            metrics['f1_weighted'].append(report['weighted avg']['f1-score'])

    avg_results = {key: [np.mean(vals)] for key, vals in metrics.items()}
    avg_results.update({
        'model': ["ANN"],
        'target': [target_col]
    })

    results_to_file(avg_results)


def xgb_model(rep_df, target_col, k=4):
    """Fits and evaluates the best XGB Model for the target"""
    X, y = preprocess(rep_df, target_col)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Applying SMOTE to prevent overfitting
    smote = SMOTE(random_state=19)
    X, y = smote.fit_resample(X, y)

    # extracting best parameters from file
    parameters = extract_model_params("XGBoost", target_col)

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    metrics = {
        'accuracy': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'precision_weighted': [],
        'recall_weighted': [],
        'f1_weighted': []
    }

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = xgb.XGBClassifier(**parameters)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision_macro'].append(report['macro avg']['precision'])
        metrics['recall_macro'].append(report['macro avg']['recall'])
        metrics['f1_macro'].append(report['macro avg']['f1-score'])
        metrics['precision_weighted'].append(report['weighted avg']['precision'])
        metrics['recall_weighted'].append(report['weighted avg']['recall'])
        metrics['f1_weighted'].append(report['weighted avg']['f1-score'])

    avg_results = {key: [np.mean(vals)] for key, vals in metrics.items()}
    avg_results.update({
        'model': ["XGB Classifier"],
        'target': [target_col]
    })

    results_to_file(avg_results)

def lasso_model(rep_df, target_col, k=4):
    """Fits and evaluates a Lasso Regression model for the target"""
    X, y = preprocess(rep_df, target_col)

    parameters = extract_model_params("Lasso Regression", target_col)

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    metrics = {
        'mae': [],
        'mse': [],
        'rmse': [],
        'r2': []
    }

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

        model = Lasso(**parameters, random_state=19)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = regression_results(y_test, y_pred)

        metrics['mae'].append(report['mae'])
        metrics['mse'].append(report['mse'])
        metrics['rmse'].append(report['rmse'])
        metrics['r2'].append(report['r2'])

    avg_results = {key: [np.mean(vals)] for key, vals in metrics.items()}
    avg_results.update({
        'model': ["Lasso Regression"],
        'target': [target_col]
    })

    results_to_file(avg_results)

def ridge_model(rep_df, target_col, k=4):
    """Fits and evaluates the best Ridge Regression Model for the target"""
    X, y = preprocess(rep_df, target_col)

    # extracting best parameters from file
    parameters = extract_model_params("Ridge Regression", target_col)

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    metrics = {
        'mae': [],
        'mse': [],
        'rmse': [],
        'r2': []
    }

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

        model = Ridge(**parameters, random_state=19)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = regression_results(y_test, y_pred)

        metrics['mae'].append(report['mae'])
        metrics['mse'].append(report['mse'])
        metrics['rmse'].append(report['rmse'])
        metrics['r2'].append(report['r2'])

    avg_results = {key: [np.mean(vals)] for key, vals in metrics.items()}
    avg_results.update({
        'model': ["Ridge Regression"],
        'target': [target_col]
    })

    results_to_file(avg_results)

def elastic_net_model(rep_df, target_col, k=4):
    """Fits and evaluates the best ElasticNet Regression Model for the target"""
    X, y = preprocess(rep_df, target_col)

    # extracting best parameters from file
    parameters = extract_model_params("ElasticNet Regression", target_col)

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    metrics = {
        'mae': [],
        'mse': [],
        'rmse': [],
        'r2': []
    }

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

        model = ElasticNet(**parameters, random_state=19)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = regression_results(y_test, y_pred)

        metrics['mae'].append(report['mae'])
        metrics['mse'].append(report['mse'])
        metrics['rmse'].append(report['rmse'])
        metrics['r2'].append(report['r2'])

    avg_results = {key: [np.mean(vals)] for key, vals in metrics.items()}
    avg_results.update({
        'model': ["ElasticNet Regression"],
        'target': [target_col]
    })

    results_to_file(avg_results)

def rf_reg_model(rep_df, target_col, k=4):
    """Fits and evaluates the best RF Regression Model for the target"""
    X, y = preprocess(rep_df, target_col)

    # extracting best parameters from file
    parameters = extract_model_params("RF Regression", target_col)

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    metrics = {
        'mae': [],
        'mse': [],
        'rmse': [],
        'r2': []
    }

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

        model = RandomForestRegressor(**parameters, random_state=19)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = regression_results(y_test, y_pred)

        metrics['mae'].append(report['mae'])
        metrics['mse'].append(report['mse'])
        metrics['rmse'].append(report['rmse'])
        metrics['r2'].append(report['r2'])

    avg_results = {key: [np.mean(vals)] for key, vals in metrics.items()}
    avg_results.update({
        'model': ["RF Regression"],
        'target': [target_col]
    })

    results_to_file(avg_results)
    
def svr_model(rep_df, target_col, k=4):
    """Fits and evaluates the best SVR Model for the target"""
    X, y = preprocess(rep_df, target_col)

    # extracting best parameters from file
    parameters = extract_model_params("SVR", target_col)

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    metrics = {
        'mae': [],
        'mse': [],
        'rmse': [],
        'r2': []
    }

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

        model = SVR(**parameters)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = regression_results(y_test, y_pred)

        metrics['mae'].append(report['mae'])
        metrics['mse'].append(report['mse'])
        metrics['rmse'].append(report['rmse'])
        metrics['r2'].append(report['r2'])

    avg_results = {key: [np.mean(vals)] for key, vals in metrics.items()}
    avg_results.update({
        'model': ["SVR"],
        'target': [target_col]
    })

    results_to_file(avg_results)

def xgb_reg_model(rep_df, target_col, k=4):
    """Fits and evaluates the best XGB Regression Model for the target"""
    X, y = preprocess(rep_df, target_col)

    # extracting best parameters from file
    parameters = extract_model_params("XGB Regressor", target_col)

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    metrics = {
        'mae': [],
        'mse': [],
        'rmse': [],
        'r2': []
    }

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_scaler = StandardScaler()
        y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

        model = xgb.XGBRegressor(**parameters)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = regression_results(y_test, y_pred)

        metrics['mae'].append(report['mae'])
        metrics['mse'].append(report['mse'])
        metrics['rmse'].append(report['rmse'])
        metrics['r2'].append(report['r2'])

    avg_results = {key: [np.mean(vals)] for key, vals in metrics.items()}
    avg_results.update({
        'model': ["XGB Regressor"],
        'target': [target_col]
    })

    results_to_file(avg_results)

def get_best_model(df, col):
    """Gets the row where the selection metric is best"""
    if col in ['emg_pca_1', 'emg_pca_2']:
        return df.sort_values('rmse').iloc[0], df.sort_values('rmse').iloc[0]['rmse']
    else:
        return df.sort_values('f1_weighted', ascending=False).iloc[0], df.sort_values('f1_weighted', ascending=False).iloc[0]['f1_weighted']

def select_best_models(filename="best_models.txt"):
    """Selects the best model for each target"""
    targets = ['emg_umap_km_class', 'emg_tsne_km_class', 'emg_pca_1', 'emg_pca_2']
    with open(filename, mode="w") as f:
        for target in targets:
            df = pd.read_csv(f"./results/{target}_model_eval.csv")
            best_row, best_score = get_best_model(df, target)
            f.write(f"{target} - {best_row['model']} - {best_score}\n")

if __name__=="__main__":
    set_up_files()
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
        ('svr', 'emg_pca_1'),
        ('svr', 'emg_pca_2'),
        ('xgb', 'emg_pca_1'),
        ('xgb', 'emg_pca_2'),
    ]

    for m, t in tqdm(options, colour="blue"):
        if m=="rf":
            if t in ['emg_pca_1', 'emg_pca_2']:
                rf_reg_model(rep_df, target_col=t)
            else:
                rf_model(rep_df, target_col=t)
        elif m=="lr":
            lr_model(rep_df, target_col=t)
        elif m=="svm":
            svm_model(rep_df, target_col=t)
        elif m=="ann":
            ann_model(rep_df, target_col=t)
        elif m=="xgb":
            if t in ['emg_pca_1', 'emg_pca_2']:
                xgb_reg_model(rep_df, target_col=t)
            else:
                xgb_model(rep_df, target_col=t)
        elif m=="las":
            lasso_model(rep_df, target_col=t)
        elif m=="rid":
            ridge_model(rep_df, target_col=t)
        elif m=="enet":
            elastic_net_model(rep_df, target_col=t)
        elif m=="svr":
            svr_model(rep_df, target_col=t)

    select_best_models()