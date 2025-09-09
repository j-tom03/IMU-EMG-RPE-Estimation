import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import re
import ast
from lstm_grid_search import add_jitter, create_grouped_kfold_split, scale_3d_data 
from plot_results import plot_confusion_matrix

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, root_mean_squared_error, r2_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input, GRU
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import xgboost as xgb
from sklearn.svm import SVC

def extract_model_params(model_name, emg, file_path="./best_parameters.txt"):
    """Reads best model file and finds parameter dictionary for specified model and EMG flag"""
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            if model_name in line and f"EMG: {str(emg)}" in line:
                # Try matching OrderedDict or normal dict
                match = re.search(rf'{re.escape(model_name)}: (OrderedDict\()?({{.*}})\)? -- Score: (.+) -- EMG: (.+)', line)
                if match:
                    try:
                        param_str = match.group(2)
                        param_dict = ast.literal_eval(param_str)
                        results.append(param_dict)
                    except Exception as e:
                        print(f"Error parsing line: {line}\n{e}")

    if not results:
        raise ValueError(f"No match found for model '{model_name}' with EMG = {emg} in file '{file_path}'")

    return results[0]

def set_up_file():
    """Resets the result output files"""

    f = open(f"./results/rpe_class_eval.csv", mode="w")
    f.write("accuracy,acc_in_1,precision_macro,recall_macro,f1_macro,precision_weighted,recall_weighted,f1_weighted,mae,rmse,abs_err_std,r2,model,target,emg\n")
    f.close()

def results_to_file(results_dict):
    """Stores metrics for each model to a file"""

    df = pd.DataFrame(results_dict)
    df.to_csv(f"./results/rpe_class_eval.csv", index=False, mode='a', header=False)

def preprocess(rep_df, emg=True):
    """Prepreocesses the dataset by removing columns and OHE categorical data"""
    rep_df = rep_df.drop(['ID', 'rep_num'], axis=1)

    if not emg:
        rep_df = rep_df.drop(['emg_tsne_km_class', 'emg_umap_km_class', 'emg_pca_1', 'emg_pca_2'], axis=1)
    else:
        cat_cols = ['emg_tsne_km_class', 'emg_umap_km_class']
        rep_df = pd.get_dummies(rep_df, columns=cat_cols, drop_first=True)

    X = rep_df.drop(["RPE"], axis=1)
    y = rep_df["RPE"]

    X = np.array(X)
    y = np.array(y)

    return X, y

def set_seed(seed=19):
    """Sets the random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def acc_within_1(y_test, y_pred):
    """Calculates the accuracy of predictions within +-1 rating"""
    if len(y_test.shape) != 1:
        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

    count = 0
    n = 0

    for yt, yp in zip(y_test, y_pred):
        n += 1
        if abs(yt - yp) <= 1:
            count += 1

    return count/n

def abs_err_std(y_test, y_pred):
    """Calculates the standard deviation of absolute error"""
    return np.std(np.abs(y_test - y_pred))

def rf_model(rep_df, emg=True, k=4):
    """Fits and evaluates an RF Model using KFold cross-validation"""
    X, y = preprocess(rep_df, emg=emg)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # extracting best parameters from file
    parameters = extract_model_params("RF Classifier", emg=emg)

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    metrics = {
        'accuracy': [],
        'acc_in_1': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'precision_weighted': [],
        'recall_weighted': [],
        'f1_weighted': [],
        'mae': [],
        'rmse': [],
        'abs_err_std': [],
        'r2': []
    }

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Applying SMOTE to prevent overfitting
        smote = SMOTE(random_state=19)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        rf = RandomForestClassifier(**parameters, random_state=19)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['mae'].append(mean_absolute_error(y_test, y_pred))
        metrics['rmse'].append(root_mean_squared_error(y_test, y_pred))
        metrics['abs_err_std'].append(abs_err_std(y_test, y_pred))
        metrics['r2'].append(r2_score(y_test, y_pred))
        metrics['acc_in_1'].append(acc_within_1(y_test, y_pred))
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
        'target': ['RPE'],
        'emg': emg
    })

    results_to_file(avg_results)

def lr_model(rep_df, emg=True, k=4):
    """Fits and evaluates the best LR Model for the target"""
    X, y = preprocess(rep_df, emg=emg)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # extracting best parameters from file
    parameters = extract_model_params("Logistic Regression Classifier", emg=emg)

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    metrics = {
        'accuracy': [],
        'acc_in_1': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'precision_weighted': [],
        'recall_weighted': [],
        'f1_weighted': [],
        'mae': [],
        'rmse': [],
        'abs_err_std': [],
        'r2': []
    }

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Applying SMOTE to prevent overfitting
        smote = SMOTE(random_state=19)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        logr = LogisticRegression(**parameters, random_state=19)
        logr.fit(X_train, y_train)
        y_pred = logr.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['mae'].append(mean_absolute_error(y_test, y_pred))
        metrics['rmse'].append(root_mean_squared_error(y_test, y_pred))
        metrics['abs_err_std'].append(abs_err_std(y_test, y_pred))
        metrics['r2'].append(r2_score(y_test, y_pred))
        metrics['acc_in_1'].append(acc_within_1(y_test, y_pred))
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
        'target': ["RPE"],
        'emg': emg,
    })

    results_to_file(avg_results)

def svm_model(rep_df, emg=True, k=4):
    """Fits and evaluates the best SVM Model for the target"""
    X, y = preprocess(rep_df, emg=emg)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # extracting best parameters from file
    parameters = extract_model_params("SVM Classifier", emg=emg)

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    metrics = {
        'accuracy': [],
        'acc_in_1': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'precision_weighted': [],
        'recall_weighted': [],
        'f1_weighted': [],
        'mae': [],
        'rmse': [],
        'abs_err_std': [],
        'r2': []
    }

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Applying SMOTE to prevent overfitting
        smote = SMOTE(random_state=19)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        svm_model = SVC(**parameters)
        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['mae'].append(mean_absolute_error(y_test, y_pred))
        metrics['rmse'].append(root_mean_squared_error(y_test, y_pred))
        metrics['abs_err_std'].append(abs_err_std(y_test, y_pred))
        metrics['r2'].append(r2_score(y_test, y_pred))
        metrics['acc_in_1'].append(acc_within_1(y_test, y_pred))
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
        'target': ["RPE"],
        'emg': emg
    })  

    results_to_file(avg_results)

def build_ann_classification(units, output_shape=11, learning_rate=0.001, input_dim=None):
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

def ann_model(rep_df, emg=True, k=4):
    """Fits and evaluates the best ANN Model for the target"""
    X, y = preprocess(rep_df, emg=emg)

    parameters = extract_model_params("ANN Classifier", emg=emg)

    batch = parameters['batch_size']
    epochs = parameters['epochs']
    learning_rate = parameters['learning_rate']
    units = parameters['units']

    num_classes = np.max(y) + 1
    metrics = {
        'accuracy': [],
        'acc_in_1': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'precision_weighted': [],
        'recall_weighted': [],
        'f1_weighted': [],
        'mae': [],
        'rmse': [],
        'abs_err_std': [],
        'r2': []
    }

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        smote = SMOTE(random_state=19)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        y_train = to_categorical(y_train, num_classes=num_classes)
        y_test = to_categorical(y_test, num_classes=num_classes)

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

        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)

        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['mae'].append(mean_absolute_error(y_test, y_pred))
        metrics['rmse'].append(root_mean_squared_error(y_test, y_pred))
        metrics['abs_err_std'].append(abs_err_std(y_test, y_pred))
        metrics['r2'].append(r2_score(y_test, y_pred))
        metrics['acc_in_1'].append(acc_within_1(y_test, y_pred))
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
        'target': ["RPE"],
        'emg': emg
    })

    results_to_file(avg_results)

def xgb_model(rep_df, emg=True, k=4):
    """Fits and evaluates the best XGB Model for the target"""
    X, y = preprocess(rep_df, emg=emg)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # extracting best parameters from file
    parameters = extract_model_params("XGBoost", emg=emg)

    kf = KFold(n_splits=k, shuffle=True, random_state=19)

    metrics = {
        'accuracy': [],
        'acc_in_1': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'precision_weighted': [],
        'recall_weighted': [],
        'f1_weighted': [],
        'mae': [],
        'rmse': [],
        'abs_err_std': [],
        'r2': []
    }

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Applying SMOTE to prevent overfitting
        smote = SMOTE(random_state=19)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        model = xgb.XGBClassifier(**parameters)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['mae'].append(mean_absolute_error(y_test, y_pred))
        metrics['rmse'].append(root_mean_squared_error(y_test, y_pred))
        metrics['abs_err_std'].append(abs_err_std(y_test, y_pred))
        metrics['r2'].append(r2_score(y_test, y_pred))
        metrics['acc_in_1'].append(acc_within_1(y_test, y_pred))
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
        'target': ["RPE"],
        'emg': emg
    })

    results_to_file(avg_results)

def build_rnn_model(layers=[(64,'l'), (8, 'd')], activation='relu', learning_rate=0.0001, min_reps=7, input_dim=59):
    """Building RNN model based onn parameters given"""
    model = Sequential()
    model.add(Input((min_reps, input_dim)))
    # dynamically assigning layers
    for i, (units, layer) in enumerate(layers):
        if layer == 'l':
            is_last_lstm = i == len(layers) - 1 or layers[i+1][1] != 'l'
            
            if is_last_lstm:
                model.add(LSTM(units, return_sequences=(not is_last_lstm)))
            else:
                model.add(LSTM(units, return_sequences=True))
        elif layer == 'g':
            is_last_lstm = i == len(layers) - 1 or layers[i+1][1] != 'g'
            
            if is_last_lstm:
                model.add(GRU(units, return_sequences=(not is_last_lstm)))
            else:
                model.add(GRU(units, return_sequences=True))
        elif layer == 'd':
            model.add(Dense(units, activation=activation))

        model.add(Dropout(0.1))

    model.add(Dense(11, activation='softmax')) #output layer
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

    return model

def rnn_model(rep_df, m="LSTM", emg=True, k=4):
    """Fits and evaluates the best RNN Model for the target"""
    X, y = preprocess(rep_df, emg=emg)

    parameters = extract_model_params(f"{m} Classification", emg=emg)

    epochs = parameters['epochs']
    learning_rate = parameters['learning_rate']
    layers = parameters['layers']
    activation = parameters['activation']
    min_reps = parameters['min_reps']

    num_classes = np.max(y) + 1
    metrics = {
        'accuracy': [],
        'acc_in_1': [],
        'precision_macro': [],
        'recall_macro': [],
        'f1_macro': [],
        'precision_weighted': [],
        'recall_weighted': [],
        'f1_weighted': [],
        'mae': [],
        'rmse': [],
        'abs_err_std': [],
        'r2': []
    }

    all_folds, all_X, all_y = create_grouped_kfold_split(rep_df, min_reps=min_reps)

    for i in range(k):
        test_idx = i
        val_idx = (i + 1) % k
        train_idxs = [x for x in range(k) if x != test_idx and x != val_idx]

        X_test = all_X[test_idx]
        y_test = all_y[test_idx]

        X_val = all_X[val_idx]
        y_val = all_y[val_idx]

        X_train = np.concatenate([all_X[j] for j in train_idxs], axis=0)
        y_train = np.concatenate([all_y[j] for j in train_idxs], axis=0)

        y_train = y_train[:, -1]
        y_val = y_val[:, -1]
        y_test = y_test[:, -1]

        y_train = y_train.astype(int)
        y_val = y_val.astype(int)
        y_test = y_test.astype(int)
    
        X_train = add_jitter(X_train) #adding jitter to data to prevent data leakage
        X_train, X_val, X_test, scaler = scale_3d_data(X_train, X_val, X_test) #z score scaling the data 

        model = build_rnn_model(
            layers=layers,
            learning_rate=learning_rate,
            activation=activation,
            input_dim=X_train.shape[2],
            min_reps=min_reps
        )

        # early stopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, callbacks=[early_stopping], verbose=0)

        y_pred = model.predict(X_test)

        y_pred = np.argmax(y_pred, axis=1)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics['mae'].append(mean_absolute_error(y_test, y_pred))
        metrics['rmse'].append(root_mean_squared_error(y_test, y_pred))
        metrics['abs_err_std'].append(abs_err_std(y_test, y_pred))
        metrics['r2'].append(r2_score(y_test, y_pred))
        metrics['acc_in_1'].append(acc_within_1(y_test, y_pred))
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision_macro'].append(report['macro avg']['precision'])
        metrics['recall_macro'].append(report['macro avg']['recall'])
        metrics['f1_macro'].append(report['macro avg']['f1-score'])
        metrics['precision_weighted'].append(report['weighted avg']['precision'])
        metrics['recall_weighted'].append(report['weighted avg']['recall'])
        metrics['f1_weighted'].append(report['weighted avg']['f1-score'])

    avg_results = {key: [np.mean(vals)] for key, vals in metrics.items()}
    avg_results.update({
        'model': [m],
        'target': ["RPE"],
        'emg': emg
    })

    results_to_file(avg_results)

if __name__=="__main__":
    set_up_file()
    set_seed()

    rep_df = pd.read_csv("imu_plus_pred_emg.csv")

    """options = [
        ('LSTM', True),
        ('LSTM', False),
        ('GRU', True),
        ('GRU', False),
        ('ann', True),
        ('ann', False),
        ('rf', True),
        ('rf', False),
        ('lr', True),
        ('lr', False),
        ('svm', True),
        ('svm', False),
        ('xgb', True),
        ('xgb', False)
    ]"""

    options = [
        ('rf', True),
        ('rf', False),
    ]

    for m, b in tqdm(options, colour="blue"):
        if m=="rf":
            rf_model(rep_df, emg=b)
        elif m=="lr":
            lr_model(rep_df, emg=b)
        elif m=="svm":
            svm_model(rep_df, emg=b)
        elif m=="ann":
            ann_model(rep_df, emg=b)
        elif m=="xgb":
            xgb_model(rep_df, emg=b)
        elif m=="LSTM" or m=="GRU":
            rnn_model(rep_df, m=m, emg=b)
