from plot_results import *

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import re
import ast
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import xgboost as xgb
from sklearn.svm import SVC


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

def store_model(model, target):
    """Stores the model to the models file"""

    dump(model, f"./models/{target}.pkl")


def pca_1_model(rep_df, target_col="emg_pca_1"):
    """Fits the best XGB Regression Model for the EMG PCA 1"""
    X, y = preprocess(rep_df, target_col)

    # extracting best parameters from file
    parameters = extract_model_params("XGB Regressor", target_col)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    model = xgb.XGBRegressor(**parameters)
    model.fit(X, y)

    store_model(model, target_col)

def pca_2_model(rep_df, target_col="emg_pca_2"):
    """Fits the best ANN Regression Model for the EMG PCA 2"""
    X, y = preprocess(rep_df, target_col)

    # extracting best parameters from file
    parameters = extract_model_params("ANN", target_col)

    batch = parameters['batch_size']
    epochs = parameters['epochs']
    learning_rate = parameters['learning_rate']
    units = parameters['units']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=19)

    ann = Sequential()
    ann.add(Input(shape=(X.shape[1],)))
    ann.add(Dense(units=units[0], activation='relu'))
    for unit in units[1:]: #dynamically adding layers
        ann.add(Dense(units=unit, activation='relu'))
        ann.add(Dropout(0.2)) #dropout for overfitting prevention

    ann.add(Dense(units=1, activation='linear')) #output layer
    optimiser = Adam(learning_rate=learning_rate)
    ann.compile(optimizer = optimiser, loss = 'mse', metrics = ["mae", RootMeanSquaredError()])

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    ann.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch, verbose=0, callbacks=[early_stopping])

    ann.save(f"./models/{target_col}.keras")

def umap_km_model(rep_df, target_col="emg_umap_km_class"):
    """Fits the best RF Classifier Model for the EMG UMAP label"""
    X, y = preprocess(rep_df, target_col)

    # extracting best parameters from file
    parameters = extract_model_params("RF Classifier", target_col)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    smote = SMOTE(random_state=19)
    X, y = smote.fit_resample(X, y)

    model = RandomForestClassifier(**parameters, random_state=19)
    model.fit(X, y)

    store_model(model, target_col)

def tsne_km_model(rep_df, target_col="emg_tsne_km_class"):
    """Fits the best XGB Classifier Model for the EMG tSNE label"""

    X, y = preprocess(rep_df, target_col)

    # extracting best parameters from file
    parameters = extract_model_params("XGBoost", target_col)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    smote = SMOTE(random_state=19)
    X, y = smote.fit_resample(X, y)

    model = xgb.XGBClassifier(**parameters)
    model.fit(X, y)

    store_model(model, target_col)

def build_model(rep_df, model):
    if model == "emg_pca_1":
        pca_1_model(rep_df)
    elif model == "emg_pca_2":
        pca_2_model(rep_df)
    elif model == "emg_umap_km_class":
        umap_km_model(rep_df)
    elif model == "emg_tsne_km_class":
        tsne_km_model(rep_df)

if __name__=="__main__":
    rep_df = pd.read_csv("rep_dataset.csv")

    models = [
        "emg_pca_1",
        "emg_pca_2",
        "emg_umap_km_class",
        "emg_tsne_km_class"
    ]

    for model in tqdm(models, colour='green'):
        build_model(rep_df, model)

        

