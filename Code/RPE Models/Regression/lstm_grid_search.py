from plot_results import *

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
from itertools import product

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input, GRU
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import root_mean_squared_error

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def set_seed(seed=19):
    """Sets the random seeds for reproducibility"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

def best_to_file(best_params, model, score, emg, filename="best_lstm_parameters.txt"):
    """Stores the best parameters to file"""
    f = open(filename, "a")
    f.write(f"{model}: {best_params} -- Score: {score} -- EMG: {emg} \n")
    f.close()

def get_all_set_overlaps(rep_df, min_reps=7):
    """Gets all sequence subsets of exercise sets"""

    ### e.g. [1,2,3,4,5,6,7,8,9], min_reps=7 -> [[1,2,3,4,5,6,7], [2,3,4,5,6,7,8], [3,4,5,6,7,8,9]] ###

    rep_dict = rep_df.groupby("ID").apply(lambda x: x.to_dict(orient="records")).to_dict()
    X = []
    y = []

    for set_id in rep_dict.keys():
        set_len = len(rep_dict[set_id])
        for i in range((set_len + 1) - min_reps):
            section = rep_dict[set_id][i:i+min_reps]
            Xs = []
            ys = []

            for rep in section:
                ys.append(rep['RPE'])
                excluded_keys = ['RPE', 'ID', 'rep_num']
                rep_features = [v for k, v in rep.items() if k not in excluded_keys]
                Xs.append(rep_features)
            
            X.append(Xs)
            y.append(ys)

    return np.array(X), np.array(y)

def add_jitter(X, noise_level=0.02):
    """Incorporates random noise into the data"""
    noise = np.random.normal(0, noise_level, size=X.shape)
    return X + noise

def preprocess(rep_df, emg=True):
    """Prepares the data by removing columns or OHE"""
    if not emg:
        rep_df = rep_df.drop(['emg_pca_1', 'emg_pca_2', 'emg_tsne_km_class', 'emg_umap_km_class'], axis=1)
    else:
        cat_cols = ['emg_tsne_km_class', 'emg_umap_km_class']
        rep_df = pd.get_dummies(rep_df, columns=cat_cols, drop_first=True)

    return rep_df

def get_smallest_set(rep_df):
    """Gets the lowest number of reps in any set"""
    counts = []
    set_id = ""
    count = 0
    for index, row in rep_df.iterrows():
        if set_id != row['ID']:
            counts += [count]
            set_id = row['ID']
            count = 0

        count += 1
    
    counts += [count]
    counts = counts[1:]

    return min(counts)

def create_grouped_kfold_split(rep_df, min_reps=7, k=4, random_state=19):
    """Creates test train validation splits for kfold -- grouping so sets are contained within each set"""
    set_ids = rep_df['ID'].unique()
    np.random.seed(random_state)
    np.random.shuffle(set_ids)
    n_sets = len(set_ids)
    
    # Calculate the fold size
    fold_size = n_sets // k
    
    # Create k folds
    all_folds = []
    all_X = []
    all_y = []
    
    for i in range(k):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < k - 1 else n_sets
        fold_set_ids = set_ids[start_idx:end_idx]
        fold_df = rep_df[rep_df['ID'].isin(fold_set_ids)]
        all_folds.append(fold_df)
        
        # Get X and y for each fold
        X_fold, y_fold = get_all_set_overlaps(fold_df, min_reps)
        all_X.append(X_fold)
        all_y.append(y_fold)
    
    return all_folds, all_X, all_y

def build_model(layers=[(64,'l'), (8, 'd')], activation='relu', learning_rate=0.0001, min_reps=7, input_dim=59):
    """Building RNN model based on parameters given"""
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

    model.add(Dense(1, 'linear')) #output layer
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=learning_rate), metrics=[RootMeanSquaredError()])

    return model

def scale_3d_data(X_train, X_val, X_test):
    """Z Score Scales the 3d data"""
    n_train_samples, n_timesteps, n_features = X_train.shape
    
    X_train_reshaped = X_train.reshape(-1, n_features)
    
    scaler = StandardScaler() #defining the scaler
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    
    X_train_scaled = X_train_scaled.reshape(n_train_samples, n_timesteps, n_features)
    
    n_val_samples = X_val.shape[0]
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features))
    X_val_scaled = X_val_scaled.reshape(n_val_samples, n_timesteps, n_features)
    
    n_test_samples = X_test.shape[0]
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features))
    X_test_scaled = X_test_scaled.reshape(n_test_samples, n_timesteps, n_features)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler #returns the scaler for future use

def lstm_search(rep_df, emg=True, m="lstm", search="random", search_samples=150, k=4):
    """Performs a grid search across RNN architectures -- either LSTM or GRU"""

    param_grid = { 
        'epochs': [50, 100, 200, 300, 400, 500, 750, 1000],
        'learning_rate': [0.01, 0.001, 0.0001, 0.00005],
        'min_reps': [3,4,5,6,7],
        'activation': ['linear', 'relu'],
    }

    if m=="lstm":
        # LSTM parameter grid
        param_grid['layers'] = [
            [(32, 'l'), (16, 'd')],
            [(64, 'l'), (8, 'd')],
            [(128, 'l'), (32, 'd')],
            [(32, 'l'), (32, 'l'), (16, 'd')],
            [(64, 'l'), (64, 'l'), (32, 'd')],
            [(128, 'l'), (64, 'l'), (32, 'd')],
            [(16, 'l'), (4, 'd')],
            [(32, 'l'), (16, 'd'), (8, 'd')],
            [(64, 'l'), (32, 'd'), (16, 'd')],
            [(16, 'l'), (32, 'l'), (8, 'd')],
            [(32, 'l'), (8, 'd'), (4, 'd')],
            [(64, 'l'), (64, 'd')],
            [(128, 'l'), (16, 'd')],
            [(32, 'l'), (64, 'd')],
            [(32, 'l'), (32, 'l'), (32, 'l'), (16, 'd')]
        ]

    elif m=="gru":
        # GRU parameter grid
        param_grid['layers'] = [
            [(32, 'g'), (16, 'd')],
            [(64, 'g'), (8, 'd')],
            [(128, 'g'), (32, 'd')],
            [(32, 'g'), (32, 'g'), (16, 'd')],
            [(64, 'g'), (64, 'g'), (32, 'd')],
            [(128, 'g'), (64, 'g'), (32, 'd')],
            [(16, 'g'), (4, 'd')],
            [(32, 'g'), (16, 'd'), (8, 'd')],
            [(64, 'g'), (32, 'd'), (16, 'd')],
            [(16, 'g'), (32, 'g'), (8, 'd')],
            [(32, 'g'), (8, 'd'), (4, 'd')],
            [(64, 'g'), (64, 'd')],
            [(128, 'g'), (16, 'd')],
            [(32, 'g'), (64, 'd')],
            [(32, 'g'), (32, 'g'), (32, 'g'), (16, 'd')]
        ]

    param_combinations = list(product(*param_grid.values()))

    param_dicts = [dict(zip(param_grid.keys(), values)) for values in param_combinations]

    # selecting random subset of the grid
    if search == "random":
        param_dicts = random.sample(param_dicts, min(len(param_dicts), search_samples))

    best_params = None
    best_score = float('inf')

    for param_dict in tqdm(param_dicts, colour='red'):
        total_rmse = 0
        all_folds, all_X, all_y = create_grouped_kfold_split(rep_df, min_reps=param_dict['min_reps'])

        # manual kfold CV
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

            model = build_model(
                layers=param_dict['layers'],
                learning_rate=param_dict['learning_rate'],
                activation=param_dict['activation'],
                input_dim=X_train.shape[2],
                min_reps=param_dict['min_reps']
            )

            # early stopping to prevent overfitting
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=param_dict['epochs'], callbacks=[early_stopping], verbose=0)

            y_pred = model.predict(X_test)
            rmse = root_mean_squared_error(y_test, y_pred)

            total_rmse += rmse

        avg_rmse = total_rmse/k

        if avg_rmse < best_score:
            best_params = {
                'layers': param_dict['layers'],
                'activation': param_dict['activation'],
                'epochs' :param_dict['epochs'],
                'learning_rate': param_dict['learning_rate'],
                'min_reps': param_dict['min_reps'],
            }

            best_score = avg_rmse
        else:
            pass

    print("Best Parameters:", best_params)
    print("Best Accuracy Score:", best_score)

    best_to_file(best_params=best_params, model=f"{m.upper()} Regression", score=best_score, emg=emg)

if __name__=="__main__":
    rep_df = pd.read_csv("imu_plus_pred_emg.csv")
    rep_df = preprocess(rep_df)

    set_seed()

    lstm_search(rep_df, m="gru", emg=True, search="random", search_samples=50)
    lstm_search(rep_df, m="gru", emg=False, search="random", search_samples=50)
    lstm_search(rep_df, m="lstm", emg=True, search="random", search_samples=50)
    lstm_search(rep_df, m="lstm", emg=False, search="random", search_samples=50)
