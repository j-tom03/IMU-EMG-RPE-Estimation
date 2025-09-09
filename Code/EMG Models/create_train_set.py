import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
from joblib import dump, load
from keras.models import load_model

from sklearn.preprocessing import StandardScaler


def remove_cols(rep_df):
    return rep_df.drop(['emg_mav', 'emg_rms', 'emg_iemg', 'emg_var', 'emg_zc', 'emg_wl', 'emg_ssc', 'emg_mean_amp', 'emg_peak_amp', 'emg_umap_1', 'emg_umap_2', 'emg_tsne_1', 'emg_tsne_2', "emg_umap_km_class", "emg_tsne_km_class", "emg_umap_db_class", "emg_tsne_db_class", "emg_pca_1", "emg_pca_2"], axis=1)

if __name__=="__main__":
    # running each of the saved models on the data to add the predicted labels to a new dataset
    rep_df = pd.read_csv("rep_dataset.csv")
    rep_df = remove_cols(rep_df)

    X = rep_df.drop(['ID', 'rep_num', 'RPE'], axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    directory = './models'

    files = [entry.path for entry in os.scandir(directory) if entry.is_file()]

    for file_path in tqdm(files, colour='YELLOW'):
        
        if file_path.endswith(".keras"):
            name = file_path[9:-6]
            model = load_model(file_path)
            pred = model.predict(X)
        elif file_path.endswith(".pkl"):
            name = file_path[9:-4]
            model = load(file_path)
            pred = model.predict(X)
        else:
            continue

        rep_df[name] = pred

    # storing to a new csv
    rep_df.to_csv("imu_plus_pred_emg.csv", index=False)