import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
from tqdm.autonotebook import tqdm
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
import ast
from scipy.signal import butter, filtfilt
from umap import UMAP
from sklearn.cluster import DBSCAN, KMeans

def get_sample_rate(filename):
    hz_df = pd.read_csv(filename, skiprows=[0,1,2,3,4])
    emg_hz = float(hz_df[" EMG 1 (mV)"][0][:-3])
    imu_hz = float(hz_df["ACC X (G)"][0][:-3])
    del hz_df
    return emg_hz, imu_hz

def get_collection_length(filename):
    collection_length_df = pd.read_csv(filename, skiprows=2)
    collection_length = float(collection_length_df.columns[1])
    del collection_length_df
    return collection_length

def get_jerk_zeros(df, axis, imu_hz):
    series = df[axis]
    times = [x/imu_hz for x in df.index]
    times_diff = np.diff(times, n=1)
    jerk = np.diff(series, n=1) / times_diff[0:len(series)]

    jerk = np.convolve(jerk.astype(float), np.ones(50)/50, mode='same')

    indexes = []
    for i in range(len(jerk)):
        if (jerk[i] <= 0.2) & (jerk[i] >= -0.2):
            indexes += [i]

    return indexes

def downsample_emg(df, emg_hz, imu_hz):
    nyquist_new = imu_hz / 2
    b, a = signal.butter(4, nyquist_new / (emg_hz / 2), btype='low')
    filtered_emg = signal.filtfilt(b, a, df[" EMG 1 (mV)"])
    downsampled_emg = signal.resample_poly(filtered_emg, up=int(imu_hz*10000), down=int(emg_hz*10000))
    df_downsampled_emg = df
    df_downsampled_emg = df_downsampled_emg.drop(" EMG 1 (mV)", axis=1)

    df_downsampled_emg = df_downsampled_emg.loc[df_downsampled_emg["ACC X (G)"].notna()]

    len_raw_imu = len(df_downsampled_emg["ACC X (G)"])
    if len_raw_imu > len(downsampled_emg):
        df_downsampled_emg.iloc[:len(downsampled_emg)]
    elif len_raw_imu < len(downsampled_emg):
        downsampled_emg = downsampled_emg[:len_raw_imu]
    else:
        pass

    df_downsampled_emg["EMG"] = downsampled_emg
    df_downsampled_emg.columns = ['ACC_X', 'ACC_Y', 'ACC_Z', 'GYRO_X', 'GYRO_Y', 'GYRO_Z', 'EMG']

    return df_downsampled_emg

def smooth_imu(df):
    df.astype(float)
    window_size = 50
    df['ACC_X'] = np.convolve(df['ACC_X'].astype(float), np.ones(window_size)/window_size, mode='same')
    df['ACC_Y'] = np.convolve(df['ACC_Y'].astype(float), np.ones(window_size)/window_size, mode='same')
    df['ACC_Z'] = np.convolve(df['ACC_Z'].astype(float), np.ones(window_size)/window_size, mode='same')

    df['GYRO_X'] = np.convolve(df['GYRO_X'].astype(float), np.ones(window_size)/window_size, mode='same')
    df['GYRO_Y'] = np.convolve(df['GYRO_Y'].astype(float), np.ones(window_size)/window_size, mode='same')
    df['GYRO_Z'] = np.convolve(df['GYRO_Z'].astype(float), np.ones(window_size)/window_size, mode='same')

    return df

def get_mids(index_pairs):
    out = []
    for tup in index_pairs:
        out += [round(0.5*(sum(tup)))]

    return out

def get_rep_indexes(df, axis, imu_hz):
    indexes = get_jerk_zeros(df, axis, imu_hz)
    pos_indexes = [x for x in indexes if df[axis].iloc[x]>0]
    neg_indexes = list(set(indexes) - set(pos_indexes))

    df["Peak"] = df.index.isin(pos_indexes)
    df["Trough"] = df.index.isin(neg_indexes)
    trough_indices = df.index[df['Trough']].to_list()
    midpoints = []
    start = 0
    for i in range(1, len(trough_indices)):
        if trough_indices[i] != trough_indices[i-1] + 1:
            midpoint = int(np.mean(trough_indices[start:i]))
            midpoints.append(midpoint)
            start = i

    if trough_indices:
        midpoint = int(np.mean(trough_indices[start:]))
        midpoints.append(midpoint)

    peaks_troughs = df.loc[df['Peak'] | df['Trough']]
    peaks_troughs = peaks_troughs.copy()
    peaks_troughs['Peak-Trough'] = np.where(
        peaks_troughs["Peak"], "Peak", np.where(peaks_troughs["Trough"], "Trough", None)
    )

    num_peak_troughs = len(peaks_troughs.index)
    peak_periods = []
    trough_periods = []
    most_recent, peak_start, trough_start = None, None, None
    for i in range(num_peak_troughs):
        current = peaks_troughs['Peak-Trough'].iloc[i]
        if current == 'Trough' and current != most_recent:
            if peak_start:
                peak_periods += [(peak_start, peaks_troughs.index[i-1])]
            most_recent = current
            trough_start = peaks_troughs.index[i]
        elif current == 'Peak' and current != most_recent:
            if trough_start:
                trough_periods += [(trough_start, peaks_troughs.index[i-1])]
            most_recent = current
            peak_start = peaks_troughs.index[i]
        else:
            pass

        if i==(num_peak_troughs-1):
            if current=='Trough':
                trough_periods += [(trough_start, peaks_troughs.index[i-1])]
            elif current=='Peak':
                peak_periods += [(peak_start, peaks_troughs.index[i-1])]

    peak_mid_indexes = get_mids(peak_periods)
    start_rep_indexes = get_mids(trough_periods)

    return peak_mid_indexes, start_rep_indexes


def plot_rep_indexes(filename, peak_mid_indexes, start_rep_indexes, df, folder):
    plt.figure(figsize=(20, 15))

    plt.plot(df.index, df["EMG"], color="yellow")
    plt.plot(df.index, df["ACC_X"], color="red")
    plt.plot(df.index, df["ACC_Y"], color="green")
    plt.plot(df.index, df["ACC_Z"], color="blue")

    plt.vlines(peak_mid_indexes, ymin=df["ACC_X"].min(), 
            ymax=df["ACC_X"].max(), color="blue", linestyle="--", label="Rep Mid")

    plt.vlines(start_rep_indexes, ymin=df["ACC_X"].min(), 
            ymax=df["ACC_X"].max(), color="orange", linestyle="--", label="Rep Start")

    plt.legend()
    plt.savefig(folder+filename+"_reps_marked.jpg")
    plt.close()

def put_indexes_to_file(start_rep_indexes, peak_mid_indexes, filename, count_dict, df):
    if len(peak_mid_indexes)==count_dict[filename]:
        plot_rep_indexes(filename, peak_mid_indexes, start_rep_indexes, df, "./correct/")
    else:
        if len(start_rep_indexes)==count_dict[filename]:
            start_rep_indexes, peak_mid_indexes = peak_mid_indexes, start_rep_indexes
            plot_rep_indexes(filename, peak_mid_indexes, start_rep_indexes, df, "./correct/")
        else:
            plot_rep_indexes(filename, peak_mid_indexes, start_rep_indexes, df, "./incorrect/")
    
    f = open('./rep_indexes.txt', mode='a')
    text = filename+", "+str(start_rep_indexes)+", "+str(peak_mid_indexes)+", "+str(len(peak_mid_indexes))+", "+str(count_dict[filename])+" \n"
    f.write(text)
    f.close()

    if len(peak_mid_indexes)==count_dict[filename] or len(start_rep_indexes)==count_dict[filename]:
        return 1
    else:
        return 0
    
def line_to_list(line):
    out = []
    s = ""
    ignore = False

    for char in line:
        if not ignore and char==",":
            out += [s]
            s = ""
            continue
        elif char==" ":
            continue 
        else:
            if char=="[":
                ignore = True
            elif char=="]":
                ignore = False

        s += str(char)

    out += [s]
    out[1] = ast.literal_eval(out[1])
    out[2] = ast.literal_eval(out[2])
    return out

    
def get_rep_indexes_from_file(filename):
    info = {}
    f = open(filename, "r")
    for line in f:
        line_list = line_to_list(line)
        info[line_list[0]] = {
            "start_indexes": line_list[1],
            "peak_indexes": line_list[2],
            "counted": line_list[3]
            }
        
    return info

def get_rep_dataset(filename):
    df = pd.read_csv(filename)
    return df

def save_rep_dataset(filename, df):
    df.to_csv(filename, index=False)

def get_index_tuples(start_indexes, peak_indexes):
    rep_tuples = []
    for i in range(len(peak_indexes)):
        rep_tuples += [(start_indexes[i], peak_indexes[i], start_indexes[i+1])]

    return rep_tuples

def read_rpes(filename):
    df = pd.read_csv(filename, skiprows=[1])
    counts_df = pd.read_csv(filename, nrows=1)
    count_dict = counts_df.iloc[0].to_dict()
    return df, count_dict

def get_rep_length(indexes, hz):
    start, end = indexes
    length = (end-start)/hz
    return length

def poly_regression_fit_score(x, y):
    coeffs = np.polyfit(x, y, 2)
    poly = np.poly1d(coeffs)
    y_pred = poly(x)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def bandpass_emg_filter(signal, lowcut=20, highcut=450, fs=1000, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def smooth_emg_signal(signal, window_size=50):
    return np.convolve(np.abs(signal), np.ones(window_size)/window_size, mode='same')

def train_pca_model(feature_matrix, n_components=1):
    pca = PCA(n_components=n_components)
    pca.fit(feature_matrix)
    return pca

def train_ica_model(feature_matrix, n_components=1):
    ica = FastICA(n_components=n_components)
    ica.fit(feature_matrix)
    return ica

def train_umap_model(feature_matrix, n_components=1):
    umap = UMAP(n_components=n_components, random_state=42)
    umap.fit(feature_matrix)
    return umap

def train_tsne_model(feature_matrix, n_components=2):
    tsne = TSNE(n_components=n_components)
    X_transformed = tsne.fit_transform(feature_matrix)
    return tsne, X_transformed

def get_emg_features(raw_emg_series):
    emg_series_filtered = bandpass_emg_filter(raw_emg_series)
    emg_series = smooth_emg_signal(emg_series_filtered)

    mav = np.mean(np.abs(emg_series))
    rms = np.sqrt(np.mean(emg_series**2))
    iemg = np.sum(np.abs(emg_series))
    var = np.var(emg_series)
    zc = np.sum(np.diff(np.sign(emg_series)) != 0)
    wl = np.sum(np.abs(np.diff(emg_series)))
    ssc = np.sum(np.diff(np.sign(np.diff(emg_series))) != 0)
    mean_amp = np.mean(emg_series)
    peak_amp = np.max(np.abs(emg_series))
    
    return (mav, rms, iemg, var, zc, wl, ssc, mean_amp, peak_amp)

def get_info_from_reps(rep_indexes, df, set_name, imu_hz, rpe_list):
    info = []
    rep = 0
    for start, mid, end in rep_indexes:
        rpe = rpe_list[rep]
        section = df.loc[start:end]
        up_section = df.loc[start:mid]
        down_section = df.loc[mid:end]

        indexes = [x for x in range(round(start), round(end+1))]
        times = [x/imu_hz for x in indexes]

        up_indexes = [x for x in range(round(start), round(mid+1))]
        up_times = [x/imu_hz for x in up_indexes]
        times_diff_up = np.diff(up_times, n=1)

        down_indexes = [x for x in range(round(mid), round(end+1))]
        down_times = [x/imu_hz for x in down_indexes]
        times_diff_down = np.diff(down_times, n=1)
        jerk_x_up = np.diff(up_section["ACC_X"], n=1) / times_diff_up[0:len(up_section["ACC_X"])-1]
        jerk_x_down = np.diff(down_section["ACC_X"], n=1) / times_diff_down[0:len(down_section["ACC_X"])-1]

        jerk_y_up = np.diff(up_section["ACC_Y"], n=1) / times_diff_up[0:len(up_section["ACC_Y"])-1]
        jerk_y_down = np.diff(down_section["ACC_Y"], n=1) / times_diff_down[0:len(down_section["ACC_Y"])-1]

        jerk_z_up = np.diff(up_section["ACC_Z"], n=1) / times_diff_up[0:len(up_section["ACC_Z"])-1]
        jerk_z_down = np.diff(down_section["ACC_Z"], n=1) / times_diff_down[0:len(down_section["ACC_Z"])-1]

        up_length = get_rep_length((start, mid), imu_hz)
        down_length = get_rep_length((mid, end), imu_hz)

        emg_mav, emg_rms, emg_iemg, emg_var, emg_zc, emg_wl, emg_ssc,  emg_mean_amp, emg_peak_amp = get_emg_features(section['EMG'])

        features = {
            "ID" : set_name,
            "rep_num" : rep,
            "rep_length" : get_rep_length((start, end), imu_hz),
            "up_length" : up_length,
            "down_length" : down_length,
            "up_down_ratio" : (up_length/down_length),

            "acc_x_avg_up" : up_section['ACC_X'].mean(),
            "acc_x_std_up" : up_section['ACC_X'].std(),
            "acc_x_max_up" : up_section['ACC_X'].max(),
            "acc_x_min_up" : up_section['ACC_X'].min(),
            "jerk_x_avg_up": np.mean(jerk_x_up),
            "jerk_x_std_up": np.std(jerk_x_up),

            "acc_x_avg_down" : down_section['ACC_X'].mean(),
            "acc_x_std_down" : down_section['ACC_X'].std(),
            "acc_x_max_down" : down_section['ACC_X'].max(),
            "acc_x_min_down" : down_section['ACC_X'].min(),
            "jerk_x_avg_down": np.mean(jerk_x_down),
            "jerk_x_std_down": np.std(jerk_x_down),

            "acc_x_p2p" : section['ACC_X'].max() - section['ACC_X'].min(),
            "acc_x_r2"  : poly_regression_fit_score(indexes, section['ACC_X'].tolist()),
            
            "acc_y_avg_up" : up_section['ACC_Y'].mean(),
            "acc_y_std_up" : up_section['ACC_Y'].std(),
            "acc_y_max_up" : up_section['ACC_Y'].max(),
            "acc_y_min_up" : up_section['ACC_Y'].min(),
            "jerk_y_avg_up": np.mean(jerk_y_up),
            "jerk_y_std_up": np.std(jerk_y_up),

            "acc_y_avg_down" : down_section['ACC_Y'].mean(),
            "acc_y_std_down" : down_section['ACC_Y'].std(),
            "acc_y_max_down" : down_section['ACC_Y'].max(),
            "acc_y_min_down" : down_section['ACC_Y'].min(),
            "jerk_y_avg_down": np.mean(jerk_y_down),
            "jerk_y_std_down": np.std(jerk_y_down),

            "acc_y_p2p" : section['ACC_Y'].max() - section['ACC_Y'].min(),
            "acc_y_r2"  : poly_regression_fit_score(indexes, section['ACC_Y'].tolist()),

            "acc_z_avg_up" : up_section['ACC_Z'].mean(),
            "acc_z_std_up" : up_section['ACC_Z'].std(),
            "acc_z_max_up" : up_section['ACC_Z'].max(),
            "acc_z_min_up" : up_section['ACC_Z'].min(),
            "jerk_z_avg_up": np.mean(jerk_z_up),
            "jerk_z_std_up": np.std(jerk_z_up),

            "acc_z_avg_down" : down_section['ACC_Z'].mean(),
            "acc_z_std_down" : down_section['ACC_Z'].std(),
            "acc_z_max_down" : down_section['ACC_Z'].max(),
            "acc_z_min_down" : down_section['ACC_Z'].min(),
            "jerk_z_avg_down": np.mean(jerk_z_down),
            "jerk_z_std_down": np.std(jerk_z_down),

            "acc_z_p2p" : section['ACC_Z'].max() - section['ACC_Z'].min(),
            "acc_z_r2"  : poly_regression_fit_score(indexes, section['ACC_Z'].tolist()),

            "gyro_x_avg" : section['GYRO_X'].mean(),
            "gyro_x_std" : section['GYRO_X'].std(),
            "gyro_x_r2"  : poly_regression_fit_score(indexes, section['GYRO_X'].tolist()),

            "gyro_y_avg" : section['GYRO_Y'].mean(),
            "gyro_y_std" : section['GYRO_Y'].std(),
            "gyro_y_r2"  : poly_regression_fit_score(indexes, section['GYRO_Y'].tolist()),

            "gyro_z_avg" : section['GYRO_Z'].mean(),
            "gyro_z_std" : section['GYRO_Z'].std(),
            "gyro_z_r2"  : poly_regression_fit_score(indexes, section['GYRO_Z'].tolist()),

            "emg_mav" : emg_mav,
            "emg_rms" : emg_rms,
            "emg_iemg": emg_iemg,
            "emg_var" : emg_var,
            "emg_zc" : emg_zc,
            "emg_wl" : emg_wl,
            "emg_ssc" : emg_ssc,
            "emg_mean_amp" : emg_mean_amp,
            "emg_peak_amp" : emg_peak_amp,

            "emg_pca_1" : "",
            "emg_pca_2" : "",
            "emg_umap_1" : "",
            "emg_umap_2" : "",
            "emg_umap_km_class" : "",
            "emg_umap_db_class": "",
            "emg_tsne_1" : "",
            "emg_tsne_2" : "",
            "emg_tsne_km_class": "",
            "emg_tsne_db_class": "",

            "RPE" : rpe,
        }

        info += [features]
        rep += 1

    return pd.DataFrame(info)

def set_up_rep_dataset(filename):
    columns = {'ID':[], 'rep_num':[], 'rep_length':[], 'up_length':[], 'down_length':[], 'up_down_ratio':[],
       'acc_x_avg_up':[], 'acc_x_std_up':[], 'acc_x_max_up':[], 'acc_x_min_up':[],
       'jerk_x_avg_up':[], 'jerk_x_std_up':[], 'acc_x_avg_down':[], 'acc_x_std_down':[],
       'acc_x_max_down':[], 'acc_x_min_down':[], 'jerk_x_avg_down':[],
       'jerk_x_std_down':[], 'acc_x_p2p':[], 'acc_x_r2':[], 'acc_y_avg_up':[],
       'acc_y_std_up':[], 'acc_y_max_up':[], 'acc_y_min_up':[], 'jerk_y_avg_up':[],
       'jerk_y_std_up':[], 'acc_y_avg_down':[], 'acc_y_std_down':[], 'acc_y_max_down':[],
       'acc_y_min_down':[], 'jerk_y_avg_down':[], 'jerk_y_std_down':[], 'acc_y_p2p':[],
       'acc_y_r2':[], 'acc_z_avg_up':[], 'acc_z_std_up':[], 'acc_z_max_up':[],
       'acc_z_min_up':[], 'jerk_z_avg_up':[], 'jerk_z_std_up':[], 'acc_z_avg_down':[],
       'acc_z_std_down':[], 'acc_z_max_down':[], 'acc_z_min_down':[], 'jerk_z_avg_down':[],
       'jerk_z_std_down':[], 'acc_z_p2p':[], 'acc_z_r2':[], 'gyro_x_avg':[], 'gyro_x_std':[],
       'gyro_x_r2':[], 'gyro_y_avg':[], 'gyro_y_std':[], 'gyro_y_r2':[], 'gyro_z_avg':[],
       'gyro_z_std':[], 'gyro_z_r2':[], "emg_mav":[],"emg_rms":[],"emg_iemg":[],"emg_var":[],
        "emg_zc":[],"emg_wl":[],"emg_ssc":[],"emg_mean_amp":[],"emg_peak_amp":[],'emg_pca_1':[],
        'emg_pca_2':[],'emg_umap_1':[],'emg_umap_2':[],'emg_tsne_1':[],'emg_tsne_2':[],
        'emg_umap_km_class':[],'emg_tsne_km_class':[],'emg_umap_db_class':[], 'emg_tsne_db_class':[], 'RPE':[]}
    df = pd.DataFrame(columns)
    df.to_csv(filename, index=False)

def make_pca_model(rep_dataset_df):
    feature_matrix = []
    for _, row in rep_dataset_df.iterrows():
        feature_matrix.append([row["emg_mav"], row["emg_rms"], row["emg_iemg"], row["emg_var"], row["emg_zc"], row["emg_wl"], row["emg_ssc"], row["emg_mean_amp"], row["emg_peak_amp"]])

    pca_model = train_pca_model(np.array(feature_matrix), 2)
    return pca_model

def make_ica_model(rep_dataset_df):
    feature_matrix = []
    for _, row in rep_dataset_df.iterrows():
        feature_matrix.append([row["emg_mav"], row["emg_rms"], row["emg_iemg"], row["emg_var"], row["emg_zc"], row["emg_wl"], row["emg_ssc"], row["emg_mean_amp"], row["emg_peak_amp"]])

    ica_model = train_ica_model(np.array(feature_matrix), 2)
    return ica_model

def make_umap_model(rep_dataset_df):
    feature_matrix = []
    for _, row in rep_dataset_df.iterrows():
        feature_matrix.append([row["emg_mav"], row["emg_rms"], row["emg_iemg"], row["emg_var"], row["emg_zc"], row["emg_wl"], row["emg_ssc"], row["emg_mean_amp"], row["emg_peak_amp"]])

    umap_model = train_umap_model(np.array(feature_matrix), 2)
    return umap_model

def make_tsne_model(rep_dataset_df):
    feature_matrix = []
    for _, row in rep_dataset_df.iterrows():
        feature_matrix.append([row["emg_mav"], row["emg_rms"], row["emg_iemg"], row["emg_var"], row["emg_zc"], row["emg_wl"], row["emg_ssc"], row["emg_mean_amp"], row["emg_peak_amp"]])

    tsne_model, X_tsne = train_tsne_model(np.array(feature_matrix))
    return tsne_model, X_tsne

def emg_stim_to_dataset(rep_dataset_df, file_path, umap_model, pca_model, tsne_embeddings, rep_tuples):
    set_df = pd.read_csv(file_path)
    filename = file_path[16:-11]
    rep = 0
    for start, _, end in rep_tuples:
        section = set_df.loc[start:end]
        features = np.array(get_emg_features(section['EMG'])).reshape(1, -1)
        pca1, pca2 = tuple(pca_model.transform(features)[0, i] for i in range(2))
        umap1, umap2 = tuple(umap_model.transform(features)[0, i] for i in range(2))
        tsne1, tsne2 = tsne_embeddings.get((filename, rep), (None, None))
        rep_dataset_df.loc[(rep_dataset_df['ID'] == filename) & (rep_dataset_df['rep_num'] == rep), ['emg_pca_1', 'emg_pca_2','emg_umap_1', 'emg_umap_2','emg_tsne_1', 'emg_tsne_2']] = pca1, pca2, umap1, umap2, tsne1, tsne2
        rep += 1
    
    return rep_dataset_df

def kmeans_emg_train(dim_method, rep_df, n_clusters=4, random_state=19):
    X = rep_df[['emg_'+dim_method+'_1', 'emg_'+dim_method+'_2']].values
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    return kmeans, cluster_labels

def dbscan_emg_train(dim_method, rep_df, eps=0.5, min_samples=5):
    X = rep_df[['emg_'+dim_method+'_1', 'emg_'+dim_method+'_2']].values
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X)

    return dbscan, cluster_labels

def plot_clusters(rep_df, dim_method, cluster_method):
    x_col, y_col = f"emg_{dim_method}_1", f"emg_{dim_method}_2"
    label_col = f"emg_{dim_method}_{cluster_method}_class"
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(rep_df[x_col], rep_df[y_col], c=rep_df[label_col], cmap="viridis", edgecolor="k", s=50)
    plt.colorbar(scatter, label="Cluster Label")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{dim_method.upper()} - {cluster_method.upper()} Clustering")
    plt.show()


def compute(file_path, down_dir, smooth_dir, count_dict):
    set_name = file_path[16:-4]
    df = pd.read_csv(file_path, skiprows=[0,1,2,3,4,6])
    emg_hz, imu_hz = get_sample_rate(file_path)
    downsampled_df = downsample_emg(df, emg_hz, imu_hz)
    downsampled_df.to_csv(down_dir+set_name+"_down.csv")
    downsampled_df = pd.read_csv(down_dir+set_name+"_down.csv")
    smoothed_df = smooth_imu(downsampled_df)
    smoothed_df.to_csv(smooth_dir+set_name+"_smooth.csv")
    smoothed_df = pd.read_csv(smooth_dir+set_name+".csv")
    start_rep_indexes, peak_mid_indexes = get_rep_indexes(smoothed_df, "ACC_X", imu_hz)
    put_indexes_to_file(start_rep_indexes, peak_mid_indexes, set_name, count_dict)
    rep_index_dict = get_rep_indexes_from_file("rep_indexes_corrected.txt")
    rpes = rpe_df[set_name].to_list()
    rpes = [int(x) for x in rpes if not np.isnan(x)]
    set_info = rep_index_dict[set_name]
    rep_tuples = get_index_tuples(set_info["start_indexes"], set_info["peak_indexes"])
    info = get_info_from_reps(rep_tuples, df, set_name, imu_hz, rpes)
    rep_df = get_rep_dataset("rep_dataset.csv")
    df_merged = pd.concat([rep_df, info], ignore_index=True)
    save_rep_dataset("rep_dataset.csv", df_merged)

def emg_feature_imputing(rep_df):
    print("Making PCA Model")
    pca_model = make_pca_model(rep_df)
    print("Making ICA Model")
    ica_model = make_ica_model(rep_df)
    print("Making UMAP Model")
    umap_model = make_umap_model(rep_df)
    print("Making T-SNE Model")
    _, X_tsne = make_tsne_model(rep_df)

    all_filenames_and_reps = rep_df[['ID', 'rep_num']].values.tolist()
    tsne_embeddings = {(filename, rep): (X_tsne[i, 0], X_tsne[i, 1]) 
                   for i, (filename, rep) in enumerate(all_filenames_and_reps)}

    return pca_model, ica_model, umap_model, tsne_embeddings

def emg_targets_to_file(rep_index_dict, rep_df, umap_model, pca_model, tsne_embeddings):
    print("Models trained - making dataset")
    for file in tqdm(files):
        set_name = file[16:-11]
        set_info = rep_index_dict[set_name]
        rep_tuples = get_index_tuples(set_info["start_indexes"], set_info["peak_indexes"])
        rep_df = emg_stim_to_dataset(rep_df, file, umap_model, pca_model, tsne_embeddings, rep_tuples)

    save_rep_dataset("rep_dataset.csv", rep_df)

def train_emg_clustering(rep_df):
    _, k_means_labels_umap = kmeans_emg_train("umap", rep_df)
    _, k_means_labels_tsne = kmeans_emg_train("tsne", rep_df)

    _, dbscan_labels_umap = dbscan_emg_train("umap", rep_df)
    _, dbscan_labels_tsne = dbscan_emg_train("tsne", rep_df, eps=0.01, min_samples=5)

    rep_df['emg_umap_km_class'] = k_means_labels_umap
    rep_df['emg_tsne_km_class'] = k_means_labels_tsne
    rep_df['emg_umap_db_class'] = dbscan_labels_umap
    rep_df['emg_tsne_db_class'] = dbscan_labels_tsne

    save_rep_dataset("rep_dataset.csv", rep_df)

def plot_all_clusters():
    cluster_methods = ["db", "km"]
    dim_red_methods = ["tsne", "umap"]
    for c in cluster_methods:
        for d in dim_red_methods:
            plot_clusters(rep_df, d, c)

if __name__=="__main__":
    directory = './Data'
    down_dir = './down_data/'
    smooth_dir = './smoothed_data/'

    rpe_df, count_dict = read_rpes("rpes.csv")

    set_up_rep_dataset("rep_dataset.csv")

    files = [entry.path for entry in os.scandir(smooth_dir) if entry.is_file()]

    for file_path in tqdm(files):
        compute(file_path, down_dir, smooth_dir, count_dict)

    rep_df = get_rep_dataset("rep_dataset.csv")
    rep_index_dict = get_rep_indexes_from_file("rep_indexes_corrected.txt")

    pca_model, ica_model, umap_model, tsne_embeddings = emg_feature_imputing(rep_df)

    emg_targets_to_file(rep_index_dict, rep_df, umap_model, pca_model, tsne_embeddings)
    rep_df = get_rep_dataset("rep_dataset.csv")
    train_emg_clustering(rep_df)
    plot_all_clusters(rep_df)
