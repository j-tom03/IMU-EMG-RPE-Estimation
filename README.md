# RPE Prediction Project  

This repository contains the code and data for my final year Computer Science project: **Estimation of Resistance Training RPE Using Inertial Sensors, Electromyography, and Machine Learning**.  

## Overview  

The project investigates whether **machine learning models** can accurately predict the **Rate of Perceived Exertion (RPE)** during resistance training.  
- Data was collected using **Inertial Measurement Units (IMUs)** (accelerometer, gyroscope) and **surface Electromyography (EMG)** sensors.  
- Repetition-level features were extracted from the raw sensor data.  
- Models were developed to predict muscle stimulation (from EMG features) and RPE (from IMU + EMG-derived features).  
- Both **classification and regression** approaches were tested, including Random Forests, XGBoost, Artificial Neural Networks, and RNNs (LSTM, GRU).  

Best performing model:  
- **Random Forest Classifier with EMG features**  
- Achieved **±1 RPE accuracy of 85.9%** and **F1 score of 0.442**.

### Code  

The `code/` folder contains:  
- **preprocessing/**: scripts for sample rate matching, smoothing, segmentation, and feature extraction.  
- **emg models/**: scripts for training models to predict the classifiations of emg labels.
- **rpe models/**: scripts for training models to predict the rpe of the rep, with and without emg input features.
- **datasets/**: some of the repwise datasets required (these are produced by the preprocessing scripts)

  
### Data  

The `data/` folder includes:  
- Raw **CSV recordings** of IMU and EMG signals.  
- Processed datasets with extracted rep-level features.  
- Label files containing RPE values provided by participants.

⚠️ **Note**: Due to size/ethics restrictions, some raw participant data may be excluded.  

## Requirements  

The project was developed using **Python 3.10+** with the following key libraries:  
- `pandas`, `numpy`, `scipy`  
- `scikit-learn`, `xgboost`, `imblearn`, `scikit-optimize`  
- `tensorflow` / `keras`  
- `umap-learn`  
- `matplotlib`, `seaborn`

## Results

* EMG label prediction models achieved high accuracy (>0.8 F1 for UMAP/t-SNE embeddings).
* RPE classification models outperformed regression and RNN models.
* Incorporating EMG features slightly improved performance compared to IMU-only models.

## Citation

If you use this work, please cite the project report:

> James Thomas. *Estimation of Resistance Training RPE Using Inertial Sensors, Electromyography, and Machine Learning*. Final Year Project Report, University of Exeter, May 2025.

