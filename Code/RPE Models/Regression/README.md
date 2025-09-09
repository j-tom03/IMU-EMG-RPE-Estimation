# RPE Regression Models

This directory contains all code used to develop models predicting EMG labels. 

* ```regression_grid_search.py``` performs hyperparameter searching across the range of models, storing the best parameters to ```best_parameters.txt```
* ```lstm_grid_search.py``` performs hyperparameter searching across LSTM and GRU, storing the best parameters to ```best_lstm_parameters.txt```
* ```all_models.py``` builds the best version of each model for RPE prediction and then evaluates it, storing the results to a csv file stored in the ```results``` directory 