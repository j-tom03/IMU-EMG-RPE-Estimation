# EMG Prediction Models

This directory contains all code used to develop models predicting EMG labels. 

* ```emg_model_grid_search.py``` performs hyperparameter searching across the range of models, for a range of EMG targets, storing the best parameters to ```best_emg_parameters.txt```
* ```all_emg_models.py``` builds the best version of each model for each target and then evaluates it, storing the results to a csv file stored in the ```results``` directory 
* ```best_emg_models.py``` builds the best model for each target and stores it to the ```models``` directory
* ```create_train_set.py``` uses the built models to create the final training data set, stored to ```imu_plus_pred_emg.csv```
