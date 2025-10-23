# Using GBMs to predict medical charges

### Gradient Boosting Regressor

The script begins by importing: 
**pandas** for data handling and preprocessing, 
**train_test_split** from scikit-learn’s model_selection module to divide the data into training and testing sets, 
**GradientBoostingRegressor** from scikit-learn’s ensemble module to build a gradient boosting model, 
**mean_absolute_error** from scikit-learn’s metrics module to evaluate prediction accuracy, and 
**GridSearchCV** from scikit-learn’s model_selection module to tune the model’s parameters.

The next section reads the dataset *insurance.csv* using pandas and displays the first few rows to verify that the data has loaded correctly. It then checks the number of rows in the dataset and calculates how many missing values exist in each column (none) to ensure the data is complete before training.

The categorical columns sex, smoker, and region are then converted into numeric codes using pandas categorical encoding. This allows the machine learning model to interpret the categorical variables as numerical values. After encoding, the features are separated into the variable x, which contains all predictors, and y, which contains the target variable charges.

The data is then split into training and testing subsets using train_test_split, reserving 20 percent of the data for testing. A GradientBoostingRegressor is created and fitted to the training data. This model builds an ensemble of shallow decision trees where each tree attempts to correct the errors of the previous ones.

Predictions are generated for both the test set and the training set using the model’s predict function. The mean_absolute_error function is used to compute the average difference between predicted and actual insurance charges for both sets, giving an indication of the model’s accuracy and potential overfitting.

A second GradientBoostingRegressor is then defined, and a parameter grid is created specifying different learning rates and numbers of estimators to test. GridSearchCV systematically trains multiple models using these combinations and selects the parameters that yield the lowest prediction error during cross-validation. The model is retrained with the best parameters, and its performance is evaluated again using MAE on both the training and test data.

Finally, a custom function named predict_charge is defined. This function allows the user to input specific feature values such as age, BMI, number of children, sex code, smoker code, and region code. The inputs are assembled into a one-row pandas DataFrame, passed into the trained model, and the predicted insurance charge is returned.

### PyTorch with XGBoost


