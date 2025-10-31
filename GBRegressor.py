# %%
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
# %%
df = pd.read_csv("insurance.csv")
df.head() # does data look normal? yes
# %%
len(df)
# %%
df.isna().sum() # how many missing vals (none)
# %%
df['sex'] = pd.Categorical(df['sex']) # categories which need numeric codes
df['sex'] = df['sex'].cat.codes

df['smoker'] = pd.Categorical(df['smoker'])
df['smoker'] = df['smoker'].cat.codes

df['region'] = pd.Categorical(df['region'])
df['region'] = df['region'].cat.codes
# %%
x = df.drop(columns= ['charges']) # separate charges (target)
y = df['charges']
# %%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9) # 20% data for test
# %%
reg = GradientBoostingRegressor()
reg.fit(x_train, y_train) # apply GB regressor to data
# %%
y_pred = reg.predict(x_test) # predict charges with trained model
mean_absolute_error(y_test, y_pred) # MAE between predicted and actual 
# %%
y_pred_train = reg.predict(x_train) # predicts charges on already seen data
mean_absolute_error(y_train, y_pred_train) # Small MAE means overfitting
# %%
gbm = GradientBoostingRegressor()
parameters = {'learning_rate': [0.1, 0.2, 0.3], # smaller = slower/ more stable learning
              'n_estimators' : [100, 90, 120]} # no. boosting stages (trees)

reg = GridSearchCV(gbm, parameters) # checks all possible param combos
reg.fit(x_train, y_train)              
# %%
reg.best_params_ # best params = lowest pred error
# %%
y_pred = reg.predict(x_test)
mean_absolute_error(y_test, y_pred) # charges predictions based on best params (should be smaller MAE)
# %%
y_pred_train = reg.predict(x_train)
mean_absolute_error(y_train, y_pred_train) # check training accuracy again
# %%
# %%
feature_order = x_train.columns  # read columns in correct order
# %%
def predict_charge(age, bmi, children, sex_code, smoker_code, region_code): # function to predict charges
    row = pd.DataFrame([{
        "age": age,
        "sex": sex_code,
        "bmi": bmi,
        "children": children,
        "smoker": smoker_code,
        "region": region_code
    }])
    row = row.reindex(columns=feature_order)  # correct order
    return float(reg.predict(row)[0])

# %%
charge = predict_charge(age=30, bmi=32.4, children=1, sex_code=0, smoker_code=0, region_code=3) # compare to real CSV entries
# %%
print(f"Predicted charge: {charge:.2f}")