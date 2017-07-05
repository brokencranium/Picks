import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import ensemble

input_data = read_csv("/home/vicky/Documents/it/AI/data/picks_2016.csv")

X = input_data.iloc[:,0:5]
y = input_data.iloc[:,5:6]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

params = {'n_estimators': 500, 'max_depth': 6, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
gb_mod = ensemble.GradientBoostingRegressor(**params)

gb_fit = gb_mod.fit(X_train, y_train['PPS'])
gb_pred = gb_fit.predict(X_test)
rms_gb = np.sqrt(mean_squared_error(y_test, gb_pred))
print(rms_gb)
print(gb_pred)