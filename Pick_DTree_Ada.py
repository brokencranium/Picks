import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

rng = np.random.RandomState(1)
input_data = read_csv("/home/vicky/Documents/it/AI/data/picks_2016.csv")

X = input_data.iloc[:,0:5]
y = input_data.iloc[:,5:6]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

dec_tree = DecisionTreeRegressor(max_depth=6)

ada_dec_tree= AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng)

dec_tree_fit = dec_tree.fit(X_train, y_train['PPS'])
ada_dec_tree_fit = ada_dec_tree.fit(X_train, y_train['PPS'])

dec_tree_pred = dec_tree_fit.predict(X_test)
ada_dec_tree_pred = ada_dec_tree_fit.predict(X_test)

rms_dec = np.sqrt(mean_squared_error(y_test, dec_tree_pred))
rms_ada_dec = np.sqrt(mean_squared_error(y_test, ada_dec_tree_pred))

print('RMS decision tree', rms_dec)
print('RMS ADA decision tree', rms_ada_dec)