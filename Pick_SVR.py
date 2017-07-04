from pandas import read_csv
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np


input_data = read_csv("/home/vicky/Documents/it/AI/data/picks_2016.csv")

X = input_data.iloc[:,0:5]
y = input_data.iloc[:,5:6]

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)

y_rbf = svr_rbf.fit(X, y['PPS']).predict(X)
y_lin = svr_lin.fit(X, y['PPS']).predict(X)
y_poly = svr_poly.fit(X, y['PPS']).predict(X)

rms = np.sqrt(mean_squared_error(y, y_rbf))
print(rms)


