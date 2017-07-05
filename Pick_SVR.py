from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np


input_data = read_csv("/home/vicky/Documents/it/AI/data/picks_2016.csv")

X = input_data.iloc[:,0:5]
y = input_data.iloc[:,5:6]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)


svr_rbf = SVR(kernel='rbf', C=1, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)

rbf_fit = svr_rbf.fit(X_train, y_train['PPS'])

rbf_pred = rbf_fit.predict(X_test)
rms = np.sqrt(mean_squared_error(y_test, rbf_pred))
print(rms)

