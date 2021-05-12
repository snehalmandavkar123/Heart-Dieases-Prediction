import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from sklearn.tree import DecisionTreeClassifier


from pandas import read_csv


import csv


def process(path,X_test):
	df = pd.read_csv(path, sep=',')
	X_train = df.drop(['AHD'], axis = 1) 
	y_train = df["AHD"]
	print(X_train)
	print(y_train)
	#X_test=np.array([67,1,4,160,286,0,2,108,1,1.5,2,3,3])
	X_test=np.array(X_test)
	X_test=X_test.reshape(1,-1)
	print(X_test)
	
	model4 = DecisionTreeClassifier()
	model4.fit(X_train, y_train)
	y_pred = model4.predict(X_test)
	print(y_pred[0])
	result=""
	if y_pred[0]==0:
		result="Normal"
	if y_pred[0]==1:
		result="High Blood Pressure"
	if y_pred[0]==2:
		result="Coronary Artery Disease"
	if y_pred[0]==3:
		result="Congestive Heart Failure"
	if y_pred[0]==4:
		result="Storke"
	print(result)
	return result
