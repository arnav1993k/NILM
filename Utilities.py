#!/usr/bin/env python3
import pandas as pd
import numpy as np
from math import sqrt
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
def get_current_dir():
    return os.getcwd()
def read_data(fname):
	df_master=pd.read_csv("/home/arnav1993k/Desktop/NILM/Data/"+fname,index_col=0)
	# df_master['Timestamp']=df_master.iloc[:,0]
	return df_master
def threshold(data,threshold):
	l,b=data.shape
	op=np.zeros((l,b))
	for i in range(l):
		for j in range(b):
			if data[i,j]>=threshold:
				op[i,j]=1
			# print(data[i])
	# print(max(op))
	return op	
def discretize(data):
	l=len(data)
	m=np.max(data)
	print(m)
	op=np.zeros(l)
	for i in range(l):
		if data[i]>=0.03*m:
			op[i]=1
			# print(data[i])
	print(max(op))
	return op
def discretize_df(df):
	l,b=df.shape
	output=np.zeros((l,b))
	for i in range(b):
		data=df.iloc[:,i]
		output[:,i]=discretize(np.array(data))
	return output
#main function
def main():
	df=read_data("UK_dale_Building1.csv")
	df=df.fillna(0)
	df=df.iloc[:5000,:]
	print(df.head())
	output=discretize_df(df[["Fridge freezer", "Kettle"]])
	# df[["Kettle"]].plot()
	plt.plot(output[:,0])
	plt.show()
	plt.close()
	plt.plot(output[:,1])
	plt.show()
	plt.close()
	df[["Site meter"]].plot()
	plt.show()
	plt.close()

if __name__=="__main__":
	main()