#!/usr/bin/env python3
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt
from Utilities import read_data,get_current_dir,discretize_df,threshold
import Models
from sklearn.metrics import accuracy_score,confusion_matrix,roc_curve
def main():
	df=read_data("UK_dale_Building1.csv")
	print(df.shape)

	mains=df[['Site meter']].fillna(0,limit=1)
	mains=mains.fillna(method='pad')
	# print(mains[1:200])
	print(mains.head())
	appliances=df[["Fridge freezer",]].fillna(0,limit=1)
	appliances=appliances.fillna(method='ffill')
	op=appliances.shape[1]
	mains=(mains)/mains.max()
	mains["diff1"]=mains[["Site meter"]].shift(1)
	mains["diff2"]=mains[["Site meter"]].shift(2)
	mains["diff3"]=mains[["Site meter"]].shift(3)
	mains["diff4"]=mains[["Site meter"]].shift(4)
	print(mains.shape)
	mains=mains.dropna(0)
	print(mains.shape)
	appliances=discretize_df(appliances)
	model=train(mains,appliances,op)
	test(model,mains,appliances,op)
	# plt.plot(mains.iloc[:5000,0])
	# plt.plot(appliances[0:5000])
	# plt.show()
def train(mains,appliances,op):
	print("********************************")
	print("Creating model")
	model1=Models.RNN_model(epochs=5,output_dim=1,appliance="Fridge")
	print("Model Created")
	if(model1.exists()):
		model1=model1.load(model1.Model_name)
	else:
		begin=int(input("Enter start point for training ==> "))
		end=int(input("Enter end point for training ==> "))
		X_t=mains.iloc[begin:end]
		y_t=appliances[begin:end]
		features = np.array(X_t)
		target = np.array(y_t).reshape((-1,1))
		print(target.shape)
		model1.train(features,target)
		model1.save()
	return model1
def test(model1,mains,appliances,op):
	begin=int(input("Enter start point for testing ==> "))
	end=int(input("Enter end point for testing ==> "))

	X=np.array(mains.iloc[begin:end])
	y=np.array(appliances[begin:end])
	y=y.reshape((len(X),op))
	print(X.shape)
	pred1=model1.test(X)
	pred1=threshold(pred1,0.5)
	plt.plot(y)
	plt.plot(pred1)
	plt.legend()
	plt.savefig(get_current_dir()+'/Outputs/'+model1.Model_name+"Fridge.png")
	plt.close()
	print("********************************")
	print("********************************")
	print("Confusion Matrix for RNN LSTM fridge")
	print(confusion_matrix(y[:,0],pred1[:,0]))
	print("********************************")
	print("Accuracy Score for RNN LSTM fridge")
	print(accuracy_score(y[:,0],pred1[:,0]))
	print("********************************")
main()

'''
model1=Models.RNN_model(epochs=5,output_dim=op)
model2=Models.GRU_model(epochs=10,output_dim=op)

features = np.array(X_t)
target = np.array(y_t).reshape((len(features),op))
if(model1.exists()):
	model1=model1.load(model1.Model_name)
else:
	model1.train(features[:,0],target)
	model1.save()
if(model2.exists()):
	model2=model2.load(model2.Model_name)
else:
	model2.train(features,target)
	model2.save()
begin=int(input("Enter start point for testing ==> "))
end=int(input("Enter end point for testing ==> "))

X=np.array(mains.iloc[begin:end])
y=np.array(appliances[begin:end])
y=y.reshape((len(X),op))
print(X.shape)
pred1=model1.test(X[:,0])
pred2=model2.test(X)
pred1=threshold(pred1,0.5)
pred2=threshold(pred2,0.5)
print("********************************")
print("********************************")
print("Confusion Matrix for RNN LSTM fridge")
print(confusion_matrix(y[:,0],pred1[:,0]))
print("********************************")
print("Accuracy Score for RNN LSTM fridge")
print(accuracy_score(y[:,0],pred1[:,0]))
print("********************************")'''


# print("Confusion Matrix for RNN LSTM kettle")
# print(confusion_matrix(y[:,1],pred1[:,1]))
# print("********************************")
# print("Accuracy Score for RNN LSTM kettle")
# print(accuracy_score(y[:,1],pred1[:,1]))
# print("********************************")
# print("Confusion Matrix for RNN LSTM DW")
# print(confusion_matrix(y[:,1],pred1[:,2]))
# print("********************************")
# print("Accuracy Score for RNN LSTM DW")
# print(accuracy_score(y[:,1],pred1[:,2]))

'''
print("********************************")
print("********************************")
print("********************************")
print("Confusion Matrix for GRU fridge")
print(confusion_matrix(y[:,0],pred2[:,0]))
print("********************************")
print("Accuracy Score for GRU fridge")
print(accuracy_score(y[:,0],pred2[:,0]))
print("********************************")
'''

# print("Confusion Matrix for GRU kettle")
# print(confusion_matrix(y[:,1],pred2[:,1]))
# print("********************************")
# print("Accuracy Score for GRU kettle")
# print(accuracy_score(y[:,1],pred2[:,1]))
# print("********************************")
# print("Confusion Matrix for GRU DW")
# print(confusion_matrix(y[:,1],pred2[:,2]))
# print("********************************")
# print("Accuracy Score for GRU DW")
# print(accuracy_score(y[:,1],pred2[:,2]))
# print("********************************")
# print("********************************")

# plt.plot(pred1[:,0],label="Predict")
# plt.plot(y[:,0],label="Actual")
# plt.title("Fridge")
# plt.legend()
# plt.savefig(get_current_dir()+'/Outputs/'+model1.Model_name+"Fridge.png")
# plt.close()
# plt.plot(pred1[:,1],label="Predict")
# plt.plot(y[:,1],label="Actual")
# plt.title("Kettle")
# plt.legend()
# plt.savefig(get_current_dir()+'/Outputs/'+model1.Model_name+"Kettle.png")
# plt.close()

'''
plt.plot(pred1[:,0],label="Predict")
plt.plot(y[:,0],label="Actual")
plt.title("Dish Washer")
plt.legend()
plt.savefig(get_current_dir()+'/Outputs/'+model1.Model_name+"dw.png")
plt.close()

# plt.plot(pred2[:,0],label="Predict")
# plt.plot(y[:,0],label="Actual")
# plt.title("Fridge")
# plt.legend()
# plt.savefig(get_current_dir()+'/Outputs/'+model2.Model_name+"Fridge.png")
# plt.close()
# plt.plot(pred2[:,1],label="Predict")
# plt.plot(y[:,1],label="Actual")
# plt.title("Kettle")
# plt.legend()
# plt.savefig(get_current_dir()+'/Outputs/'+model2.Model_name+"Kettle.png")
# plt.close()
plt.plot(pred2[:,0],label="Predict")
plt.plot(y[:,0],label="Actual")
plt.title("Dishwasher")
plt.legend()
plt.savefig(get_current_dir()+'/Outputs/'+model2.Model_name+"dw.png")
plt.close()'''