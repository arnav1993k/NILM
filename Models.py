#!/usr/bin/env python3
import pandas as pd
import numpy as np
import h5py
import os
from keras.models import Sequential,load_model
from keras.layers import Dense, Conv1D, LSTM, Bidirectional, Dropout, Flatten, Reshape,GRU,MaxPooling1D
from keras.layers import LSTM, ThresholdedReLU
from keras.utils import plot_model
from Utilities import read_data,get_current_dir
from keras.utils import to_categorical


class RNN_model():
    def __init__(self,epochs=10,batchsize=5,input_dim=1,output_dim=1,appliance="Fridge"):
        self.Model_name="RNN_LSTM"+appliance
        self.epochs=epochs
        self.batchsize=batchsize
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.model=self.create_model()
    def train(self,X,y):
        X=X.reshape((len(X),5,1))
        self.model.fit(X,y,epochs=self.epochs,batch_size=self.batchsize,shuffle=True,validation_split=0.2)
    def test(self,X):
        X=X.reshape((len(X),5,1))
        print(X.shape)
        y=self.model.predict(X)
        return y
    def create_model(self):
        model = Sequential()
        # 1D Conv
        model.add(Conv1D(10, 3, input_shape=(5,1), padding="same", strides=1,activation='relu'))
        model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid'))
        #Bi-directional LSTMs
        model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
        model.add(Bidirectional(LSTM(256, return_sequences=False, stateful=False), merge_mode='concat'))
        model.add(Dropout(0.05))
        # Fully Connected Layers
        model.add(Dense(128, activation='tanh'))
        model.add(Dense(self.output_dim, activation='sigmoid'))
        # model.add(ThresholdedReLU(theta=0.5))

        model.compile(loss='binary_crossentropy', optimizer='adam')
        plot_model(model, to_file=get_current_dir()+'/Models/'+self.Model_name+'.png', show_shapes=True)

        return model
    def save(self):
        self.model.save(get_current_dir()+'/Models/'+self.Model_name+'.h5')

    def load(self,name,dir=get_current_dir()):
        self.model=load_model(str(dir+"/Models/"+name+".h5"))
        return self

    def exists(self):
        if (os.path.isfile(get_current_dir()+'/Models/'+self.Model_name+'.h5')):
            return True
        else:
            return False

class GRU_model():
    def __init__(self,epochs=10,batchsize=5,input_dim=1,output_dim=1):
        self.Model_name="GRU"
        self.epochs=epochs
        self.batchsize=batchsize
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.model=self.create_model()
    def train(self,X,y):
        X=X.reshape((len(X),4,1))
        self.model.fit(X,y,epochs=self.epochs,batch_size=self.batchsize,shuffle=True)
    def test(self,X):
        X=X.reshape((len(X),4,1))
        print(X.shape)
        y=self.model.predict(X)
        return y
    def create_model(self):
        model = Sequential()
        # 1D Conv
        model.add(Conv1D(16, 4, activation="relu", padding="same", strides=1, input_shape=(4,1)))
        model.add(Conv1D(8, 4, activation="relu", padding="same", strides=1))

        # Bi-directional LSTMs
        model.add(Bidirectional(GRU(64, return_sequences=True, stateful=False), merge_mode='concat'))
        model.add(Bidirectional(GRU(256, return_sequences=False, stateful=False), merge_mode='concat'))
        # model.add(Dropout(0.1))
        # Fully Connected Layers
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.output_dim, activation='sigmoid'))
        # model.add(ThresholdedReLU(theta=0.5))

        model.compile(loss='mse', optimizer='adam')
        plot_model(model, to_file=get_current_dir()+'/Models/'+self.Model_name+'.png', show_shapes=True)

        return model
    def save(self):
        self.model.save(get_current_dir()+'/Models/'+self.Model_name+'.h5')

    def load(self,name,dir=get_current_dir()):
        self.model=load_model(str(dir+"/Models/"+name+".h5"))
        return self

    def exists(self):
        if (os.path.isfile(get_current_dir()+'/Models/'+self.Model_name+'.h5')):
            return True
        else:
            return False

class AutoEncoder_model():
    def __init__(self,epochs=10,batchsize=5,input_dim=1,output_dim=1):
        self.Model_name="AutoEncoder"
        self.epochs=epochs
        self.batchsize=batchsize
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.model=self.create_model()
    def train(self,X,y):
        # additional=self.batchsize-(len(X)%self.batchsize)
        # X=np.append(X,np.zeros(additional))
        # y=np.append(y,np.zeros((additional,self.output_dim)))
        y=y.reshape((len(X),self.output_dim))
        print(X.shape,y.shape)
        X=X.reshape((int(len(X)/self.batchsize),self.batchsize,1))
        y=y.reshape((int(y.shape[0]/self.batchsize),self.batchsize,self.output_dim))
        self.model.fit(X,y,epochs=self.epochs,batch_size=self.batchsize,shuffle=True)
    def test(self,X):
        additional=self.batchsize-(len(X)%self.batchsize)
        X=np.append(X,np.zeros(additional))
        X=X.reshape((int(len(X)/self.batchsize),self.batchsize,1))
        y=self.model.predict(X)
        return y
    def create_model(self):
        model = Sequential()

        # 1D Conv
        model.add(Conv1D(8, 4, activation="linear", input_shape=(1, 1), padding="same", strides=1))
        model.add(Flatten())
        # Fully Connected Layers
        model.add(Dropout(0.2))
        # model.add(Dense((self.batchsize-0)*8, activation='relu'))
        model.add(Dense((1-0)*8, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        # model.add(Dense((self.batchsize-0)*8, activation='relu'))
        model.add(Dense((1-0)*8, activation='relu'))
        model.add(Dropout(0.2))
        # 1D Conv
        # model.add(Reshape(((self.batchsize-0), 8)))
        model.add(Reshape(((1-0), 8)))
        model.add(Conv1D(self.output_dim, 4, activation="linear", padding="same", strides=1))
        model.compile(loss='mse', optimizer='adam')
        plot_model(model, to_file=get_current_dir()+'/Models/'+self.Model_name+'.png', show_shapes=True)

        return model
    def save(self):
        self.model.save(get_current_dir()+'/Models/'+self.Model_name+'.h5')
    def load(self,name,dir=get_current_dir()):
        self.model=load_model(str(dir+'/Models/'+name+'.h5'))
        return self
    def exists(self):
        if (os.path.isfile(get_current_dir()+'/Models/'+self.Model_name+'.h5')):
            return True
        else:
            return False
