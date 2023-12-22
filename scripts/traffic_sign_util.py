# -*- coding: utf-8 -*-

import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as tfl
from keras.utils import to_categorical
import cv2
import pandas as pd


# read image data from folders, resize and vectorize them
def process_data_from_folder(src_dir,width,height,classes):    
    
    data = []
    labels = []
    
    for i in range(classes):
        
        path = os.path.join(src_dir,str(i))
        print(path)
        
        files = os.listdir(path)       
    
        for file in files:
        
            filename = os.path.join(path, file)
            if (os.path.isdir(filename)):
                #print(filename, "  is directory")
                continue
            if(os.path.getsize(filename) ==0):
                print(file, "  is zero length, so ignoring")
                continue
            
            image = cv2.imread(filename)
            res_im = cv2.resize(image,(height,width))
            data.append(np.array(res_im))
            labels.append(i)
    
    data = np.array(data)
    labels = np.array(labels)
    randomize = np.arange(len(data))
    np.random.shuffle(randomize)
    data = data[randomize]
    labels = labels[randomize]    
        
    return data, labels
 
    
 
# read image data from csv file, resize and vectorize them
def process_data_from_file(src_dir,filename,width,height):    
    
    data = []
    file = pd.read_csv(filename)
    labels = file["ClassId"]
    filenames = file["Path"]  
    
    
    for i in range(len( filenames)):
                
        filename = os.path.join(src_dir,filenames[i])
        # print(path)               
        
        if (os.path.isdir(filename)):
            #print(filename, "  is directory")
            continue
        if(os.path.getsize(filename) ==0):
            print(file, "  is zero length, so ignoring")
            continue
        
        image = cv2.imread(filename)
        res_im = cv2.resize(image,(height,width))
        data.append(np.array(res_im))        

    data = np.array(data)
    data = data.astype('float32')/255
    labels = np.array(labels)
    randomize = np.arange(len(data))
    np.random.shuffle(randomize)
    data = data[randomize]
    labels = labels[randomize]
        
    return data, labels 

# read csv file for file locations and labels
def parse_data_from_csv(filename):
    
    file = pd.read_csv(filename)
    labels = file["ClassId"]
    filenames = file["Path"]  
    
    return filenames,labels



# read imag data & labels from h5 file
def read_data(filename):
        
    hfile = h5py.File(filename,'r')    
    data = np.array(hfile.get('data'))
    labels = np.array(hfile.get('labels'))
    hfile.close()
    
    return data,labels

# split data into train-test split
def split_data(data, labels, split_size):
  
    randomize = np.arange(len(data))
    np.random.shuffle(randomize)
    data = data[randomize]
    labels = labels[randomize]
    
    ratio = int(len(data)*split_size)
    x_train,x_val = data[0:ratio,:],data[ratio:len(data),:]
    y_train,y_val = labels[0:ratio],labels[ratio:len(data)]         
        
    return x_train,x_val,y_train,y_val


# initialize all preliminary data
def initialize_prelims(src_dir, train_file_name, test_file_name, labels_file_name):
        
    # read train data from h5 file
    train_file = os.path.join(src_dir,train_file_name)
    train_data, train_labels = read_data(train_file)
    split_size = 0.8
    x_train,x_val,y_train,y_val = split_data(train_data, train_labels, split_size)
    
    # normalize data & one-hot vectoring labels
    x_train = x_train.astype('float32')/255 
    x_val = x_val.astype('float32')/255
    y_train = to_categorical(y_train, 43)
    y_val = to_categorical(y_val, 43)
    
    
    # read test data from h5 file
    test_file = os.path.join(src_dir,test_file_name)
    x_test,y_test = read_data(test_file)
    x_test = x_test.astype('float32')/255
    # y_test = to_categorical(y_test, 43)
    
    
    #  Read label names for claasification report
    labelnames_file = os.path.join(src_dir,labels_file_name)
    labelNames = open(labelnames_file).read().strip().split("\n")[1:]
    labelNames = [l.split(",")[1] for l in labelNames]

    return x_train,x_val,y_train,y_val,x_test,y_test,labelNames


#model definition
def create_model():
     
    model = tf.keras.models.Sequential([
                    tfl.Conv2D(32, (3,3), activation='relu', input_shape=(50, 50,3),padding="valid"),
                      tfl.Conv2D(32, (3,3), activation='relu', input_shape=(50, 50,3),padding="valid"),                            \
                      tfl.MaxPooling2D(2,2),                              
                   
                      tfl.Conv2D(64, (3,3), activation='relu',padding="valid"),
                      tfl.Conv2D(64, (3,3), activation='relu',padding="valid"),                             
                      tfl.MaxPooling2D(2,2),                                
                    
                      tfl.Conv2D(128, (3,3), activation='relu',padding="valid"),
                      tfl.Conv2D(128, (3,3), activation='relu',padding="valid"),
                      tfl.MaxPooling2D(2,2),                                
                    
                      tfl.Dropout(0.5),                                                                
                   
                      tfl.Flatten(), 
                      tfl.Dense(256, activation='relu'),                                
                      tfl.Dense(43, activation='softmax')  
                    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics = ['accuracy'])
    
    model.summary()
    
    return model


def train_model(model,x_train,y_train,x_val,y_val):
        
    # Train your model
    history = model.fit(x_train,y_train, 
                        epochs=20, 
                        batch_size=512,
                        verbose = 1, 
                        validation_data=(x_val,y_val)                    
                        )
    return history


def visualize_plots(history):
    
    acc      = history.history[     'accuracy' ]
    val_acc  = history.history[ 'val_accuracy' ]
    loss     = history.history[    'loss' ]
    val_loss = history.history['val_loss' ]
    epochs   = range(len(acc)) # Get number of epochs
    
    #------------------------------------------------
    # Plot training and validation accuracy per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     acc, label = 'Train Acc' )
    plt.plot  ( epochs, val_acc, label = 'Val Acc' )
    plt.title ('Training and validation accuracy')
    plt.xlabel ('epochs')
    plt.legend(loc="upper right")
    plt.figure()    
    
    #------------------------------------------------
    # Plot training and validation loss per epoch
    #------------------------------------------------
    plt.plot  ( epochs,     loss,label = 'Train loss' )
    plt.plot  ( epochs, val_loss , label = 'Val loss')
    plt.title ('Training and validation loss'   )
    plt.xlabel ('epochs')
    plt.legend(loc="lower right")

    return