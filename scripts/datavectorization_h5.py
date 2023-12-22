# -*- coding: utf-8 -*-


import os
import h5py
from traffic_sign_util import process_data_from_folder,process_data_from_file
import gc


gc.enable()
# parent/ source directory where all the data and csv files are on your local machine
src_dir = 'D:/Data/germantrafficsign/'
data_dir = os.path.join(src_dir,'Train')
train_file = os.path.join(src_dir,'Train.csv')
test_file = os.path.join(src_dir,'Test.csv')

# image resize parameters
height = 50 
width = 50
num_classes = 43

# two methods to read image data - 1.From image folders, 2.From a CSV file
data, labels = process_data_from_folder(data_dir, height, width, num_classes)
# data, labels = process_data_from_file(src_dir,train_file, height, width)


# save train data to h5 file format
trf = h5py.File('D:/Data/germantrafficsign/signtraindata.h5','w')
trf.create_dataset('data',data=data)
trf.create_dataset('labels',data = labels)
trf.close()

# read form h5 to verify if all data is written
# trhf = h5py.File('D:\Data\germantrafficsign\signtraindata.h5','r')
# data = np.array(trhf.get('data'))
# labels = np.array(trhf.get('labels'))
# trhf.close()

# Read test image data from csv file
x_test,y_test =  process_data_from_file(src_dir,test_file,width,height)

# save test data to h5 file format
test = h5py.File('D:/Data/germantrafficsign/signtestdata.h5','w')
test.create_dataset('data',data=x_test)
test.create_dataset('labels',data = y_test)
test.close()


# # read form h5 to verify if all data is written
# testhf = h5py.File('D:\Data\germantrafficsign\signtestdata.h5','r')
# tdata = np.array(testhf.get('data'))
# tlabel = np.array(testhf.get('labels'))
# testhf.close()
gc.collect()