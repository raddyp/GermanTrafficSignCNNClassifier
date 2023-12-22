# -*- coding: utf-8 -*-

# Run datavectorization_h5.py before running this script
# Main classifier training and evaluation script

from sklearn.metrics import classification_report
import gc

from traffic_sign_util import initialize_prelims,create_model,train_model,visualize_plots


gc.enable()

# parent/ source directory where all the data h5 files are on your local machine
src_dir = 'D:/Data/germantrafficsign'
train_file_name = 'signtraindata.h5'
test_file_name ='signtestdata.h5'
labels_file_name = 'signnames.csv'


x_train,x_val,y_train,y_val,x_test,y_test,labelNames = initialize_prelims(src_dir, train_file_name, test_file_name, labels_file_name)
 
# define tensorflow model
model = create_model()

# train model
history = train_model(model,x_train,y_train,x_val,y_val)

# save your model
# model.save('D:/Data/germantrafficsign/model.h5')

# visualize model train/validation accuracy & loss plots
visualize_plots(history)

#  prediction testing on test data

# load model from directory
# model = load_model('D:/Data/germantrafficsign/model.h5')

predictions = model.predict(x_test,batch_size=256)
pred_report = classification_report(y_test,predictions.argmax(axis=1), target_names=labelNames)
print(pred_report)


gc.collect()

