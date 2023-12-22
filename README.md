# GermanTrafficSignCNNClassifier
CNN classifier for German traffic sign dataset with 98% Test-F1 score and 99% train and validation accuracy

Related Repositories  
German Traffic Sign Dataset  
https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign  

Dataset Directory Structure  
./Meta/  
./Train/0  
./Train/1  
.  
.  
.  
./Train/42  
./Test/  
./Meta.csv  
./Train.csv  
./Test.csv  

Active Directory structure  
./Train/~  
./Test/~  
./Train.csv  
./Test.csv  
./signnames.csv  
./datavectorization_h5.py  
./traffic_sign_util.py  
./Traffic_sign_classifier.py  
--------------------------------------------------------------------------------
How to:
1. Download dataset from url
2. Download scripts from repo
3. Setup directory structure as shown above "Active Directory structure".
4. Run "datavectorization_h5.py" in a python ide to generate .h5 data files.(change paths, filenames as necessary)
5. Run "Traffic_sign_classifier.py" in a python ide to train, save, test model, generate plots and reports.

Params.txt - network architecture parameters and values
signnames.csv - class names for classification report

Have Fun!
