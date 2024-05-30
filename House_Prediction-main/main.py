from imports import *
from information import Information
from houseObjectOriented import HouseObjectOriented
from preprocess import  Pre_processing
from preprocessor import Preprocessor
from ML import ML

#collect the data
train = pd.read_csv('/Users/wenda/Downloads/dataset/train.csv')
test = pd.read_csv('/Users/wenda/Downloads/dataset/test.csv')


#create HouseObjectOriented object
HOOP = HouseObjectOriented()

#adding the data 
HOOP.add_data(train, test)

#Display Information about the data
HOOP.information()

#Pre-Process the data
HOOP.preprocessing()

#Display Information about the data after Pre-Processing
HOOP.information()

#Create a Machine Learning object
ML = HOOP.ml(HOOP)

#Show the available algorithms
ML.show_available_algorithms()

#Initialize the ML Regressors
ML.init_regressors('all')

#Train-Test Validation
ML.train_test_validation()

#Visualize the results of train-test validation
ML.visualize_trai_test()

#Applying Cross-Validation
ML.cross_validation('all')

#Visualize the results of Cross-Validation
#ML.visualize_cv()

#Find the best model and fit it to the data
ML.fit_best_model()

# Predict and show the prediction
ML.show_predictions()