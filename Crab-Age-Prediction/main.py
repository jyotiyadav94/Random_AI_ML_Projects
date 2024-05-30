from imports import *
from objectOriented import ObjectOriented
from information import Information
from ML import MLUtilities

#collect the data
data = pd.read_csv('dataset/CrabAgePrediction.csv')

#create ObjectOriented object
HOOP = ObjectOriented()

#adding the data
HOOP.add_data(data)

#Display Information about the data
HOOP.print_data(data,2)

#Display Information about the data
HOOP.information()

#Create a Machine Learning object
ML = HOOP.ml(HOOP)

#Show the available algorithms
X_train, y_train,X_test, y_test=ML.train_test_split(data,'Age',0.8)

#Train all the models in the pipeline
ML.train(X_train, y_train,X_test,y_test)

#Predict the models
rf_pred, svr_pred, dt_pred=ML.predict(X_test)

#Evaluation of the model
evaluation_result=ML.evaluation(y_test, rf_pred, svr_pred, dt_pred)

#save the best models prediction
ML.save_predictions_to_csv(rf_pred,'predictions.csv')














