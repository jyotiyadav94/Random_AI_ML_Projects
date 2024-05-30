from imports import *
from information import Information
from ML import MLUtilities
from preprocess import Preprocess

class ObjectOriented:
    """
    param train: train data will be used for modelling
    param test:  test data will be used for model evaluation
    """
    def __init__(self):
        #properties
        self.data=None
        self._info=Information()
        self._models = MLUtilities()
        self._preprocess=Preprocess()
        
        print()
        print('ObjectOriented object is created')
        print()
        
    def add_data(self, data):
        #properties
        self.data=data.copy()
        self.orig_data=self.data.copy()
        print()
        print('Your data has been added')
        print()

    def print_data(self,data,rows):
        "This function printing the dataset"
        print()
        print('printing the dataset')
        print()
        print(self._info.get_head(self.data,rows))

    def information(self):
        "This function shows some information about the data like Feature names,data type, number of missing values for each feature and ten samples of each feature"
        print()
        print('Information about the data is displayed')
        print()
        data=self._preprocess.label_encode_column(self.data,'Sex')
        print(self._info.get_head(data,5))
        print(self._info._info_(self.data))       


    class ml:

        def __init__(self,ObjectOriented):
            self.hp=ObjectOriented
            self.data=self.hp.data
            self._models = MLUtilities()
            print("class ml is created")

        
        def X_train_y_train(self,data,target_column):
            # Assuming 'X' contains the features and 'y' contains the target variable
            X = self.data.drop(columns=[target_column])
            y = self.data[target_column]
            return X,y

        # train test split 
        def train_test_split(self,data,target_column,train_size=0.8):
            "This function splits the data into train and test"
            X,y=self.X_train_y_train(data,target_column)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
            print("Training and test data has been split")
            return X_train, y_train,X_test, y_test

        # models definition 
        def train(self,X_train, y_train,X_test,y_test):
            "This function trains all the models in the pipeline"
            self._models.train(X_train, y_train)
            print("Training of the models completed")

        def predict(self,X_test):
            "This function predicts the models"
            rf_pred, svr_pred, dt_pred =  self._models.predict(X_test)
            return rf_pred, svr_pred, dt_pred

        # model training 
        def evaluation(self,y_true,rf_pred,svr_pred,dt_pred):
            "This function evaluates the models using mean squared error"
            print("evaluation of the models")
            return self._models.evaluate(y_true, rf_pred, svr_pred, dt_pred)

        # saved models prediction  
        def save_predictions_to_csv(self,predictions,output_file):
            """
            Save predictions to a CSV file.

            Parameters:
            - predictions (numpy.ndarray or list): Predicted values.
            - output_file (str): Path to the output CSV file.

            Returns:
            - None
            """
            # Flatten predictions if needed
            predictions = predictions.flatten() if hasattr(predictions, 'flatten') else predictions
            
            # Create a DataFrame for submission
            submission = pd.DataFrame({'Age': predictions})
            submission.to_csv(output_file, index=True, index_label='id')
            print(f'Predictions saved to {output_file}')