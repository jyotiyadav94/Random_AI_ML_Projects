from imports import *

class Preprocess:
    """
    param train: train data will be used for modelling
    param test:  test data will be used for model evaluation
    """
    def __init__(self):
        #properties
        self.data=None
        
        print()
        print('Preprocess object is created')
        print()
        
    def label_encode_column(self,data, column_name):
        """
        Function to label encode a categorical column in a DataFrame.
        
        Args:
        - df: pandas DataFrame containing the column to be encoded.
        - column_name: str, name of the column to be encoded.
        
        Returns:
        - df: pandas DataFrame with the specified column label encoded.
        """
        label_encoder = LabelEncoder()
        data[column_name] = label_encoder.fit_transform(data[column_name])
        return data
         