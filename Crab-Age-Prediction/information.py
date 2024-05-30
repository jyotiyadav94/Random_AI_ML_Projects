from imports import *

class Information:
    "This class shows some information about the dataset"
    def __init__(self):

        print()
        print('Information object is created')
        print()
    
    def get_head(self,data,rows):
        "This function returns the first 5 rows of the dataset"
        return data.head(rows)

    def remove_missing_values(self,data):
        """
        Remove missing values from a numerical dataset where 'Age' is the target variable.
        
        Parameters:
        - dataset: pandas DataFrame containing the dataset
        
        Returns:
        - dataset_cleaned: pandas DataFrame with missing values removed
        """
        # Drop rows with missing values
        dataset_cleaned = self.data.dropna()
        
        return dataset_cleaned

    def _info_(self, data):
        "This function shows some information about the data like Feature names,data type, number of missing values for each feature and ten samples of each feature"
        self.data=data
        rows, columns=self.data.shape
        print("=" * 50)
        print('====> This data contains {} rows and {} columns'.format(rows,columns))
        print("=" * 50)
        print()
        data_cleaned = self.remove_missing_values(self.data)
        print(data_cleaned.head(5))
        rows, columns=data_cleaned.shape
        print("=" * 50)
        print('====> This data contains {} rows and {} columns'.format(rows,columns))
        print("=" * 50)
        return data_cleaned


