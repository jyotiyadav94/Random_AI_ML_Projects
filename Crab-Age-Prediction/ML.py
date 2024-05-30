from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from imports import *

class MLUtilities:
    def __init__(self):
        self.rf_model = RandomForestRegressor()
        self.svr_model = SVR()
        self.dt_model = DecisionTreeRegressor()     
        print()
        print('Machine Learning object is created')
        print()

    def train(self, X_train, y_train):
        """Train all the models in the pipeline."""
        self.rf_model.fit(X_train, y_train)
        self.svr_model.fit(X_train, y_train)
        self.dt_model.fit(X_train, y_train)

    def predict(self, X_test):
        """Make predictions using all the models in the pipeline."""
        rf_pred = self.rf_model.predict(X_test)
        svr_pred = self.svr_model.predict(X_test)
        dt_pred = self.dt_model.predict(X_test)
        return rf_pred, svr_pred, dt_pred

    def evaluate(self, y_true, rf_pred, svr_pred, dt_pred):
        """Evaluate the models using mean squared error."""
        mse_rf = mae(y_true, rf_pred)
        mse_svr = mae(y_true, svr_pred)
        mse_dt = mae(y_true, dt_pred)
        return {'Random Forest': mse_rf, 'SVR': mse_svr, 'Decision Tree': mse_dt}

