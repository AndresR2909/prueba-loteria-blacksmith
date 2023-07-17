import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error,r2_score

from local.competition import MetricsCompetition


from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from pmdarima import ARIMA
from sklearn.preprocessing import MinMaxScaler
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from sklearn.pipeline import make_pipeline
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.model_selection_sarimax import grid_search_sarimax




class ForecasterModel:
    def __init__(self,steps,lags_grid,metric, scaler=MinMaxScaler(), random_state:int =123) -> None:
        self.random_state = random_state
        self.steps = steps
        self.lags_grid = lags_grid
        self.scalers = scaler
        self.models = {
            "ridge":{ 
                "model": Ridge(random_state=random_state),
                "hiperparameters": {'ridge__alpha': np.logspace(-3, 5, 10)}},
        
            "xgbregressor":{ 
                "model": XGBRegressor(random_state = random_state),
                "hiperparameters": {'xgbregressor__min_child_weight': [1,5],
                                    'xgbregressor__gamma': [0.5,1,2], 
                                    'xgbregressor__subsample': [0.6,1],
                                    'xgbregressor__colsample_bytree': [0.6,1],
                                    'xgbregressor__max_depth': [3,5]}},
            "arima":{
                "model": ARIMA(order=(12, 1, 1), seasonal_order=(0, 0, 0, 0), maxiter=200),
                "hiperparameters": {'arima__order': [(7,0,0),(7,1,0),(7,1,1),(7,0,7),(7,1,7),
                                                (14,0,0),(14,1,0),(14,1,1),(14,0,7),(14,1,7),
                                                (21,0,0),(21,1,0),(21,1,1),(21,0,7),(21,1,7),
                                                (60,0,0),(60,1,0),(60,1,1),(60,0,7),(60,1,7)],
                                    'arima__seasonal_order': [(0, 0, 0, 0)],
                                    'arima__trend': [None, 'n', 'c']}}
        }
        self.y_name = None
        self.sel_exog = None
        self.metric = metric
        self.results = {}
        self.best_model = None
        self.best_score = 0.0
        self.best_name = ""

    #Creamos función para entrenar los modelos
    def train_model(self, data_train, y_name, sel_exog = None,transformer_exog=None):
        self.y_name=y_name
        self.sel_exog=sel_exog
        for name,object  in self.models.items():
            model = object["model"]
            param_grid = object["hiperparameters"]
            
            pipe = make_pipeline(self.scaler, model)
            if name == "arima":

                forecaster = ForecasterSarimax(
                                regressor=pipe,
                                transformer_exog = transformer_exog
                            )
                results_grid = grid_search_sarimax(
                   forecaster         = forecaster,
                   y                  = data_train[y_name],
                   exog               = data_train[sel_exog] if sel_exog else None,
                   param_grid         = param_grid,
                   steps              = self.steps,
                   refit              = True,
                   metric             = 'mean_absolute_error',
                   initial_train_size = int(len(data_train)*0.5),
                   fixed_train_size   = False,
                   return_best        = True,
                   n_jobs             = -1,
                   verbose            = False,
                   show_progress      = True
               )

            
            else:
                forecaster = ForecasterAutoreg(
                                regressor = pipe,
                                lags = 1,  # This value will be replaced in the grid search
                                transformer_exog= transformer_exog
                            )
            
                results_grid = grid_search_forecaster(
                    forecaster         = forecaster,
                    y                  = data_train[y_name],
                    exog               = data_train[sel_exog] if sel_exog else None ,
                    param_grid         = param_grid,
                    lags_grid          = self.lags_grid,
                    steps              = self.steps,
                    metric             = ['mean_absolute_error','mean_squared_error'],
                    refit              = False,
                    initial_train_size = int(len(data_train)*0.8),
                    fixed_train_size   = True,
                    return_best        = True,
                    verbose            = False,
                    show_progress      = True
                )
                
                
                    

            object["model"] = forecaster
            print(f"Modelo {name} ha sido entrenado")

    #Creamos una función para guardar el modelo entrenado
    def save_model(self, path):
        #model = self.models["random_forest"]
        #joblib.dump(model, path)

        print("El modelo ha sido serializado con exito.")


    def evaluate_models(self, data_test):
        
        for name, config in self.models.items():
            model = config["model"]
            y_pred = model.predict(steps=self.steps,exog=data_test[self.sel_exog])
            mse = mean_squared_error(data_test[self.y_name], y_pred)
            mae = mean_absolute_percentage_error(data_test[self.y_name], y_pred)
            r2 = r2_score(data_test[self.y_name], y_pred)
            self.results[name] = {'mean squared error': mse, 'mean absolute percentage error': mae, 'r2_score': r2}

            print("-"*40)
            print(f"Métricas de modelo {name}:")
            print(f"--mse: {mse}")
            print(f"--mae: {mae}")
            print(f"--r2: {r2}")
    
    def select_best_model(self):
        competition = MetricsCompetition(self.results,'count')
        winner = competition.evaluated_best_model()
        self.best_name = winner
        self.best_score = self.results["winner"]
        self.best_model = self.models[winner]["model"]




    