import argparse
import os
import pandas as pd
import mlflow
from forecaster_model import ForecasterModel
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer


def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])


# Start Logging
mlflow.start_run()

os.makedirs("./outputs", exist_ok=True)

def parser_args():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--y_name", required=False, default='Cantidad', type=str)
    parser.add_argument("--steps", required=False, default=7, type=int)
    parser.add_argument("--lags_grid", required=False, default="7,21,60", type=str,help='Lags used as predictors')
    parser.add_argument("--sel_exog", required=False, default=None, type=str,help='Lags used as predictors')
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()
    return args
def main(args):
    """Main function of the script."""

    # paths are mounted as folder, therefore, we are selecting the file from folder
    #data_train = pd.read_csv(select_first_file(args.train_data))

    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    data_train = pd.read_csv(select_first_file(args.train_data), delimiter=",",index_col='index',parse_dates=['index'],date_parser=dateparse)
    data_train = data_train.asfreq('d')
    # paths are mounted as folder, therefore, we are selecting the file from folder
    #data_test = pd.read_csv(select_first_file(args.test_data))
    data_test = pd.read_csv(select_first_file(args.test_data), delimiter=",",index_col='index',parse_dates=['index'],date_parser=dateparse)
    data_test = data_test.asfreq('d')

    metric = ['mean_absolute_error','mean_squared_error']

    print(f"Training with data {data_train.head(4)}")

    print(f"Training with data {data_test.head(4)}")

    lags= [int(numero) for numero in args.lags_grid.split(',')]
    

    forecaster_model = ForecasterModel(int(args.steps),lags,metric)

    print(forecaster_model.lags_grid)
    print(forecaster_model.steps)
    print(forecaster_model.metric)

    transformer_exog = ColumnTransformer(
                       [#('scale_1', OneHotEncoder(), ['DiaSemana']),
                        ('scale_2', MinMaxScaler(), ['Mes']),
                        ('scale_3', MinMaxScaler(), ['Dia']),
                        ('scale_4', MinMaxScaler(), ['media_movil']),
                        #('onehot_1', OneHotEncoder(), ['EsFestivo']),
                        #('onehot_2', OneHotEncoder(), ['EsQuincena'])
                       ],
                       remainder = 'drop', #passthrough',
                       verbose_feature_names_out = False
                   )
    sel_exog = args.sel_exog.split(',')
    print(args.sel_exog.split(','))
    forecaster_model.train_model(data_train,y_name=args.y_name,sel_exog=None,transformer_exog = None)

    forecaster_model.evaluate_models(data_test)

    forecaster_model.select_best_model()

    print(f'winner: {forecaster_model.best_model_name}')
    print(f'winner metrics: {forecaster_model.best_model_metrics}')
    
    for k,v in forecaster_model.best_model_metrics.items():
        mlflow.log_metric(k, v)
    for k,v in forecaster_model.best_model.params.items():
        mlflow.log_param(k, v)


    fig, ax = plt.subplots(figsize=(9, 4))
    df_grafico=pd.DataFrame([])
    df_grafico['predicciones']= forecaster_model.best_model.predict(steps =args.steps)
    df_grafico['data test']=data_test[args.y_name]
    df_grafico.plot(ax=ax)
    ax.legend()
    ax.set_title(f'prediccion del valor: {forecaster_model.best_model_name}')
    mlflow.log_figure(fig, "test.png")

    # Registering the model to the workspace
    print("Registering the model via MLFlow")
    if forecaster_model.best_model_name == 'xxx':
        mlflow.pyfunc.log_model(
            sk_model=forecaster_model.best_model,
            registered_model_name=args.registered_model_name,
            artifact_path=args.registered_model_name,
        )
        # Saving the model to a file
        mlflow.pyfunc.save_model(
            sk_model=forecaster_model.best_model,
            path=os.path.join(args.model, "trained_model"),
        )
    
    else:
        mlflow.sklearn.log_model(
            sk_model=forecaster_model.best_model,
            registered_model_name=args.registered_model_name,
            artifact_path=args.registered_model_name,
        )

        # Saving the model to a file
        mlflow.sklearn.save_model(
            sk_model=forecaster_model.best_model,
            path=os.path.join(args.model, "trained_model"),
        )

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    args = parser_args()
    main(args)