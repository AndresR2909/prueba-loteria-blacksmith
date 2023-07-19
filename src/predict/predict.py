import argparse
import mlflow
from mlflow import MlflowClient
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt

def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    return os.path.join(path, files[0])

os.makedirs("./outputs", exist_ok=True)

def extract_registered_model(args):
    model_name=args.model_name

    client = MlflowClient()
    model =client.get_registered_model(name=model_name)
    print(model.latest_versions[0])
    model_uri = f"models:/{model_name}/{model.latest_versions[0].version}"
    print(model_uri)
    try:
        load_model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print('no se pudo cargar sklearn model')
        print(type(e))
        print(e.args)
        try:
            load_model = mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            print('no se pudo cargar pyfunc model')
            print(type(e))
            print(e.args)
    
    return load_model
    
def get_args():
    parser= argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str,default='.',help='the path of the model input')
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument('--model_name',type=str,default='.',help='Model name')
    parser.add_argument('--steps',type=int,default=7,help='number steps to forecasting')
    parser.add_argument('--predict_output',type=str,default='.',help='the path of the forecasting')

    args = parser.parse_args()
    return args

def main(args):

    mlflow.start_run()
    run = mlflow.active_run()
    print("Active run_id: {}".format(run.info.run_id))
    
    print('**********************last_window************************')
    dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')
    data = pd.read_csv(select_first_file(args.test_data), delimiter=",",index_col='index',parse_dates=['index'],date_parser=dateparse)
    last_window = data['Cantidad'].asfreq('d')
   

    load_model = extract_registered_model(args)

    print(type(load_model))

    print(load_model)
    try: 
        last_window = last_window.iloc[-len(load_model.lags):]
    except Exception as e:
        print(e)
        last_window = last_window.iloc[-args.steps:]
    
    print(last_window)

    pred = load_model.predict(steps=args.steps,last_window=last_window)
    
    print('**********************forecasting************************')
    print(pred)

    current_date = datetime.now()
    pred.to_csv(args.predict_output + f"/{args.model_name}_predictions_{current_date}.csv",index=True, sep=",")
    
    fig, ax = plt.subplots(figsize=(9, 4))

    last_window.plot(ax=ax, label='last_window')
    pred.plot(ax=ax, label='predictions')
    ax.set_title(f'prediccion del valor SDV: {args.model_name}')
    mlflow.log_figure(fig, "forecasting.png")

    mlflow.end_run()

    
if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)

