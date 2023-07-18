import argparse
import os
import pandas as pd
import mlflow
from train.forecaster_model import ForecasterModel


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
    parser.add_argument("--lags_grid", required=False, default="[7, 21, 60]", type=str,help='Lags used as predictors')
    parser.add_argument("--sel_exog", required=False, default="['Mes', 'Dia','media_movil']", type=str,help='Lags used as predictors')
    parser.add_argument("--registered_model_name", type=str, help="model name")
    parser.add_argument("--model", type=str, help="path to model file")
    args = parser.parse_args()
    return args
def main(args):
    """Main function of the script."""

    # paths are mounted as folder, therefore, we are selecting the file from folder
    data_train = pd.read_csv(select_first_file(args.train_data))

    # paths are mounted as folder, therefore, we are selecting the file from folder
    data_test = pd.read_csv(select_first_file(args.test_data))
    
    metric = ['mean_absolute_error','mean_squared_error']

    print(f"Training with data {data_train.head()}")

    forecaster_model = ForecasterModel(args.steps,args.lags_grid,metric)

    forecaster_model.train_model(data_train,args.y_name)

    forecaster_model.evaluate_models(data_test)

    forecaster_model.select_best_model()

    print(f'winner: {forecaster_model.best_name}')
    print(f'winner metrics: {forecaster_model.best_score}')

    # Registering the model to the workspace
    print("Registering the model via MLFlow")
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