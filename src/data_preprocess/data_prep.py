import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow
import matplotlib.pyplot as plt
from data_preprocess import DataPreprocess

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="path to input data")
    parser.add_argument("--test_steps", type=int, required=False, default=7)
    parser.add_argument("--index_column", type=str, required=False, default='FechaTx')
    parser.add_argument("--target_column", type=str, required=False, default='Cantidad')
    parser.add_argument("--filter_column", type=str, required=False, default='CodSDV')
    parser.add_argument("--filter_value", type=str, required=False, default='109216')
    parser.add_argument("--del_columns", type=str, required=False, default='IdCliente,NomProducto,CodProducto')
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    parser.add_argument("--output_data", type=str, help="path to data")
    args = parser.parse_args()
    return args

def main(args):
    """Main function of the script."""

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.input_data)

    data_prepo = DataPreprocess(args.input_data, index_column=args.index_column, target_column = args.target_column)
    data_prepo.remove_irrelevant_features(del_columns=args.del_columns.split(','))
    data_prepo.filter_dataframe_by_feature(filter_column=args.filter_column,filter_value=int(args.filter_value))
    data_prepo.grouped_dataframe_by_feature(grouped_column =args.filter_column)
    data_prepo.completed_timeserie_df()
    data_prepo.feature_generation()
    data_prepo.handle_missing_values()
    data_prepo.split_data(split = 7)

    df_in = data_prepo.dataframe
    mlflow.log_metric("num_registros_data_in", df_in.shape[0])
    mlflow.log_metric("num_caracteristicas_data_in", df_in.shape[1] - 1)
 
    df_out = data_prepo.output_dataframe
    mlflow.log_metric("num_registros_data_out", df_out.shape[0])
    mlflow.log_metric("num_caracteristicas_data_out", df_out.shape[1] - 1)
    mlflow.log_param("fecha minima_data_out", df_out.index.min())
    mlflow.log_param("fecha_maxima_data_out", df_out.index.max())
    mlflow.log_param("frecuencia_data_out", df_out.index.freq)
    train_df = data_prepo.data_train
    test_df = data_prepo.data_test


    fig, ax = plt.subplots(figsize=(9, 4))
    test_df[[args.target_column,'media_movil']].plot(ax=ax)
    train_df[[args.target_column,'media_movil']].plot(ax=ax)
    ax.legend()
    ax.set_title(f'serie de tiempo SDV: {args.filter_value}')
    mlflow.log_figure(fig, "test.png")

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    train_df.to_csv(os.path.join(args.train_data, "data_train.csv"), index=True)

    test_df.to_csv(os.path.join(args.test_data, "data_test.csv"), index=True)

    df_out.to_csv(os.path.join(args.output_data, "data.csv"), index=True)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    # input and output arguments
    args = parser_args()
    main(args)