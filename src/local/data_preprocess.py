
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataCleaning:
    #Inicializamos cargando la data
    def __init__(self, data_path) -> None:
        self.data_path = data_path
        self.dataframe = pd.read_csv(data_path, delimiter=",")

    #Creamos un método para manejar los valores que estén nulos
    def handle_missing_values(self):
        self.dataframe = self.dataframe.dropna() #Método para eliminar las filas con valores faltantes de (ojo!! pueden interpolarse o rellenarse)

    #Creamos un método para manejar variables que no son numéricas
    def handle_categorical_features(self):
        df_copy = self.dataframe.copy()

        for col in df_copy.columns:
            if df_copy[col].dtype == object:
                le = LabelEncoder()

                df_copy[col] = le.fit_transform(df_copy[col])

        self.dataframe =  df_copy
    
    #Creamos un método para manejar la clase objetivo de manera desbalanceada (si lo está)
    def handle_imbalanced_data(self, objetive):
        pass

    #Creamos un método para escalar las caracteristicas que queramos
    def scalate_features(self, feature):
        pass

    #Creamos un método para identificar y eliminar caracteristicas que sean irrelevantes
    def remove_irrelevant_features(self):
        pass

    #Creamos un método para realizar el split de la data
    def split_data(self, target_feature, size):
        X = self.dataframe.drop(columns=[target_feature])
        y = self.dataframe[target_feature]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=size, random_state=42)