
import pandas as pd
import holidays_co

class DataPreprocess:
    #Inicializamos cargando la data
    def __init__(self, data_path, index_column: str = 'Fecha Tx', target_column: str = 'Cantidad') -> None:
        self.data_path = data_path
        self.dataframe = pd.read_excel(data_path) #, header=1, index_col=0)
        self.index_column = index_column
        self.target_column = target_column
        self.output_dataframe = None
        self.data_train = None
        self.data_test = None

    #Creamos un método para manejar los valores que estén nulos
    def handle_missing_values(self):
        self.output_dataframe =self.output_dataframe.dropna()
        
    def handle_categorical_features(self):
        pass
    
    def handle_imbalanced_data(self, objetive):
        pass

    def scalate_features(self, feature):
        pass

    #método para identificar y eliminar caracteristicas que sean irrelevantes
    def remove_irrelevant_features(self,del_columns:list|str=['Id Cliente','Nom Producto','Cod Producto']):
        "metodo para eliminar columnas del dataframe"
        self.output_dataframe = self.dataframe.drop(del_columns,axis=1)

    def filter_dataframe_by_feature(self,filter_column:str='Cod SDV',filter_value:str=109216):
        "metodo para filtar registros por un valor de una columna especifica del dataframe"
        self.output_dataframe = self.output_dataframe[self.output_dataframe[filter_column]==filter_value]

    def grouped_dataframe_by_feature(self,grouped_column:str = 'Cod SDV'):
        "metodo para agrupar registros dataframe por un valor de una columna especifica del dataframe"
        target_column = [self.target_column]
        sel_columns = [self.index_column,self.target_column]
        self.output_dataframe = self.output_dataframe.groupby(by=[grouped_column, self.index_column])[target_column].sum().reset_index()[sel_columns]

    def completed_timeserie_df(self):
        df_copy = self.output_dataframe.copy()
        # Convertir la columna a tipo datetime si no lo está
        df_copy[self.index_column] = pd.to_datetime(df_copy[self.index_column])

        # Establecer la columna como índice del DataFrame
        df_copy = df_copy.set_index(self.index_column)

        # Generar un rango de fechas desde la primera hasta la última fecha en el DataFrame
        dates_range = pd.date_range(start=df_copy.index.min(), end=df_copy.index.max(), freq='D')

        # Reindexar el DataFrame con el rango de fechas generado
        df_copy = df_copy.reindex(dates_range, fill_value=0)
        
        self.output_dataframe = df_copy

    def feature_generation(self):

        df_copy = self.output_dataframe.copy().reset_index()

        # Agregar la columna de día de la semana en formato numérico (Lunes = 0, Domingo = 6)
        df_copy['DiaSemana'] = df_copy['index'].dt.weekday

        # Agregar la columna de mes en formato numérico
        df_copy['Mes'] = df_copy['index'].dt.month

        # Agregar la columna de mes en formato numérico
        df_copy['Dia'] = df_copy['index'].dt.day

        # quincena
        quincena =[15,30,31]
        df_copy['EsQuincena'] = df_copy['Dia'].map(lambda x: x in quincena)

        # Agregar la columna de festivo o día laboral
        df_copy['EsFestivo'] = df_copy['index'].map(lambda x: holidays_co.is_holiday_date(x))

        # Establecer la columna "Fecha Tx" como índice del DataFrame
        df_copy = df_copy.set_index('index')

        df_copy['media_movil'] = df_copy['Cantidad'].rolling(window=7).mean()

        df_copy['media_movil'] = df_copy['media_movil'].shift(1)

        df_copy = df_copy.asfreq('d')
        df_copy = df_copy.sort_index()
        
        self.output_dataframe = df_copy

    #Creamos un método para realizar el split de la data
    def split_data(self,split = 7):
        self.data_train = self.output_dataframe[:-split]
        self.data_test = self.output_dataframe[-split:]