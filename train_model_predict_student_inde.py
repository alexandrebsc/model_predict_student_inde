import random
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import shapiro
from pathlib import Path

DATA_FODLER_PATH = f'{Path(__file__).parent.resolve()}\\data'
RANDOM_SEED = 1

class c:
    first_year = 'INGRESSANTE'
    name = 'NOME'
    year = 'ANO'
    level = 'FASE'
    age = 'IDADE'
    inde = 'INDE'
    ian = 'IAN'
    iaa = 'IAA'
    ieg = 'IEG'
    ida = 'IDA'
    ips = 'IPS'
    ipp = 'IPP'
    ipv = 'IPV'

def main():
    set_random_seeds(RANDOM_SEED)
    df = get_students_df()
    df = data_cleaning(df)
    
    #df_analysis(df)
    #df_vizulization(df)
    
    df_train, df_test = get_df_train_and_test(df)
    #df_train = data_augmentation(df_train) #intentionally commented, not optimal
    df_train, df_test, df_complete = pre_processing(df_train), pre_processing(df_test), pre_processing(df)
    
    train_and_dump_model(df_train, df_test, df_complete)

def set_random_seeds(seed_value: int) -> None:
    np.random.seed(seed_value)
    random.seed(seed_value)

def get_students_df() -> pd.DataFrame:
    file_path = './PEDE_PASSOS_DATASET_FIAP.csv'
    pd.set_option('display.max_columns', None)
    return pd.read_csv(
            file_path, 
            encoding='utf-8',
            delimiter=';',
            engine="python",
        )

def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    def __main(df: pd.DataFrame) -> pd.DataFrame:
        df = df[['NOME', 'IDADE_ALUNO_2020', 'ANOS_PM_2020', 'FASE_TURMA_2020', 'INDE_2020', 'IAA_2020','IAN_2020','IEG_2020','IDA_2020', 'SINALIZADOR_INGRESSANTE_2021', 'FASE_2021',  'INDE_2021',  'IEG_2021', 'IAA_2021', 'IDA_2021',  'IAN_2021', 'ANO_INGRESSO_2022', 'FASE_2022', 'INDE_2022', 'IEG_2022','IAA_2022', 'IDA_2022','IAN_2022', 'IPS_2020', 'IPS_2021', 'IPS_2022', 'IPP_2020', 'IPP_2021', 'IPP_2022', 'IPV_2020', 'IPV_2021', 'IPV_2022']]
        
        df = drop_corrupted_date(df)
        df = fill_age_data(df)
        df = preprocess_is_first_year(df)
        df = preprocess_level_data(df)
        df = cast_values(df)
        df = unpivot_by_year(df)
        df = df.dropna(subset=df.columns.difference([c.name, c.year, c.first_year, c.age]), how='all')
        df = df.astype({
            c.year:int,
            c.level:int,
            c.age:int,
            c.inde:float,
            c.iaa:float,
            c.ian:float,
            c.ida:float,
            c.ieg:float,
            c.ipp:float,
            c.ips:float,
            c.ipv:float,
            c.first_year:bool,
        })
        
        return df.loc[df[c.level] < 8]
        
    def drop_corrupted_date(df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index('NOME')
        return df.drop(['ALUNO-1259','ALUNO-506','ALUNO-71'], axis=0)
    def fill_age_data(df: pd.DataFrame) -> pd.DataFrame:
        def expected_age_in_2021(age_in_2020: int):
            AVERAGE_2021_AGE = 13
            return (age_in_2020 + 1) if not np.isnan(age_in_2020) else AVERAGE_2021_AGE
        def expected_age_in_2022(age_in_2020: int):
            AVERAGE_2023_AGE = 13
            return (age_in_2020 + 2) if not np.isnan(age_in_2020) else AVERAGE_2023_AGE
        
        df['IDADE_2020'] = pd.to_numeric(df['IDADE_ALUNO_2020'], downcast='integer')
        df['IDADE_2021'] = df['IDADE_2020'].apply(expected_age_in_2021)
        df['IDADE_2022'] = df['IDADE_2020'].apply(expected_age_in_2022)
        return df
    def preprocess_is_first_year(df: pd.DataFrame) -> pd.DataFrame:
        df['INGRESSANTE_2020'] = df['ANOS_PM_2020'].apply(lambda x: x!='0' if x else None)
        df['INGRESSANTE_2021'] = df['SINALIZADOR_INGRESSANTE_2021'].apply(lambda x: x=='Ingressante' if x else None)
        df['INGRESSANTE_2022'] = pd.to_numeric(df['ANO_INGRESSO_2022'], downcast='integer').apply(lambda x: x==2022 if x else None)
        df = df.drop(['ANOS_PM_2020', 'ANO_INGRESSO_2022', 'SINALIZADOR_INGRESSANTE_2021'], axis=1)
        return df
    def preprocess_level_data(df: pd.DataFrame) -> pd.DataFrame:
        df['FASE_2020'] = pd.to_numeric(df['FASE_TURMA_2020'].str[0], downcast='integer')
        df = df.drop('FASE_TURMA_2020', axis=1)
        df['FASE_2021'] = pd.to_numeric(df['FASE_2021'], downcast='integer')
        df['FASE_2022'] = pd.to_numeric(df['FASE_2022'], downcast='integer')
        return df
    def cast_values(df: pd.DataFrame) -> pd.DataFrame:
        return df.astype({
            'IDADE_2020':'Int32',
            'IDADE_2021':'Int32',
            'IDADE_2022':'Int32',
            'FASE_2020':'Int32',
            'FASE_2021':'Int32',
            'FASE_2022':'Int32',
            'INDE_2020':float,
            'IAA_2020':float,
            'IAN_2020':float,
            'IEG_2020':float,
            'IDA_2020':float,
            'IPS_2020':float,
            'IPP_2020':float,
            'IPV_2020':float,
            'INDE_2021':float,
        })
    def unpivot_by_year(df: pd.DataFrame) -> pd.DataFrame:
        df = df.reset_index()
        df = pd.melt(df,
                    id_vars = 'NOME',
                    value_vars=['IDADE_2020', 'IDADE_2021', 'IDADE_2022',
                                'INGRESSANTE_2020', 'INGRESSANTE_2021', 'INGRESSANTE_2022',
                                'FASE_2020', 'FASE_2021', 'FASE_2022',
                                'INDE_2020', 'INDE_2021', 'INDE_2022',
                                'IAA_2020', 'IAA_2021', 'IAA_2022',
                                'IAN_2020', 'IAN_2021', 'IAN_2022',
                                'IEG_2020', 'IEG_2021', 'IEG_2022',
                                'IDA_2020', 'IDA_2021', 'IDA_2022',
                                'IPS_2020', 'IPS_2021', 'IPS_2022', 
                                'IPP_2020', 'IPP_2021', 'IPP_2022', 
                                'IPV_2020', 'IPV_2021', 'IPV_2022',],
                    var_name='COLUNA_ANO', 
                    value_name='value')
        
        df[['COLUNA', 'ANO']] = df['COLUNA_ANO'].str.split('_', expand=True)
        df = df.drop('COLUNA_ANO', axis=1)
        df = df.pivot_table(index=['NOME','ANO'], columns='COLUNA', values='value').reset_index()
        return df
    
    return __main(df)

def df_analysis(df: pd.DataFrame) -> None:
    def shapiro_test() -> None:
        warnings.filterwarnings("ignore", category=UserWarning, message="p-value may not be accurate for N > 5000.")
        
        stat, p = shapiro(df[c.inde])
        
        print(f'Statistics (W): {stat:.4f}\n'
            f'Value p: {p:.4f}')

    def null_analysis() -> None:
        df_nulls = df.isnull()
        
        print('Quantidade de nulos: ', df_nulls.sum())
        
    def duplicated_analysis() -> None:
        df_duplicated = df.duplicated()
        
        print('Quantidade de duplicados: ',df_duplicated.sum())
        print(df[df_duplicated])
    
    print(f"> Info:\n")
    df.info()
    print('\n')
    print(f"> Análise dados nulos:\n\n")
    null_analysis()
    print(f"> Análise dados duplicados:\n\n")
    duplicated_analysis()
    print(f"> Df sample:\n\n{df.head(3)}", end='\n\n')
    print(f"> Df statistics:\n\n{df.describe()}", end='\n\n')
    print(f"> Shapiro test on percentual variation by day:\n\n")
    shapiro_test()
    print('\n')

def data_augmentation(df: pd.DataFrame) -> pd.DataFrame:
    def get_augmented_df(base_df: pd.DataFrame, column: str, value: float) -> pd.DataFrame:
        __df = base_df.copy()
        __df[column] = __df[column] + value
        __df[column] = __df[column].clip(lower=0.0, upper=10.0)
        return __df
    
    for augmentation_group in [ ((c.iaa, 1), (c.iaa, -1)), 
                                ((c.ieg, 0.66), (c.ieg, -0.66)),
                                ((c.ida, 1), (c.ida, -1)),]:
        dfs_to_concat = []
        for augmentation in augmentation_group:
            dfs_to_concat.append(get_augmented_df(df, augmentation[0], augmentation[1]))
        
        df = pd.concat(dfs_to_concat + [df], ignore_index=True)
        df = df.drop_duplicates()
    
    def calculate_inde(row):
        return (row[c.ian] * 0.1 + 
                row[c.ida] * 0.2 + 
                row[c.ieg] * 0.2 +
                row[c.iaa] * 0.1 +
                row[c.ips] * 0.1 +
                row[c.ipp] * 0.1 +
                row[c.ipv] * 0.2) 
    
    df[c.inde] = df.apply(calculate_inde, axis=1)
    
    df = df.drop_duplicates()
    return df

def get_df_train_and_test(df: pd.DataFrame, test_percentage = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = shuffle(df)
    df_len = len(df)
    head_count = int(df_len*0.25)
    return df.head(df_len - head_count), df.tail(head_count)

def df_vizulization(df: pd.DataFrame) -> None:
    def hist_plot():
        fig, axs = plt.subplots(5, 1, figsize=(8, 8))
        
        df[[c.iaa, c.inde]].plot.hist(bins=100, alpha=0.5, ax=axs[0])
        axs[0].set_title('iaa vs inde')
        
        df[[c.ieg, c.inde]].plot.hist(bins=100, alpha=0.5, ax=axs[1])
        axs[1].set_title('ieg vs inde')

        df[[c.level, c.inde]].plot.hist(bins=100, alpha=0.5, ax=axs[2])
        axs[2].set_title('level vs inde')
        
        df[[c.age, c.inde]].plot.hist(bins=100, alpha=0.5, ax=axs[3])
        axs[3].set_title('age vs inde')
    
    def correlation_plot():
        corr_columns = list(df.columns)
        corr_columns.remove(c.name)
        correlation_matrix = df[corr_columns].corr().round(2)

        fig, ax = plt.subplots(figsize=(8,8))
        sns.heatmap(data=correlation_matrix, annot=True, linewidths=.5, ax=ax)
    
    #hist_plot()
    correlation_plot()
    plt.tight_layout()
    plt.show()

def pre_processing(df: pd.DataFrame) -> pd.DataFrame:
    class KeepFeatures(BaseEstimator,TransformerMixin):
        def __init__(self,feature_to_keep: list):
            self.feature_to_keep = feature_to_keep
        
        def fit(self,df):
            return self
        
        def transform(self,df):
            to_columns_drop = set(df.columns) - set(self.feature_to_keep)
            if to_columns_drop:
                return df.drop(to_columns_drop,axis=1)
            return df
    
    class DivideByScaler(BaseEstimator, TransformerMixin):
        def __init__(self, divisor:float, features:list):
            self.divisor = divisor
            self.features = features

        def fit(self,df):
            return self

        def transform(self,df):
            for feature in self.features:
                df[feature] = df[feature] / self.divisor
            return df
    
    class Shuffle(BaseEstimator,TransformerMixin):
        def fit(self,df):
            return self
        
        def transform(self,df):
            return shuffle(df)
    
    pre_processor = ColumnTransformer(
        transformers=[
            ('label_encoder',OrdinalEncoder(),[c.first_year]),],
        remainder='passthrough',
        verbose_feature_names_out = False)
    pre_processor.set_output(transform = 'pandas')
    
    feature_to_keep = [c.age,c.level,c.inde,c.ian,c.iaa,c.ieg,c.first_year]
    pipeline = Pipeline([
        ('keep_features', KeepFeatures(feature_to_keep)),
        ('divide_by_scaler_i', DivideByScaler(10.0, [c.inde,c.ian,c.iaa,c.ieg])),
        ('divide_by_scaler_level', DivideByScaler(7.0, [c.level])),
        ('divide_by_scaler_age', DivideByScaler(30.0, [c.age])),
        ('pre_processor', pre_processor),
        ('shuffle', Shuffle())
    ])
    
    df = pipeline.fit_transform(df)
    return df

def train_and_dump_model(df_train: pd.DataFrame, df_test: pd.DataFrame, df_complete: pd.DataFrame) -> None:
    def get_model_results(model, x, y):
        y_pred = model.predict(x)
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        max_diff = np.max(np.abs(y_pred - y))
        return mae, mse, r2, max_diff
    
    def grid_search():
        best_r2 = 0
        for n_estimators in [50]:
            for max_features in ['sqrt']:#['sqrt', 'log2']:
                for max_depth in [7,8,9,10,11,12,]:
                    model = RandomForestRegressor(
                        n_estimators = n_estimators, 
                        max_features = max_features, 
                        max_depth = max_depth, 
                        random_state = RANDOM_SEED)
                    model.fit(x_train, y_train)
                    
                    mae, mse, r2, max_diff = get_model_results(model,x_train,y_train)
                    print(f'train -> {n_estimators:03d}, {max_features}, {max_depth:02d}, {mae:0.3f}, {mse:0.4f}, {r2:0.3f}, {max_diff:0.3f}')
                    mae, mse, r2, max_diff = get_model_results(model,x_test,y_test)
                    print(f'test  -> {n_estimators:03d}, {max_features}, {max_depth:02d}, {mae:0.3f}, {mse:0.4f}, {r2:0.3f}, {max_diff:0.3f}')
                    mae, mse, r2, max_diff = get_model_results(model,x_complete,y_complete)
                    print(f'compt -> {n_estimators:03d}, {max_features}, {max_depth:02d}, {mae:0.3f}, {mse:0.4f}, {r2:0.3f}, {max_diff:0.3f}')
                    print()
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_result = [n_estimators, max_features, max_depth]
        
        print(f'Best results {best_result}')
    
    def train_final_model() -> RandomForestRegressor:
        model = RandomForestRegressor(
            n_estimators = 50, 
            max_features = 'sqrt', 
            max_depth = 11, 
            random_state = RANDOM_SEED)
        model.fit(x_train, y_train)
        
        for log in [('Comp', (x_complete,y_complete)),
                    ('Test', (x_test,y_test)),
                    ('Train', (x_train,y_train)),]:
            mae, mse, r2, max_diff = get_model_results(model,*log[1])
            print(f'{log[0]}  -> MAE: {mae:0.3f}; MSE: {mse:0.4f}; R2: {r2:0.3f}; MAX DIFF: {max_diff:0.3f}')
        
        return model
    
    def get_x_y_from_df(df):
        return df[[c.age,c.level,c.ian,c.iaa,c.ieg,c.first_year]].values, df[c.inde].values
    
    x_train, y_train = get_x_y_from_df(df_train)
    x_test, y_test = get_x_y_from_df(df_test)
    x_complete, y_complete = get_x_y_from_df(df_complete)
    
    #grid_search()
    model = train_final_model()
    joblib.dump(model,'random_forest_regressor_predict_student_inde.pkl')

if __name__ == '__main__':
    main()