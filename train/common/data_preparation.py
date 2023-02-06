from typing import Optional, Tuple, List
import os
from src.io.path_definition import get_file
import pandas as pd
from src.common.tools import load_yaml_file
from src.api import logger
from src.common.load_data import load_data
from src.common.feature_engineering import create_total_stays_night, create_number_of_allpeople, create_nationality_code, create_if_comment, create_check_in_month, stays_night_is_national_holiday, create_important_sp_date, stays_night_is_holiday ,stays_night_is_weekday, create_is_weekday
from sklearn.impute import SimpleImputer
from src import config


def load_training_data(hotel_ids: Optional[List], remove_business_booking: bool=True) -> Tuple[pd.DataFrame, pd.Series]:

    '''
    Load sellout data as a pandas DataFrame
    '''

    df = load_data()

    if hotel_ids:
        df = df[df['pms_hotel_id'].isin(hotel_ids)]

    if remove_business_booking:
        df = df[df["source"] != "BUSINESS_BOOKING"]

    df = df[~(df['status'] == 'UPCOMING')]

    # feature engineering:
    df = create_total_stays_night(df=df)
    df = create_number_of_allpeople(df=df)
    df = create_nationality_code(df=df)
    df = create_if_comment(df=df)
    df = create_check_in_month(df=df)
    df = stays_night_is_national_holiday(df=df)
    df = stays_night_is_holiday(df=df)
    df = create_important_sp_date(df=df)
    df = create_is_weekday(df=df)
    df = stays_night_is_weekday(df=df)


    features_configuration = \
    load_yaml_file(get_file(os.path.join('config', 'training_config.yml')))['features_configuration'][config.configuration]
    onehot = features_configuration['onehot']
    numerical = features_configuration['numerical']

    df.loc[:, numerical] = df[numerical].fillna(0)

    simpleimputer = SimpleImputer(strategy='most_frequent')
    df.loc[:, onehot] = simpleimputer.fit_transform(df[onehot])


    df['label'] = 0
    df.loc[df['status'] == 'CHECKED_IN', 'label'] = 0
    df.loc[df['status'] == 'CHECKED_OUT', 'label'] = 0
    df.loc[df['status'] == 'NO_SHOW', 'label'] = 0
    df.loc[df['status'] == 'CANCELED', 'label'] = 1

    y = df['label']

    return df, y