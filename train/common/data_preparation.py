from typing import Optional, Tuple, List

import pandas as pd

from src.api import logger
from src.common.load_data import load_data
from src.common.feature_engineering import create_total_stays_night, create_number_of_allpeople, create_nationality_code

def load_training_data(hotel_ids: Optional[List], remove_business_booking: bool=True) -> Tuple[pd.DataFrame, pd.Series]:

    '''
    Load sellout data as a pandas DataFrame
    '''

    df = load_data()

    if hotel_ids:
        df = df[df['pms_hotel_id'].isin(hotel_ids)]

    if remove_business_booking:
        df = df[df["source"] != "BUSINESS_BOOKING"]

    df = df[df["source"] != "BUSINESS_BOOKING"]

    # feature engineering:
    df = create_total_stays_night(df=df)
    df = create_number_of_allpeople(df=df)
    df = create_nationality_code(df=df)

    df['label'] = 0
    df.loc[df['status'] == 'CHECKED_IN', 'label'] = 0
    df.loc[df['status'] == 'CHECKED_OUT', 'label'] = 0
    df.loc[df['status'] == 'NO_SHOW', 'label'] = 0
    df.loc[df['status'] == 'CANCELED', 'label'] = 1

    y = df['label']

    return df, y