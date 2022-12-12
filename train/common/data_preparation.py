from typing import Optional, Tuple, List

import pandas as pd

from src.api import logger
from src.common.load_data import load_data


def load_training_data(hotel_ids: Optional[List], remove_business_booking: bool=True) -> Tuple[pd.DataFrame, pd.Series]:

    '''
    Load sellout data as a pandas DataFrame
    '''

    df = load_data()

    if hotel_ids:
        df = df[df['pms_hotel_id'].isin(hotel_ids)]

    if remove_business_booking:
        pass

    df['label'] = 0
    df.loc[df['status']=='CANCELED', 'label'] = 1

    y = df['label']

    return df, y