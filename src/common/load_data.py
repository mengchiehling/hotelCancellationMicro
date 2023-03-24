import os
import re
from datetime import datetime
from typing import List
from src.io.path_definition import get_file

import pandas as pd
#from sklearn.impute import SimpleImputer

from src.api import logger
from src import config
from src.io.path_definition import get_datafetch


def retrieve_hyperparameter_files(algorithm: str, last: bool = False) -> List:

    dir_ = os.path.join(get_datafetch(), 'optimization')
    hotel_ids = config.hotel_ids

    if isinstance(hotel_ids, list):
        if config.ts_split:

            search_pattern = 'logs_' + algorithm + f"_{config.configuration}" + f"_tssplit_{hotel_ids[0]}"+ "_[\d]{8}-[\d]{4}.json"
        else:

            search_pattern = 'logs_' + algorithm + f"_{config.configuration}" + f"_{hotel_ids[0]}" + "_[\d]{8}-[\d]{4}.json"
    else:
        if config.ts_split:

            search_pattern = 'logs_' + algorithm + f"_{config.configuration}" + "_tssplit_unification_[\d]{8}-[\d]{4}.json"
        else:

            search_pattern = 'logs_' + algorithm + f"_{config.configuration}" + "_unification_[\d]{8}-[\d]{4}.json"

    logger.debug(f"retrieve file pattern {search_pattern}")

    res = [f for f in os.listdir(dir_) if re.search(search_pattern, f)]
    files = [os.path.join(dir_, f) for f in res]

    files_with_time = [(file, datetime.fromtimestamp(os.path.getmtime(file))) for file in files]

    files_with_time.sort(key=lambda x: x[1])

    if last:
        files = [files_with_time[-1][0]]
    else:
        files = [f[0] for f in files_with_time]

    return files


def load_data() -> pd.DataFrame:

    filename = os.path.join(get_datafetch(), '訂單資料_20221229.csv')
    booking_data = pd.read_csv(filename, index_col=0)
    booking_data.set_index('number', inplace=True)

    filename = os.path.join(get_datafetch(), '訂房資料_20221202.csv')
    room_data = pd.read_csv(filename, index_col=0)
    room_data = room_data.drop_duplicates(subset=['number'], keep='first').set_index('number')

    booking_data = booking_data.join(room_data[['lead_time', 'platform', 'holiday', 'weekday', 'pms_room_type_id', 'lead_time_range']], how='inner')

    filename = os.path.join(get_datafetch(), 'date_features.csv')
    date_features = pd.read_csv(filename)
    date_features['date'] = date_features['date'].apply(
     lambda x: datetime.strptime(x, '%Y/%m/%d').strftime("%Y-%m-%d"))
    booking_data = booking_data.merge(date_features, how='left', left_on='check_in', right_on='date')

    filename = os.path.join(get_datafetch(), '房型資料_20221229.csv')
    room_type_data = pd.read_csv(filename, index_col=0)
    room_type_data1 = room_type_data.set_index(["room_type_id"])['type'].to_dict()
    booking_data['type'] = booking_data['pms_room_type_id'].map(room_type_data1)

    # Attach the hotel information
    filename = os.path.join(get_datafetch(), 'hotel_info.csv')
    hotel_data = pd.read_csv(filename, index_col=0, encoding='utf-8')
    # 這邊要寫pms hotel id ,還是hotel id
    booking_data = booking_data.join(hotel_data, how='left', on='pms_hotel_id')
    booking_data.dropna(subset=['所在縣市'], inplace=True)
    booking_data['date_filter'] = pd.to_datetime(booking_data['date'])
    booking_data = booking_data[booking_data['date_filter'] < pd.to_datetime('2022-12-29')]
    return booking_data


