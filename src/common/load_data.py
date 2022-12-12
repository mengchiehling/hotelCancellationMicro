import os
import re
from datetime import datetime
from typing import List

import pandas as pd

from src.io.path_definition import get_datafetch


def retrieve_hyperparameter_files(algorithm: str, last: bool=False) -> List:

    dir_ = os.path.join(get_datafetch(), 'optimization')

    search_pattern = 'logs_' + algorithm + "_[\d]{8}-[\d]{4}.json"

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

    filename = os.path.join(get_datafetch(), '訂單資料_20221202.csv')
    df_booking_info = pd.read_csv(filename, index_col=0)

    filename = os.path.join(get_datafetch(), '訂房資料_20221202.csv')
    df_booking_detail = pd.read_csv(filename, index_col=0)

    columns = ['platform', 'pms_room_type_id', 'lead_time', 'lead_time_range', 'weekday', 'week', 'month', 'year',
               'season', 'holiday', 'status']

    df_booking_info = df_booking_info.join(df_booking_detail[columns].drop_duplicates(subset=['number'], keep='first'),
                                           on=['number'])

    return df_booking_info

