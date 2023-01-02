import os
import re
from datetime import datetime
from typing import List

import pandas as pd

from src.api import logger
from src import config
from src.io.path_definition import get_datafetch


def retrieve_hyperparameter_files(algorithm: str, last: bool=False) -> List:

    dir_ = os.path.join(get_datafetch(), 'optimization')

    if isinstance(config.hotel_ids, list):
        search_pattern = 'logs_' + algorithm + f"_{config.hotel_ids[0]}" + "_[\d]{8}-[\d]{4}.json"
    else:
        search_pattern = 'logs_' + algorithm + "_unification_[\d]{8}-[\d]{4}.json"

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
    booking_data = pd.read_csv(filename, index_col=0)
    booking_data.set_index('number', inplace=True)

    filename = os.path.join(get_datafetch(), '訂房資料_20221202.csv')
    room_data = pd.read_csv(filename, index_col=0)
    room_data = room_data.drop_duplicates(subset=['number'], keep='first').set_index('number')

    booking_data = booking_data.join(room_data[['lead_time', 'platform', 'season', 'holiday', 'weekday']], how='inner')

    return booking_data

