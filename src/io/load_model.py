import os
import joblib
from typing import Optional
from src import config
from src.io.path_definition import get_datafetch


def load_lightgbm_model(hotel_id: Optional[int]):

    dir_ = os.path.join(get_datafetch(), 'model')
    if hotel_id is not None:
        model = joblib.load(os.path.join(dir_, f'micro_{config.configuration}_{hotel_id}.sav'))
    else:
        model = joblib.load(os.path.join(dir_, f'micro_{config.configuration}_unification.sav'))

    return model