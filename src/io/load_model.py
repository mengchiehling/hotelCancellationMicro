import os
import joblib
from typing import Optional
from src import config
from src.io.path_definition import get_datafetch


def load_model(hotel_id: Optional[int]):

    algorithm = config.algorithm
    dir_ = os.path.join(get_datafetch(), 'model')
    if hotel_id is not None:
        model = joblib.load(os.path.join(dir_, f'{algorithm}_{config.configuration}_{hotel_id}_evaluation.sav'))
    else:
        model = joblib.load(os.path.join(dir_, f'{algorithm}_{config.configuration}_unification_evaluation.sav'))

    return model
