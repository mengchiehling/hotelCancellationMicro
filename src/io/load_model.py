import os
import joblib
from typing import Optional
from src import config
from src.io.path_definition import get_datafetch


def load_model(hotel_id: Optional[int]):

    algorithm = config.algorithm
    dir_ = os.path.join(get_datafetch(), 'model')

    if config.ts_split:
        basic_filename = os.path.join(dir_, f"{config.algorithm}_{config.configuration}_tssplit")
    else:
        basic_filename = os.path.join(dir_, f"{config.algorithm}_{config.configuration}")

    hotel_ids = config.hotel_ids

    if isinstance(hotel_ids, list):
        basic_filename = basic_filename + f"_{hotel_ids[0]}"
    else:
        basic_filename = basic_filename + "_unification"

    if config.class_weight:
        basic_filename = basic_filename + f"_{config.class_weight}"


    filename_ = basic_filename + "_evaluation.sav"
    model = joblib.load(filename_)

    return model
