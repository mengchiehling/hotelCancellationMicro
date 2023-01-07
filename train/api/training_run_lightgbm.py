import argparse
import os
import joblib
from functools import partial

import pandas as pd
from sklearn.model_selection import train_test_split

from src.api import logger
from src import config
from src.io.path_definition import get_datafetch
from src.io.load_model import load_lightgbm_model
from src.common.tools import load_x_labels, load_pbounds, load_optimized_parameters
from train.common.optimization import optimization_process
from train.common.model_selection import cross_validation
from train.common.data_preparation import load_training_data
from train.logic.training_process_lightgbm import process
from train.common.evaluation import run_evaluation


def create_dataset(dataset: pd.DataFrame, test_size):

    RANDOM_SEED = 42
    y = dataset['label']
    train_dataset, eval_dataset, train_target, eval_target = train_test_split(dataset, y, stratify=y,
                                                                              test_size=test_size, random_state=RANDOM_SEED)

    return train_dataset, eval_dataset, train_target, eval_target


def export_final_model(dataset, test_size: float, evaluation:bool=False):

    train_target = dataset['label']

    filename = f'{model_name}'

    params, _ = load_optimized_parameters(algorithm=filename)

    model = process(dataset, train_target, test_size=test_size, **params)

    dir_ = os.path.join(get_datafetch(), 'model')
    if not os.path.isdir(dir_):
        os.makedirs(dir_)

    if evaluation:
        filename = os.path.join(dir_, f'{filename}_{hotel_id}_evaluation.sav')
    else:
        filename = os.path.join(dir_, f'{filename}_{hotel_id}.sav')

    joblib.dump(model, filename)


def set_configuration():

    config.class_weight = args.class_weight
    config.algorithm = 'lightgbm'
    config.hotel_ids = args.hotel_ids


if __name__ == "__main__":

    model_name = 'micro'

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_size', type=float, help='Fraction of data for model testing')
    parser.add_argument('--env', type=str, help='"dev", "prod"')
    parser.add_argument('--configuration', type=str, help='"A", please check config/training_config.yml')
    parser.add_argument('--hotel_ids', nargs='+', type=int, help='hotel ids')
    parser.add_argument('--class_weight', type=str)

    args = parser.parse_args()

    set_configuration()

    pbounds = load_pbounds()

    dataset, _ = load_training_data(hotel_ids=args.hotel_ids, remove_business_booking=True)

    train_dataset, test_dataset, train_target, test_target = create_dataset(dataset, test_size=args.test_size)

    x_labels = load_x_labels(configuration=args.configuration)

    cross_validation_fn = partial(cross_validation, data=train_dataset, x_labels=x_labels,
                                  y_label='label', optimization=True, test_size=args.test_size)

    filename = f'{model_name}'
    _ = optimization_process(cross_validation_fn, pbounds, algorithm=filename, env=args.env)

    if isinstance(args.hotel_ids, list):
        hotel_id = args.hotel_ids[0]
        filename= str(hotel_id)
    else:
        hotel_id = None
        filename = 'unification'

    export_final_model(dataset=dataset, test_size=args.test_size)

    export_final_model(dataset=train_dataset, test_size=args.test_size, evaluation=True)

    model = load_lightgbm_model(hotel_id=hotel_id)

    run_evaluation(model=model, eval_dataset=test_dataset, filename=filename)


