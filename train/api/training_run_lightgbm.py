import argparse
import os
import joblib
from functools import partial

import pandas as pd
from sklearn.model_selection import train_test_split

from src.api import logger
from src.io.path_definition import get_datafetch
from src.common.tools import load_x_labels, load_pbounds, load_optimized_parameters
from train.common.optimization import optimization_process
from train.common.model_selection import cross_validation
from train.common.data_preparation import load_training_data
from train.logic.training_process_lightgbm import process


algorithm = 'lightgbm'

def create_dataset(dataset: pd.DataFrame, test_size):

    y = dataset['label']
    train_dataset, eval_dataset, train_target, eval_target = train_test_split(dataset, y, stratify=y,
                                                                              test_size=test_size, random_state=42)

    return train_dataset, eval_dataset, train_target, eval_target


def export_final_model(train_dataset, test_size: float, evaluation:bool=False):

    train_target = train_dataset['label']

    filename = f'{model_name}'

    params, _ = load_optimized_parameters(algorithm=filename)

    model = process(train_dataset, train_target, test_size=test_size, configuration=args.configuration, **params)

    dir_ = os.path.join(get_datafetch(), 'model')
    if not os.path.isdir(dir_):
        os.makedirs(dir_)

    if evaluation:
        filename = os.path.join(dir_, f'{filename}_evaluation.sav')
    else:
        filename = os.path.join(dir_, f'{filename}.sav')

    joblib.dump(model, filename)


if __name__ == "__main__":

    model_name = 'micro'

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_size', type=float, help='Fraction of data for model testing')
    parser.add_argument('--env', type=str, help='"dev", "prod"')
    parser.add_argument('--configuration', type=str, help='"A", please check config/training_config.yml')

    args = parser.parse_args()

    pbounds = load_pbounds()

    dataset, _ = load_training_data(hotel_ids=None, remove_business_booking=True)

    train_dataset, _, train_target, _ = create_dataset(dataset, test_size=args.test_size)

    x_labels = load_x_labels(configuration=args.configuration)

    cross_validation_fn = partial(cross_validation, algorithm=algorithm, data=train_dataset, x_labels=x_labels,
                                  y_label='label', optimization=True, test_size=args.test_size,
                                  configuration=args.configuration)

    filename = f'{model_name}'
    _ = optimization_process(cross_validation_fn, pbounds, algorithm=filename, env=args.env)

    export_final_model(dataset, args.test_size)

    export_final_model(train_dataset, args.test_size, evaluation=True)
