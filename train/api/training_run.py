import argparse
import os
import joblib
import importlib
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.io.path_definition import get_datafetch, get_file
from src.common.tools import load_pbounds, load_optimized_parameters, load_yaml_file
from train.common.optimization import optimization_process
from train.common.model_selection import cross_validation
from train.common.data_preparation import load_training_data
# from train.logic.training_process_lightgbm import process
# from train.common.evaluation import run_evaluation


def create_dataset(dataset: pd.DataFrame, test_size):

    """
    等同於網路上的train_test_split步驟
    :param dataset:
    :param test_size:
    :return:
    """

    RANDOM_SEED = 42

    unique_hotel_ids = np.unique(dataset['pms_hotel_id'].values)

    train_dataset_list = []
    eval_dataset_list = []
    train_target_list = []
    eval_target_list = []

    for hotel_id in unique_hotel_ids:
        df_hotel = dataset[dataset['pms_hotel_id'] == hotel_id]
        y = df_hotel['label']
        x_train, x_eval, y_train, y_eval = train_test_split(df_hotel, y, test_size=test_size, shuffle=True,
                                                            random_state=RANDOM_SEED)
        train_dataset_list.append(x_train)
        eval_dataset_list.append(x_eval)
        train_target_list.append(y_train)
        eval_target_list.append(y_eval)

    x_train = pd.concat(train_dataset_list)
    x_eval = pd.concat(eval_dataset_list)
    y_train = pd.concat(train_target_list)
    y_eval = pd.concat(eval_target_list)

    return x_train, x_eval, y_train, y_eval


def export_final_model(dataset, test_size: float, evaluation: bool = False):

    y = dataset['label']

    params, _ = load_optimized_parameters(algorithm=args.algorithm)

    module = importlib.import_module(f'train.logic.training_process_{config.algorithm}')
    process = getattr(module, 'process')

    model = process(dataset, y, test_size=test_size, **params)
    dir_ = os.path.join(get_datafetch(), 'model')
    if not os.path.isdir(dir_):
        os.makedirs(dir_)

    if evaluation:
        filename = os.path.join(dir_, f'{args.algorithm}_{config.configuration}_evaluation.sav')
    else:
        filename = os.path.join(dir_, f'{args.algorithm}_{config.configuration}.sav')

    joblib.dump(model, filename)


def set_configuration():

    config.class_weight = args.class_weight
    config.algorithm = args.algorithm
    config.configuration = args.configuration

    features_configuration = \
        load_yaml_file(get_file(os.path.join('config', 'training_config.yml')))['features_configuration'][
            args.configuration]
    for key, values in features_configuration.items():
        config.features_configuration[key] = values


if __name__ == "__main__":

    model_name = 'micro'

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_size', type=float, help='Fraction of data for model testing')
    parser.add_argument('--env', type=str, help='"dev", "prod"')
    parser.add_argument('--configuration', type=str, help='"A", please check config/training_config.yml')
    parser.add_argument('--class_weight', type=str)
    parser.add_argument('--algorithm', type=str)

    args = parser.parse_args()

    set_configuration()

    pbounds = load_pbounds(algorithm=args.algorithm)

    training_data, _ = load_training_data(remove_business_booking=True)

    training_data.sort_values(by='check_in', inplace=True)

    train_dataset, test_dataset, train_target, test_target = create_dataset(training_data, test_size=args.test_size)

    cross_validation_fn = partial(cross_validation, data=train_dataset, y_label='label',
                                  optimization=True, test_size=args.test_size)

    _ = optimization_process(cross_validation_fn, pbounds, env=args.env)

    export_final_model(dataset=training_data, test_size=args.test_size)

    export_final_model(dataset=train_dataset, test_size=args.test_size, evaluation=True)
