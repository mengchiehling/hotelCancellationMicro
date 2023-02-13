import argparse
import os
import joblib
from functools import partial
import importlib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.api import logger
from src import config
from src.io.path_definition import get_datafetch, get_file
from src.io.load_model import load_model
from src.common.tools import load_x_labels, load_pbounds, load_optimized_parameters, load_yaml_file
from train.common.optimization import optimization_process
from train.common.model_selection import cross_validation
from train.common.data_preparation import load_training_data
#from train.logic.training_process_lightgbm import process
#from train.common.evaluation import run_evaluation

#等同於網路上的train_test_split步驟
def create_dataset(dataset: pd.DataFrame, test_size):

    RANDOM_SEED = 42
    y = dataset['label']
    train_dataset, eval_dataset, train_target, eval_target = train_test_split(dataset, y,
                                                                              test_size=test_size, shuffle=True,random_state=RANDOM_SEED)

    return train_dataset, eval_dataset, train_target, eval_target

#儲存模型
def export_final_model(dataset, test_size: float, evaluation:bool=False):

    train_target = dataset['label']

    params, _ = load_optimized_parameters(algorithm=args.algorithm)

    module = importlib.import_module(f'train.logic.training_process_{config.algorithm}')
    process = getattr(module, 'process')

    model = process(dataset, train_target, test_size=test_size, **params)
    dir_ = os.path.join(get_datafetch(), 'model')
    if not os.path.isdir(dir_):
        os.makedirs(dir_)

    if evaluation:
        filename = os.path.join(dir_, f'{args.algorithm}_{config.configuration}_{hotel_id}_evaluation.sav')
    else:
        filename = os.path.join(dir_, f'{args.algorithm}_{config.configuration}_{hotel_id}.sav')

    joblib.dump(model, filename)


def set_configuration():

    config.class_weight = args.class_weight
    config.algorithm =  args.algorithm  #'lightgbm'
    config.hotel_ids = args.hotel_ids
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
    parser.add_argument('--hotel_ids', nargs='+', type=int, help='hotel ids')
    parser.add_argument('--class_weight', type=str)
    parser.add_argument('--algorithm', type=str)

    args = parser.parse_args()

    set_configuration()

    pbounds = load_pbounds(algorithm=args.algorithm)
    # load_training_data: 載入數據集
    dataset, _ = load_training_data(hotel_ids=args.hotel_ids, remove_business_booking=True)

    dataset.sort_values(by='check_in', inplace=True)
    # train test split
    train_dataset, test_dataset, train_target, test_target = create_dataset(dataset, test_size=args.test_size)

    #x_labels = load_x_labels(configuration=args.configuration)
    #cross_validation
    cross_validation_fn = partial(cross_validation, data=train_dataset,
                                  y_label='label', optimization=True, test_size=args.test_size)   #x_labels=x_labels

    #filename = f'{model_name}'
    _ = optimization_process(cross_validation_fn, pbounds, env=args.env)

    if isinstance(args.hotel_ids, list):
        hotel_id = args.hotel_ids[0]
        filename= str(hotel_id)
    else:
        hotel_id = None
        filename = 'unification'
    #export_final_model
    export_final_model(dataset=dataset, test_size=args.test_size)

    export_final_model(dataset=train_dataset, test_size=args.test_size, evaluation=True)


