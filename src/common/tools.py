import os
import yaml

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.api import logger
from src import config
from src.io.path_definition import get_file
from src.common.load_data import retrieve_hyperparameter_files


def load_pbounds():

    pbounds = load_yaml_file(get_file(os.path.join('config', 'training_config.yml')))['pbounds']
    for key, value in pbounds.items():
        pbounds[key] = eval(value)

    return pbounds


def load_x_labels(configuration: str):

    features_configuration = \
    load_yaml_file(get_file(os.path.join('config', 'training_config.yml')))['features_configuration'][configuration]
    x_labels = []
    for key, values in features_configuration.items():
        config.features_configuration[key] = values
        if type(values) == str:
            x_labels.append(values)
        else:
            x_labels += values

    return x_labels


def load_yaml_file(filepath: str):

    with open(filepath, 'r') as stream:
        map_ = yaml.safe_load(stream)

    return map_


def load_optimized_parameters(algorithm: str, last: bool=False):

    logger.debug(f"loading optimized parameters for algorithm={algorithm}")

    files = retrieve_hyperparameter_files(algorithm=algorithm, last=last)

    target_max = 0

    for f in files:
        with open(f, 'rb') as f:
            while True:
                data = f.readline()
                if not data:
                    break
                data = eval(data)
                target = data['target']
                if target > target_max:
                    target_max = target
                    params = data['params']

    return params, target

#創建虛構日期
def create_fictitous_date(df: pd.DataFrame):

    date_list = df['check_in'].values.tolist()
    date_list.sort()
    date_start = date_list[0]
    date_end = date_list[-1]
    idx = pd.date_range(date_start, date_end)
    idx = [t.strftime("%Y-%m-%d") for t in idx]
    date_feature = pd.DataFrame(data=np.zeros([len(idx), 1]), index=idx)

    return date_feature


def timeseries_train_test_split(df, test_size):
    '''
    #切法1
    date_feature = create_fictitous_date(df)
    train_time, eval_time = train_test_split(date_feature, test_size=test_size, shuffle=False)
    train_dataset = df[df['check_in'].isin(train_time.index)]
    eval_dataset = df[df['check_in'].isin(eval_time.index)]
    train_target = train_dataset['label']
    eval_target = eval_dataset['label']
    '''

    #切法2，
    train_time, test_time = train_test_split(np.unique(df['check_in']), test_size=test_size, shuffle=True, random_state=0)
    # train_date_feature = create_fictitous_date(df[df['check_in'].isin(train_time)])
    # test_date_feature = create_fictitous_date(df[df['check_in'].isin(test_time)])
    train_dataset = df[df['check_in'].isin(train_time)]
    eval_dataset = df[df['check_in'].isin(test_time)]
    train_target = train_dataset['label']
    eval_target = eval_dataset['label']

    return train_dataset, eval_dataset, train_target, eval_target