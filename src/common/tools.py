import os
import yaml

import pandas as pd
import numpy as np
from scipy.stats import chisquare

from src.api import logger
from src import config
from src.io.path_definition import get_file
from src.common.load_data import retrieve_hyperparameter_files
from sklearn.model_selection import train_test_split


def load_pbounds(algorithm: str):

    training_config = load_yaml_file(get_file(os.path.join('config', 'training_config.yml')))
    pbounds = training_config[f'pbounds'][algorithm]
    for key, value in pbounds.items():
        pbounds[key] = eval(value)

    return pbounds


def load_yaml_file(filepath: str):

    with open(filepath, 'r', encoding='utf-8') as stream:
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

    return params, target_max


def timeseries_train_test_split(df, test_size):

    # 切法1
    # date_feature = create_fictitous_date(df)
    # train_time, eval_time = train_test_split(date_feature, test_size=test_size, shuffle=False)
    # train_dataset = df[df['check_in'].isin(train_time.index)]
    # eval_dataset = df[df['check_in'].isin(eval_time.index)]
    # train_target = train_dataset['label']
    # eval_target = eval_dataset['label']

    # 切法2
    train_time, test_time = train_test_split(np.unique(df['check_in']), test_size=test_size, shuffle=False, random_state=0)
    train_dataset = df[df['check_in'].isin(train_time)]
    eval_dataset = df[df['check_in'].isin(test_time)]
    train_target = train_dataset['label']
    eval_target = eval_dataset['label']

    return train_dataset, eval_dataset, train_target, eval_target


def chi2(dataset: pd.DataFrame, column: str):

    """
    Reference: https://www.youtube.com/watch?v=rpKzq64GA9Y
    :param dataset:
    :param column:
    :return:
    """

    unique_categories = np.unique(dataset[column])
    kept_categories = []
    for category in unique_categories:
        if dataset[dataset[column] == category]['label'].nunique() != 1:
            kept_categories.append(category)

    dataset = dataset[dataset[column].isin(kept_categories)]

    if len(dataset) == 0:
        return 0, 0

    df_group = dataset.groupby([column, 'label']).agg(count=('label', 'count')).reset_index()

    count_of_label_1 = df_group[df_group['label'] == 1]['count'].sum()
    count_of_label_2 = df_group[df_group['label'] == 0]['count'].sum()

    for category in kept_categories:

        df_group.loc[(df_group[column] == category) & (df_group['label'] == 1), 'expectation'] = \
            count_of_label_1 / (df_group['count'].sum()) * \
            df_group[df_group[column] == category]['count'].sum()

        df_group.loc[(df_group[column] == category) & (df_group['label'] == 0), 'expectation'] = \
            count_of_label_2 / (df_group['count'].sum()) * \
            df_group[df_group[column] == category]['count'].sum()
    # df = (n_col - 1) * (n_row - 1)
    # df = n_col * n_row - n_col - n_row + 1

    # chisquare, df = k - 1 - ddof

    ddof = 2 + len(np.unique(dataset[column])) - 2

    chisq, p = chisquare(df_group['count'].values, df_group['expectation'].values, ddof=ddof)

    return chisq, np.round(p, 7)


def chi2_pipeline(dataset: pd.DataFrame) -> pd.DataFrame:

    data = []

    for c in config.features_configuration['onehot']:
        try:
            chisq, p = chi2(dataset=dataset, column=c)
        except ValueError:
            chi2(dataset=dataset, column=c)
        data.append((c, chisq, p))

    df = pd.DataFrame(data=data, columns=['feature', 'chi2', 'p'])

    return df