import argparse
import os

from src.common.tools import chi2_pipeline

from src import config
from src.io.path_definition import get_file
from src.common.tools import load_yaml_file
from train.common.data_preparation import load_training_data


def set_configuration():

    config.algorithm = args.algorithm
    config.hotel_ids = args.hotel_ids
    config.configuration = args.configuration

    features_configuration = \
        load_yaml_file(get_file(os.path.join('config', 'training_config.yml')))['features_configuration'][
            args.configuration]
    for key, values in features_configuration.items():
        config.features_configuration[key] = values


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--configuration', type=str, help='"A", please check config/training_config.yml')
    parser.add_argument('--hotel_ids', nargs='+', type=int, help='hotel ids')
    parser.add_argument('--algorithm', type=str)

    args = parser.parse_args()

    set_configuration()

    # load_training_data: 載入數據集
    dataset, _ = load_training_data(hotel_ids=args.hotel_ids, remove_business_booking=True)
    chi2_pipeline(dataset)






