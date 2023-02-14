import os
import yaml

from src.api import logger
from src import config
from src.io.path_definition import get_file
from src.common.load_data import retrieve_hyperparameter_files


def load_pbounds(algorithm: str):

    training_config = load_yaml_file(get_file(os.path.join('config', 'training_config.yml')))
    pbounds = training_config[f'pbounds'][algorithm]
    for key, value in pbounds.items():
        pbounds[key] = eval(value)

    return pbounds

#限制哪些feature可以放到模型內
#def load_x_labels(configuration: str):

    #features_configuration = \
    #load_yaml_file(get_file(os.path.join('config', 'training_config.yml')))['features_configuration'][configuration]
    #x_labels = []
    #for key, values in features_configuration.items():
        #config.features_configuration[key] = values
        #if type(values) == str:
            #x_labels.append(values)
        #else:
            #x_labels += values

    #return x_labels


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

    return params, target_max
#
# if __name__ == "__main__":
#
#     load_optimized_parameters(algorithm='unification')