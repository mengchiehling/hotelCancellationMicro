import os
import re
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from src import config
from src.io.path_definition import get_datafetch, get_file
from src.common.tools import load_yaml_file


def optimization_process(fn, pbounds: Dict, env: str) -> Tuple[Dict, np.ndarray]:
    """
    Bayesian optimization process interface. Returns hyperparameters of machine learning algorithms and the
    corresponding out-of-fold (oof) predictions. The progress will be saved into a json file.
    Args:
        fn: functional that will be optimized
        pbounds: a dictionary having the boundary of parameters of fn
        env
    Returns:
        A tuple of dictionary containing optimized hyperparameters and oof-predictions
    """

    training_config = load_yaml_file(get_file(os.path.join('config', 'training_config.yml')))

    bayesianoptimization = training_config['bayesianOptimization'][env]

    random_seed = 42
    optimizer = BayesianOptimization(
        f=fn,
        pbounds=pbounds,
        random_state=random_seed)

    export_form = datetime.now().strftime("%Y%m%d-%H%M")

    optimization_file_dir = os.path.join(get_datafetch(), 'optimization')

    if not os.path.isdir(optimization_file_dir):
        os.makedirs(optimization_file_dir)

    hotel_ids = config.hotel_ids
    algorithm = config.algorithm

    # 是否為單一間旅店
    if isinstance(hotel_ids, list):
        if config.ts_split:
            logs = f"{optimization_file_dir}/logs_{algorithm}_{config.configuration}_tssplit_{hotel_ids[0]}_{export_form}.json"
            search_pattern = 'logs_' + algorithm + f"_{config.configuration}" + f"_tssplit_{hotel_ids[0]}"+ "_[\d]{8}-[\d]{4}.json"
        else:
            logs = f"{optimization_file_dir}/logs_{algorithm}_{config.configuration}_{hotel_ids[0]}_{export_form}.json"
            search_pattern = 'logs_' + algorithm + f"_{config.configuration}" + f"_{hotel_ids[0]}" + "_[\d]{8}-[\d]{4}.json"
    else:
        if config.ts_split:
            logs = f"{optimization_file_dir}/logs_{algorithm}_{config.configuration}_tssplit_unification_{export_form}.json"
            search_pattern = 'logs_' + algorithm + f"_{config.configuration}" + "_tssplit_unification_[\d]{8}-[\d]{4}.json"
        else:
            logs = f"{optimization_file_dir}/logs_{algorithm}_{config.configuration}_unification_{export_form}.json"
            search_pattern = 'logs_' + algorithm + f"_{config.configuration}" + "_unification_[\d]{8}-[\d]{4}.json"

    res = [f for f in os.listdir(optimization_file_dir) if re.search(search_pattern, f)]
    previous_logs = [os.path.join(optimization_file_dir, f) for f in res]

    if previous_logs:
        load_logs(optimizer, logs=previous_logs)
        bayesianoptimization['init_points'] = 0

    logger = JSONLogger(path=logs)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        **bayesianoptimization
    )
    optimized_parameters = optimizer.max['params']

    return optimized_parameters
