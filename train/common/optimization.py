import os
import re
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from src.io.path_definition import get_datafetch, get_file
from src.common.tools import load_yaml_file


def optimization_process(fn, pbounds: Dict, algorithm: str, env: str) -> Tuple[Dict, np.ndarray]:
    """
    Bayesian optimization process interface. Returns hyperparameters of machine learning algorithms and the
    corresponding out-of-fold (oof) predictions. The progress will be saved into a json file.
    Args:
        fn: functional that will be optimized
        pbounds: a dictionary having the boundary of parameters of fn
    Returns:
        A tuple of dictionary containing optimized hyperparameters and oof-predictions
    """

    training_config = load_yaml_file(get_file(os.path.join('config', 'training_config.yml')))

    bayesianOptimization = training_config['bayesianOptimization'][env]

    optimizer = BayesianOptimization(
        f=fn,
        pbounds=pbounds,
        random_state=1)

    export_form = datetime.now().strftime("%Y%m%d-%H%M")

    optimization_file_dir = os.path.join(get_datafetch(), 'optimization')

    if not os.path.isdir(optimization_file_dir):
        os.makedirs(optimization_file_dir)

    logs = f"{optimization_file_dir}/logs_{algorithm}_{export_form}.json"

    search_pattern = 'logs_' + algorithm + "_[\d]{8}-[\d]{4}.json"
    res = [f for f in os.listdir(optimization_file_dir) if re.search(search_pattern, f)]
    previous_logs = [os.path.join(optimization_file_dir, f) for f in res]

    if previous_logs:
        load_logs(optimizer, logs=previous_logs)
        bayesianOptimization['init_points'] = 0

    logger = JSONLogger(path=logs)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

    optimizer.maximize(
        **bayesianOptimization
    )
    optimized_parameters = optimizer.max['params']

    return optimized_parameters