import os
from typing import Union

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.config import features_configuration, class_weight


def build_transformer_pipeline(stats=None) -> Union[ColumnTransformer, Pipeline]:

    """

    Returns:
        a scikit-learn pipeline
    """

    transformers = []
    for key, value in features_configuration.items():
        if key == 'numerical':
            transformers.append((key, 'passthrough', features_configuration[key]))
        elif key == 'onehot':
            transformers.append((key, OneHotEncoder(handle_unknown="ignore"), features_configuration[key]))

    column_transformer = ColumnTransformer(transformers=transformers, remainder="drop")

    feature_transformer = Pipeline([('column_transformer', column_transformer),
                                    ('feature_selector', SelectKBest(stats))])

    return feature_transformer


def process(X: pd.Series, y: pd.Series, test_size: float, **kwargs):
    '''
    For binary classfication, choose objective='binary'
    :param X:
    :param y:
    :param test_size:
    :param kwargs:
    :return:
    '''
    RANDOM_SEED = 42
    feature_extractor = build_transformer_pipeline(stats=chi2)

    model =  Pipeline([('feature_extractor', feature_extractor),
                       ('model', LGBMClassifier(boosting_type='gbdt', random_state=RANDOM_SEED, class_weight=class_weight,
                                                n_estimators=3000, objective='binary', n_jobs=1))
                       ])

    k = kwargs.get('k', None)
    if k:
        k = int(k)
        model.set_params(**{f"feature_extractor__feature_selector__k": k})

    reg_alpha = kwargs['reg_alpha']
    reg_lambda = kwargs['reg_lambda']
    learning_rate = kwargs['learning_rate']
    min_child_samples = int(kwargs['min_child_samples'])
    num_leaves = int(kwargs['num_leaves'])

    model.set_params(**{"model__reg_alpha": reg_alpha,
                        "model__reg_lambda": reg_lambda,
                        "model__learning_rate": learning_rate,
                        "model__min_child_samples": min_child_samples,
                        "model__num_leaves": num_leaves,
                        })

    #ToDo
    y = y.astype(np.int32).values
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=test_size, random_state=RANDOM_SEED)

    callbacks = [early_stopping(stopping_rounds=100), log_evaluation(period=100)]

    try:
        model_temp = Pipeline(model.steps[:-1])
        model_temp.fit_transform(X_train, y_train)
        eval_set = [(model_temp.transform(X_val), y_val)]
    except ValueError:
        model.set_params(**{f"feature_extractor__feature_selector__k": 'all'})
        model_temp = Pipeline(model.steps[:-1])
        model_temp.fit_transform(X_train, y_train)
        eval_set = [(model_temp.transform(X_val), y_val)]

    # For example, setting it to 100 means we stop the training if the predictions have not improved for
    # the last 100 rounds.
    # https://stackoverflow.com/questions/40329576/sklearn-pass-fit-parameters-to-xgboost-in-pipeline/55711752#55711752

    model.fit(X_train, y_train, model__eval_set=eval_set, model__callbacks=callbacks, model__eval_metric=['binary'])

    return model