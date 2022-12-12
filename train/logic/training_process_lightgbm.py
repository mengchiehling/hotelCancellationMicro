import os
from typing import Union

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.io.path_definition import get_file
from src.common.tools import load_yaml_file


def build_transformer_pipeline(configuration: str, stats=None) -> Union[ColumnTransformer, Pipeline]:

    """

    Returns:
        a scikit-learn pipeline
    """

    features_configuration = load_yaml_file(get_file(os.path.join('config', 'training_config.yml')))['features_configuration'][configuration]

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


def process(X: pd.Series, y: pd.Series, test_size: float, configuration: str, **kwargs):


    feature_extractor = build_transformer_pipeline(stats=chi2, configuration=configuration)

    model =  Pipeline([('feature_extractor', feature_extractor),
                       ('model', LGBMClassifier(boosting_type='gbdt', random_state=0, class_weight='balanced',
                                                n_estimators=3000, objective='softmax', n_jobs=-1))
                       ])

    k = kwargs.get('k', None)
    if k:
        k = int(k)
        model.set_params(**{f"feature_extractor__feature_selector__feature_selector__k": k})

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

    # ToDo replace with time series splitting
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=test_size, random_state=42)

    callbacks = [early_stopping(stopping_rounds=100), log_evaluation(period=100)]

    try:
        model_temp = Pipeline(model.steps[:-1])
        model_temp.fit_transform(X_train, y_train)
        eval_set = [(model_temp.transform(X_val), y_val)]
    except ValueError:
        model.set_params(**{f"feature_extractor__feature_selector__feature_selector__k": 'all'})
        model_temp = Pipeline(model.steps[:-1])
        model_temp.fit_transform(X_train, y_train)
        eval_set = [(model_temp.transform(X_val), y_val)]

    # For example, setting it to 100 means we stop the training if the predictions have not improved for
    # the last 100 rounds.
    # https://stackoverflow.com/questions/40329576/sklearn-pass-fit-parameters-to-xgboost-in-pipeline/55711752#55711752

    model.fit(X_train, y_train, model__eval_set=eval_set, model__callbacks=callbacks, model__eval_metric=['softmax'])

    return model