import os
import argparse
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, \
    mean_absolute_percentage_error, roc_auc_score, roc_curve

from src import config
from train.common.timeseries_prediction import timeseries_prediction
from src.api import logger
from src.io.path_definition import get_datafetch
from train.common.data_preparation import load_training_data
from train.api.training_run import create_dataset
from src.io.load_model import load_model


def run_mape_evaluation(df: pd.DataFrame, pic_name):

    y_true = df['label'].values
    y_pred = df['y_pred'].values

    mape = mean_absolute_percentage_error(y_true + 1, y_pred + 1)
    logger.debug("MAPE值: {:.2f}".format(mape))

    y_abs_diff = np.abs(y_true - y_pred)
    wmape = y_abs_diff.sum() / y_true.sum()
    logger.debug("WMAPE值: {:.2f}".format(wmape))

    fig, ax = plt.subplots()
    ax.plot(y_true, color="red", label="The actual number of canceled orders")
    ax.plot(y_pred, color="blue", label="The predict number of canceled orders")
    ax.set_xlabel("Check in date")
    ax.set_ylabel("Canceled orders")
    plt.legend()
    plt.savefig(f"{pic_name}.png")


def run_evaluation_log(y_true, y_pred, y_pred_proba):

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)

    logger.debug("測試準確度: {:.2f}".format(acc))
    logger.debug("-----------------------------")
    logger.debug("F1值: {:.2f}".format(f1))
    logger.debug("-----------------------------")
    logger.debug("Recall值: {:.2f}".format(recall))
    logger.debug("-----------------------------")
    logger.debug("Precision值: {:.2f}".format(precision))
    logger.debug("-----------------------------")
    logger.debug("AUC值: {:.2f}".format(auc))
    logger.debug("-----------------------------")
    logger.debug("混淆矩陣如下: ")
    logger.debug("\n")
    logger.debug(cm)


def run_timeseries_aggregation(df: pd.DataFrame, hotel_id: Optional[str] = None):

    df_grouped = df.groupby(by="check_in")[["y_pred", 'label']].sum()
    algorithm = args.algorithm

    if hotel_id is not None:
        if config.ts_split:
            filepath = os.path.join(get_datafetch(),
                                f'predictResult(no fill zero)_{algorithm}_tssplit_{hotel_id}_{config.configuration}.csv')
        else:
            filepath = os.path.join(get_datafetch(),
                                f'predictResult(no fill zero)_{algorithm}_{hotel_id}_{config.configuration}.csv')
    else:
        if config.ts_split:
            filepath = os.path.join(get_datafetch(),
                                f'predictResult(no fill zero)_{algorithm}_tssplit_unification_{config.configuration}.csv')
        else:
            filepath = os.path.join(get_datafetch(),
                                f'predictResult(no fill zero)_{algorithm}_unification_{config.configuration}.csv')
    df_grouped.to_csv(filepath)

    run_mape_evaluation(df_grouped, "no_fill_zero")
    df_grouped = timeseries_prediction(df_grouped)
    run_mape_evaluation(df_grouped, "fill_zero")
    if hotel_id is not None:
        if config.ts_split:
            filepath = os.path.join(get_datafetch(),
                                    f'predictResult(fill zero)_{algorithm}_tssplit_{hotel_id}_{config.configuration}.csv')
        else:
            filepath = os.path.join(get_datafetch(),
                                    f'predictResult(fill zero)_{algorithm}_{hotel_id}_{config.configuration}.csv')
    else:
        if config.ts_split:
            filepath = os.path.join(get_datafetch(),
                                    f'predictResult(fill zero)_{algorithm}_tssplit_unification_{config.configuration}.csv')
        else:
            filepath = os.path.join(get_datafetch(),
                                    f'predictResult(fill zero)_{algorithm}_unification_{config.configuration}.csv')
    df_grouped.to_csv(filepath)


def run_evaluation(model_, df: pd.DataFrame, filename: str):

    y_pred_proba = model_.predict_proba(df)
    y_pred = (y_pred_proba[:,1] > 0.5) * 1
    # y_pred = model.predict(eval_dataset)

    df['y_pred'] = y_pred
    df['y_pred_proba'] = y_pred_proba[:, 1]

    # 全部旅館訂房模型表現
    logger.debug("全旅館")
    y_true = df['label']
    run_evaluation_log(y_true, y_pred, y_pred_proba[:, 1])
    run_timeseries_aggregation(df)

    # 個別旅館
    unique_hotel_ids = np.unique(df['pms_hotel_id'].values)

    for hotel_id in unique_hotel_ids:
        logger.debug(f"旅館-{hotel_id}")
        eval_dataset = df[df['pms_hotel_id'] == hotel_id]
        eval_y = eval_dataset['label']
        y_pred = eval_dataset['y_pred']
        y_pred_proba = eval_dataset['y_pred_proba']
        run_evaluation_log(eval_y, y_pred, y_pred_proba)
        run_timeseries_aggregation(eval_dataset)
    print("\n")
    df_grouped = df.groupby(by="check_in")[["y_pred", 'label']].sum()
    algorithm = args.algorithm
    if config.ts_split:
        df_grouped.to_csv(os.path.join(get_datafetch(), f'predictResult(no fill zero)_tssplit_{algorithm}_{filename}_{config.configuration}.csv'))
        run_mape_evaluation(df_grouped,"no_fill_zero")
        df_grouped = timeseries_prediction(df_grouped)
        run_mape_evaluation(df_grouped,"fill_zero")
        df_grouped.to_csv(os.path.join(get_datafetch(), f'predictResult(fill zero)_tssplit_{algorithm}_{filename}_{config.configuration}.csv'))
    else:
        df_grouped.to_csv(os.path.join(get_datafetch(), f'predictResult(no fill zero)_{algorithm}_{filename}_{config.configuration}.csv'))
        run_mape_evaluation(df_grouped,"no_fill_zero")
        df_grouped = timeseries_prediction(df_grouped)
        run_mape_evaluation(df_grouped,"fill_zero")
        df_grouped.to_csv(os.path.join(get_datafetch(), f'predictResult(fill zero)_{algorithm}_{filename}_{config.configuration}.csv'))

    df['mismatch'] = (df['y_pred'] != df['label']).astype(int)
    df = df[df['mismatch'] == 1]
    df.to_csv(os.path.join(get_datafetch(), f'QA_{filename}.csv'))


def set_configuration():

    config.algorithm = args.algorithm
    config.hotel_ids = args.hotel_ids
    config.configuration = args.configuration
    config.ts_split = args.ts_split


if __name__ == "__main__":

    model_name = 'micro'

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_size', type=float, help='Fraction of data for model testing')
    parser.add_argument('--configuration', type=str, help='"A", please check config/training_config.yml')
    parser.add_argument('--hotel_ids', nargs='+', type=int, help='hotel ids')
    parser.add_argument('--algorithm', type=str)
    parser.add_argument('--ts_split', action='store_true')
    args = parser.parse_args()

    set_configuration()

    dataset, _ = load_training_data(hotel_ids=args.hotel_ids, remove_business_booking=True)

    train_dataset, test_dataset, train_target, test_target = create_dataset(dataset, test_size=args.test_size)

    if isinstance(args.hotel_ids, list):
        hotel_id = args.hotel_ids[0]
        filename = str(hotel_id)
    else:
        hotel_id = None
        filename = 'unification'

    model = load_model(hotel_id=hotel_id)

    run_evaluation(model_=model, df=test_dataset, filename=filename)
