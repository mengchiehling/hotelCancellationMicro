import argparse
import os
import joblib
from typing import Optional
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src import config
from src.io.path_definition import get_datafetch
from src.io.load_model import load_model


def set_configuration():

    config.algorithm = 'lightgbm'
    config.hotel_ids = args.hotel_id
    config.configuration = args.configuration


def run_prediction(model_, eval_dataset: pd.DataFrame):

    y_pred_proba = model_.predict_proba(eval_dataset)
    y_pred = (y_pred_proba[:,1] > 0.5) * 1

    eval_dataset['pred'] = y_pred
    df_grouped = eval_dataset.groupby(by="check_in")[["pred"]].sum()
    df_grouped = df_grouped.reindex(idx).fillna(0)

    filename = f"{model_name}_{args.hotel_id}"

    df_grouped.to_csv(os.path.join(get_datafetch(), f'predict_{filename}.csv'))


if __name__ == "__main__":

    model_name = 'micro'

    parser = argparse.ArgumentParser()

    parser.add_argument('--configuration', type=str, help='"A", please check config/training_config.yml')
    parser.add_argument('--hotel_id', type=int, help='hotel id')
    parser.add_argument('--time_start', type=str)  # 'YYYY-mm-dd'
    parser.add_argument('--timespan', type=int)

    args = parser.parse_args()

    set_configuration()

    model = load_model(hotel_id=args.hotel_id)

    time_start = datetime.strptime(args.time_start, "%Y-%m-%d")
    time_end = (time_start + timedelta(days=args.timespan)).strftime("%Y-%m-%d")

    idx = pd.date_range(time_start, time_end)
    idx = [t.strftime("%Y-%m-%d") for t in idx]

    # load input data, sql dependent

    from train.common.data_preparation import load_training_data

    dataset, _ = load_training_data(hotel_ids=[args.hotel_id], remove_business_booking=True)
    pred_dataset = dataset[dataset['check_in'].isin(idx)]

    run_prediction(model_=model, eval_dataset=pred_dataset)