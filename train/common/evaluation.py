import os
import argparse
from src import config
import matplotlib.pyplot as plt
from typing import Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, \
    mean_absolute_percentage_error, roc_auc_score, roc_curve
from train.common.timeseries_prediction import timeseries_prediction
from src.api import logger
from src.io.path_definition import get_datafetch
from train.common.data_preparation import load_training_data
from train.api.training_run_lightgbm import create_dataset
#from src.io.load_model import load_lightgbm_model


def load_lightgbm_model(hotel_id: Optional[int]):

    dir_ = os.path.join(get_datafetch(), 'model')
    if hotel_id is not None:
        model = joblib.load(os.path.join(dir_, f'micro_{config.configuration}_{hotel_id}_evaluation.sav'))
    else:
        model = joblib.load(os.path.join(dir_, f'micro_{config.configuration}_unification_evaluation.sav'))

    return model

def run_mape_evaluation(df: pd.DataFrame, pic_name):

    #df_grouped = df.groupby(by="check_in")[["pred", 'label']].sum()
    #df_grouped = timeseries_prediction(df_grouped)
    y_true = df['label'].values
    y_pred = df['pred'].values

    mape = mean_absolute_percentage_error(y_true + 1, y_pred + 1)

    logger.debug("MAPE值: {:.2f}".format(mape))

    plt.plot(y_true, color="red", label="The actual number of canceled orders")
    plt.plot(y_pred, color="blue", label="The predict number of canceled orders")
    #plt.title("Hotel 294 canceled orders prediction: LinearSVC")
    plt.xlabel("Check in date")
    plt.ylabel("Canceled orders")
    plt.legend()
    plt.savefig(f"{pic_name}.png")


def run_evaluation(model, eval_dataset: pd.DataFrame, filename: str):

    y_pred_proba = model.predict_proba(eval_dataset)
    y_pred = (y_pred_proba[:,1] > 0.5) * 1
    #y_pred = model.predict(eval_dataset)

    eval_y = eval_dataset['label']

    acc = accuracy_score(eval_y, y_pred)
    f1 = f1_score(eval_y, y_pred)
    recall = recall_score(eval_y, y_pred)
    precision = precision_score(eval_y, y_pred)
    cm = confusion_matrix(eval_y, y_pred)
    auc = roc_auc_score(eval_y, y_pred_proba[:,1])

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

    eval_dataset['pred'] = y_pred
    df_grouped = eval_dataset.groupby(by="check_in")[["pred", 'label']].sum()

    df_grouped.to_csv(os.path.join(get_datafetch(), f'predictResult(no fill zero)_{filename}.csv'))

    run_mape_evaluation(df_grouped,"no_fill_zero")
    df_grouped = timeseries_prediction(df_grouped)
    run_mape_evaluation(df_grouped,"fill_zero")
    df_grouped.to_csv(os.path.join(get_datafetch(), f'predictResult(fill zero)_{filename}.csv'))

    eval_dataset['mismatch'] = (eval_dataset['pred'] != eval_dataset['label']).astype(int)
    eval_dataset = eval_dataset[eval_dataset['mismatch'] == 1]
    eval_dataset.to_csv(os.path.join(get_datafetch(), f'QA_{filename}.csv'))


def set_configuration():

    config.algorithm = 'lightgbm'
    config.hotel_ids = args.hotel_ids
    config.configuration = args.configuration

if __name__ == "__main__":

    model_name = 'micro'

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_size', type=float, help='Fraction of data for model testing')
    parser.add_argument('--configuration', type=str, help='"A", please check config/training_config.yml')
    parser.add_argument('--hotel_ids', nargs='+', type=int, help='hotel ids')

    args = parser.parse_args()

    set_configuration()

    dataset, _ = load_training_data(hotel_ids=args.hotel_ids, remove_business_booking=True)

    train_dataset, test_dataset, train_target, test_target = create_dataset(dataset, test_size=args.test_size)

    #x_labels = load_x_labels(configuration=args.configuration)

    filename = f'{model_name}'

    if isinstance(args.hotel_ids, list):
        hotel_id = args.hotel_ids[0]
        filename= str(hotel_id)
    else:
        hotel_id = None
        filename = 'unification'

    #export_final_model(dataset=dataset, test_size=args.test_size)

    #export_final_model(dataset=train_dataset, test_size=args.test_size, evaluation=True)

    model = load_lightgbm_model(hotel_id=hotel_id)

    run_evaluation(model=model, eval_dataset=test_dataset, filename=filename)