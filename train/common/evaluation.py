import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, \
    mean_absolute_percentage_error, roc_auc_score, roc_curve

from src.api import logger
from src.io.path_definition import get_datafetch


def run_mape_evaluation(df: pd.DataFrame):

    df_grouped = df.groupby(by="check_in")[["pred", 'label']].sum()

    y_true = df_grouped['label'].values
    y_pred = df_grouped['pred'].values

    mape = mean_absolute_percentage_error(y_true + 1, y_pred + 1)

    logger.debug("MAPE值: {:.4f}".format(mape))


def run_evaluation(model, eval_dataset: pd.DataFrame, filename: str):

    y_pred_proba = model.predict_proba(eval_dataset)
    y_pred = (y_pred_proba > 0.5) * 1
    #y_pred = model.predict(eval_dataset)

    eval_y = eval_dataset['label']

    acc = accuracy_score(eval_y, y_pred)
    f1 = f1_score(eval_y, y_pred)
    recall = recall_score(eval_y, y_pred)
    precision = precision_score(eval_y, y_pred)
    cm = confusion_matrix(eval_y, y_pred)
    auc = roc_auc_score(eval_y, y_pred_proba)

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

    run_mape_evaluation(eval_dataset)

    eval_dataset['mismatch'] = (eval_dataset['pred'] != eval_dataset['label']).astype(int)
    eval_dataset = eval_dataset[eval_dataset['mismatch'] == 1]
    eval_dataset.to_csv(os.path.join(get_datafetch(), f'QA_{filename}.csv'))