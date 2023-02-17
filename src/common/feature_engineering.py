import os
from functools import partial
from datetime import datetime

import pandas as pd

from src.io.path_definition import get_datafetch


def create_total_stays_night(df: pd.DataFrame):

    nums_of_stays_nights = pd.to_datetime(df.check_out) - pd.to_datetime(df.check_in)
    df['total_stays_night'] = nums_of_stays_nights
    df.loc[:, 'total_stays_night'] = df.loc[:, 'total_stays_night'].apply(lambda x: x.days)

    return df


def create_number_of_allpeople(df: pd.DataFrame):

    df['adults'].fillna(0, inplace=True)
    df['children'].fillna(0, inplace=True)
    df["number_of_allpeople"] = df.adults + df.children
    filter_ = (df.adults == 0) & (df.children == 0)

    return df[~filter_]


def create_nationality_code(df: pd.DataFrame):

    df['nationality_code'] = 0
    df.loc[df['nationality'] == 'TW', 'nationality_code'] = 1

    return df


def create_new_currency_code(df: pd.DataFrame):

    df['new_currency_code'] = 0
    df.loc[df['currency_code'] == 'TWD', 'new_currency_code'] = 1

    return df


def create_if_comment(df: pd.DataFrame):

    df['if_comment'] = ~pd.isnull(df['comment'])
    df['if_comment'] = df['if_comment'].astype(int)

    return df


def create_check_in_month(df: pd.DataFrame):

    df['check_in_month'] = pd.to_datetime(df['check_in'], errors='coerce')
    df['check_in_month'] = df['check_in_month'].dt.month

    return df


def create_important_sp_date(df: pd.DataFrame):

    df.loc[df['sp_date'].isin(['白色情人節', '西洋情人節', '七夕情人節', '父親節', '母親節', '聖誕節']), "important_sp_date"] = 1

    return df


def stays_night_is_national_holiday(df: pd.DataFrame):

    """
    #入住日當中是否有遇到國定連假 (遇到幾次)
    :param df:
    :return:
    """

    filename = os.path.join(get_datafetch(), '有影響的國定假日表格(到2023年底).csv')
    create_national_holiday_name = pd.read_csv(filename)
    all_holidays = create_national_holiday_name['date'].values.tolist()
    all_holidays = [datetime.strptime(c, "%Y-%m-%d") for c in all_holidays]
    f = partial(create_is_national_holiday, all_holidays=all_holidays)
    df['stay_night_is_national_holiday'] = df.apply(lambda x: f(x), axis=1)

    return df


def create_is_national_holiday(x, all_holidays):

    check_in = x['check_in']
    check_out = x['check_out']
    check_in = datetime.strptime(check_in, "%Y-%m-%d")
    check_out = datetime.strptime(check_out, "%Y-%m-%d")

    acc_range = pd.date_range(check_in, check_out)

    n = 0

    for d in acc_range:
        if d in all_holidays:
            n += 1

    return n


def stays_night_is_holiday(df: pd.DataFrame):

    """
    #入住日當中是否有遇到五六也就是假日 (遇到幾次)
    :param df:
    :return:
    """

    df['stay_night_is_holiday'] = df.apply(lambda x: _stay_night_is_holiday_fn(x), axis=1)

    return df


def _stay_night_is_holiday_fn(x):

    check_in = x['check_in']
    check_out = x['check_out']
    check_in = datetime.strptime(check_in, "%Y-%m-%d")
    check_out = datetime.strptime(check_out, "%Y-%m-%d")

    acc_range = pd.date_range(check_in, check_out)

    n = 0

    for d in acc_range:
        if d.weekday() in [4, 5]:
            n += 1

    return n


def create_is_weekday(df: pd.DataFrame):

    """
    與working day不太一樣，working day是指六日為非工作日，一到五為需要工作日。而create is weekday是對應holiday，指一二三四日為weekday，五六為holiday
    :param df:
    :return:
    """

    df['create_is_weekday'] = 1
    df.loc[df['weekday'] == '4', 'create_is_weekday'] = 0
    df.loc[df['weekday'] == '5', 'create_is_weekday'] = 0

    return df


def stays_night_is_weekday(df: pd.DataFrame):
    """
    入住日當中是否有遇到一二三四日也就是平日 (遇到幾次)
    :param df:
    :return:
    """

    df['stay_night_is_weekday'] = df.apply(lambda x: _stay_night_is_weekday_fn(x), axis=1)

    return df


def _stay_night_is_weekday_fn(x):

    check_in = x['check_in']
    check_out = x['check_out']
    check_in = datetime.strptime(check_in, "%Y-%m-%d")
    check_out = datetime.strptime(check_out, "%Y-%m-%d")

    acc_range = pd.date_range(check_in, check_out)

    n = 0

    for d in acc_range:
        if d.weekday() in [0, 1, 2, 3, 6]:
            n += 1

    return n
