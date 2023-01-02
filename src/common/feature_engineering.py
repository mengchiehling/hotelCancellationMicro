import pandas as pd


def create_total_stays_night(df: pd.DataFrame):

    nums_of_stays_nights = pd.to_datetime(df.check_out) - pd.to_datetime(df.check_in)
    df['total_stays_night'] = nums_of_stays_nights
    df.loc[:, 'total_stays_night'] = df.loc[:, 'total_stays_night'].apply(lambda x: x.days)

    return df


def create_number_of_allpeople(df: pd.DataFrame):

    df["number_of_allpeople"] = df.adults + df.children

    filter = (df.adults == 0) & (df.children == 0)

    return df[~filter]


def create_nationality_code(df: pd.DataFrame):

    df['nationality_code'] = 0
    df.loc[df['nationality'] == 'TW', 'nationality_code'] = 1

    return df


def create_new_currency_code(df: pd.DataFrame):

    df['new_currency_code'] = 0
    df.loc[df['currency_code'] == 'TWD', 'new_currency_code'] = 1

    return df
