import pandas as pd


def select_columns(df, columns=[]):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be pandas.DataFrame')
    return df[columns]


def identify(df):
    return df
