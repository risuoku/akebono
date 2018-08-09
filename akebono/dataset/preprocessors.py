import pandas as pd


def select_columns(df, columns=[]):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be pandas.DataFrame')
    return df[columns]


def exclude_columns(df, columns=[]):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be pandas.DataFrame')
    s_columns = list(df.columns())
    for c in columns:
        if c in s_columns:
            s_columns.remove(c)
    return df[s_columns]


def identify(data, **param):
    return data
