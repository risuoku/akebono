import pandas as pd


def identify(df):
    """
    入力データをそのまま返す関数

    :param df: データセットが持つ説明変数
    :type df: pandas.DataFrame
    :return: pandas.DataFrame object
    """

    return df


def select_columns(df, columns=[]):
    """
    入力データから、指定したカラムのみを選択して返す関数

    :param df: データセットが持つ説明変数
    :type df: pandas.DataFrame
    :param columns: 選択対象カラム名のリスト
    :type columns: list[str]
    :return: pandas.DataFrame object
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be pandas.DataFrame')
    return df[columns]


def exclude_columns(df, columns=[]):
    """
    入力データから、指定したカラムのみを除外して返す関数

    :param df: データセットが持つ説明変数
    :type df: pandas.DataFrame
    :param columns: 選択対象カラム名のリスト
    :type columns: list[str]
    :return: pandas.DataFrame object
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be pandas.DataFrame')
    s_columns = list(df.columns)
    for c in columns:
        if c in s_columns:
            s_columns.remove(c)
    return df[s_columns]
