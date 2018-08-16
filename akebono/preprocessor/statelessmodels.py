from akebono.logging import getLogger
import pandas as pd


logger = getLogger(__name__)


class StatelessPreprocessor:
    """
    状態を持たないPreprocessor
    """

    def process(self, df_train, df_test):
        """
        前処理を実行するためのメソッド

        :param df_train: 訓練データセットが持つ説明変数
        :type df_train: pandas.DataFrame
        :param df_test: テストデータセットが持つ説明変数
        :type df_test: pandas.DataFrame or None
        :return: tuple(pandas.DataFrame, pandas.DataFrame) object
        """
        raise NotImplementedError()


class Identify(StatelessPreprocessor):
    """
    入力データをそのまま返すPreprocessor
    """

    def process(self, df_train, df_test):
        logger.debug('Identify#process invoked.')
        return df_train, df_test


class SelectColumns(StatelessPreprocessor):
    """
    入力データから、指定したカラムのみを選択して返すPreprocessor

    :param columns: 選択対象カラム名のリスト
    :type columns: list[str]
    """
    def __init__(self, columns=[]):
        self._columns = columns

    def process(self, df_train, df_test):
        logger.debug('SelectColumns#process invoked.')
        if not isinstance(df_train, pd.DataFrame):
            raise TypeError('df_train must be pandas.DataFrame')
        r_dftrain = r_dftest = None
        r_dftrain = df_train[self._columns]
        if df_test is not None:
            r_dftest = df_test[self._columns] 
        return r_dftrain, r_dftest


class ExcludeColumns(StatelessPreprocessor):
    """
    入力データから、指定したカラムのみを除外して返すPreprocessor

    :param columns: 選択対象カラム名のリスト
    :type columns: list[str]
    """
    def __init__(self, columns=[]):
        self._columns = columns

    def process(self, df_train, df_test):
        if not isinstance(df_train, pd.DataFrame):
            raise TypeError('df_train must be pandas.DataFrame')
        s_columns = list(df_train.columns)
        for c in self._columns:
            if c in s_columns:
                s_columns.remove(c)

        r_dftrain = r_dftest = None
        r_dftrain = df_train[s_columns]
        if df_test is not None:
            r_dftest =df_test[s_columns] 
        return r_dftrain, r_dftest
