import pandas as pd


class Dataset:
    """
    Dataset class
    """

    def __init__(self, value, target_column):
        if not isinstance(value, pd.DataFrame):
            raise TypeError('value must be pandas.DataFrame')
        self._value = value
        self._target_column = target_column
    
    def get_predictor_target(self):
        """
        Datasetが管理するデータの説明変数と目的変数のtupleを返す

        Dataset生成時に目的変数が設定されていない場合は例外発生

        :return: tuple(pandas.DataFrame, pandas.Series) object
        """
        if self._target_column is None:
            raise Exception('target done not exist.')
        y = self._value[self._target_column]
        columns = list(self._value.columns.copy())
        columns.remove(self._target_column)
        X = self._value[columns]
        return X, y
    
    @property
    def value(self):
        """
        Datasetが管理するデータの実体

        :return: pandas.DataFrame object
        """
        return self._value
