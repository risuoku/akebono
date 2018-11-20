import pandas as pd


class Dataset:
    """
    Dataset class
    """

    def __init__(self, fname, value, target_column, evacuated_columns):
        if not isinstance(value, pd.DataFrame):
            raise TypeError('value must be pandas.DataFrame')
        self._fname = fname
        self._value = value
        self._target_column = target_column
        self._evacuated_columns = evacuated_columns
    
    def get_predictor_target(self):
        """
        Datasetが管理するデータの説明変数と目的変数のtupleを返す

        Dataset生成時に目的変数が設定されていない場合は例外発生

        :return: tuple(pandas.DataFrame, pandas.Series) object
        """
        if self._target_column is None:
            raise Exception('target done not exist.')
        y = self._value[self._target_column]
        X = self.get_predictor()
        return X, y

    def get_predictor(self):
        columns = list(self._value.columns.copy())
        if self._target_column is not None:
            columns.remove(self._target_column)
        for c in self._evacuated_columns:
            if c in columns:
                columns.remove(c)
        return self._value[columns]

    @property
    def name(self):
        return self._fname

    def get_evacuated(self):
        return self._value[self._evacuated_columns]
    
    @property
    def value(self):
        """
        Datasetが管理するデータの実体

        :return: pandas.DataFrame object
        """
        return self._value
