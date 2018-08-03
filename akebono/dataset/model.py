import pandas as pd


class Dataset:
    def __init__(self, value, target_column=None):
        if not isinstance(value, pd.DataFrame):
            raise TypeError('value must be pandas.DataFrame')
        self._value = value
        self._target_column = target_column
    
    def get_predictor_target(self):
        if self._target_column is None:
            raise Exception('target done not exist.')
        y = self._value[self._target_column]
        columns = list(self._value.columns.copy())
        columns.remove(self._target_column)
        X = self._value[columns]
        return X, y
    
    @property
    def value(self):
        return self._value
