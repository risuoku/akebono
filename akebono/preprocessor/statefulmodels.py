from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from akebono.io.operation.dumper import dump_sklearn_model
from akebono.io.operation.loader import load_sklearn_model
from akebono.logging import getLogger
import pandas as pd


logger = getLogger(__name__)


class StatefulPreprocessor:
    """
    状態を持つPreprocessor
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

    def reset(self):
        """
        前処理実体を初期化するメソッド
        """
        raise NotImplementedError()

    def set_operation_mode(self, mode):
        """
        operation_modeを設定するメソッド

        :param mode: 設定可能な値は `train` or `predict`
        :type mode: str
        """
        if mode not in ('train', 'predict'):
            raise ValueError('invalid mode')
        self._opmode = mode

    @property
    def operation_mode(self):
        if not hasattr(self, '_opmode'):
            self._opmode = None
        return self._opmode

    @property
    def value(self):
        """
        前処理実体
        """
        return self._value
    
    def dump(self, dirpath, name):
        """
        Preprocessorをストレージに永続化するためのメソッド

        :param dirpath: ストレージのパス
        :type dirpath: str
        :param name: ファイル名
        :type name: str
        """
        raise NotImplementedError()

    def load(self, dirpath, name):
        """
        ストレージに永続化されてるPreprocessorを復元するためのメソッド

        :param dirpath: ストレージのパス
        :type dirpath: str
        :param name: ファイル名
        :type name: str
        """
        raise NotImplementedError()


class ApplyStandardScaler(StatefulPreprocessor):
    """
    入力データを正規化するPreprocessor

    :param init_kwargs: 前処理実体の初期化パラメータ
    :type init_kwargs: dict
    :param columns: 正規化する対象のカラム名
    :type columns: list[str] or str
    """

    def __init__(self, init_kwargs={}, columns='all'):
        self._init_kwargs = init_kwargs
        self._columns = columns
        self._value = None
        self.reset()

    def process(self, df_train, df_test):
        logger.debug('ApplyStandardScaler#process invoked')
        if self._columns == 'all':
            if not self.operation_mode == 'predict':
                self.value.fit(df_train)
            columns = list(df_train.columns)
            r_df_test = r_df_train = None
            r_df_train = pd.DataFrame(self.value.transform(df_train), columns=columns)
            if df_test is not None:
                r_df_test = pd.DataFrame(self.value.transform(df_test), columns=columns)
            return r_df_train, r_df_test
        else:
            if not isinstance(self._columns, list):
                raise TypeError('columns must be list.')
            t_df = df_train[self._columns]
            if not self.operation_mode == 'predict':
                self.value.fit(t_df)
            r_df_test = r_df_train = None
            r_df_train = pd.DataFrame(self.value.transform(t_df), columns=self._columns)
            for c in self._columns:
                df_train[c] = r_df_train[c]
            if df_test is not None:
                r_df_test = pd.DataFrame(self.value.transform(dt_test[self._columns]), columns=self._columns)
                for c in self._columns:
                    df_test[c] = r_df_test[c]
            return df_train, df_test

    def reset(self):
        self._value = StandardScaler(**self._init_kwargs)
        return self

    dump = dump_sklearn_model
    load = load_sklearn_model


class ApplyPca(StatefulPreprocessor):
    """
    入力データにPCAをかけるPreprocessor

    :param init_kwargs: 前処理実体の初期化パラメータ
    :type dict
    """

    def __init__(self, init_kwargs={}):
        self._init_kwargs = init_kwargs
        self._value = None
        self.reset()

    def process(self, df_train, df_test):
        logger.debug('ApplyPca#process invoked')
        if not self.operation_mode == 'predict':
            self.value.fit(df_train)
        n_components = self._init_kwargs.get('n_components', len(df_train.columns))
        columns = ['x'+str(i) for i in range(n_components)]
        r_df_test = r_df_train = None
        r_df_train = pd.DataFrame(self.value.transform(df_train), columns=columns)
        if df_test is not None:
            r_df_test = pd.DataFrame(self.value.transform(df_test), columns=columns)
        return r_df_train, r_df_test

    def reset(self):
        self._value = PCA(**self._init_kwargs)
        return self

    dump = dump_sklearn_model
    load = load_sklearn_model
