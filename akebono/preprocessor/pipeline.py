from .statefulmodels import StatefulPreprocessor
from .statelessmodels import StatelessPreprocessor

from akebono.logging import getLogger


logger = getLogger(__name__)


class PreprocessorPipeline:
    """
    複数のPreprocessor連結を抽象化するためのクラス
    """

    def __init__(self, preprocessors):
        for p in preprocessors:
            if not (isinstance(p, StatefulPreprocessor) or isinstance(p, StatelessPreprocessor)):
                raise TypeError('p must be Preprocessor')
        self._preprocessors = preprocessors
        self._opmode = None

    def reset(self):
        for p in self._preprocessors:
            if isinstance(p, StatefulPreprocessor):
                p.reset()
        return self

    def set_operation_mode(self, mode):
        if mode not in ('train', 'predict'):
            raise ValueError('invalid mode')
        self._opmode = mode
        for p in self._preprocessors:
            if isinstance(p, StatefulPreprocessor):
                p.set_operation_mode(mode)
        return self

    @property
    def operation_mode(self):
        if not hasattr(self, '_opmode'):
            self._opmode = None
        return self._opmode

    def dump_with_operation_rule(self, dirpath, train_id):
        """
        Pipelineに含まれるStatefulPreprocessorを永続化するためのメソッド

        :param dirpath: ストレージのパス
        :type dirpath: str
        :param train_id: train_id
        :type train_id: str
        """

        logger.debug('dump_with_operation_rule invoked.')
        for idx, p in enumerate(self._preprocessors):
            if isinstance(p, StatefulPreprocessor):
                fname = 'train_result_preprocessor_{}_{}_{}'.format(p.__class__.__name__, train_id, idx)
                p.dump(dirpath, fname)
        return self

    def load_with_operation_rule(self, dirpath, train_id):
        """
        ストレージに永続化されているStatefulPreprocessorをPipelineに復元するためのメソッド

        :param dirpath: ストレージのパス
        :type dirpath: str
        :param train_id: train_id
        :type train_id: str
        """

        logger.debug('load_with_operation_rule invoked.')
        for idx, p in enumerate(self._preprocessors):
            if isinstance(p, StatefulPreprocessor):
                fname = 'train_result_preprocessor_{}_{}_{}'.format(p.__class__.__name__, train_id, idx)
                p.load(dirpath, fname)
        return self

    def process(self, df_train, df_test):
        """
        Pipelineに含まれるPreprocessorのprocessを順次直列に実行するためのメソッド

        :param df_train: 訓練データセットが持つ説明変数
        :type df_train: pandas.DataFrame
        :param df_test: テストデータセットが持つ説明変数
        :type df_test: pandas.DataFrame or None
        :return: tuple(pandas.DataFrame, pandas.DataFrame) object
        """

        logger.debug('processing .. operation_mode: {}'.format(self.operation_mode))
        r1 = df_train.copy()
        r2 = df_test.copy() if df_test is not None else None
        for p in self._preprocessors:
            r1, r2 = p.process(r1, r2)
        return r1, r2
