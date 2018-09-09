from akebono.model.evaluator import evaluate


class WrappedModel:
    """
    WrappedModel interface

    :param init_kwargs: モデル実体の初期化時に渡されるパラメータ
    :type init_kwargs: dict
    :param fit_kwargs: モデル実体の訓練時に渡されるパラメータ
    :type fit_kwargs: dict
    :param evaluate_kwargs: モデルの評価時に渡されるパラメータ
    :type evaluate_kwargs: dict
    :param pos_index: モデル実体が確率を返す予測(predict_proba)をした場合の、正例に相当するindex
    :type pos_index: int or None
    """

    def __init__(self, init_kwargs={}, fit_kwargs={}, evaluate_kwargs={}, pos_index=None):
        self._init_kwargs = init_kwargs
        self._fit_kwargs = fit_kwargs
        self._evaluate_kwargs = evaluate_kwargs
        self._value = None
        self._pos_index = pos_index
        self.model_type = None
        self.base_init_finished()
    
    def base_init_finished(self):
        """
        self.__init__() の終了後に実行されるメソッド
        """
        pass
    
    def fit(self, X, y):
        """
        モデルの訓練に用いるメソッド

        :param X: 説明変数
        :type X: numpy array-like
        :param y: 目的変数
        :type y: numpy array-like
        :return: WrappedModel object
        """
        raise NotImplementedError()
    
    def reset(self):
        """
        モデルの初期化に用いるメソッド

        :return: WrappedModel object
        """
        raise NotImplementedError()
    
    def dump(self, dirpath, name):
        """
        モデルをストレージに永続化するためのメソッド

        :param dirpath: ストレージのパス
        :type dirpath: str
        :param name: ファイル名
        :type name: str
        """
        raise NotImplementedError()

    def load(self, dirpath, name):
        """
        ストレージに永続化されてるモデルを復元するためのメソッド

        :param dirpath: ストレージのパス
        :type dirpath: str
        :param name: ファイル名
        :type name: str
        """
        raise NotImplementedError()

    def predict(self, X):
        """
        予測結果を返すメソッド

        :param X: 説明変数
        :type X: numpy array-like
        :return: numpy array object
        """
        raise NotImplementedError()

    def predict_proba(self, X):
        """
        予測結果（確率）を返すメソッド

        :param X: 説明変数
        :type X: numpy array-like
        :return: numpy array object
        """
        raise NotImplementedError()

    def set_model_type(self, y=None, model_type=None):
        """
        モデルのタイプを設定するためのメソッド

        :param y: 目的変数
        :type y: numpy array-like
        :param model_type: モデルのタイプ。設定可能値は、`binary_classifier` or `multiple_classifier` or `regressor`
        :type model_type: str
        """
        if (y is None and model_type is None) or (y is not None and model_type is not None):
            raise Exception('ambiguous argument .. one of y and model_type is None.')
        if y is not None:
            if len(set(y)) < 3:
                self.model_type = 'binary_classifier'
            else:
                self.model_type = 'multiple_classifier'
        else:
            if model_type not in ('binary_classifier', 'multiple_classifier', 'regressor'):
                raise ValueError('invalid model_type')
            self.model_type = model_type
    
    @property
    def value(self):
        """
        モデルの実体
        """
        return self._value

    def evaluate(self, X, y, preprocessor, format_func_for_predictor, format_func_for_target):
        """
        モデルを評価するためのメソッド

        :param X: 説明変数
        :type X: numpy array-like
        :param y: 目的変数
        :type y: numpy array-like
        :param preprocessor: Preprocessor object
        :type preprocessor: StatefulPreprocessor or StatelessPreprocessor
        :param format_func_for_predictor: 説明変数を整形するための関数
        :type format_func_for_predictor: function
        :param format_func_for_target: 目的変数を整形するための関数
        :type format_func_for_target: function

        :return: list
        """
        return evaluate(self, X, y, preprocessor, format_func_for_predictor, format_func_for_target, **self._evaluate_kwargs)
