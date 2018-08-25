# akebonoの設定

## 開発の基本

akebonoを使った開発は、akebonoの設定ファイル(このドキュメントの色々なところで登場する`config.py`)を作成する作業が基本になります。
設定ファイルといっても、実体はpythonモジュールですから実行時に値が決定されます。

`config.py`は、akebonoが実行すべきオペレーションの内容、ストレージの場所、プロジェクト名などakebonoの振る舞いを決定する役割を担います。

## config.pyの記法

config.pyは、python処理系から見るとpythonモジュールで、各プロパティはpythonのオブジェクトとして定義されます。

ただし、一部のプロパティで可読性のためオブジェクトをstring形式で表すことがあります。これを、「@記法」と呼ぶことにします。
@記法は、形式的には `<オブジェクト名>@<オブジェクトが生成されるモジュール名>` で表されます。例えば、
`load_iris@akebono.dataset.generator.sklearn`という@記法の文字列は、`akebono.dataset.generator.sklearn`モジュールで生成される
`load_iris`オブジェクトを表します。

## config.pyのプロパティ

### グローバル

`config.py`で設定可能なプロパティは下表のようになります。

|設定可能なプロパティ|型|デフォルト値|意味|
|:-------------------|:--|:-----------|:---|
|storage_root_dir|string|`_storage`|ストレージのrootパス|
|storage_type|string|`local`|ストレージのタイプ。設定可能値は`local`or`gcs`。|
|storage_option|dict|`{}`|ストレージの振る舞いを決定する設定値。ストレージタイプごとに異なる。|
|project_name|string|`default`|akebonoの操作を適用する対象プロジェクトの名前。|
|project_root_dir|string|`os.getcwd()`|プロジェクトのrootパス。|
|train_config|dict or list|`{}`|trainの処理内容。|
|predict_config|dict or list|`{}`|predictの処理内容。|

### storage_option

#### local

該当設定無し

#### gcs

|key|型|デフォルト値|制約|意味|
|:--|:--|:-----------|:-------|:---|
|bucket_name|string|無し|required:`true`|ストレージとして使うバケット名|

### train_config

下表は、train_configの型がdictの場合です。listの場合は、その要素が下表の仕様に従います。

|key|型|デフォルト値|制約|意味|
|:--|:--|:-----|:----|:---|
|dataset_config|dict|無し|required:`true`|Datasetの設定|
|dataset_config.name|string|無し|dataset_config.cache_enabledが`true`の場合はrequired:`true`, そうでなければrequired:`false`|Datasetの名前|
|dataset_config.target_column|string|`target`|required:`false`|目的変数のカラム名|
|dataset_config.cache_enabled|bool|`False`|required:`false`|データセットのキャッシュを作成するかどうかのフラグ|
|dataset_config.loader_config|dict|無し|required:`true`|Dataset loaderの設定|
|dataset_config.loader_config.name|string|無し|required:`true`|loaderが実行する関数。@記法で書かれる。|
|dataset_config.loader_config.kwargs|dict|`{}`|required:`false`|loaderが実行する関数に渡されるキーワード引数。|
|dataset_config.loader_config.param|dict|`{}`|required:`false`|loaderの振る舞いを決定するパラメータ。|
|dataset_config.preprocess_func|string|`identify@akebono.dataset.preprocessors`|required:`false`|Datasetの前処理を実行する関数。@記法で書かれる。|
|dataset_config.preprocess_func_kwargs|dict|`{}`|required:`false`|Datasetの前処理を実行する関数に渡されるキーワード引数。|
|model_config|dict|無し|required:`true`|Modelの設定|
|model_config.name|string|無し|required:`true`|Modelの名前。主に、手法名。|
|model_config.model_type|string|無し|required:`false`|Modelのタイプ。基本的に不要だが、データが少数の場合は設定しておいたほうが安全。|
|model_config.init_kwargs|dict|`{}`|required:`false`|モデル初期化の実行に渡すキーワード引数。|
|model_config.fit_kwargs|dict|`{}`|required:`false`|モデル訓練の実行時に渡すキーワード引数。|
|model_config.evaluate_kwargs|dict|`{}`|required:`false`|モデル評価の実行時に渡すキーワード引数|
|model_config.evaluate_kwargs.train_test_split_func|string|`train_test_split@sklearn.model_selection`|required:`false`|評価時に、モデル訓練用と評価用でデータを分割するための関数。@記法で書かれる。`cross_val_iterator`が指定されている場合は無視される。|
|model_config.evaluate_kwargs.train_test_split_func_kwargs|dict|`{}`|required:`false`|train_test_split_funcの実行時に渡されるキーワード引数。|
|model_config.evaluate_kwargs.cross_val_iterator|string|`None`|required:`false`|交差検定をする際のデータ生成に使うイテレータ。@記法で書かれる。|
|model_config.evaluate_kwargs.cross_val_iterator_kwargs|dict|`{}`|required:`false`|交差検定用イテレータの生成時に渡されるキーワード引数。|
|model_config.pos_index|int|`None`|モデルの評価や予測で`predict_proba`を実行する場合はrequired:`true`。そうでなければrequired:`false`|正例のインデックス|
|preprocessor_config|dict or list|`{'name':'identify','kwargs':{}}`|required:`false`|前処理の設定|
|preprocessor_config.name|string|`identify`|required:`false`|前処理の名前|
|preprocessor_config.kwargs|dict|`{}`|required:`false`|前処理に渡すキーワード引数|
|evaluate_enabled|bool|`False`|required:`false`|モデルの評価を実行するかのフラグ。|
|fit_model_enabled|bool|`False`|required:`false`|モデルの訓練を実行するかのフラグ。|
|dump_result_enabled|bool|`False`|required:`false`|trainオペレーションの結果や訓練済みモデルをストレージに保存するかどうかのフラグ。|

### predict_config

下表は、predict_configの型がdictの場合です。listの場合は、その要素が下表の仕様に従います。

|key|型|デフォルト値|制約|意味|
|:--|:--|:-----|:----|:---|
|dataset_config|dict|無し|required:`true`|Datasetの設定|
|dataset_config.name|string|無し|dataset_config.cache_enabledが`true`の場合はrequired:`true`, そうでなければrequired:`false`|Datasetの名前|
|dataset_config.cache_enabled|bool|`False`|required:`false`|データセットのキャッシュを作成するかどうかのフラグ|
|dataset_config.loader_config|dict|無し|required:`true`|Dataset loaderの設定|
|dataset_config.loader_config.name|string|無し|required:`true`|loaderが実行する関数。@記法で書かれる。|
|dataset_config.loader_config.kwargs|dict|`{}`|required:`false`|loaderが実行する関数に渡されるキーワード引数。|
|dataset_config.loader_config.param|dict|`{}`|required:`false`|loaderの振る舞いを決定するパラメータ。|
|dataset_config.preprocess_func|string|`identify@akebono.dataset.preprocessors`|required:`false`|Datasetの前処理を実行する関数。@記法で書かれる。|
|dataset_config.preprocess_func_kwargs|dict|`{}`|required:`false`|Datasetの前処理を実行する関数に渡されるキーワード引数。|
|method_type|string|`predict`|required:`false`|予測のタイプ。`predict`or`predict_proba`。|
|train_id|int or string|`0`|required:`false`|予測で使うモデルのtrain_id。|
|dump_result_enabled|bool|`False`|required:`false`|predictオペレーションの結果をストレージに保存するかどうかのフラグ。|
|dumper_config|dict|`{}`|required:`false`|predictオペレーションの結果をストレージに保存する際の設定。|
|result_target_columns|string or list|`all`|required:`false`|predictの結果に含める説明変数のカラム名のリスト。`all`の場合は全て含める。|
|result_predict_column|string|`predicted`|required:`false`|predictの結果を表すカラム名。|
