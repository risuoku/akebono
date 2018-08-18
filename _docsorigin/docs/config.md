# akebonoの設定

## 開発の基本

akebonoを使った開発は、akebonoの設定ファイル(このドキュメントの色々なところで登場する`config.py`)を作成する作業が基本になります。
設定ファイルといっても、実体はpythonモジュールですから実行時に値を決定することもできます。

`config.py`は、akebonoが実行すべきオペレーションの内容、ストレージの場所、プロジェクト名などakebonoの振る舞いを決定する役割を担います。

## config.pyの仕様

`config.py`で設定可能なプロパティは下表のようになります。

|設定可能なプロパティ|型|デフォルト値|意味|
|:-------------------|:--|:-----------|:---|
|storage_root_dir|string|`_storage`|ストレージのrootパス|
|storage_type|string|`local`|ストレージのタイプ。設定可能値は`local`or`gcs`。|
|storage_option|dict|`{}`|ストレージの振る舞いを決定する設定値。ストレージタイプごとに異なる。|
|project_name|string|`default`|akebonoの操作を適用する対象プロジェクトの名前。|
|project_root_dir|string|`os.getcwd()`|プロジェクトのrootパス。|
|bq_sql_template_dir|string|`os.path.join(project_root_dir, '_dataset/bq_sql_templates')`|BigQuery loaderが読み込む対象のsqlテンプレートファイルを置く場所。|
|train_config|dict or list|`{}`|trainの処理内容。|
|predict_config|dict or list|`{}`|predictの処理内容。|
