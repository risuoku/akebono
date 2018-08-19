# akebonoの構成


akebonoが管理する構成は、物理的には大きく２つに分けられ、それぞれ`processor`,`storage`と呼んでおきます。

`processor`は、akebonoを使った開発における処理に関わる全ての領域です。
例えば、`config.py`やカスタムモデルが書かれたpythonモジュールファイル、BigQuery loaderが読むSQLファイルなどがこれにあたります。

`storage`は、ファイル、ディレクトリ、これらの要素に対するオペレーション等のファイルシステム相当の機能を持った外部記憶システムを抽象化したものです。
例えば、各種OSが持つファイルシステムと記憶領域がこれにあたります。また、Google Cloud PlatformのGCSやAWSのS3も`storage`にあたります。
`storage`は、訓練済みモデルを保存したり、データセットのキャッシュを置いておくといった使い方をします。


akebonoが管理する構成は、論理的にはトップレベルにまず`project`がきます。設定は`project`ごとに存在します。
ストレージも`project`ごと独立した領域が存在し、`storage_root_dir`と`storage_type`が同じ２つのプロジェクトの場合は
同じストレージの中に`project`ごと別のパスが割り当てられ、区別されて管理されます。

akebonoの内部コンポーネントは、大きく分けて `Settings`,`Dataset`,`Preprocessor`,`Model`,`Operator`があり、
これらを組み合わせた`Command`があります。


## Settings

`Settings`は、akebonoの設定を管理するためのコンポーネントです。`config.py`に設定した内容は実行時に`Settings`に反映され、akebonoに存在する各コンポーネントは
`Settings`を参照して動作します。`Settings`が管理するプロパティは`config.py`とほぼ同一で、詳細は[akebonoの設定](config.md)を参照してください。

## Dataset

`Dataset`は、教師あり学習で利用するデータを管理するためのコンポーネントです。具体的には、以下のような役割を持ちます。

* 外部記憶システムに保存されたデータやあるルールに基づいて生成されたデータを読み込む。読み込むためのコンポーネントを`loader`と呼びます。
    - 様々な形式に対応(csv,BigQueryなど)
    - `sklearn.datasets`等に含まれる自動生成されるデータや広く使われるデータセットの読み込みに対応
* 名前の管理
    - `Dataset`には、各プロジェクトごと、loaderごとに一意に名前をつけることができます。例えば、`dataset1`という名前のDatasetのloaderをBigQuery loaderとすると、
    `dataset1.sql`というファイルにSQLを書くことができて、別のSQLで記述される`dataset2`とは区別されます。
* 読み込んだデータに対する前処理。これは、何らかの理由により`loader`の後に処理を加えたものを`Dataset`として管理したく、かつ
後述の`Preprocessor`とは区別したい場合に利用します。ありそうなユースケースとしては、SQLでは書くのが難しかったり不可能な前処理を加えた後に
説明変数として意味のある形式になるデータに対する適用です。この場合は、次元削減等のチューニングのための前処理（こういった目的では`Preprocessor`の利用を推奨します）とは区別する意味があります。
* 目的変数のカラム名設定
* キャッシュ利用の有無
    - `Dataset`はキャッシュを作成することができ、具体的にはloaderで読み込んだ後、前処理後のpandas.DataFrameをpickle化してストレージに保存しておき、次の読み込み時にはpickleから読むようになります。
    キャッシュは`Dataset`についた名前と前処理関数の名前のハッシュ、loaderへ渡したパラメータ、前処理関数に渡したパラメータの組み合わせから一意に生成され、これらの組み合わせが一致するとキャッシュヒットします。


## Preprocessor

`Preprocessor`は、データの前処理の役割を担うコンポーネントです。
`Preprocessor`は、大きく２つに分けることができ、それぞれ`StatelessPreprocessor`、`StatefulPreprocessor`と呼びます。
`StatelessPreprocessor`は訓練データに依存しない前処理、`StatefulPreprocessor`は訓練データに依存する前処理です。
`StatelessPreprocessor`の例としては、カラム選択処理があります。あるモデルをチューニングするに際し、データのカラムの一部だけを使ってノイズを減らそうとするアプローチがあります。
カラム選択の処理は訓練データの値に依存しないのでこれは`StatelessPreprocessor`になります。
一方、`StatefulPreprocessor`の例としてはデータの標準正規化があります。この変換処理は訓練データの平均・分散が必要になるため、訓練データの値に依存し、`StatefulPreprocessor`になります。

`StatefulPreprocessor`は、訓練、予測のフローの中でやや複雑な振る舞いをし、以下のとおり留意すべきポイントがあります。

1. 訓練データに依存して振る舞いが変わるということは、評価に際してデータを分割するたびに振る舞いが変わる。交差検定の場合は分割するたびモデルを訓練しなおすだけでなく前処理を実行する必要がある。
2. 予測の際は、訓練に使ったデータに依存した前処理が必要になる。このため、訓練時に`StatefulPreprocessor`の振る舞いを決定するパラメータを保存しておく必要がある。

1,2に対して、後述の`Operator`経由で実行する場合は自動で配慮されます（チュートリアルで使っているコマンド経由でも同様です）。

また、`Preprocessor`には`Pipeline`という仕組みがあり、これは複数の`Preprocessor`連結を１つの`Preprocessor`として扱う抽象化機構です。
`Operator`は`Pipeline`として`Preprocessor`を扱うため、`config.py`には複数の`Preprocessor`を記述することが可能です。


## Model

## Operator

## Command
