# akebonoをはじめる

akebonoは、機械学習における教師あり学習のワークフロー定義と、その実装が含まれるフレームワークです。以下のような特徴があります。

* 入力データ、モデルといった機械学習で用いるコンポーネントに対する高水準API
* ワークフローの宣言的設定
* ストレージの抽象化

## akebonoの役割

まず、一般的な教師あり学習のワークフローの中で発生する典型的なタスクと、akebonoの関わりの有無をまとめると下表のようになります。

|タスク|例|akebonoの関わり|
|:----|:--|:------------|
|入力データの準備|ログデータを蓄積する。分類したい画像を収集する。ラベリング作業をする。|なし|
|入力データを、読み込み可能な保存領域に保存・転送する|DBにあるデータをBigqueryに転送する。画像を数値ベクトル化してpickle形式で保存する。|なし|
|入力データを読み込む|機械学習用プログラムからBigqueryにクエリを発行してデータを取得する。|あり|
|入力データの前処理|データを正規化する。訓練に使うカラムだけ選択する。|あり|
|モデルの訓練|線形回帰による回帰モデルの作成。ランダムフォレストによるクラス分類モデルの作成。|あり|
|モデルの評価|二値分類モデルを、１０分割交差検定のaucで評価する。|あり|
|モデルの保存|訓練済みのランダムフォレストをストレージに保存する。|あり|
|既存モデルを使った予測処理|訓練済みランダムフォレストをストレージから読み込んで、新しい入力に対して予測結果を出力する。|あり|

akebonoの目標は、入力データを読み込み、モデルを訓練・評価し、外部に保存、作ったモデルを読み込んで予測するところまでの統合的なワークフローを構築することです。

## インストール

```
$ pip install git+https://github.com/risuoku/akebono
```

## プロジェクトの初期化

まず、適当な作業用ディレクトリを作って移動します。MacやLinuxの場合は例えばこうします。

```
$ mkdir demo-akebono
$ cd demo-akebono
```

次に、`config.py` という名前のファイルを作成します。中身は空で大丈夫です。

```
$ touch config.py
```

次に、以下のコマンドを入力して実行します。

```
$ akebono init
```

これは、akebonoを動作させるために必要な初期化処理を実行します。コマンド実行後、直下に `_storage` というディレクトリが作成されたでしょうか？
これは、akebonoがデータを保存したり読み込むためのスペースとなります。この場所は設定で変更することが可能です。

## irisデータセットを分類する

最初のタスクとして、irisデータセットの分類機を作ってみます。
より具体的には、以下のような作業をやります。

1. irisデータセットを取得する。
2. irisデータセットのpandas.DataFrameを作成する。
3. scikit-learnのロジスティック回帰を使ってモデルを訓練する。評価は１０分割の交差検定による正解率を用いる。

akebonoを使って上記を実行する手順を説明します。

まず、`config.py` を以下のとおりに編集します。

```python
train_config = {
    'dataset_config': {
        'loader_config': {
            'name': 'iris',
        },
    },
    'model_config': {
        'name': 'SklearnLogisticRegression',
        'evaluate_kwargs': {
            'cross_val_iterator': 'KFold@sklearn.model_selection',
            'cross_val_iterator_kwargs': {
                'n_splits': 10,
                'shuffle': True,
                'random_state': 0,
            },
        },
        'model_type': 'multiple_classifier',
    },
    'evaluate_enabled': True,
    'fit_model_enabled': True,
    'dump_result_enabled': True,
}
```

次に、以下のコマンドを入力して実行します。

```
$ akebono train
```

実行ログが流れましたか？このコマンドは `config.py` を読み込んで上記1から3を実行した後、実行結果や訓練済みモデルを保存するところまでやりました。

モデルの評価を確認するには以下のコマンドを実行します。

```
$ akebono inspect

### 以下、実行結果
=== scenario summary .. tag: default ===

------------------------------------------------------------
train_id: 0

accuracy
0.94667 
```

accuracy、つまり正解率が約95%のモデルを訓練できたことがわかりました。

また、構築した分類機を使って予測処理を実行することも可能です。

まず、予測用の設定を準備します。`config.py` に、以下の設定を追加してください。 
この設定は、訓練データと同じデータで予測もしようとしています。用いるデータは設定で変更することが可能です。

```python
predict_config = {
    'method_type': 'predict',
    'dataset_config': {
        'loader_config': {
            'name': 'iris',
        },  
    },
    'train_id': '0',
    'dump_result_enabled': True,
    'dumper_config': {
        'name': 'csv',
    },
    'result_target_columns': 'all',
}
```

次に、以下のコマンドを入力して実行します。

```
$ akebono predict
```

`_storage/default/operation_results/default` 以下に `predict_result_*` というファイルが生成されていたら成功です。
今回の予測結果は、 `_storage/default/operation_results/default/predict_result_JBIjqknXCIv5SNm2.csv` に入っており、
予測結果がcsvフォーマットで保存されています。ただし、ファイル名のランダム文字列部分はpredict実行時に生成されます。

```
$ head -10 _storage/default/operation_results/default/predict_result_JBIjqknXCIv5SNm2.csv 

### 以下、実行結果
sepal_length,sepal_width,petal_length,petal_width,predicted
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
4.7,3.2,1.3,0.2,0
4.6,3.1,1.5,0.2,0
5.0,3.6,1.4,0.2,0
5.4,3.9,1.7,0.4,0
4.6,3.4,1.4,0.3,0
5.0,3.4,1.5,0.2,0
4.4,2.9,1.4,0.2,0
```
