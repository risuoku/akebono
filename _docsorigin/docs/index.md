# akebono

akebonoは、機械学習における教師あり学習のワークフロー定義と、その実装が含まれるフレームワークです。以下のような特徴があります。

* 入力データ、モデルといった機械学習で用いるコンポーネントに対する高水準API
* ワークフローの宣言的設定
* ストレージの抽象化

## akebonoをはじめる

### インストール

```
pip install git+https://github.com/risuoku/akebono
```

### プロジェクトの初期化

まず、適当な作業用ディレクトリを作って移動します。MacやLinuxの場合は例えばこうします。

```
mkdir demo-akebono
cd demo-akebono
```

次に、`config.py` という名前のファイルを作成します。中身は空で大丈夫です。

```
touch config.py
```

次に、以下のコマンドを入力して実行します。

```
akebono init
```

これは、akebonoを動作させるために必要な初期化処理を実行します。コマンド実行後、直下に `_storage` というディレクトリが作成されたでしょうか？
これは、akebonoがデータを保存したり読み込むためのスペースとなります。この場所は設定で変更することが可能です。

### irisデータセットを分類する

最初のタスクとして、irisデータセットの分類機を作ってみます。
より具体的には、以下のような作業をやります。

1. irisデータセットを取得する。
2. irisデータセットのpandas.DataFrameを作成する。
3. scikit-learnのロジスティック回帰を使ってモデルを訓練する。評価は１０分割の交差検定による正解率を用いる。

akebonoを使って上記を実行する手順を説明します。

まず、`config.py` を以下のとおりに編集します。

```python
train_operations = [
    {
        'dataset_config': {
            'loader_config': {
                'func': 'load_iris@akebono.dataset.generator.sklearn',
                'func_kwargs': {
                },
            },
        },
        'model_config': {
            'name': 'SklearnLogisticRegression',
            'init_kwargs': {},
            'fit_kwargs': {},
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
]
```

次に、以下のコマンドを入力して実行します。

```
akebono train
```

実行ログが流れましたか？このコマンドは `config.py` を読み込んで上記1から3を実行した後、実行結果や訓練済みモデルを保存するところまでやりました。

モデルの評価を確認するには以下のコマンドを実行します。

```
akebono inspect

### 以下、実行結果
=== performance summary .. scenario_tag: latest ===

------------------------------
train_id: 0

accuracy
0.94667 
```

accuracy、つまり正解率が約95%のモデルを訓練できたことがわかりました。
