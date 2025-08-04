# コード修正ノート: Batch および MetricsBoard の dataclasses 化

## 概要
本修正は、`wisp.datasets.batch.Batch` および `wisp.trainers.tracker.metrics.MetricsBoard` クラスの基底クラスを `attrdict.AttrDict` から `dataclasses.dataclass` へと変更し、それに伴うコードの調整を行ったものです。これにより、型ヒントの強化、コードの可読性向上、および保守性の向上が図られました。

## 修正内容

### 1. `wisp/datasets/batch.py`
-   **`Batch` クラス**:
    -   `AttrDict` から `dataclasses.dataclass` へ変更。
    -   内部データ管理のため、`_data: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)` を追加。
    -   初期データ投入のロジックを `__init__` から `__post_init__` へ移行し、`setattr` を使用するように変更。
    -   辞書ライクなアクセス（`[]`）を維持しつつ、ドット属性アクセスを促進するため、`__setitem__`, `__getitem__`, `keys()` メソッドを実装。
-   **`MultiviewBatch` および `SDFBatch` クラス**:
    -   `super().__init__` の呼び出しを `Batch` の新しい初期化ロジックに合わせて `super().__post_init__` に変更。
    -   `ray_values` および `coord_values` メソッド内で使用されていた辞書式属性アクセス（例: `self['rgb']`）をドット属性アクセス（例: `self.rgb`）に修正。

### 2. `wisp/trainers/tracker/metrics.py`
-   **`MetricsBoard` クラス**:
    -   `AttrDict` から `dataclasses.dataclass` へ変更。
    -   内部データ管理のため、`_data: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)` を追加。
    -   初期化ロジックを `__init__` から `__post_init__` へ移行。
    -   辞書ライクなアクセスを維持するため、`__setitem__`, `__getitem__`, `keys()` メソッドを実装。
    -   `clear`, `define_metric`, `log_metric`, `average_metric`, `finalize_epoch`, `active_metrics` メソッド内の全ての辞書式属性アクセス（例: `self[k]`）をドット属性アクセス（例: `getattr(self, k)`, `setattr(self, k, ...)`）に修正。
### 3. `wisp/trainers/tracker/tracker.py`
-   **`_WandB` クラス:
    -   `log_table` メソッドにおいて、`wandb.Table` の `data` 引数に渡す形式が正しくなかったため、`data.values()` を `[list(data.values())]` に変更しました。これにより、`wandb.Table` が期待する「行のリスト」形式でデータが渡され、`wandb` へのログ記録が正しく行われるようになりました。

## 修正中に遭遇し解決した問題

1.  **`NameError: name 'Optinal' is not defined`**:
    -   `wisp/trainers/tracker/tracker.py` の379行目における `Optional` のスペルミス（`Optinal`）を修正。
2.  **`TypeError: expected str, bytes or os.PathLike object, not NoneType`**:
    -   `nerf_hash_fused.yaml` 設定ファイルで `tracker.tensorboard` の `log_dir`, `exp_name`, `log_fname` が `null` に設定されていたため、`wisp/trainers/tracker/tracker.py` の `_setup_dashboards` メソッド内で `_Tensorboard` をインスタンス化する際に、`Tracker` クラスの対応する値を明示的に渡すように修正。
3.  **`NameError: name 'self' is not defined`**:
    -   `wisp/trainers/tracker/tracker.py` の `_setup_dashboards` メソッド内で `self.cfg.log_dir` と誤って参照されていた箇所を、引数として渡されている `cfg.log_dir` に修正。

これらの修正により、コードベースは `dataclasses` を活用したより堅牢な構造となり、機能的な互換性を維持しつつ、以前のエラーも解消されました。