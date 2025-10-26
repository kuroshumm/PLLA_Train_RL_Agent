# PLLA_Custom_RL_Agent
<pre>
.
├── .gitignore
├── README.md
├── base
│   └── learning_algorithm_base.py  # 強化学習アルゴリズムの共通インターフェース
├── classDiagram.md                 # クラス間の関係性を示すUMLクラス図
├── common
│   ├── checkpoint_manager.py       # 学習の進捗（モデル）を保存・読み込みする管理クラス
│   ├── reward_normalizer.py        # 報酬を正規化するためのクラス
│   ├── running_mean_std.py         # データの移動平均と標準偏差を計算
│   └── seed_utils.py               # 乱数シードを固定し、実験の再現性を保つためのクラス
├── config.yaml                     # 学習率など、プロジェクト全体の設定を管理するファイル
├── data
│   ├── action_info.py              # エージェントが決定した行動の情報を格納するデータクラス
│   ├── buffer.py                   # 経験（状態、行動、報酬など）を一時的に保存するバッファ
│   └── transition_data.py          # 1ステップの遷移データ（状態、報酬など）の型を定義
├── main.py                         # プログラム全体を起動するエントリーポイント
├── ppo
│   ├── gae_calculator.py           # PPOのGAEを専門に行うクラス
│   ├── model
│   │   ├── actor_critic.py         # Actor（行動）とCritic（価値）のニューラルネットワーク定義
│   │   └── ppo_network.py          # ActorとCriticを組み合わせたPPOネットワーク全体を構築
│   ├── ppo.py                      # PPOアルゴリズム本体
│   └── ppo_loss_calculator.py      # PPOの損失計算（Actor/Critic損失）を専門に行うクラス
├── settings
│   ├── loader.py                   # config.yaml を読み込み、Pythonオブジェクトに変換するローダー
│   └── setting.py                  # config.yaml の構造に対応するPydantic（設定）モデル
├── train.bat                       # Windows環境で学習（main.py）を開始するための実行ファイル
└── trainer
    └── trainer.py                  # 環境とのやり取りや学習ループ全体を管理・実行するクラス
</pre>
