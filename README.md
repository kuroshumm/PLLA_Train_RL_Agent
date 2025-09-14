# PLLA_Custom_RL_Agent
<pre>
.
├── .gitignore                      
├── README.md                       
├── agent
│   ├── buffer.py                   # 強化学習の経験（状態、行動、報酬など）を一時的に保存するバッファ
│   └── ppo_agent.py                # PPOアルゴリズムに基づいたエージェントの行動選択や学習処理
├── base
│   └── base_agent.py               # 強化学習エージェントの基本的な構造（設計図）を定義する基底クラス
├── common
│   ├── normalizer.py               # 報酬などのデータの正規化（平均0、分散1に近づける）
│   └── seed_utils.py               # 実験の再現性を保つために、乱数のシード（種）を固定する関数
├── config.py                       # 学習率や割引率など、強化学習の各種パラメータを設定
├── log
│   └── best_ppo_model.onnx         # 学習済みモデルをONNX形式で保存したファイル
├── main.py                         # プログラム全体のエントリーポイント（実行開始点）
├── Model
│   ├── actor_critic.py             # Actor（行動選択）とCritic（価値評価）のニューラルネットワークモデル
│   └── ppo_network.py              # ActorとCriticを組み合わせたPPO用のネットワーク全体を構築
├── Sample
│   └── plla_sample.py              # Unity環境と連携し、ランダムな行動をとるエージェントのサンプルコード
├── trainer
│   └── ppo_trainer.py              # PPOエージェントの学習ループ全体を管理
└── train.bat                       # Windows環境で学習を開始するためのバッチファイル
</pre>