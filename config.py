class DQNConfig:
    """DQN用の設定クラス"""
    
    def __init__(self):
        # 学習パラメータ
        self.lr = 1e-4
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 0.995
        
        # ネットワークパラメータ
        self.batch_size = 32
        self.memory_capacity = 50000
        self.min_experiences = 1000
        self.target_update = 10
        
        # 学習設定
        self.max_steps = 10000
        self.max_steps_per_episode = 1000
        self.save_interval = 1000

class Config:
    """完全分離型PPO設定クラス（ハイブリッド対応）"""
    
    def __init__(self):
        # エージェント識別子
        self.agent_name = "default_agent"
        
        # Unity環境の設定
        self.env_path = None
        self.time_scale = 8
        self.seed = 1

        # --- 使用するアクションの制御フラグ ---
        self.use_continuous = True
        self.use_discrete = False
        # ------------------------------------
        
        # アクション空間の次元数（UnityEnvWrapperから自動設定されるが、デフォルト値を定義）
        self.continuous_action_size = 0
        self.discrete_action_size = 0

        # 学習パラメータ
        self.lr = 3e-4
        self.lr_end = 1e-5  # 終了時の学習率
        self.gamma = 0.99
        
        # 分離されたネットワーク用の設定
        self.hidden_size = 128

        # PPO固有のパラメータ
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.ppo_epochs = 5
        self.gae_lambda = 0.95
        self.max_grad_norm = 0.5
        
        # 連続値行動空間のパラメータ
        self.action_scale = 1.0
        self.action_bias = 0.0
        
        # バッチサイズ
        self.batch_size = 64
        self.buffer_size = 1024
        self.min_experiences = 10
        
        # 学習設定
        self.max_steps = 30000
        self.max_steps_per_episode = 1000
        self.save_interval = 1000
        self.lr_decay_steps = self.max_steps / 2