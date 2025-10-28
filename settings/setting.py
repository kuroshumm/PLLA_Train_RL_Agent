from pydantic import BaseModel, Field
from typing import Literal, Optional

# ===================================================
# algorithm セクション内の詳細なデータクラス
# ===================================================

class ActionSpaceSettings(BaseModel):
    """行動空間に関する設定"""
    use_continuous: bool
    use_discrete: bool
    # continuous_action はネストした辞書に対応
    continuous_action: dict[str, float] = Field(default_factory=dict)
    discrete_action: dict[str, int] = Field(default_factory=dict)

class ModelSettings(BaseModel):
    """ニューラルネットワークモデルの構造に関する設定"""
    hidden_size: int

class PPOHyperparameters(BaseModel):
    """PPO固有のハイパーパラメータ"""
    gae_lambda: float
    clip_epsilon: float
    value_coef: float
    entropy_coef: float
    epochs: int
    max_grad_norm: float

class HyperparameterSettings(BaseModel):
    """アルゴリズムのハイパーパラメータ設定"""
    gamma: float
    learning_rate: float
    learning_rate_end: float
    lr_decay_steps: int
    buffer_size: int
    batch_size: int
    # ppo固有のパラメータをネストして保持
    ppo: PPOHyperparameters

# ===================================================
# トップレベルのデータクラス
# ===================================================

class TrainerSettings(BaseModel):
    """Trainer（学習全体）に関する設定"""
    agent_name: str
    max_steps: int
    max_steps_per_episode: int
    save_interval: int

class EnvironmentSettings(BaseModel):
    """Unity環境に関する設定"""
    # env_pathはNoneの場合もあるためOptionalを使用
    env_path: Optional[str] = None
    time_scale: int
    seed: int
    normalize_observations: bool = False

class AlgorithmSettings(BaseModel):
    """アルゴリズム全体に関する設定"""
    # nameは "PPO" または将来追加される "SAC" のみ受け付ける
    name: Literal["PPO", "SAC"]
    action_space: ActionSpaceSettings
    model: ModelSettings
    hyperparameters: HyperparameterSettings

class AppConfig(BaseModel):
    """
    アプリケーション全体の設定を保持するトップレベルクラス。
    config.yamlの構造と1対1で対応する。
    """
    trainer: TrainerSettings
    environment: EnvironmentSettings
    algorithm: AlgorithmSettings