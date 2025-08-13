from abc import ABC, abstractmethod
from typing import Tuple, Any
from config import Config

class BaseAgent(ABC):
    """強化学習エージェントの基底クラス（PPO向けに更新）"""
    
    def __init__(self, state_size: int, config: Config):
        """
        Args:
            state_size (int): 状態空間のサイズ
            config (Config): 設定オブジェクト
        """
        self.config = config

        self.state_size = state_size
        self.continuous_action_size = getattr(config, 'continuous_action_size', None)
        self.discrete_action_size = getattr(config, 'discrete_action_size', None)
        self.hidden_size = getattr(config, 'hidden_size', 128)
        
    
    @abstractmethod
    def select_action(self, state: Any, training: bool = True) -> Tuple[Any, ...]:
        """状態に基づいて行動を選択する"""
        pass
    
    @abstractmethod
    def store_transition(self, state: Any, continuous_action: Any, discrete_action: Any, reward: float, next_state: Any, done: bool, log_prob: float = None, value: float = None, raw_continuous_action: Any = None):
        """経験（遷移）をバッファに保存する"""
        pass
    
    @abstractmethod
    def update(self) -> dict:
        """収集した経験を使ってモデルを更新する"""
        pass

    # --- モデルの保存と読み込み --- 

    @abstractmethod
    def save_checkpoint(self, filepath: str):
        """学習再開用のチェックポイント（全体モデル）を保存する"""
        pass

    @abstractmethod
    def load_checkpoint(self, filepath: str):
        """学習再開用のチェックポイント（全体モデル）を読み込む"""
        pass

    @abstractmethod
    def save_actor_model(self, filepath: str):
        """推論用のActorモデルを保存する"""
        pass

    @abstractmethod
    def load_actor_model(self, filepath: str):
        """推論用のActorモデルを読み込む"""
        pass

    @abstractmethod
    def export_onnx(self, filepath: str):
        """推論用のActorモデルをONNX形式でエクスポートする"""
        pass