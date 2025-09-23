import torch
import numpy as np
from typing import List, Dict, NamedTuple, Optional

class Trajectory(NamedTuple):
    """
    エージェントが経験した1ステップ分の遷移データを格納するデータクラス。
    連続値・離散値アクションに対応。
    """
    state: np.ndarray
    continuous_action: Optional[np.ndarray] # 環境に渡された連続値アクション
    discrete_action: Optional[np.ndarray]   # 環境に渡された離散値アクション
    reward: float
    done: bool
    next_state: np.ndarray
    value: float
    log_prob: float
    
    # PPOの学習で必要になる、分布からサンプリングされた生の連続値アクション
    raw_continuous_action: Optional[np.ndarray] = None