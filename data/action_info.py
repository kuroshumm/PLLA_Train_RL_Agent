import numpy as np
from typing import NamedTuple, Optional

class ActionInfo(NamedTuple):
    """
    行動決定メソッドの戻り値を格納するデータクラス。
    連続値・離散値アクションに対応。
    """
    continuous_action: Optional[np.ndarray] # 連続値アクション
    # PPOの学習で必要になる、分布からサンプリングされた生の連続値アクション
    raw_continuous_action: Optional[np.ndarray]

    value: float
    log_prob: float