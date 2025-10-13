import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict

from data.buffer import Buffer
from data.action_info import ActionInfo

# ===================================================
# アルゴリズムのインターフェース（抽象基底クラス）定義
# ===================================================

class LearningAlgorithmBase(ABC):
    """
    強化学習アルゴリズムのインターフェースを定義する抽象基底クラス。

    具体的なアルゴリズム（PPO, SACなど）は、このクラスを継承し、
    以下の抽象メソッド（decide_action, train）を必ず実装する必要があります。
    """

    @abstractmethod
    def decide_action(self, state: np.ndarray) -> ActionInfo:
        """
        現在の状態に基づき、行動を決定する。

        Args:
            state (np.ndarray): 環境から観測された現在の状態。

        Returns:
            ActionInfo: 決定された行動、その時の価値、対数確率を含むオブジェクト。
        """
        pass

    @abstractmethod
    def train(self, buffer: Buffer) -> Dict[str, float]:
        """
        収集された遷移データのリストを用いて、モデルの学習（更新）を行う。

        Args:
            trajectories (List[Trajectory]): 学習に使用する遷移データのリスト。

        Returns:
            Dict[str, float]: 学習結果の統計情報（損失など）を格納した辞書。
        """
        pass