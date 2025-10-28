import torch
import numpy as np
from typing import List, Dict

from data.transition_data import TransitionData

class Buffer:
    """
    経験（TransitionData）を蓄積し、学習用のTensorバッチを提供する責務を持つクラス。
    """
    
    def __init__(self, capacity: int):
        self.capacity: int = capacity
        self.count: int = 0

        self.state: List[np.ndarray] = []
        self.reward: List[float] = []
        self.done: List[bool] = []
        self.next_state: List[np.ndarray] = []
        self.value: List[float] = []
        self.log_prob: List[float] = []
        self.raw_continuous_action: List[np.ndarray] = []
        self.discrete_action: List[np.ndarray] = []

    def store(self, transition_data: TransitionData):
        
        """遷移データ(Trajectory)を1つ保存する"""
        if self.count >= self.capacity:
            self.remove_old_data()  # 古いものから削除

        self.state.append(transition_data.states)
        self.reward.append(transition_data.rewards)
        self.done.append(transition_data.dones)
        self.next_state.append(transition_data.next_states)
        self.value.append(transition_data.values)
        self.log_prob.append(transition_data.log_probs)
        self.raw_continuous_action.append(transition_data.raw_continuous_actions)
        self.discrete_action.append(transition_data.discrete_actions)

        self.count = len(self.state)

    def remove_old_data(self):
        self.state.pop(0)
        self.reward.pop(0)
        self.done.pop(0)
        self.next_state.pop(0)
        self.value.pop(0)
        self.log_prob.pop(0)
        self.raw_continuous_action.pop(0)
        self.discrete_action.pop(0)

    def get_tensor_data(self) -> TransitionData:
        """
        蓄積された全Trajectoryを、学習で利用可能なTensorの辞書に変換して返す。
        """
        if not self.state:
            return None
        
        states = torch.tensor(np.array(self.state), dtype=torch.float32)
        rewards = torch.tensor(np.array(self.reward), dtype=torch.float32)
        dones = torch.tensor(np.array(self.done), dtype=torch.float32)
        next_states = torch.tensor(np.array(self.next_state), dtype=torch.float32)
        old_log_probs = torch.tensor(np.array(self.log_prob), dtype=torch.float32)
        values = torch.tensor(np.array(self.value), dtype=torch.float32)

        # 連続値アクションが存在する場合の処理
        if self.raw_continuous_action and self.raw_continuous_action[0] is not None:
            continuous_actions_np = np.vstack(self.raw_continuous_action)
        else:
            continuous_actions_np = np.empty((len(self.state), 0), dtype=np.float32)
        raw_continuous_actions = torch.tensor(continuous_actions_np, dtype=torch.float32)

        # 離散アクションが存在する場合の処理
        if self.discrete_action and self.discrete_action[0] is not None:
            discrete_actions_np = np.vstack(self.discrete_action)
        else:
            discrete_actions_np = np.empty((len(self.state), 0), dtype=np.int64)
        discrete_actions = torch.tensor(discrete_actions_np, dtype=torch.int64)

        transition_data = TransitionData(
            states=states,
            rewards=rewards,
            dones=dones,
            next_states=next_states,
            values=values,
            log_probs=old_log_probs,
            raw_continuous_actions=raw_continuous_actions,
            discrete_actions=discrete_actions
        )

        return transition_data

    def clear(self):
        """バッファを空にする"""
        self.state.clear()
        self.reward.clear()
        self.done.clear()
        self.next_state.clear()
        self.value.clear()
        self.log_prob.clear()
        self.raw_continuous_action.clear()
        self.discrete_action.clear()
        self.count = 0

    def is_ready(self, min_size: int) -> bool:
        """学習に十分なデータが蓄積されているかチェック"""
        return self.count >= min_size

    def __len__(self) -> int:
        return self.count