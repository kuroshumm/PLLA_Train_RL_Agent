import torch
import numpy as np
from typing import List, Dict

from data.trajectory import Trajectory

class Buffer:
    """
    経験（Trajectory）を蓄積し、学習用のTensorバッチを提供する責務を持つクラス。
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Trajectory] = []
    
    def store(self, trajectory: Trajectory):
        """遷移データ(Trajectory)を1つ保存する"""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0) # 古いものから削除
        self.buffer.append(trajectory)

    def get_tensor_data(self) -> Dict[str, torch.Tensor]:
        """
        蓄積された全Trajectoryを、学習で利用可能なTensorの辞書に変換して返す。
        """
        if not self.buffer:
            return {}

        # 必須のフィールドは常に変換
        states = torch.tensor([t.state for t in self.buffer], dtype=torch.float32)
        rewards = torch.tensor([t.reward for t in self.buffer], dtype=torch.float32)
        dones = torch.tensor([t.done for t in self.buffer], dtype=torch.float32)
        old_log_probs = torch.tensor([t.log_prob for t in self.buffer], dtype=torch.float32)
        values = torch.tensor([t.value for t in self.buffer], dtype=torch.float32)
        next_states = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float32)
        
        batch_dict = {
            "states": states, "rewards": rewards, "dones": dones,
            "old_log_probs": old_log_probs, "values": values, "next_states": next_states,
        }

        # オプショナルなフィールドは、存在する場合のみ辞書に追加する
        first_trajectory = self.buffer[0]
        
        # 環境に渡した（クリップ済み）連続値アクション
        if first_trajectory.continuous_action is not None:
            actions = [t.continuous_action for t in self.buffer]
            batch_dict["continuous_actions"] = torch.tensor(actions, dtype=torch.float32)
        
        # 学習に使う、生の（クリップ前）連続値アクション
        if first_trajectory.raw_continuous_action is not None:
            raw_actions = [t.raw_continuous_action for t in self.buffer]
            batch_dict["raw_continuous_actions"] = torch.tensor(raw_actions, dtype=torch.float32)

        # 離散値アクション
        if first_trajectory.discrete_action is not None:
            # discrete_actionがNoneでないTrajectoryのみをリスト内包表記の対象にする
            disc_actions = [t.discrete_action for t in self.buffer if t.discrete_action is not None]
            batch_dict["discrete_actions"] = torch.tensor(disc_actions, dtype=torch.int64).squeeze(-1)
            
        return batch_dict

    def clear(self):
        """バッファを空にする"""
        self.buffer = []

    def is_ready(self, min_size: int) -> bool:
        """学習に十分なデータが蓄積されているかチェック"""
        return len(self.buffer) >= min_size

    def __len__(self) -> int:
        return len(self.buffer)