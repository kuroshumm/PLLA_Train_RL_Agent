import torch
import numpy as np
from typing import List, Dict, NamedTuple, Optional
from torch.utils.data import Dataset

class TransitionData(NamedTuple):
    """
    1つの遷移データを格納するデータクラス。
    """
    states: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_states: torch.Tensor
    values: torch.Tensor
    log_probs: torch.Tensor
    raw_continuous_actions: torch.Tensor
    discrete_actions: torch.Tensor

class TrainData(NamedTuple):
    """
    学習に使用するデータを格納するデータクラス。
    """
    states: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    values: torch.Tensor
    continuous_actions: Optional[torch.Tensor]
    discrete_actions: Optional[torch.Tensor]

class TrainDataset(Dataset):
    """
    TrainDataをDatasetとして扱うためのラッパークラス
    """
    def __init__(self, trainData: TrainData):
        self.trainData = trainData

    def __len__(self) -> int:
        return self.trainData.states.size(0)

    def __getitem__(self, idx: int) -> TrainData:
        return TrainData(
            states=self.trainData.states[idx],
            old_log_probs=self.trainData.old_log_probs[idx],
            advantages=self.trainData.advantages[idx],
            returns=self.trainData.returns[idx],
            values=self.trainData.values[idx],
            continuous_actions=self.trainData.continuous_actions[idx],
            discrete_actions=self.trainData.discrete_actions[idx],
        )