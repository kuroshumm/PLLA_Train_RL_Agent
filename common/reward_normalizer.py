import torch
from common.running_mean_std import RunningMeanStd

class RewardNormalizer:
    """
    報酬を正規化する責務を持つクラス。
    報酬の移動平均と標準偏差を計算し、報酬を正規化する。"""
    def __init__(self):
        self.reward_rms = RunningMeanStd(shape=(1,))

    def update(self, rewards: torch.Tensor):
        """統計情報を更新する"""
        self.reward_rms.update(rewards.numpy().reshape(-1, 1))

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        """報酬を正規化して返す"""
        normalized = self.reward_rms.normalize(rewards.numpy().reshape(-1, 1))
        return torch.tensor(normalized, dtype=torch.float32).flatten()