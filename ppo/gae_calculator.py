import torch

class GAECalculator:
    """
    GAE (Generalized Advantage Estimation) を計算する責務を持つクラス。
    """
    def __init__(self, gamma: float, gae_lambda: float):
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    @torch.no_grad()
    def calculate(self, 
                rewards: torch.Tensor,
                values: torch.Tensor,
                next_values: torch.Tensor,
                dones: torch.Tensor) -> torch.Tensor:

        """
        GAEを再帰的に計算する
        逆順で計算することで、次状態の価値を簡単に参照できる
        """
        advantages = []
        gae = 0.0
        for i in reversed(range(len(rewards))):
            
            if i == len(rewards) - 1:
                next_value = next_values[-1] if not dones[i] else 0
            else:
                next_value = values[i + 1]

            # i時点のtd誤差を計算
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            # i+1時点のtd誤差を使ってGAEを更新（i+1時点のtd誤差には割引率を適用）
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32)