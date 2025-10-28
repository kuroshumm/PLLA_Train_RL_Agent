import pdb
import torch
import torch.nn.functional as F
from typing import Dict
from data.transition_data import TrainData
from ppo.model.ppo_network import PPONetwork

class PPOLossCalculator:
    """
    PPOの損失を計算する責務を持つクラス。
    """
    def __init__(self, clip_epsilon: float, value_coef: float, entropy_coef: float):
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def compute(self, 
                network: PPONetwork,
                batch: TrainData) -> torch.Tensor:

        """
        バッチデータから合計損失を計算する
        """
        
        # ネットワークから現在の予測値を取得
        log_probs, entropy, values = network.evaluate_actions(
            batch.states,
            batch.continuous_actions,
            batch.discrete_actions
        )
        values = values.squeeze()

        # Actor loss (Policy loss)

        # 現在と過去(データ収集した時)の間で、ある行動が選ばれる確率の比率を計算
        # これにより、過去の行動選択と現在の方策の違いを考慮する
        # 　ratioが大きい=新しい方策ではその行動が選ばれやすい
        # 　ratioが小さい=新しい方策ではその行動が選ばれにくい
        ratio = torch.exp(log_probs - batch.old_log_probs)
        # 比率にアドバンテージを掛けることで、行動選択の優位性を考慮
        # 　アドバンテージが正（良い行動）ならratioを大きくする
        # 　アドバンテージが負（悪い行動）ならratioを小さくする
        # これにより、過去の行動選択と現在の方策の違いを考慮した損失を計算
        surr1 = ratio * batch.advantages
        # クリッピングを行うことで、過度な更新を防ぐ
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch.advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic loss (Value loss) with clipping
        # 現在の状態価値とリターン（収益）の差を計算
        value_losses = F.mse_loss(batch.returns, values, reduction='none')
        # クリッピングを行うことで、価値関数の過度な更新を防ぐ
        # values_clipped = batch['values'] + (values - batch['values']).clamp(-self.clip_epsilon, self.clip_epsilon)
        # value_losses_clipped = F.mse_loss(values_clipped, batch['returns'], reduction='none')
        #critic_loss = torch.max(value_losses, value_losses_clipped).mean()
        critic_loss = value_losses.mean()

        # Entropy loss
        # エントロピー損失の計算
        # エントロピーを最大化することで、行動の多様性を促進
        # これにより、エージェントが特定の行動に偏らず、探索を続けることができる
        entropy_loss = entropy.mean()

        # Total loss
        total_loss = (actor_loss + 
                      self.value_coef * critic_loss - 
                      self.entropy_coef * entropy_loss)
        
        return total_loss