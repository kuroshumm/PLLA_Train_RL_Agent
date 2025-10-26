import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from .actor_critic import ContinuousActorNetwork, CriticNetwork
import pdb

class PPONetwork(nn.Module):
    """ActorとCriticを完全分離したハイブリッドPPOネットワーク"""
    
    def __init__(self, state_size, continuous_action_size, hidden_size=128):
        super(PPONetwork, self).__init__()
        self.continuous_action_size = continuous_action_size

        self.actor = ContinuousActorNetwork(state_size, continuous_action_size, hidden_size)
        self.critic = CriticNetwork(state_size, hidden_size)
        
    def forward(self, x):
        # Actorから行動の平均と標準偏差を取得
        continuous_mean, continuous_std = self.actor(x, self.continuous_action_size)
        # Criticから状態価値を取得
        state_value = self.critic(x)

        return continuous_mean, continuous_std, state_value

    def evaluate_actions(self, state, continuous_action):
        """
        与えられた状態xに対して、指定された行動continuous_actionの評価を行う
        Args:
            x (torch.Tensor): 入力状態
            continuous_action (torch.Tensor): 評価する連続値アクション
        Returns:
            log_prob (torch.Tensor): 指定された行動の対数確率
            entropy (torch.Tensor): 行動のエントロピー（不確実性の尺度）
            state_value (torch.Tensor): 状態の価値
        """
        
        continuous_mean, continuous_std, state_value = self(state)

        continuous_dist = Normal(continuous_mean, continuous_std)
        log_prob = continuous_dist.log_prob(continuous_action).sum(axis=-1)
        entropy = continuous_dist.entropy().sum(axis=-1)

        return log_prob, entropy, state_value

    def get_value(self, x):
        return self.critic(x)
    
    def get_action_distribution(self, x):
        """
        状態xに対する行動の確率分布を取得
        連続値アクションと離散値アクションの両方を考慮

        Args:
            x (torch.Tensor): 入力状態
        Returns:
            continuous_dist (Normal): 連続値アクションの分布
            discrete_dist (Categorical): 離散値アクションの分布
        """

        continuous_mean, continuous_std = self.actor(x, self.continuous_action_size)
        
        continuous_dist = Normal(continuous_mean, continuous_std)
            
        return continuous_dist