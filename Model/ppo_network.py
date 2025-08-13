import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
from .actor_critic import HybridActorNetwork, CriticNetwork
import pdb

class PPONetwork(nn.Module):
    """ActorとCriticを完全分離したハイブリッドPPOネットワーク"""
    
    def __init__(self, state_size, continuous_action_size, discrete_action_size, hidden_size=128):
        super(PPONetwork, self).__init__()
        self.continuous_action_size = continuous_action_size
        self.discrete_action_size = discrete_action_size

        self.actor = HybridActorNetwork(state_size, continuous_action_size, discrete_action_size, hidden_size)
        self.critic = CriticNetwork(state_size, hidden_size)
        
    def forward(self, x):
        continuous_mean, continuous_std, discrete_logits = self.actor(x, self.continuous_action_size, self.discrete_action_size)
        state_value = self.critic(x)
        return continuous_mean, continuous_std, discrete_logits, state_value
    
    def get_action_and_value(self, x, continuous_action=None, discrete_action=None, use_continuous=True, use_discrete=True):
        continuous_mean, continuous_std, discrete_logits, state_value = self(x)
        
        continuous_dist, discrete_dist = None, None
        log_prob = 0
        entropy = 0

        if use_continuous and self.continuous_action_size > 0:
            continuous_dist = Normal(continuous_mean, continuous_std)
            if continuous_action is None:
                continuous_action = continuous_dist.sample()
            log_prob += continuous_dist.log_prob(continuous_action).sum(axis=-1)
            entropy += continuous_dist.entropy().sum(axis=-1)

        if use_discrete and self.discrete_action_size > 0:
            discrete_dist = Categorical(logits=discrete_logits)
            if discrete_action is None:
                discrete_action = discrete_dist.sample()
                log_prob += discrete_dist.log_prob(discrete_action.squeeze(-1))
            entropy += discrete_dist.entropy()
            
        return continuous_action, discrete_action, log_prob, entropy, state_value
    
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

        continuous_mean, continuous_std, discrete_logits = self.actor(x, self.continuous_action_size, self.discrete_action_size)
        
        continuous_dist, discrete_dist = None, None
        if self.continuous_action_size > 0:
            continuous_dist = Normal(continuous_mean, continuous_std)
        if self.discrete_action_size > 0:
            discrete_dist = Categorical(logits=discrete_logits)
            
        return continuous_dist, discrete_dist
