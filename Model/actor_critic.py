import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    """Actor専用ネットワーク（行動選択）"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # 行動の平均値
        self.action_mean = nn.Linear(hidden_size, action_size)
        
        # 行動の標準偏差（学習可能なパラメータ）
        self.action_log_std = nn.Parameter(torch.zeros(action_size))
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 行動の平均値
        action_mean = self.action_mean(x)
        
        # 標準偏差（常に正の値）
        action_std = torch.exp(self.action_log_std.expand_as(action_mean))
        
        return action_mean, action_std

class HybridActorNetwork(nn.Module):
    """連続値と離散値を出力するActorネットワーク"""
    def __init__(self, state_size, continuous_action_size, discrete_action_size, hidden_size=128):
        super(HybridActorNetwork, self).__init__()
        
        # 隠れ層の定義
        self.hidden_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self._initialize_weights()

        # 連続値アクション用の出力層
        if continuous_action_size > 0:
            self.continuous_mean = nn.Linear(hidden_size, continuous_action_size)
            # action_log_stdは「対数を取った標準偏差」として扱われる学習可能なパラメータ。
            # この値自体が統計的な標準偏差ではなく、探索の幅を決定するために学習される。
            # 指数（exp）を取ることで、標準偏差が常に正の値になることを保証する。
            self.action_log_std = nn.Parameter(torch.zeros(continuous_action_size))

            nn.init.uniform_(self.continuous_mean.weight, -0.01, 0.01)
            nn.init.constant_(self.continuous_mean.bias, 0)

        # 離散値アクション用の出力層
        if discrete_action_size > 0:
            self.discrete_logits = nn.Linear(hidden_size, discrete_action_size)

            nn.init.uniform_(self.discrete_logits.weight, -0.01, 0.01)
            nn.init.constant_(self.discrete_logits.bias, 0)

    def _initialize_weights(self):
        # 隠れ層の重みをKaiming初期化、バイアスを0で初期化
        for module in self.hidden_layers:
            if isinstance(module, nn.Linear):
                # Kaiming UniformをReLUと組み合わせて使用
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                # バイアスは0で初期化
                nn.init.constant_(module.bias, 0)

    def forward(self, x, continuous_action_size, discrete_action_size):
        x = self.hidden_layers(x)
        
        continuous_mean, continuous_std, discrete_logits = None, None, None

        if continuous_action_size > 0:
            continuous_mean = self.continuous_mean(x)
            continuous_std = torch.exp(self.action_log_std.expand_as(continuous_mean))
        
        if discrete_action_size > 0:
            discrete_logits = self.discrete_logits(x)
        
        return continuous_mean, continuous_std, discrete_logits

class CriticNetwork(nn.Module):
    """Critic専用ネットワーク（価値関数）"""
    
    def __init__(self, state_size, hidden_size=128):
        super(CriticNetwork, self).__init__()
        
        # 隠れ層の定義
        self.hidden_layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.value = nn.Linear(hidden_size, 1)

        self._initialize_weights()
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.hidden_layers(x)

        # 状態価値
        state_value = self.value(x)
        
        return state_value
    
    def _initialize_weights(self):
        # 隠れ層の重みをKaiming初期化、バイアスを0で初期化
        for module in self.hidden_layers:
            if isinstance(module, nn.Linear):
                # Kaiming UniformをReLUと組み合わせて使用
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                # バイアスは0で初期化
                nn.init.constant_(module.bias, 0)

        # 価値関数の出力層
        nn.init.uniform_(self.value.weight, -0.01, 0.01)
        nn.init.constant_(self.value.bias, 0)
