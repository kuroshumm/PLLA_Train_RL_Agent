import torch
import numpy as np

class Buffer:
    """P経験バッファクラス（ハイブリッドアクション対応）"""
    
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.reset()
    
    def reset(self):
        """バッファをリセット"""
        self.states = []
        self.continuous_actions = []
        self.discrete_actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.raw_continuous_actions = []
        self.size = 0
    
    def store_transition(self, state, continuous_action, discrete_action, reward, next_state, done, log_prob, value, raw_continuous_action=None):
        """遷移データを1つ保存"""
        if self.size >= self.capacity:
            self._remove_oldest()
        
        self.states.append(state)
        self.continuous_actions.append(continuous_action)
        self.discrete_actions.append(discrete_action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.raw_continuous_actions.append(raw_continuous_action if raw_continuous_action is not None else continuous_action)
        self.size += 1
    
    def get_tensor_data(self):
        """テンソル形式でデータを取得"""
        if self.size == 0:
            return None
        
        states = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in self.states])
        continuous_actions = torch.tensor(np.array(self.continuous_actions), dtype=torch.float32)
        discrete_actions = torch.tensor(np.array(self.discrete_actions), dtype=torch.int64)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        next_states = torch.stack([s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32) for s in self.next_states])
        raw_continuous_actions = torch.tensor(np.array(self.raw_continuous_actions), dtype=torch.float32)
        
        return {
            'states': states,
            'continuous_actions': continuous_actions,
            'discrete_actions': discrete_actions,
            'old_log_probs': old_log_probs,
            'values': values,
            'rewards': rewards,
            'dones': dones,
            'next_states': next_states,
            'raw_continuous_actions': raw_continuous_actions
        }

    def _remove_oldest(self):
        """最も古いデータを削除"""
        if self.size > 0:
            self.states.pop(0)
            self.continuous_actions.pop(0)
            self.discrete_actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.log_probs.pop(0)
            self.values.pop(0)
            self.raw_continuous_actions.pop(0)
            self.size -= 1

    def is_ready(self, min_size):
        """学習に十分なデータが蓄積されているかチェック"""
        return self.size >= min_size

    def __len__(self):
        return self.size
