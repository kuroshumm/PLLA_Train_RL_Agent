import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from base.base_agent import BaseAgent
from Model.ppo_network import PPONetwork
from agent.buffer import Buffer
from common.normalizer import RunningMeanStd
from common.seed_utils import set_seed
from config import Config
import numpy as np
import os

class PPOAgent(BaseAgent):

    def __init__(self, state_size: int, config: Config):
        super().__init__(state_size, config)

        # 保存用ディレクトリの作成
        os.makedirs("log/checkpoint", exist_ok=True)

        # PPOネットワークの初期化
        self.network = PPONetwork(state_size, self.continuous_action_size, self.discrete_action_size, self.hidden_size)
        self.buffer = Buffer(getattr(config, 'buffer_capacity', 10000))

        self._init_optimizers()
        self._init_params()

        # 報酬正規化のためのインスタンス
        self.reward_rms = RunningMeanStd(shape=(1,))

    # ----------------------------------------------------------------------
    # BaseAgent 関数
    # ----------------------------------------------------------------------

    def select_action(self, state, training=True):
        """受け取った状態から、方策に基づいて次の行動を選択する"""

        self.network.eval() if not training else self.network.train()
        
        with torch.no_grad():
            # ニューラルネットワーク（方策関数）から、現在の状態に対する行動の確率分布を取得
            # 行動の確率分布は連続値と離散値の両方を含む
            continuous_dist, discrete_dist = self.network.get_action_distribution(state)

            # アクションの初期化
            raw_continuous_action = np.zeros(self.continuous_action_size, dtype=np.float32)
            continuous_action_clipped = np.zeros(self.continuous_action_size, dtype=np.float32)
            discrete_action = np.zeros(1, dtype=np.int64) # Assuming single discrete branch
            log_prob = 0

            # 行動の確率分布から行動をサンプリング（連続値）
            if self.config.use_continuous and continuous_dist is not None:
                continuous_action_sample = continuous_dist.sample()
                raw_continuous_action = continuous_action_sample.cpu().numpy().flatten()
                # サンプリングした行動が現在の確率分布においてどの位の確率で選ばれたかを計算（対数確率）
                log_prob += continuous_dist.log_prob(continuous_action_sample).sum(axis=-1)
                # サンプリングした連続値アクションをクリッピング
                continuous_action_clipped = (torch.tanh(continuous_action_sample) * self.action_scale + self.action_bias).cpu().numpy().flatten()

            # 行動の確率分布から行動をサンプリング（離散値）
            if self.config.use_discrete and discrete_dist is not None:
                discrete_action_sample = discrete_dist.sample()
                # サンプリングした行動が現在の確率分布においてどの位の確率で選ばれたかを計算（対数確率）
                log_prob += discrete_dist.log_prob(discrete_action_sample)
                discrete_action = discrete_action_sample.cpu().numpy().flatten()

            # 学習時
            # Criticネットワークから現在の状態価値（将来得られる報酬の期待値）を取得
            if training:
                state_value = self.network.get_value(state).item()
                return (continuous_action_clipped, 
                        discrete_action,
                        log_prob.item() if torch.is_tensor(log_prob) else log_prob, 
                        state_value, 
                        raw_continuous_action)
            else:
                continuous_mean, _, discrete_logits, _ = self.network(state)
                if self.config.use_continuous and continuous_mean is not None:
                    continuous_action_clipped = (torch.tanh(continuous_mean) * self.action_scale + self.action_bias).cpu().numpy().flatten()
                if self.config.use_discrete and discrete_logits is not None:
                    discrete_action = torch.argmax(discrete_logits, dim=-1).cpu().numpy().flatten()
                return continuous_action_clipped, discrete_action

    def store_transition(self, state, continuous_action, discrete_action, reward, next_state, done, log_prob=None, value=None, raw_continuous_action=None):
        self.buffer.store_transition(state, continuous_action, discrete_action, reward, next_state, done, log_prob, value, raw_continuous_action)

    def update(self):
        """PPOアルゴリズムに基づいてモデルを更新する"""

        # バッファが十分なデータを持っているか確認
        if not self.buffer.is_ready(self.config.buffer_size):
            return None
        
        # バッファからデータを取得
        data = self.buffer.get_tensor_data()
        if data is None: return None
            
        # データをテンソルに変換
        states = data['states']
        continuous_actions = data['raw_continuous_actions']
        discrete_actions = data['discrete_actions']
        old_log_probs = data['old_log_probs']
        values = data['values']
        rewards = data['rewards']
        dones = data['dones']
        next_states = data['next_states']
        
        # GAE計算前に報酬を正規化
        self.reward_rms.update(rewards.numpy().reshape(-1, 1))
        normalized_rewards = torch.tensor(self.reward_rms.normalize(rewards.numpy().reshape(-1, 1)), dtype=torch.float32).flatten()

        # 次状態の価値を計算
        with torch.no_grad():
            next_values = self.network.get_value(next_states).squeeze()
        
        # GAEを使用してアドバンテージを計算
        # アドバンテージ＝ある状態sにおいて行動aを選択したとき、平均的な行動と比べてどれだけ優れているか
        advantages = self._compute_gae(normalized_rewards, values, next_values, dones)

        # Criticの学習ターゲットとなるリターン（収益）を計算。
        # GAE（アドバンテージ）に現在の状態価値を足し合わせることで、
        # より安定したQ値の推定値を算出し、価値関数の学習を効率化する。
        returns = advantages + values

        # アドバンテージの値は学習効率化のため正規化する
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        
        dataset = TensorDataset(states, continuous_actions, discrete_actions, old_log_probs, advantages, returns, values)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        for _ in range(self.ppo_epochs):
            for batch_states, batch_continuous_actions, batch_discrete_actions, batch_old_log_probs, batch_advantages, batch_returns, batch_values in dataloader:

                # ここでネットワークを通して現在の価値を取得
                _, _, current_log_probs, entropy, current_values = self.network.get_action_and_value(
                    batch_states, 
                    batch_continuous_actions, 
                    batch_discrete_actions,
                    use_continuous=self.config.use_continuous,
                    use_discrete=self.config.use_discrete
                )
                
                # -------------------------------------
                # Lossの計算
                # -------------------------------------

                # Actorの損失計算

                # 現在と過去(データ収集した時)の間で、ある行動が選ばれる確率の比率を計算
                # これにより、過去の行動選択と現在の方策の違いを考慮する
                # 　ratioが大きい=新しい方策ではその行動が選ばれやすい
                # 　ratioが小さい=新しい方策ではその行動が選ばれにくい
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                # 比率にアドバンテージを掛けることで、行動選択の優位性を考慮
                # 　アドバンテージが正（良い行動）ならratioを大きくする
                # 　アドバンテージが負（悪い行動）ならratioを小さくする
                # これにより、過去の行動選択と現在の方策の違いを考慮した損失を計算
                surr1 = ratio * batch_advantages
                # クリッピングを行うことで、過度な更新を防ぐ
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Criticの損失計算

                # 現在の状態価値とリターン（収益）の差を計算
                value_losses = F.mse_loss(current_values.squeeze(), batch_returns, reduction='none')
                # クリッピングを行うことで、価値関数の過度な更新を防ぐ
                value_pred_clipped = batch_values + (current_values.squeeze() - batch_values).clamp(-self.clip_epsilon, self.clip_epsilon)
                value_losses_clipped = F.mse_loss(value_pred_clipped, batch_returns, reduction='none')
                critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                
                # エントロピー損失の計算
                # エントロピーを最大化することで、行動の多様性を促進
                # これにより、エージェントが特定の行動に偏らず、探索を続けることができる
                entropy_loss = entropy.mean()
                
                # Actor、Critic、エントロピーの損失を合成
                # これにより、PPOの目的関数を最小化する
                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy_loss.item()
        
        self.buffer.reset()
        
        # スケジューラのステップを更新
        if self.scheduler is not None:
            self.scheduler.step()
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        num_updates = self.ppo_epochs * len(dataloader)
        return {
            'actor_loss': total_actor_loss / num_updates,
            'critic_loss': total_critic_loss / num_updates,
            'entropy': total_entropy_loss / num_updates
        }

    def _compute_gae(self, rewards, values, next_values, dones):
        """
        Generalized Advantage Estimation (GAE)を計算する
        
        td誤差_t = r_t + γ * V(s') - V(s)
        gae_t = td誤差_t + γ * λ * gae_{t+1}

        Args:
            rewards (torch.Tensor): 報酬のテンソル
            values (torch.Tensor): 状態価値のテンソル
            next_values (torch.Tensor): 次状態の価値のテンソル
            dones (torch.Tensor): エピソード終了フラグのテンソル
        Returns:
            torch.Tensor: GAEのテンソル
        """

        gae_lambda = getattr(self.config, 'gae_lambda', 0.95)
        advantages = []
        gae = 0
        # GAEを再帰的に計算
        # 逆順で計算することで、次状態の価値を簡単に参照できる
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = next_values[-1] if not dones[i] else 0
            else:
                next_value = values[i + 1]
            
            # i時点のtd誤差を計算
            delta = rewards[i] + self.config.gamma * next_value * (1 - dones[i]) - values[i]
            # i+1時点のtd誤差を使ってGAEを更新（i+1時点のtd誤差には割引率を適用）
            gae = delta + self.config.gamma * gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages, dtype=torch.float32)

    def export_onnx(self, filepath):
        self.network.eval()
        dummy_input = torch.randn(1, self.state_size)
        
        class HybridActorOnly(nn.Module):
            def __init__(self, actor_network, continuous_size, discrete_size):
                super().__init__()
                self.actor = actor_network
                self.continuous_size = continuous_size
                self.discrete_size = discrete_size
                
            def forward(self, x):
                continuous_mean, _, discrete_logits = self.actor(x, self.continuous_size, self.discrete_size)
                continuous_action, discrete_action = None, None
                if self.continuous_size > 0:
                    continuous_action = torch.tanh(continuous_mean)
                if self.discrete_size > 0:
                    discrete_action = torch.argmax(discrete_logits, dim=-1, keepdim=True)
                return continuous_action, discrete_action
        
        actor_only = HybridActorOnly(self.network.actor, self.continuous_action_size, self.discrete_action_size)

        filepath = "log/" + filepath
        output_names = []
        if self.continuous_action_size > 0:
            output_names.append('continuous_actions')
        if self.discrete_action_size > 0:
            output_names.append('discrete_actions')

        torch.onnx.export(
            actor_only,
            dummy_input,
            filepath,
            verbose=False,
            input_names=['observations'],
            output_names=output_names,
            opset_version=11,
            do_constant_folding=True
        )

    def save_checkpoint(self, filepath):
        """学習再開用のチェックポイント（全体モデル）を保存する"""
        filepath = "log/checkpoint/" + filepath
        torch.save(self.network.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        """学習再開用のチェックポイント（全体モデル）を読み込む"""
        filepath = "log/checkpoint/" + filepath
        self.network.load_state_dict(torch.load(filepath))

    def save_actor_model(self, filepath):
        """推論用のActorモデルを保存する"""
        filepath = "log/checkpoint/" + filepath
        torch.save(self.network.actor.state_dict(), filepath)

    def load_actor_model(self, filepath):
        """推論用のActorモデルを読み込む"""
        filepath = "log/checkpoint/" + filepath
        self.network.actor.load_state_dict(torch.load(filepath))


    # ----------------------------------------------------------------------
    # 非公開 関数
    # ----------------------------------------------------------------------

    def _init_optimizers(self):
        """オプティマイザを初期化する"""

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.lr)

        # 学習率スケジューラの設定
        self.scheduler = LinearLR(
            self.optimizer, 
            start_factor=1.0,
            end_factor=self.config.lr_end / self.config.lr,
            total_iters=self.config.max_steps
        )
    
    def _init_params(self):
        """ネットワークのパラメータを設定する"""

        self.clip_epsilon = getattr(self.config, 'clip_epsilon', 0.2)
        self.value_coef = getattr(self.config, 'value_coef', 0.5)
        self.entropy_coef = getattr(self.config, 'entropy_coef', 0.01)
        self.ppo_epochs = getattr(self.config, 'ppo_epochs', 4)
        self.max_grad_norm = getattr(self.config, 'max_grad_norm', 0.5)

        self.action_scale = getattr(self.config, 'action_scale', 1.0)
        self.action_bias = getattr(self.config, 'action_bias', 0.0)
