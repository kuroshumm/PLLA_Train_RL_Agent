import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Optional, Dict
import pdb

from data.action_info import ActionInfo
from data.buffer import Buffer
from base.learning_algorithm_base import LearningAlgorithmBase
from settings.setting import AlgorithmSettings, TrainerSettings
from common.reward_normalizer import RewardNormalizer
from common.checkpoint_manager import CheckpointData
from .gae_calculator import GAECalculator
from .ppo_loss_calculator import PPOLossCalculator
from .model.ppo_network import PPONetwork

from data.transition_data import TrainData, TrainDataset

class PPOAlgorithm(LearningAlgorithmBase):

    """
    PPOアルゴリズムの実装
    連続値・離散値の両方の行動空間に対応
    """
    def __init__(self, state_size: int, algorithm_settings: AlgorithmSettings, trainer_settings: TrainerSettings):
        
        self.settings = algorithm_settings
        trainer_settings = trainer_settings
        hp = self.settings.hyperparameters
        ppo_hp = hp.ppo
        action_space = self.settings.action_space

        self.continuous_action_size = action_space.continuous_action.get("size", 0) if action_space.use_continuous else 0
        self.discrete_action_size = action_space.discrete_action.get("size", 0) if action_space.use_discrete else 0

        # モデルと専門家コンポーネントのインスタンス化
        self.network = PPONetwork(
            state_size,
            self.continuous_action_size,
            self.discrete_action_size,
            self.settings.model.hidden_size
        )
        self.reward_normalizer = RewardNormalizer()
        self.gae_calculator = GAECalculator(gamma=hp.gamma, gae_lambda=ppo_hp.gae_lambda)
        self.loss_calculator = PPOLossCalculator(
            clip_epsilon=ppo_hp.clip_epsilon,
            value_coef=ppo_hp.value_coef,
            entropy_coef=ppo_hp.entropy_coef
        )
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=hp.learning_rate)
        # 学習率スケジューラの設定
        self.scheduler = LinearLR(
            self.optimizer, 
            start_factor=1.0,
            end_factor=hp.learning_rate_end / hp.learning_rate,
            total_iters=trainer_settings.max_steps
        )
        
        print("PPOAlgorithm is initialized.")

    def decide_action(self, state: np.ndarray) -> ActionInfo:
        """
        現在の状態に基づき、行動を決定する。
        """
        self.network.eval()

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            
            continuous_dist, discrete_dist = self.network.get_action_distribution(state_tensor)
            value = self.network.get_value(state_tensor).item()
            
            log_probs = []
            continuous_action_clipped = None
            raw_continuous_action = None
            discrete_action_np = None

            # 連続値アクションのサンプリング
            if continuous_dist is not None:
                continuous_action_sample = continuous_dist.sample()
                raw_continuous_action = continuous_action_sample.cpu().numpy().flatten()
                # サンプリングした行動が現在の確率分布においてどのくらいの確率で選ばれたか計算（対数確率）
                log_probs.append(continuous_dist.log_prob(continuous_action_sample).sum(axis=-1))
                # サンプリングした連続値アクションをクリップ
                continuous_action_clipped = torch.clip(continuous_action_sample, -1.0, 1.0).cpu().numpy()
            
            # 離散値アクションのサンプリング
            if discrete_dist is not None:
                discrete_action = discrete_dist.sample()
                discrete_action_np = discrete_action.cpu().numpy()
                log_probs.append(discrete_dist.log_prob(discrete_action))

            # 対数確率を合算することで、連続値・離散値アクションの両方を考慮
            log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1).item()

        return ActionInfo(
            continuous_action=continuous_action_clipped,
            discrete_action=discrete_action_np,
            raw_continuous_action=raw_continuous_action,
            value=value,
            log_prob=log_prob,
        )

    def train(self, buffer: Buffer) -> Dict[str, float]:
        """
        収集された経験を用いて、モデルを学習させるプロセスを指揮する。
        """
        self.network.train()
        
        # --- 1. データの準備 ---
        batch_tensors = buffer.get_tensor_data()
        if not batch_tensors:
            return {}
        
        # データ取得
        states = batch_tensors.states
        values = batch_tensors.values
        rewards = batch_tensors.rewards
        dones = batch_tensors.dones
        next_states = batch_tensors.next_states
        old_log_probs = batch_tensors.log_probs
        raw_continuous_actions = batch_tensors.raw_continuous_actions
        discrete_actions = batch_tensors.discrete_actions

        # GAE計算前に報酬を正規化
        #self.reward_normalizer.update(rewards)
        #normalized_rewards = self.reward_normalizer.normalize(rewards)
        normalized_rewards = rewards

        with torch.no_grad():
            next_values = self.network.get_value(next_states).squeeze()

        # GAEを使用してアドバンテージを計算
        # アドバンテージ＝ある状態sにおいて行動aを選択したとき、平均的な行動と比べてどれだけ優れているか
        advantages = self.gae_calculator.calculate(
            normalized_rewards, values, next_values, dones
        )

        # Criticの学習ターゲットとなるリターン（収益）を計算。
        # GAE（アドバンテージ）に現在の状態価値を足し合わせることで、
        # より安定したQ値の推定値を算出し、価値関数の学習を効率化する。
        returns = advantages + values

        # データセットの準備
        # 必須のTensorをリストに追加
        dataset_tensors = TrainDataset(
            TrainData(
                states=states,
                old_log_probs=old_log_probs,
                advantages=advantages,
                returns=returns,
                values=values,
                continuous_actions=raw_continuous_actions,
                discrete_actions=discrete_actions,
            )
        )
        dataloader = DataLoader(dataset_tensors, batch_size=self.settings.hyperparameters.batch_size, shuffle=True)

        # 学習の実行 (ミニバッチ学習)
        loss_list = []
        epochs = self.settings.hyperparameters.ppo.epochs
        for _ in range(epochs):
            for batch in dataloader:

                # advantageを正規化
                advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
                batch_data = batch._replace(advantages=advantages)

                loss = self.loss_calculator.compute(
                    self.network, 
                    batch_data, 
                )

                loss_list.append(loss.item())
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.settings.hyperparameters.ppo.max_grad_norm)
                self.optimizer.step()
                
        # スケジューラのステップを更新
        if self.scheduler is not None:
            self.scheduler.step()

        return {"total_loss": sum(loss_list) / len(loss_list)} if loss_list else {"total_loss": 0}
    
    def get_checkpoint_state(self) -> CheckpointData:
        """
        チェックポイント保存用の状態を CheckpointData インスタンスとして返す
        """
        
        return CheckpointData(
            network_state_dict=self.network.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict()
        )

    def load_from_checkpoint(self, checkpoint_data: CheckpointData):
        """
        CheckpointData インスタンスから状態を読み込む
        """
        
        self.network.load_state_dict(checkpoint_data.data['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint_data.data['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint_data.data['scheduler_state_dict'])
        print("✓ PPOAlgorithm state has been loaded from checkpoint.")