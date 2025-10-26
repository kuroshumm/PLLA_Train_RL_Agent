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
    PPOアルゴリズムのオーケストレータークラス。
    連続値・離散値・ハイブリッドアクションに対応。
    """
    def __init__(self, state_size: int, algorithm_settings: AlgorithmSettings, trainer_settings: TrainerSettings):
        
        self.settings = algorithm_settings
        trainer_settings = trainer_settings
        hp = self.settings.hyperparameters
        ppo_hp = hp.ppo
        action_space = self.settings.action_space
        
        # --- 1. モデルと専門家コンポーネントのインスタンス化 ---
        self.network = PPONetwork(
            state_size,
            action_space.continuous_action.get("size", 0),
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
        
        print("PPOAlgorithm (Hybrid Action Enabled) is initialized.")

    def decide_action(self, state: np.ndarray) -> ActionInfo:
        """
        現在の状態に基づき、行動を決定する。
        (Source: ppo_agent.py の select_action メソッドを移植)
        """
        self.network.eval()

        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            
            continuous_dist = self.network.get_action_distribution(state_tensor)
            value = self.network.get_value(state_tensor).item()
            
            # --- アクションのサンプリングと情報収集 ---
            # 連続値アクションの処理
            continuous_action_sample = continuous_dist.sample()
            raw_continuous_action = continuous_action_sample.cpu().numpy().flatten()
            # サンプリングした行動が現在の確率分布においてどの位の確率で選ばれたかを計算（対数確率）
            log_prob = continuous_dist.log_prob(continuous_action_sample).sum(axis=-1)
            # サンプリングした連続値アクションをクリッピング
            # action_scale = self.settings.action_space.continuous_action.get("scale", 1.0)
            # action_bias = self.settings.action_space.continuous_action.get("bias", 0.0)
            # continuous_action_clipped = (torch.tanh(continuous_action_sample) * action_scale + action_bias).cpu().numpy()
            continuous_action_clipped = torch.clip(continuous_action_sample, -1.0, 1.0).cpu().numpy()

        return ActionInfo(
            continuous_action=continuous_action_clipped,
            raw_continuous_action=raw_continuous_action,
            value=value,
            log_prob=log_prob.item(),
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
        
        # データをテンソルに変換
        states = batch_tensors.states
        values = batch_tensors.values
        rewards = batch_tensors.rewards
        dones = batch_tensors.dones
        next_states = batch_tensors.next_states
        old_log_probs = batch_tensors.log_probs
        raw_continuous_actions = batch_tensors.raw_continuous_actions

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

        # アドバンテージの値は学習効率化のため正規化する
        #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- 3. データセットの準備 ---
        # 必須のTensorをリストに追加
        dataset_tensors = TrainDataset(
            TrainData(
                states=states,
                old_log_probs=old_log_probs,
                advantages=advantages,
                returns=returns,
                values=values,
                actions=raw_continuous_actions,
            )
        )
        dataloader = DataLoader(dataset_tensors, batch_size=self.settings.hyperparameters.batch_size, shuffle=True)
        
        # --- 4. 学習の実行 (ミニバッチ学習) ---
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
        """チェックポイント保存用の状態を CheckpointData インスタンスとして返す"""
        
        # CheckpointDataのコンストラクタにキーワード引数として渡す
        return CheckpointData(
            network_state_dict=self.network.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict()
        )

    def load_from_checkpoint(self, checkpoint_data: CheckpointData):
        """CheckpointData インスタンスから状態を読み込む"""
        
        # .data 属性経由で辞書にアクセス
        self.network.load_state_dict(checkpoint_data.data['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint_data.data['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint_data.data['scheduler_state_dict'])
        print("✓ PPOAlgorithm state has been loaded from checkpoint.")