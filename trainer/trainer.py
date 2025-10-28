from collections import deque
import numpy as np
from typing import Dict, Tuple

from common.checkpoint_manager import CheckpointData, CheckpointManager, CheckpointFileGenerator
from settings.setting import TrainerSettings
from base.learning_algorithm_base import LearningAlgorithmBase
from data.buffer import Buffer
from data.transition_data import TransitionData
from enviroment.unity_env_wrapper import UnityEnvWrapper
# from checkpoint_manager import CheckpointManager # 将来的に実装

import pdb

class Trainer:
    """
    学習ループ全体を管理し、アルゴリズム、環境、バッファの橋渡しを行うオーケストレーター。
    """
    def __init__(self,
                algorithm: LearningAlgorithmBase,
                env: UnityEnvWrapper,
                buffer_capacity: int):

        self.algorithm = algorithm
        self.env = env
        self.buffer = Buffer(capacity=buffer_capacity)
        self.checkpoint_manager = CheckpointManager()
        self.checkpoint_file_generator = CheckpointFileGenerator()

        self.best_avg_reward = -np.inf

        print("Trainer initialized.")

    def run(self, 
            trainer_settings: TrainerSettings,
            buffer_size_to_train: int):
        
        """学習ループを実行する"""
        # 最近100エピソードの報酬を保持するためのリスト
        episode_rewards = deque(maxlen=100)
        total_steps = 0
        episode_count = 0
        avg_reward = 0

        try:
            while total_steps < trainer_settings.max_steps:
                # ---------------------------------
                episode_reward = 0
                state = self.env.reset()
                
                for episode_step in range(trainer_settings.max_steps_per_episode):
                    # 1-1. アルゴリズムに行動を決定させる
                    action_info = self.algorithm.decide_action(state)
                    
                    # --- 変更点: 実際の環境ステップ実行 ---
                    next_state, reward, done = self.env.step(
                        action_info.continuous_action,
                        action_info.discrete_action
                    )

                    transition_data = TransitionData(
                        states=state,
                        rewards=reward,
                        dones=done,
                        next_states=next_state,
                        values=action_info.value,
                        log_probs=action_info.log_prob,
                        raw_continuous_actions=action_info.raw_continuous_action,
                        discrete_actions=action_info.discrete_action
                    )

                    # -----------------------------------
                    # 1-2. バッファにデータを保存
                    self.buffer.store(transition_data)

                    state = next_state
                    episode_reward += reward
                    total_steps += 1

                    if self.buffer.is_ready(buffer_size_to_train):
                        print(f"\n--- Step {total_steps}: Buffer is full. Starting training... ---")
                        train_results = self.algorithm.train(self.buffer)
                        print(f"Training finished. Results: {train_results}")
                        self.buffer.clear()
                        print("----------------------------------------------------")

                    if total_steps % trainer_settings.save_interval == 0:
                        avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                        print(f"Step {total_steps}/{trainer_settings.max_steps} | Episode {episode_count} | Avg Reward (Last {len(episode_rewards)} episodes): {avg_reward:.2f}")
                        print(f"--- Step {total_steps}: Saving model checkpoint... ---")

                        # 1. アルゴリズムからベースとなるCheckpointDataを取得
                        state_to_save: CheckpointData = self.algorithm.get_checkpoint_state()
                        
                        # 2. CheckpointDataにTrainer管理下の情報を追加
                        state_to_save.set('total_steps', total_steps)
                        state_to_save.set('episode_count', episode_count) # このエピソード完了時の回数
                        state_to_save.set('best_avg_reward', self.best_avg_reward) # 現時点のベストも保存

                        # 3. CheckpointManagerを使って保存（ベース名とエピソード数を渡す）
                        base_filename = "checkpoint" # configなどから取得しても良い
                        checkpoint_filepath = self.checkpoint_file_generator.generate_filename(base_filename, total_steps)
                        self.checkpoint_manager.save(state_to_save, checkpoint_filepath)

                    # 現在の平均報酬が、これまでの最高記録を上回った場合
                    if avg_reward > self.best_avg_reward:
                        self.best_avg_reward = avg_reward

                        # 保存する状態（CheckpointData）を取得
                        best_state: CheckpointData = self.algorithm.get_checkpoint_state()
                        best_state.set('total_steps', total_steps)
                        best_state.set('episode_count', episode_count)
                        best_state.set('best_avg_reward', self.best_avg_reward)
                        
                        # CheckpointManagerの新しいメソッドで "best_model.pth" として保存
                        best_model_filepath = self.checkpoint_file_generator.generate_best_filename()
                        self.checkpoint_manager.save_best(best_state, best_model_filepath)

                    if done or total_steps >= trainer_settings.max_steps:
                        break
                
                episode_count += 1
                episode_rewards.append(episode_reward)

        finally:
            self.env.close()
            print("\n--- Training Run Finished ---")
            # -----------------------------------

    def load_checkpoint(self, checkpoint_filepath: str) -> Tuple[int, int]:
        """指定されたベース名とエピソード番号からチェックポイントを読み込む"""
        
        # CheckpointManagerのloadに変更。ベース名とエピソードを渡す
        checkpoint_data = self.checkpoint_manager.load(checkpoint_filepath)
        
        if checkpoint_data:
            self.algorithm.load_from_checkpoint(checkpoint_data)
            
            # CheckpointDataインスタンスの .get() メソッドで値を取得
            start_step = checkpoint_data.get('total_steps', 0)
            start_episode = checkpoint_data.get('episode_count', 0)

            # ベストモデルの報酬も読み込んで反映する
            self.best_avg_reward = checkpoint_data.get('best_avg_reward', -np.inf)
            if self.best_avg_reward > -np.inf:
                print(f"Loaded best average reward: {self.best_avg_reward:.2f}")
            
            return start_step, start_episode
        
        # ロード失敗時は0, 0を返す
        return 0, 0