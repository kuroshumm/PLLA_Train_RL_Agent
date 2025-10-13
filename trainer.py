import numpy as np

from settings.setting import TrainerSettings
from base.learning_algorithm_base import LearningAlgorithmBase
from data.buffer import Buffer
from data.transition_data import TransitionData
from env.unity_env_wrapper import UnityEnvWrapper
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
        # self.checkpoint_manager = CheckpointManager()

        print("Trainer initialized.")

    def run(self, 
            trainer_settings: TrainerSettings,
            buffer_size_to_train: int):
        
        """学習ループを実行する"""
        total_steps = 0
        episode_count = 0

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
                        None
                    )

                    transition_data = TransitionData(
                        states=state,
                        rewards=reward,
                        dones=done,
                        next_states=next_state,
                        values=action_info.value,
                        log_probs=action_info.log_prob,
                        raw_continuous_actions=action_info.raw_continuous_action
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
                        print(f"Step {total_steps}/{trainer_settings.max_steps} | Episode {episode_count}")
                        print(f"--- Step {total_steps}: Saving model checkpoint... ---")

                    if done or total_steps >= trainer_settings.max_steps:
                        break
                
                episode_count += 1
                #print(f"Episode {episode_count} finished. Reward: {episode_reward:.2f}")

        finally:
            self.env.close()
            print("\n--- Training Run Finished ---")
            # -----------------------------------