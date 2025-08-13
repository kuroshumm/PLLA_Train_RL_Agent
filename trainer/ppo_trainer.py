import numpy as np

from base.base_agent import BaseAgent
from env.unity_env_wrapper import UnityEnvWrapper
from config import Config 

class PPOTrainer:
    """単一のエージェントで学習ループを管理するクラス"""

    def __init__(self, agent: BaseAgent, env_wrapper: UnityEnvWrapper, config: Config):
        self.agent = agent
        self.env_wrapper = env_wrapper
        self.config = config
        
        self.best_avg_reward = -np.inf
        self.best_model_path = ""
        print(f"PPOTrainer initialized for agent: '{self.config.agent_name}'")

    def train(self):
        """学習の実行"""
        total_steps = 0
        episode = 0
        episode_rewards = []
        last_log_step = 0
        last_log_episode = 0
        
        while total_steps < self.config.max_steps:
            episode_reward, episode_steps, last_log_step, last_log_episode = self._run_episode(
                total_steps, episode, episode_rewards, last_log_step, last_log_episode
            )
            
            total_steps += episode_steps
            episode += 1
            episode_rewards.append(episode_reward)

        print("\n--- Finalizing Training ---")
        self._log_progress(total_steps, episode, episode_rewards, last_log_episode)
        
        print("Saving final model...")
        final_checkpoint_path = f"ppo_checkpoint_{total_steps}.pth"
        self.agent.save_checkpoint(final_checkpoint_path)

        if self.best_model_path:
            print(f"Exporting best model to ONNX from log/checkpoint/{self.best_model_path}")
            self.agent.load_checkpoint(self.best_model_path)
            self.agent.export_onnx("best_ppo_model.onnx")
        else:
            print("No best model was saved during training. Exporting final model to ONNX.")
            self.agent.export_onnx(f"ppo_actor_{total_steps}.onnx")

        print("---------------------------")
        
        return episode_rewards
    
    def _run_episode(self, total_steps, current_episode, all_rewards, last_log_step, last_log_episode):
        """1エピソードの実行と、ステップごとのログ・モデル保存チェック"""
        episode_reward = 0
        episode_step = 0
        state, _ = self.env_wrapper.reset()

        while True:
            current_total_steps = total_steps + episode_step
            if current_total_steps >= self.config.max_steps:
                break

            continuous_action_clipped, discrete_action, log_prob, value, raw_continuous_action = self.agent.select_action(state, training=True)
            next_state, reward, done, _ = self.env_wrapper.step(continuous_action_clipped, discrete_action)

            self.agent.store_transition(state, continuous_action_clipped, discrete_action, reward, next_state, done, log_prob, value, raw_continuous_action)
            self.agent.update()
            
            if current_total_steps >= last_log_step + self.config.save_interval:
                avg_reward = self._log_progress(current_total_steps, current_episode, all_rewards, last_log_episode)
                
                checkpoint_path = f"ppo_checkpoint_{current_total_steps}.pth"
                self.agent.save_checkpoint(checkpoint_path)
                self.agent.save_actor_model(f"ppo_actor_{current_total_steps}.pth")

                if avg_reward is not None and avg_reward > self.best_avg_reward:
                    self.best_avg_reward = avg_reward
                    self.best_model_path = checkpoint_path
                    print(f"New best model found! Average reward: {avg_reward:.2f}. Checkpoint: {checkpoint_path}")

                last_log_step = current_total_steps
                last_log_episode = current_episode + 1

            state = next_state
            episode_reward += reward
            episode_step += 1
            
            if done:
                break
        
        return episode_reward, episode_step, last_log_step, last_log_episode
    
    def _log_progress(self, total_steps, current_episode, all_rewards, last_log_episode):
        """学習の進捗をコンソールに表示し、平均報酬を返す"""
        rewards_since_last_log = all_rewards[last_log_episode:]
        
        avg_reward = None
        if not rewards_since_last_log:
            avg_reward_str = "N/A"
            num_episodes_str = "0"
        else:
            avg_reward = np.mean(rewards_since_last_log)
            num_episodes_since_log = len(rewards_since_last_log)
            avg_reward_str = f"{avg_reward:.2f}"
            num_episodes_str = str(num_episodes_since_log)

        print(f"Step {total_steps}/{self.config.max_steps}, "
            f"Episode {current_episode + 1}, "
            f"Avg Reward ({num_episodes_str} episodes): {avg_reward_str}")
        
        return avg_reward