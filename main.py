import sys

from settings.loader import load_config
from trainer.trainer import Trainer
from ppo.ppo import PPOAlgorithm
from base.learning_algorithm_base import LearningAlgorithmBase
from env.unity_env_wrapper import UnityEnvWrapper # インポートを追加
from common.seed_utils import set_seed # 既存のseed_utils.pyをそのまま利用

def main():
    """アプリケーションのエントリーポイント"""
    print("--- 1. Loading Configuration & Initializing Environment ---")
    try:
        config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
        config = load_config(config_path)
        print(f"Configuration loaded from '{config_path}'.")
        
        env_wrapper = UnityEnvWrapper(config.environment)
        
        state_size = env_wrapper.get_state_size()
        continuous_action_size = env_wrapper.get_continuous_action_size()
        
        config.algorithm.action_space.continuous_action["size"] = continuous_action_size

        print(f"State Size: {state_size}, Continuous Actions: {continuous_action_size}")

    except Exception as e:
        print(f"✗ Error during initialization: {e}")
        import traceback
        traceback.print_exc()
        return

    set_seed(config.environment.seed)

    print("\n--- 2. Initializing Algorithm ---")
    algorithm: LearningAlgorithmBase
    
    if config.algorithm.name == "PPO":
        algorithm = PPOAlgorithm(state_size, config.algorithm, config.trainer)
    else:
        print(f"✗ Error: Unknown algorithm '{config.algorithm.name}'.")
        env_wrapper.close() # エラー時も環境を閉じる
        return
    
    print(f"✓ Algorithm '{config.algorithm.name}' initialized.")

    print("\n--- 3. Initializing Trainer ---")
    trainer = Trainer(
        algorithm=algorithm,
        env=env_wrapper,
        buffer_capacity=config.algorithm.hyperparameters.buffer_size * 2
    )
    print("✓ Trainer initialized.")

    print("\n--- 4. Starting Training Run ---")
    try:
        trainer.run(
            trainer_settings=config.trainer,
            buffer_size_to_train=config.algorithm.hyperparameters.buffer_size
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\n✗ An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n--- Application Finished ---")

if __name__ == "__main__":
    main()