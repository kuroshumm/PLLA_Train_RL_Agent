import sys
import argparse

from settings.loader import load_config
from trainer.trainer import Trainer
from ppo.ppo import PPOAlgorithm
from base.learning_algorithm_base import LearningAlgorithmBase
from env.unity_env_wrapper import UnityEnvWrapper # インポートを追加
from common.seed_utils import set_seed # 既存のseed_utils.pyをそのまま利用
from common.checkpoint_manager import CheckpointFileGenerator

def main():
    """アプリケーションのエントリーポイント"""
    args = parse_arguments()

    print("--- 1. Loading Configuration & Initializing Environment ---")
    try:
        config_path = args.config
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

    start_step = 0
    start_episode = 0
    # ベース名とエピソード番号が両方指定されている場合のみロードを試みる
    if args.resume_from:
        base_name_to_load = args.resume_from
        step_to_load = args.resume_step

        checkpoint_file_generator = CheckpointFileGenerator()
        checkpoint_file_path = None
        if step_to_load is None:
            checkpoint_file_path = checkpoint_file_generator.generate_best_filename()
            print(f"\n--- Resuming Training from fixed file: {base_name_to_load} ---")
        else:
            checkpoint_file_path = checkpoint_file_generator.generate_filename(base_name_to_load, step_to_load)
            print(f"\n--- Resuming Training from: {base_name_to_load} (Step {step_to_load}) ---")

        start_step, start_episode = trainer.load_checkpoint(
            checkpoint_file_path
        )
        
        if start_step > 0:
            print(f"✓ Resuming from Step {start_step}. Starting next episode {start_episode + 1}.")
        else:
            print("✗ Failed to load checkpoint. Starting from scratch.")

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

def parse_arguments():
    """コマンドライン引数を解析する"""
    parser = argparse.ArgumentParser(description="PLLA Custom RL Agent Trainer")
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="Path to the configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Base name of the checkpoint (e.g., 'checkpoint' or 'best_model.pth')"
    )
    parser.add_argument(
        "--resume_step",
        type=int,
        default=None,
        help="Step number of the checkpoint to resume from (e.g., 500)"
    )
    # --- ▲ここまで変更▲ ---
    return parser.parse_args()

if __name__ == "__main__":
    main()