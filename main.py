import argparse
import os
from agent.ppo_agent import PPOAgent
from env.unity_env_wrapper import UnityEnvWrapper
from config import Config
from trainer.ppo_trainer import PPOTrainer
from common.seed_utils import set_seed

def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='PPO Training for Unity ML-Agents')
    
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='学習を再開するためのチェックポイントファイルパス')
    parser.add_argument('--load_actor', type=str, default=None,
                       help='推論/転移学習のためにActorの重みのみを読み込む')
    parser.add_argument('--env_path', type=str, default="../PLLA_Sample/Build/PLLA_Sample", help='Unity環境のパス')
    parser.add_argument('--max_steps', type=int, default=None, help='最大学習ステップ数')
    parser.add_argument('--save_interval', type=int, default=None, help='モデル保存間隔')
    
    # アクション空間に関する引数を追加
    parser.add_argument('--use_continuous', action='store_true', help='連続値アクションを使用する')
    parser.add_argument('--use_discrete', action='store_true', help='離散値アクションを使用する')
    parser.add_argument('--agent_name', type=str, default="PPO_Agent", help='エージェントの名前')

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # --- 単一の設定を定義 ---
    config = Config()
    config.agent_name = args.agent_name

    # コマンドライン引数から設定を上書き
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.save_interval:
        config.save_interval = args.save_interval

    # アクション空間の設定
    # 引数が指定されていない場合は、configのデフォルト値を使用
    if args.use_continuous or args.use_discrete:
        config.use_continuous = args.use_continuous
        config.use_discrete = args.use_discrete


    set_seed(config.seed)

    env_wrapper = UnityEnvWrapper(args.env_path, config.time_scale, config.seed)

    # 環境からアクション空間のサイズを取得して設定
    config.continuous_action_size = env_wrapper.get_continuous_action_size()
    config.discrete_action_size = env_wrapper.get_discrete_action_size()

    # --- 単一のエージェントを初期化 ---
    agent = PPOAgent(
        env_wrapper.get_state_size(),
        config.continuous_action_size,
        config.discrete_action_size,
        config
    )
    print(f"Initialized agent: '{config.agent_name}' with action space: Continuous({config.use_continuous}), Discrete({config.use_discrete})")
    # ----------------------------------------

    # モデルの読み込み処理
    if args.resume_checkpoint:
        if os.path.exists(args.resume_checkpoint):
            agent.load_checkpoint(args.resume_checkpoint)
        else:
            print(f"✗ Checkpoint not found at: {args.resume_checkpoint}")
    elif args.load_actor:
        if os.path.exists(args.load_actor):
            agent.load_actor_model(args.load_actor)
        else:
            print(f"✗ Actor model not found at: {args.load_actor}")

    # PPOTrainerに単一のエージェントと設定を渡す
    trainer = PPOTrainer(agent, config)
    
    try:
        print("Starting training...")
        trainer.train(env_wrapper)
        print("Training completed!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        env_wrapper.close()

if __name__ == "__main__":
    main()
