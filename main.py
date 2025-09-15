import argparse
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_to_gym_wrapper import UnityToGymWrapper
from config import Config

def parse_arguments():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description='PPO Training for Unity ML-Agents with stable-baselines3')
    
    parser.add_argument('--load_model', type=str, default=None,
                       help='学習を再開するためのモデルファイルパス (.zip)')
    parser.add_argument('--env_path', type=str, default="../PLLA_Sample/Build/PLLA_Sample",
                        help='Unity環境のパス')
    parser.add_argument('--max_steps', type=int, default=30000,
                        help='最大学習ステップ数')
    parser.add_argument('--save_interval', type=int, default=10000,
                        help='モデル保存間隔 (ステップ数)')
    parser.add_argument('--save_name', type=str, default="ppo_unity_model",
                        help='保存するモデルのベース名')
    parser.add_argument('--time_scale', type=int, default=8,
                        help='Unity環境のタイムスケール')

    return parser.parse_args()

def export_to_onnx(model, onnx_path, env):
    """
    stable-baselines3モデルのActorをONNX形式でエクスポートする
    """
    try:
        # PPOモデルからポリシー（Actor-Criticネットワーク）を取得
        policy = model.policy
        # 評価モードに設定
        policy.set_training_mode(False)

        # ダミーの入力を作成
        dummy_input = torch.randn(1, *env.observation_space.shape)

        # ONNXエクスポート
        torch.onnx.export(
            policy.actor,
            dummy_input,
            onnx_path,
            opset_version=11,
            input_names=['observation'],
            output_names=['action'],
            dynamic_axes={'observation': {0: 'batch_size'}, 'action': {0: 'batch_size'}}
        )
        print(f"✅ Model successfully exported to {onnx_path}")
    except Exception as e:
        print(f"✗ Error exporting to ONNX: {e}")


def main():
    args = parse_arguments()

    # --- ログとモデルの保存ディレクトリを作成 ---
    log_dir = "log/"
    os.makedirs(log_dir, exist_ok=True)

    print("--- Initializing Unity Environment ---")
    # --- Unity環境の初期化 ---
    # `file_name`にUnity実行ファイルのパスを指定
    # `seed`で環境の乱数シードを固定
    # `side_channels`でUnityとPython間の追加の通信チャネルを設定可能
    unity_env = UnityEnvironment(
        file_name=args.env_path,
        seed=1,
        side_channels=[],
        no_graphics=False # Trueにするとヘッドレスモード
    )

    # Unity環境をGymnasium互換のインターフェースに変換
    env = UnityToGymWrapper(unity_env, uint8_visual=False)
    print("✅ Unity Environment Initialized.")

    # --- PPOモデルの設定 ---
    # `stable-baselines3`のPPOモデルを初期化
    # "MlpPolicy"は、観測とアクションがベクトル形式の場合に使用する標準的なポリシー
    # `env`で学習対象の環境を指定
    # `verbose=1`で学習中の情報を表示
    # その他のハイパーパラメータは`config.py`から読み込むか、ここで直接指定
    config = Config()

    # コールバックの設定: 一定ステップごとにモデルを保存
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_interval,
        save_path=log_dir,
        name_prefix=args.save_name
    )

    if args.load_model and os.path.exists(args.load_model):
        print(f"--- Loading pre-trained model from {args.load_model} ---")
        model = PPO.load(args.load_model, env=env)
        print("✅ Model loaded.")
    else:
        print("--- Initializing new PPO model ---")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=config.lr,
            n_steps=config.buffer_size,
            batch_size=config.batch_size,
            n_epochs=config.ppo_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_epsilon,
            ent_coef=config.entropy_coef,
            vf_coef=config.value_coef,
            max_grad_norm=config.max_grad_norm,
            tensorboard_log=log_dir
        )
        print("✅ New model initialized.")

    try:
        print("\n--- Starting training ---")
        # モデルの学習を開始
        # `total_timesteps`で総学習ステップ数を指定
        # `callback`で学習中に行う処理（モデルの保存など）を指定
        model.learn(
            total_timesteps=args.max_steps,
            callback=checkpoint_callback
        )
        print("✅ Training completed!")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        
    except Exception as e:
        print(f"✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # --- モデルの保存とエクスポート ---
        final_model_path = os.path.join(log_dir, f"{args.save_name}_final.zip")
        print(f"--- Saving final model to {final_model_path} ---")
        model.save(final_model_path)

        onnx_path = os.path.join(log_dir, f"{args.save_name}_final.onnx")
        print(f"--- Exporting final model to {onnx_path} ---")
        export_to_onnx(model, onnx_path, env)

        # 環境を閉じる
        env.close()
        print("--- Environment closed. ---")

if __name__ == "__main__":
    main()
