import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple

class UnityEnvWrapper:
    """
    Unity ML-Agentsののラッパークラス。
    """
    def __init__(self, env_path: str, time_scale: int = 20, seed: int = 0):
        """
        Unity環境を初期化し、接続を確立します。

        Args:
            env_path (str): Unity環境のビルドファイルへのパス。
            time_scale (int): Unityのシミュレーション速度。
            seed (int): 環境の乱数シード。
        """
        print("Initializing Unity Environment...")
        self.channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(
            file_name=env_path,
            seed=seed,
            side_channels=[self.channel]
        )
        self.channel.set_configuration_parameters(time_scale=time_scale)
        print("✓ Unity Environment initialized.")

        # --- APIの主要な初期化処理 ---
        # 環境を一度リセットして、振る舞い(Behavior)の名前と仕様を取得します。
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs.keys())[0]
        self.behavior_spec = self.env.behavior_specs[self.behavior_name]
        
        print(f"Behavior Name: {self.behavior_name}")

    def get_state_size(self) -> int:
        """状態空間のサイズを取得します。"""
        # 最初のObservation Specの形状を取得
        return self.behavior_spec.observation_specs[0].shape[0]

    def get_action_spec(self):
        """行動の仕様(ActionSpec)を取得します。"""
        return self.behavior_spec.action_spec

    def reset(self):
        """
        【API核心部分①】環境をリセットし、最初のエージェントの状態を返します。
        """
        self.env.reset()
        # エージェントからの意思決定ステップ情報を取得
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        
        # 最初の状態(観測)を返す (エージェントが1体と仮定)
        state = decision_steps.obs[0][0]
        return state

    def step(self, action: ActionTuple):
        """
        【API核心部分②】行動を環境に送信し、次の状態・報酬・完了フラグを返します。
        """
        # 行動を環境に設定
        self.env.set_actions(self.behavior_name, action)

        # 環境の時間を1ステップ進める
        self.env.step()

        # ステップ後の情報を取得
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

        # エピソードが終了したかどうかの判定 (terminal_stepsに情報があるか)
        done = len(terminal_steps) > 0

        if done:
            # エピソードが終了した場合 (エージェントが1体と仮定)
            next_state = terminal_steps.obs[0][0]
            reward = terminal_steps.reward[0]
        else:
            # エピソードが継続中の場合 (エージェントが1体と仮定)
            next_state = decision_steps.obs[0][0]
            reward = decision_steps.reward[0]
            
        return next_state, reward, done

    def close(self):
        """環境をクローズします。"""
        self.env.close()
        print("Unity Environment closed.")


class RandomAgent:
    """
    強化学習アルゴリズムのロジックを省略したダミーエージェント。
    APIの動作確認のため、ランダムな行動を選択します。
    """
    def __init__(self, action_spec):
        self.action_spec = action_spec

    def select_action(self) -> ActionTuple:
        """
        現在の状態に関わらず、ランダムなアクションを生成します。
        
        Returns:
            ActionTuple: 生成されたランダムな行動
        """
        # ActionSpecを使ってランダムな行動を生成
        return self.action_spec.random_action(n_agents=1)

# --- メインの学習ループ ---
if __name__ == "__main__":
    
    # --- 設定 ---
    # ご自身のUnity環境ビルドへのパスを指定してください
    UNITY_ENV_PATH = "../../PLLA_Sample/Build/PLLA_Sample" # 例: "C:/Users/YourName/Build/MyGame.exe"
    MAX_EPISODES = 10

    if UNITY_ENV_PATH is None:
        print("Please set the `UNITY_ENV_PATH` variable to your Unity environment build file.")
    else:
        env = None
        try:
            # 1. 環境ラッパーの初期化
            env = UnityEnvWrapper(env_path=UNITY_ENV_PATH)

            # 2. 環境から各種仕様を取得
            state_size = env.get_state_size()
            action_spec = env.get_action_spec()
            
            print("\n--- Environment Info ---")
            print(f"State size: {state_size}")
            print(f"Action Spec: {action_spec}")
            print("------------------------\n")

            # 3. ダミーエージェントの初期化
            agent = RandomAgent(action_spec)

            # 4. 学習ループの開始
            for episode in range(MAX_EPISODES):
                # 環境をリセットし、初期状態を取得
                current_state = env.reset()
                episode_reward = 0
                
                while True:
                    # エージェントが行動を選択
                    action = agent.select_action()
                    
                    # 選択した行動を環境に渡して1ステップ進める
                    next_state, reward, done = env.step(action)
                    
                    # 報酬と状態を更新
                    episode_reward += reward
                    current_state = next_state
                    
                    # エピソードが終了したらループを抜ける
                    if done:
                        break
                
                print(f"Episode {episode + 1}: Total Reward = {episode_reward:.2f}")

        except Exception as e:
            print(f"An error occurred: {e}")
        
        finally:
            # 5. 環境を閉じる
            if env:
                env.close()
