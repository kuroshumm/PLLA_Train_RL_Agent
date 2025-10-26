# checkpoint_manager.py
import torch
import os
from pathlib import Path
from typing import Dict, Any, Optional

class CheckpointData:
    """
    学習のチェックポイント状態を保持する自作データクラス。
    **kwargs を使って任意のキーと値のペアを辞書として初期化する。
    """
    
    def __init__(self, **kwargs):
        """
        コンストラクタで渡された任意のキーワード引数を
        'data' 属性の辞書に格納する。
        
        例: CheckpointData(network_state=..., optimizer_state=...)
        """
        self.data: Dict[str, Any] = kwargs

    def get(self, key: str, default: Any = None) -> Any:
        """辞書からキーを指定して値を取得するヘルパーメソッド"""
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        """辞書にキーと値を追加・更新するヘルパーメソッド"""
        self.data[key] = value

    def __str__(self):
        """インスタンスをprintした際の表示を定義"""
        keys = self.data.keys()
        return f"CheckpointData(keys={list(keys)})"

class CheckpointFileGenerator:
    """
    チェックポイントファイル名を生成する責務を持つクラス。
    """
    def __init__(self, checkpoint_dir: str = "log/checkpoints"):
        """
        Args:
            checkpoint_dir (str): チェックポイントファイルを保存するディレクトリのパス。
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        # ディレクトリが存在しない場合は作成
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"directory initialized at: {self.checkpoint_dir}")

    def generate_filename(self, base_name: str, step: int) -> str:
        """
        チェックポイントファイル名を生成する静的メソッド。

        Args:
            base_name (str): ベースとなるファイル名（例: "checkpoint"）。
            step (int): 学習ステップ数。

        Returns:
            str: 生成されたファイル名（例: "checkpoint_step_1000.pth"）。
        """
        return self.checkpoint_dir / f"{base_name}_step_{step}.pth"
    
    def generate_best_filename(self, best_name: str = "best_model.pth") -> str:
        """
        最高モデル用の固定ファイル名を生成する静的メソッド。

        Args:
            best_name (str): 最高モデル用のベースファイル名（デフォルトは "best_model.pth"）。

        Returns:
            str: 生成された最高モデル用のファイル名。
        """
        return self.checkpoint_dir / best_name

class CheckpointManager:
    """
    自作のCheckpointDataクラスのインスタンスを保存・復元する責務を持つクラス。
    """
    def __init__(self):
        pass

    def save(self, checkpoint_data: CheckpointData, filepath: str):
        """
        現在の学習状態（CheckpointDataインスタンス）をファイルに保存する。
        ファイル名にはステップ数が含まれる。

        Args:
            checkpoint_data (CheckpointData): 保存する状態を含む自作クラスのインスタンス。
            filepath (str): 保存先のファイルパス。
        """
        try:
            # CheckpointDataインスタンスを丸ごと保存
            # torch.saveはpickleを使用するため、自作クラスインスタンスも保存可能
            torch.save(checkpoint_data, filepath)
            print(f"✓ Checkpoint successfully saved to {filepath}")
        except Exception as e:
            print(f"✗ Failed to save checkpoint to {filepath}. Error: {e}")

    def save_best(self, checkpoint_data: CheckpointData, filepath: str):
        """
        最高のパフォーマンスモデルを別枠で保存する。

        Args:
            checkpoint_data (CheckpointData): 保存する状態を含むインスタンス。
            filepath (str): 保存先のファイルパス。
        """
        try:
            torch.save(checkpoint_data, filepath)
            print(f"✓ New BEST model saved successfully to {filepath}")
        except Exception as e:
            print(f"✗ Failed to save best model to {filepath}. Error: {e}")

    def load(self, filepath: str) -> Optional[CheckpointData]:
        """
        ファイルから学習状態（CheckpointDataインスタンス）を読み込む。

        Args:
            filepath (str): "checkpoint" などのベース名。

        Returns:
            Optional[CheckpointData]: 復元されたCheckpointDataインスタンス。失敗時はNone。
        """
        if not filepath.is_file():
            print(f"✗ Checkpoint file not found at: {filepath}")
            return None
        
        try:
            print(f"Loading checkpoint from {filepath}...")
            # インスタンスを丸ごとロード
            state = torch.load(filepath, weights_only=False)
            
            # ロードしたものが本当にCheckpointDataか確認
            if isinstance(state, CheckpointData):
                print(f"✓ Checkpoint loaded successfully: {state}")
                return state
            else:
                print(f"✗ Loaded file is not an instance of CheckpointData. Type: {type(state)}")
                return None
        except Exception as e:
            print(f"✗ Failed to load checkpoint from {filepath}. Error: {e}")
            return None