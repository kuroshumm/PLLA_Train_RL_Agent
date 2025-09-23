import yaml
from pathlib import Path
from settings.setting import AppConfig

def load_config(config_path: str | Path) -> AppConfig:
    """
    YAMLファイルを読み込み、内容を検証してAppConfigオブジェクトに変換する。

    Args:
        config_path (str | Path): 設定ファイルのパス。

    Returns:
        AppConfig: 検証済みの設定情報を格納したオブジェクト。
    
    Raises:
        FileNotFoundError: 指定されたパスにファイルが存在しない場合。
        yaml.YAMLError: YAMLの解析に失敗した場合。
        pydantic.ValidationError: YAMLの内容がAppConfigのスキーマと一致しない場合。
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        # PyYAMLでYAMLをPythonの辞書に変換
        config_dict = yaml.safe_load(f)

    # Pydanticモデルに辞書を渡して、データの検証とオブジェクトへの変換を行う
    # ここで設定内容が settings.py の定義と異なればエラーが発生する
    return AppConfig(**config_dict)