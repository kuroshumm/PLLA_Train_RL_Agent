import random
import numpy as np
import torch

def set_seed(seed):
    """全ライブラリのシード値を固定"""
    print(f"Setting seed to {seed}")
    
    # Python標準ライブラリ
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorchの決定論的動作を有効化（性能は若干低下）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print("✓ All random seeds have been fixed")

def get_random_state():
    """現在のランダム状態を取得"""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state()
    }

def set_random_state(state):
    """ランダム状態を復元"""
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])