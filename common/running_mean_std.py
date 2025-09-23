import numpy as np
import pdb

class RunningMeanStd:
    """Welford's online algorithmを利用して移動平均と標準偏差を計算する"""
    def __init__(self, shape=()):
        """
        Args:
            shape (tuple): 正規化するデータの形状 (例: 観測値なら(state_size,))
        """
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = 1e-4  # ゼロ除算を避けるための小さな値

    def update(self, x):
        """
        新しいデータxを受け取り、統計量を更新する
        Args:
            x (np.ndarray): 新しいデータ。形状は (N, *shape) または (*shape) である必要がある
        """

        # 入力xをNumPy配列に変換し、必ず2次元配列（バッチ）として扱う
        batch_x = np.asarray(x)
        if batch_x.ndim == 1:
            # 1次元配列の場合、(1, 6)のような形状の2次元配列に変換する
            batch_x = batch_x.reshape(1, -1)

        batch_mean = np.mean(batch_x, axis=0)
        batch_var = np.var(batch_x, axis=0)
        batch_count = batch_x.shape[0]
        
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x, clip_range=10.0):
        """データxを正規化する"""
        # 標準偏差 (sqrt(var)) がゼロにならないように小さな値 (epsilon) を加える
        std = np.sqrt(self.var) + 1e-8
        normalized_x = (x - self.mean) / std
        return np.clip(normalized_x, -clip_range, clip_range)

    def get_stats(self):
        """現在の統計情報を返す"""
        return {
            'mean': self.mean,
            'var': self.var,
            'std': np.sqrt(self.var),
            'count': self.count
        }
