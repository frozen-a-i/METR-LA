import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def build_tensors(snaps):
    print("\n[2/5] Ma'lumot tayyorlanmoqda...")
    X_list, y_list = [], []
    for snap in snaps:
        X_list.append(snap.x[:, :, 0].numpy())   # [N, T_in]
        y_list.append(snap.y.numpy())             # [N, T_out]

    X = np.stack(X_list, axis=0)   # [T, N, T_in]
    y = np.stack(y_list, axis=0)   # [T, N, T_out]

    x_min, x_max = X.min(), X.max()
    X_n = (X - x_min) / (x_max - x_min + 1e-8)
    y_n = (y - x_min) / (x_max - x_min + 1e-8)

    n       = len(X_n)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.10)

    splits = {
        'train': (X_n[:n_train],              y_n[:n_train]),
        'val':   (X_n[n_train:n_train+n_val], y_n[n_train:n_train+n_val]),
        'test':  (X_n[n_train+n_val:],        y_n[n_train+n_val:]),
    }
    print(f"  Train  : {splits['train'][0].shape[0]}")
    print(f"  Val    : {splits['val'][0].shape[0]}")
    print(f"  Test   : {splits['test'][0].shape[0]}")
    return splits, x_min, x_max


def make_dataloaders(splits, batch_size=64):
    """Per-sensor: tensors keep the [T, N, T_in/T_out] shape intact."""
    def to_tensor(Xn, yn):
        return (torch.FloatTensor(Xn),   # [T, N, T_in]
                torch.FloatTensor(yn))   # [T, N, T_out]

    Xt_tr, yt_tr = to_tensor(*splits['train'])
    Xt_va, yt_va = to_tensor(*splits['val'])
    Xt_te, yt_te = to_tensor(*splits['test'])

    train_dl = DataLoader(TensorDataset(Xt_tr, yt_tr), batch_size=batch_size,
                          shuffle=True,  drop_last=True)
    val_dl   = DataLoader(TensorDataset(Xt_va, yt_va), batch_size=batch_size,
                          shuffle=False, drop_last=False)
    test_dl  = DataLoader(TensorDataset(Xt_te, yt_te), batch_size=batch_size,
                          shuffle=False, drop_last=False)
    return train_dl, val_dl, test_dl


def simulate_missing_data(X, missing_rate=0.2, strategy='random'):
    """
    Simulate missing data in traffic sensor readings.

    Args:
        X: numpy array [T, N, T_in]
        missing_rate: fraction of data to drop (0.0 – 0.9)
        strategy:
          'random'         – randomly zero out a fraction of all values
          'sensor_dropout' – randomly drop entire sensors with prob=missing_rate
          'block'          – drop contiguous time blocks of length 3–6 per sensor

    Returns:
        X_sparse: same shape as X, missing entries set to 0.0
        mask:     boolean array [T, N, T_in], True = observed
    """
    X_sparse = X.copy()
    mask = np.ones_like(X, dtype=bool)

    if strategy == 'random':
        drop = np.random.rand(*X.shape) < missing_rate
        X_sparse[drop] = 0.0
        mask[drop] = False

    elif strategy == 'sensor_dropout':
        T, N, Tin = X.shape
        for n in range(N):
            if np.random.rand() < missing_rate:
                X_sparse[:, n, :] = 0.0
                mask[:, n, :] = False

    elif strategy == 'block':
        T, N, Tin = X.shape
        avg_block = 4.5
        n_blocks = max(1, int(T * N * missing_rate / avg_block))
        for _ in range(n_blocks):
            t = np.random.randint(0, T)
            n = np.random.randint(0, N)
            length = np.random.randint(3, 7)
            t_end = min(t + length, T)
            X_sparse[t:t_end, n, :] = 0.0
            mask[t:t_end, n, :] = False

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    return X_sparse, mask
