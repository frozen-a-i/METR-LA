import os
import numpy as np
import torch
import h5py
from collections import namedtuple

Snapshot = namedtuple('Snapshot', ['x', 'y'])

_H5_PATH = os.path.join(os.path.dirname(__file__), '_cache', 'DCRNN', 'metr-la.h5')


def load_metr_la(timesteps_in=12, timesteps_out=3):
    print("[1/5] METR-LA yuklanmoqda...")

    with h5py.File(_H5_PATH, 'r') as f:
        data = f['df/block0_values'][:].astype(np.float32)  # [T, N]

    # Forward-fill then backward-fill any zeros/NaNs
    mask = data == 0
    data[mask] = np.nan
    for i in range(data.shape[1]):
        col = data[:, i]
        nans = np.isnan(col)
        if nans.any():
            idx = np.where(~nans)[0]
            data[:, i] = np.interp(np.arange(len(col)), idx, col[idx])

    T, N = data.shape
    snaps = []
    for t in range(T - timesteps_in - timesteps_out + 1):
        x_win = data[t : t + timesteps_in, :].T        # [N, T_in]
        y_win = data[t + timesteps_in : t + timesteps_in + timesteps_out, :].T  # [N, T_out]
        snaps.append(Snapshot(
            x=torch.FloatTensor(x_win).unsqueeze(-1),  # [N, T_in, 1]
            y=torch.FloatTensor(y_win),                # [N, T_out]
        ))

    print(f"  Vaqt qadamlari : {len(snaps)}")
    print(f"  Tugunlar       : {N}")
    print(f"  x shakli       : {snaps[0].x.shape}  →  [N, T_in, F]")
    print(f"  y shakli       : {snaps[0].y.shape}  →  [N, T_out]")
    return snaps
