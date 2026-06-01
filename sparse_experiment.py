"""
Sparse Coverage Degradation Experiment — Table 3 filler
Loads trained PerSensorLSTM, evaluates on METR-LA test set
under three missing-data strategies x four missing rates.
"""
import numpy as np
import torch
import torch.nn as nn

from data.load_data    import load_metr_la
from data.preprocess   import build_tensors, simulate_missing_data
from models.lstm_model import PerSensorLSTM
from utils.metrics     import mae, rmse, mape
from torch.utils.data  import DataLoader, TensorDataset

DEVICE    = torch.device('cpu')
PT_PATH   = 'results/best_lstm_persensor.pt'
RATES     = [0.10, 0.30, 0.50, 0.70]
STRATEGIES = ['random', 'sensor_dropout', 'block']
HORIZON_LABELS = ['5 min', '10 min', '15 min']


def evaluate_sparse(model, X_test, y_test, x_min, x_max,
                    strategy, rate, batch_size=32):
    """Apply missing-data mask, run model, return per-horizon metrics."""
    # Apply sparsity on normalised test set
    X_sparse, _ = simulate_missing_data(X_test, missing_rate=rate,
                                         strategy=strategy)
    Xt = torch.FloatTensor(X_sparse)
    yt = torch.FloatTensor(y_test)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                    shuffle=False, drop_last=False)

    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
            trues.append(yb.numpy())

    P = np.concatenate(preds, axis=0)  # [T, N, 3]
    T = np.concatenate(trues, axis=0)

    # Denormalise
    P_r = P * (x_max - x_min) + x_min
    T_r = T * (x_max - x_min) + x_min

    results = {}
    for i, h in enumerate(HORIZON_LABELS):
        p_flat = P_r[:, :, i].flatten()
        t_flat = T_r[:, :, i].flatten()
        results[h] = {
            'MAE':  mae(t_flat, p_flat),
            'RMSE': rmse(t_flat, p_flat),
            'MAPE': mape(t_flat, p_flat),
        }
    return results


def main():
    print("=" * 62)
    print("  SPARSE COVERAGE EXPERIMENT — METR-LA PerSensorLSTM")
    print("=" * 62)

    # ── Load data ──────────────────────────────────────────────
    snaps = load_metr_la()
    splits, x_min, x_max = build_tensors(snaps)
    X_test = splits['test'][0]   # already normalised numpy
    y_test = splits['test'][1]
    print(f"  Test set : {X_test.shape[0]} windows, {X_test.shape[1]} sensors")

    # ── Load model ─────────────────────────────────────────────
    ckpt   = torch.load(PT_PATH, map_location=DEVICE, weights_only=False)
    model  = PerSensorLSTM().to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    print(f"  Model    : {ckpt['n_params']:,} params loaded from {PT_PATH}")

    # ── Baseline (no missing data) ─────────────────────────────
    baseline = evaluate_sparse(model, X_test, y_test,
                                x_min, x_max, 'random', 0.0)
    print(f"\n  Baseline (r=0%):  "
          f"MAE={baseline['15 min']['MAE']:.4f}  "
          f"RMSE={baseline['15 min']['RMSE']:.4f}  "
          f"MAPE={baseline['15 min']['MAPE']:.2f}%  [15-min]")

    # ── Sparse experiments ─────────────────────────────────────
    table = {}   # table[strategy][rate] = {horizon: metrics}

    total = len(STRATEGIES) * len(RATES)
    done  = 0
    for strategy in STRATEGIES:
        table[strategy] = {}
        for rate in RATES:
            res = evaluate_sparse(model, X_test, y_test,
                                   x_min, x_max, strategy, rate)
            table[strategy][rate] = res
            done += 1
            m15 = res['15 min']
            print(f"  [{done:2d}/{total}] {strategy:<15} r={int(rate*100):2d}%  "
                  f"MAE={m15['MAE']:.4f}  RMSE={m15['RMSE']:.4f}  "
                  f"MAPE={m15['MAPE']:.2f}%")

    # ── Pretty table (15-min horizon, all strategies) ──────────
    print("\n" + "=" * 70)
    print("  TABLE 3  —  15-min MAE / RMSE / MAPE  (per-sensor evaluation)")
    print("=" * 70)
    print(f"  {'Strategy':<18} {'Rate':>5}  {'MAE':>7}  {'RMSE':>7}  {'MAPE':>7}")
    print("-" * 70)
    strat_labels = {
        'random':         'Random',
        'sensor_dropout': 'Sensor dropout',
        'block':          'Block removal',
    }
    for strategy in STRATEGIES:
        for rate in RATES:
            m = table[strategy][rate]['15 min']
            label = strat_labels[strategy] if rate == RATES[0] else ''
            print(f"  {label:<18} {int(rate*100):>4}%  "
                  f"{m['MAE']:>7.4f}  {m['RMSE']:>7.4f}  {m['MAPE']:>6.2f}%")
        print("-" * 70)

    # ── Save raw results ───────────────────────────────────────
    torch.save({'table': table, 'baseline': baseline,
                'rates': RATES, 'strategies': STRATEGIES},
               'results/sparse_experiment.pt')
    print("\n  Results saved → results/sparse_experiment.pt")
    print("  [DONE]")

    return table, baseline


if __name__ == "__main__":
    main()
