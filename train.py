import time
import numpy as np
import torch
import torch.nn as nn

from data.load_data    import load_metr_la
from data.preprocess   import build_tensors, make_dataloaders
from models.lstm_model import PerSensorLSTM
from utils.metrics     import mae, rmse, mape
from utils.plot_results import save_plots

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 50
LR     = 1e-3
BS     = 32   # smaller batch — each sample is now [N, T_in] = [207, 12]


def train_epoch(model, dl, opt, crit):
    model.train()
    total = 0.0
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * xb.size(0)
    return total / len(dl.dataset)


@torch.no_grad()
def eval_epoch(model, dl, crit):
    model.eval()
    total = 0.0
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        total += crit(model(xb), yb).item() * xb.size(0)
    return total / len(dl.dataset)


def main():
    print(f"PyTorch  : {torch.__version__}")
    print(f"Device   : {DEVICE}")

    snaps                     = load_metr_la()
    splits, x_min, x_max     = build_tensors(snaps)
    train_dl, val_dl, test_dl = make_dataloaders(splits, batch_size=BS)

    print("\n[3/5] Model qurilmoqda (PerSensorLSTM)...")
    model    = PerSensorLSTM().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametrlar : {n_params:,}")

    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    crit  = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode='min', factor=0.5, patience=5)

    best_val, best_wts = float('inf'), None
    patience_cnt = 0
    tr_losses, va_losses = [], []

    print("\n[4/5] O'qitish boshlanmoqda...")
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        tl = train_epoch(model, train_dl, opt, crit)
        vl = eval_epoch(model, val_dl, crit)
        sched.step(vl)
        tr_losses.append(tl)
        va_losses.append(vl)

        if vl < best_val:
            best_val     = vl
            best_wts     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep:3d}/{EPOCHS}  train={tl:.5f}  val={vl:.5f}  "
                  f"best={best_val:.5f}  [{time.time()-t0:.0f}s]")

        if patience_cnt >= 10:
            print(f"  Early stopping: epoch {ep}")
            break

    model.load_state_dict(best_wts)
    print(f"\n  O'qitish tugadi. Eng yaxshi val loss: {best_val:.6f}")

    # ── Test baholash ──────────────────────────────────────────────────────
    print("\n[5/5] Test natijalari (per-sensor)...")
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())  # [B, N, 3]
            trues.append(yb.numpy())                          # [B, N, 3]

    P = np.concatenate(preds, axis=0)   # [T_test, N, 3]
    T = np.concatenate(trues, axis=0)   # [T_test, N, 3]

    # Denormalize back to mph
    P_r = P * (x_max - x_min) + x_min
    T_r = T * (x_max - x_min) + x_min

    horizons = ['5 daqiqa', '10 daqiqa', '15 daqiqa']
    res = {}
    print("\n" + "═"*58)
    print(f"  {'Ufq':<14} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
    print("─"*58)
    for i, h in enumerate(horizons):
        # Flatten over all time steps AND all 207 sensors → fair per-sensor metric
        p_flat = P_r[:, :, i].flatten()
        t_flat = T_r[:, :, i].flatten()
        m_  = mae(t_flat, p_flat)
        r_  = rmse(t_flat, p_flat)
        mp_ = mape(t_flat, p_flat)
        res[h] = {'MAE': m_, 'RMSE': r_, 'MAPE': mp_}
        print(f"  {h:<14} {m_:>7.4f}  {r_:>7.4f}  {mp_:>6.2f}%")
    print("─"*58)
    p_all = P_r.flatten()
    t_all = T_r.flatten()
    ortacha = "O'rtacha"
    res[ortacha] = {'MAE': mae(t_all, p_all),
                    'RMSE': rmse(t_all, p_all),
                    'MAPE': mape(t_all, p_all)}
    avg = res[ortacha]
    print(f"  {ortacha:<14} {avg['MAE']:>7.4f}  {avg['RMSE']:>7.4f}  {avg['MAPE']:>6.2f}%")
    print("═"*58)

    # Pass sensor-averaged series to plot (for visualization only)
    T_r_plot = T_r.mean(axis=1)   # [T_test, 3]
    P_r_plot = P_r.mean(axis=1)   # [T_test, 3]
    save_plots(tr_losses, va_losses, T_r_plot, P_r_plot, res, horizons,
               save_path='results/metr_la_persensor.png')

    torch.save({
        'state_dict': model.state_dict(),
        'results'   : res,
        'x_min'     : float(x_min),
        'x_max'     : float(x_max),
        'n_params'  : n_params,
    }, 'results/best_lstm_persensor.pt')
    print("  Model saqlandi: results/best_lstm_persensor.pt")

    # ── Maqola uchun jadval ────────────────────────────────────────────────
    print("\n" + "═"*58)
    print("  MAQOLA UCHUN JADVAL  (per-sensor, adolatli taqqoslash)")
    print("═"*58)
    print(f"  Dataset  : METR-LA, 207 sensor, Mar–Jun 2012")
    print(f"  Model    : PerSensorLSTM (hidden=64, 2-layer, dropout=0.2)")
    print(f"  Kirish   : 12 × 5 min = 60 daqiqa tarix, har sensor alohida")
    print(f"  Chiqish  : 5 / 10 / 15 daqiqalik bashorat, har sensor uchun")
    print(f"  Parametr : {n_params:,}")
    print("─"*58)
    print(f"  {'Ufq':<12} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
    print("─"*58)
    for h in horizons:
        print(f"  {h:<12} {res[h]['MAE']:>7.4f}  {res[h]['RMSE']:>7.4f}  {res[h]['MAPE']:>6.2f}%")
    print("─"*58)
    print(f"  {ortacha:<12} {avg['MAE']:>7.4f}  {avg['RMSE']:>7.4f}  {avg['MAPE']:>6.2f}%")
    print("═"*58)

    lstm_15 = res['15 daqiqa']
    print("\n  Taqqoslash (15 daqiqa, PER-SENSOR — adolatli):")
    print("  DCRNN         MAE=2.77, RMSE=5.38  [Li et al. 2018]")
    print("  Graph WaveNet MAE=2.69, RMSE=5.15  [Wu et al. 2019]")
    print("  MSTGACN       MAE=2.35, RMSE=4.71  [Liu et al. 2021]")
    print(f"  Bizning LSTM  MAE={lstm_15['MAE']:.4f}, RMSE={lstm_15['RMSE']:.4f}")
    print("\n  NOT: Endi taqqoslash ADOLATLI — barcha modellar har sensor")
    print("  uchun alohida bashorat qiladi. LSTM sodda bo'lgani uchun")
    print("  graf modellardan past natija kutiladi va bu to'g'ri.")
    print("\n[TUGADI]")


if __name__ == "__main__":
    main()
