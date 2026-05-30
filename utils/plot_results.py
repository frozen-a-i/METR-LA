import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def save_plots(tr_losses, va_losses, T_r, P_r, res, horizons, save_path='results/metr_la_results.png'):
    fig = plt.figure(figsize=(13, 9), facecolor='#F8FAFC')
    fig.suptitle("METR-LA  ·  LSTM Baseline  ·  Natijalar",
                 fontsize=13, fontweight='bold', color='#0D2137', y=0.98)
    gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.30,
                           left=0.07, right=0.96, top=0.93, bottom=0.07)

    # A — Training curve
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(tr_losses, color='#1565C0', lw=2.0, label="Train")
    ax.plot(va_losses, color='#C62828', lw=2.0, label="Val")
    ax.set_title("A.  O'qitish jarayoni", fontweight='bold', color='#0D2137')
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    # B — Bashorat vs Haqiqat
    ax2 = fig.add_subplot(gs[0, 1])
    n_show = min(200, len(T_r))
    t_ax   = np.arange(n_show) * 5
    ax2.plot(t_ax, T_r[:n_show, 0], color='#1565C0', lw=1.5, label="Haqiqiy")
    ax2.plot(t_ax, P_r[:n_show, 0], color='#C62828', lw=1.8,
             ls='--', label="Bashorat")
    ax2.set_title("B.  Bashorat vs Haqiqat (5 daq.)", fontweight='bold', color='#0D2137')
    ax2.set_xlabel("Vaqt (daqiqa)"); ax2.set_ylabel("Tezlik (km/soat)")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)

    # C — MAE / RMSE
    ax3 = fig.add_subplot(gs[1, 0])
    hh     = ['5 daq', '10 daq', '15 daq']
    mae_v  = [res[h]['MAE']  for h in horizons]
    rmse_v = [res[h]['RMSE'] for h in horizons]
    xp = np.arange(3)
    b1 = ax3.bar(xp - 0.2, mae_v,  0.38, label='MAE',  color='#1565C0', alpha=0.85)
    b2 = ax3.bar(xp + 0.2, rmse_v, 0.38, label='RMSE', color='#2E7D32', alpha=0.85)
    for b in list(b1) + list(b2):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.04,
                 f'{b.get_height():.3f}', ha='center', fontsize=8.5, fontweight='bold')
    ax3.set_title("C.  MAE / RMSE — ufqlar bo'yicha", fontweight='bold', color='#0D2137')
    ax3.set_xticks(xp); ax3.set_xticklabels(hh)
    ax3.set_ylabel("Xato (km/soat)"); ax3.legend(fontsize=9)
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.spines[['top', 'right']].set_visible(False)

    # D — Xato taqsimoti
    ax4 = fig.add_subplot(gs[1, 1])
    errors = (P_r[:, 0] - T_r[:, 0]).flatten()
    ax4.hist(errors, bins=60, color='#1565C0', alpha=0.75, edgecolor='white')
    ax4.axvline(0,             color='#C62828', lw=2.0, ls='--', label='Nol')
    ax4.axvline(errors.mean(), color='#F57F17', lw=1.8, ls='-.',
                label=f"O'rtacha: {errors.mean():.3f}")
    ax4.set_title("D.  Xato taqsimoti (5 daq.)", fontweight='bold', color='#0D2137')
    ax4.set_xlabel("Xato (km/soat)"); ax4.set_ylabel("Chastota")
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)
    ax4.spines[['top', 'right']].set_visible(False)

    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#F8FAFC')
    plt.close()
    print(f"\n  Grafik saqlandi: {save_path}")
