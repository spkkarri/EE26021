
import numpy as np
for track in ['solar', 'wind']:
    oof  = np.load(f'data/processed/oof/lightgbm/{track}/lgbm_oof.npy')
    mask = np.load(f'data/processed/oof/lightgbm/{track}/lgbm_mask.npy')
    y    = np.load(f'data/processed/oof/lightgbm/{track}/y_true.npy')
    y_t  = y[mask].flatten()
    y_p  = oof[mask].flatten()
    ss_res = np.sum((y_t - y_p)**2)
    ss_tot = np.sum((y_t - y_t.mean())**2)
    r2 = 1 - ss_res/ss_tot
    print(f'{track}: R²={r2:.4f} ({r2*100:.1f}%)')
