import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

#  Load Data
print("Loading data")
scores       = np.load("data/train_scores.npy").astype(float)
game_ids     = np.load("data/train_game_ids.npy")
player_elos  = np.load("data/train_player_elos.npy")
eco_codes    = np.load("data/train_eco_codes.npy",  allow_pickle=True)
side_to_move = np.load("data/train_side_to_move.npy")

print(f"  Positions   : {len(scores):,}")
print(f"  Unique games: {len(set(game_ids)):,}")
print(f"  ELO range   : {player_elos.min()} {player_elos.max()}")
real_acpl_available = False
try:
    acpl_df = pd.read_csv("analysis_results.csv")
    real_human_acpl = acpl_df["human_cpl"].mean()
    real_model_acpl = acpl_df["model_cpl"].dropna().mean()
    real_acpl_available = True
    print(f" Real Stockfish ACPL loaded from analysis_results.csv")
    print(f"   Human ACPL : {real_human_acpl:.1f} cp")
    print(f"   Model ACPL : {real_model_acpl:.1f} cp")
    print(f"   Human best : {(acpl_df['human_class']=='Best').mean()*100:.1f}%")
    print(f"   Model best : {(acpl_df['model_class']=='Best').mean()*100:.1f}%")
except FileNotFoundError:
    print("  analysis_results.csv not found — using proxy ACPL")
print("\nGrouping positions by game")

game_data = defaultdict(lambda: {
    "scores": [], "elos": [], "eco": "UNK", "sides": [],
})

for i in range(len(scores)):
    gid = game_ids[i]
    game_data[gid]["scores"].append(scores[i])
    game_data[gid]["elos"].append(player_elos[i])
    game_data[gid]["eco"]   = eco_codes[i]
    game_data[gid]["sides"].append(side_to_move[i])

print(f"  Total games: {len(game_data):,}")


# 3. Feature Engineering 

print("Engineering features")

def extract_features(side_scores, eco_str):
    if len(side_scores) < 4:
        return None

    sc   = np.array(side_scores)
    n    = len(sc)
    swings      = np.abs(np.diff(sc)) * 1000
    acpl        = swings.mean()
    acpl_med    = np.median(swings)
    acpl_std    = swings.std()
    blunder_rate    = (swings > 300).mean() * 100
    mistake_rate    = ((swings >= 100) & (swings < 300)).mean() * 100
    inaccuracy_rate = ((swings >= 50)  & (swings < 100)).mean() * 100
    best_rate       = (swings < 50).mean() * 100
    worst_move      = swings.max()
    top3_avg        = np.sort(swings)[-3:].mean()

    # Phase features
    third     = max(1, n // 3)
    open_acpl = np.abs(np.diff(sc[:third])).mean() * 1000   if third > 1    else acpl
    mid_acpl  = np.abs(np.diff(sc[third:2*third])).mean() * 1000 if third > 1 else acpl
    end_acpl  = np.abs(np.diff(sc[2*third:])).mean() * 1000 if 2*third < n-1 else acpl

    score_trend = (sc[-1] - sc[0]) * 1000
    eco_letter  = ord(eco_str[0]) - ord('A') if eco_str != "UNK" else -1
    eco_num     = int(eco_str[1:3]) if eco_str != "UNK" and len(eco_str) >= 3 else 0

    return [
        acpl, acpl_med, acpl_std,
        blunder_rate, mistake_rate, inaccuracy_rate, best_rate,
        worst_move, top3_avg,
        open_acpl, mid_acpl, end_acpl,
        score_trend, eco_letter, eco_num, n,
    ]

feature_names = [
    "ACPL (mean)", "ACPL (median)", "ACPL (std)",
    "Blunder Rate (%)", "Mistake Rate (%)", "Inaccuracy Rate (%)", "Best Move Rate (%)",
    "Worst Single Move", "Avg 3 Worst Moves",
    "Opening Phase CPL", "Middlegame Phase CPL", "Endgame Phase CPL",
    "Score Trend", "Opening Letter (ECO)", "Opening Number (ECO)", "Moves Played",
]

X_rows, y_rows = [], []

for gid, data in game_data.items():
    sc    = np.array(data["scores"])
    elos  = np.array(data["elos"])
    sides = np.array(data["sides"])
    eco   = data["eco"]

    for side_val in [1, 0]:
        mask = sides == side_val
        if mask.sum() < 4:
            continue
        feats = extract_features(sc[mask].tolist(), eco)
        if feats is None:
            continue
        X_rows.append(feats)
        y_rows.append(int(elos[mask][0]))

X = np.array(X_rows, dtype=np.float32)
y = np.array(y_rows, dtype=np.float32)
print(f"  Samples before balancing: {len(X):,}")

#Stratified Sampling
print("Balancing ELO distribution ...")
brackets    = [(1500,1700),(1700,1900),(1900,2100),(2100,2300),(2300,2500)]
per_bracket = 2000

balanced_X, balanced_y = [], []
for lo, hi in brackets:
    mask     = (y >= lo) & (y < hi)
    idx      = np.where(mask)[0]
    n_sample = min(per_bracket, len(idx))
    if n_sample == 0:
        continue
    chosen = np.random.choice(idx, n_sample, replace=False)
    balanced_X.append(X[chosen])
    balanced_y.append(y[chosen])
    print(f"  ELO {lo}–{hi}: {len(idx):,} available → {n_sample} sampled")

X_bal = np.vstack(balanced_X)
y_bal = np.concatenate(balanced_y)
print(f"  Balanced: {len(X_bal):,} samples")



# 5. Train / Test Split + Scale
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=42
)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
print(f"\n  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print("\nTraining Random Forest")
rf = RandomForestRegressor(
    n_estimators=300, max_depth=15,
    min_samples_leaf=3, max_features="sqrt",
    n_jobs=-1, random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf    = mean_absolute_error(y_test, y_pred_rf)
r2_rf     = r2_score(y_test, y_pred_rf)

print("Training Gradient Boosting ...")
gb = GradientBoostingRegressor(
    n_estimators=200, max_depth=5,
    learning_rate=0.05, random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
mae_gb    = mean_absolute_error(y_test, y_pred_gb)
r2_gb     = r2_score(y_test, y_pred_gb)

# Pick better model
if mae_rf <= mae_gb:
    y_pred, mae, r2  = y_pred_rf, mae_rf, r2_rf
    model_name       = "Random Forest"
    importances      = rf.feature_importances_
else:
    y_pred, mae, r2  = y_pred_gb, mae_gb, r2_gb
    model_name       = "Gradient Boosting"
    importances      = gb.feature_importances_

sorted_idx = np.argsort(importances)[::-1]

print(f"\n{'═'*55}")
print(f"  ELO ESTIMATION RESULTS  (Slide 7)")
print(f"{'═'*55}")
print(f"  Best model  : {model_name}")
print(f"  MAE         : {mae:.1f} ELO points")
print(f"  R² Score    : {r2:.4f}")
print(f"  RF  — MAE: {mae_rf:.1f}  R²: {r2_rf:.4f}")
print(f"  GB  — MAE: {mae_gb:.1f}  R²: {r2_gb:.4f}")
print(f"{'─'*55}")
print(f"  Top features:")
for i in sorted_idx[:5]:
    bar = "█" * int(importances[i] * 60)
    print(f"    {feature_names[i]:<30}: {importances[i]:.4f}  {bar}")
print(f"{'═'*55}")

print(f"\n  ELO Bracket Performance:")
for lo, hi in brackets:
    mask = (y_test >= lo) & (y_test < hi)
    if mask.sum() < 5:
        continue
    b_mae = mean_absolute_error(y_test[mask], y_pred[mask])
    print(f"    ELO {lo}–{hi}: MAE={b_mae:.1f}  (n={mask.sum()})")

# 7. Plots  (Slide 7)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(
    f"ELO Estimation — {model_name} (Slide 7)\n"
    "NIT Andhra Pradesh | EE2621 Introduction to Machine Learning",
    fontsize=13, fontweight='bold'
)

ax      = axes[0]
top8    = sorted_idx[:8]
top8imp = importances[top8]
top8nm  = [feature_names[i] for i in top8]
colors  = ['#185FA5' if i == 0 else '#B5D4F4' for i in range(8)]
ax.barh(top8nm[::-1], top8imp[::-1], color=colors[::-1])
ax.set_title("Feature Importance (Slide 7)", fontweight='bold')
ax.set_xlabel("Importance")
ax.grid(True, alpha=0.3, axis='x')

# Plot 2 — Predicted vs Actual
ax = axes[1]
ax.scatter(y_test, y_pred, alpha=0.3, s=10, color='#185FA5')
ax.plot([y_bal.min(), y_bal.max()],
        [y_bal.min(), y_bal.max()],
        'r--', linewidth=2, label='Perfect')
ax.set_title(f"Predicted vs Actual ELO\nMAE={mae:.1f} | R²={r2:.3f}",
             fontweight='bold')
ax.set_xlabel("Actual ELO")
ax.set_ylabel("Predicted ELO")
ax.legend()
ax.grid(True, alpha=0.3)
ax = axes[2]
X_bal_raw = np.vstack(balanced_X)
acpl_raw  = X_bal_raw[:, 0]
mask_viz  = acpl_raw < 300

ax.scatter(acpl_raw[mask_viz], y_bal[mask_viz],
           alpha=0.15, s=6, color='#1D9E75', label='Training data')
coeffs = np.polyfit(acpl_raw[mask_viz], y_bal[mask_viz], 1)
x_line = np.linspace(acpl_raw[mask_viz].min(),
                     acpl_raw[mask_viz].max(), 100)
ax.plot(x_line, np.polyval(coeffs, x_line),
        color='#D85A30', linewidth=2.5, label='Trend')
if real_acpl_available:
    ax.axvline(x=real_human_acpl, color='#185FA5',
               linewidth=2, linestyle='--',
               label=f'Human ACPL={real_human_acpl:.1f}cp')
    ax.axvline(x=real_model_acpl, color='#534AB7',
               linewidth=2, linestyle='--',
               label=f'Model ACPL={real_model_acpl:.1f}cp')

for elo, label in [(1600,'1600'),(1800,'1800'),(2000,'2000'),(2200,'2200')]:
    ax.axhline(y=elo, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.text(acpl_raw[mask_viz].max()*0.95, elo+10, label,
            fontsize=8, color='gray', ha='right')

ax.set_title("ACPL vs ELO Rating (Slide 7)\nLower ACPL = Stronger Player",
             fontweight='bold')
ax.set_xlabel("Average Centipawn Loss (ACPL)")
ax.set_ylabel("ELO Rating")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("elo_estimation_results.png", dpi=150, bbox_inches='tight')
plt.show()
print(" Saved → elo_estimation_results.png")
print(f"\n{'═'*55}")
print(f"  COPY THESE INTO SLIDE 7")
print(f"{'═'*55}")
print(f"  Model      : {model_name}")
print(f"  MAE        : {mae:.1f} ELO points")
print(f"  R² Score   : {r2:.4f}")
print(f"  Top feature: {feature_names[sorted_idx[0]]}")
if real_acpl_available:
    print(f"  Human ACPL : {real_human_acpl:.1f} cp  (Stockfish depth 12)")
    print(f"  Model ACPL : {real_model_acpl:.1f} cp  (Stockfish depth 12)")
print(f"{'═'*55}")