import Code.app_ui as app_ui
import chess.engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from Code.model import ChessGPT
from Code.config import ModelCFG

STOCKFISH_PATH  = "/opt/homebrew/bin/stockfish"
DEPTH           = 12
N_POSITIONS     = 1000
MIN_MOVE_NUMBER = 8    # skip pure opening book
MAX_MOVE_NUMBER = 25   # skip complex endgames
DEVICE          = "mps" if torch.backends.mps.is_available() else "cpu"

def safe_score(score_obj) -> float | None:
    if score_obj is None:
        return None
    raw = score_obj.white().score(mate_score=10000)
    if raw is None:
        return None
    return max(-1000, min(1000, raw))

model_config = ModelCFG()
model        = ChessGPT(model_config).to(DEVICE)
ckpt         = torch.load("model_best.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()
test_features = np.load("data/test_features.npy")
test_moves    = np.load("data/test_moves.npy")
test_fens     = np.load("data/test_fens.npy", allow_pickle=True).tolist()

filtered_indices = []
for idx in range(len(test_fens)):
    board = app_ui.Board(test_fens[idx])
    if MIN_MOVE_NUMBER <= board.fullmove_number <= MAX_MOVE_NUMBER:
        filtered_indices.append(idx)

# Sample evenly from filtered positions
if len(filtered_indices) >= N_POSITIONS:
    sample_idx = np.linspace(
        0, len(filtered_indices) - 1, N_POSITIONS, dtype=int
    )
    indices = [filtered_indices[i] for i in sample_idx]
else:
    indices = filtered_indices

features = test_features[indices]
moves    = test_moves[indices]
fens     = [test_fens[i] for i in indices]

# Model Prediction 

def get_model_prediction(feat_np, fen_str):
    x = torch.tensor(feat_np, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(x)

    board       = app_ui.Board(fen_str)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return None

    best_move, best_score = None, float('-inf')
    for move in legal_moves:
        idx   = move.from_square * 64 + move.to_square
        score = logits[0][idx].item()
        if score > best_score:
            best_score = score
            best_move  = move
    return best_move

model_moves = []
for i, (feat, fen) in enumerate(zip(features, fens)):
    model_moves.append(get_model_prediction(feat, fen))
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(fens)} done")


#Stockfish Analysis
engine  = app_ui.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
results = []
skipped = 0

def classify_cpl(cpl):
    if cpl is None:  return "Unknown"
    if cpl < 50:     return "Best"
    if cpl < 100:    return "Inaccuracy"
    if cpl < 300:    return "Mistake"
    return "Blunder"

for i, (fen, human_move_idx, model_move) in enumerate(
        zip(fens, moves, model_moves)):

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(fens)} | kept={len(results)} skipped={skipped}")

    if model_move is None:
        skipped += 1
        continue

    try:
        board = app_ui.Board(fen)

        # Eval BEFORE move
        info_before = engine.analyse(board, app_ui.engine.Limit(depth=DEPTH))
        eval_before = safe_score(info_before["score"])
        engine_move = info_before["pv"][0]

        if eval_before is None:
            skipped += 1
            continue

        # Human move
        from_sq    = int(human_move_idx) // 64
        to_sq      = int(human_move_idx) % 64
        human_move = app_ui.Move(from_sq, to_sq)

        if human_move not in board.legal_moves:
            skipped += 1
            continue

        # Eval AFTER human move
        board_h    = board.copy()
        board_h.push(human_move)
        info_h     = engine.analyse(board_h, app_ui.engine.Limit(depth=DEPTH))
        eval_human = safe_score(info_h["score"])

        if eval_human is None:
            skipped += 1
            continue

        # Eval AFTER model move
        eval_model = None
        if model_move in board.legal_moves:
            board_m    = board.copy()
            board_m.push(model_move)
            info_m     = engine.analyse(board_m, app_ui.engine.Limit(depth=DEPTH))
            eval_model = safe_score(info_m["score"])

        # CPL
        is_white = board.turn == app_ui.WHITE
        if is_white:
            human_cpl = max(0, eval_before - eval_human)
            model_cpl = max(0, eval_before - eval_model) if eval_model is not None else None
        else:
            human_cpl = max(0, eval_human  - eval_before)
            model_cpl = max(0, eval_model  - eval_before) if eval_model is not None else None

    
        if human_cpl > 400:
            skipped += 1
            continue
        if model_cpl is not None and model_cpl > 400:
            skipped += 1
            continue

        human_matches_engine = (human_move == engine_move)
        model_matches_engine = (model_move  == engine_move)
        model_matches_human  = (model_move  == human_move)

        results.append({
            "fen":                  fen,
            "turn":                 "White" if is_white else "Black",
            "move_number":          board.fullmove_number,
            "engine_move":          engine_move.uci(),
            "human_move":           human_move.uci(),
            "model_move":           model_move.uci(),
            "eval_before":          eval_before,
            "eval_after_human":     eval_human,
            "eval_after_model":     eval_model,
            "human_cpl":            human_cpl,
            "model_cpl":            model_cpl,
            "human_class":          classify_cpl(human_cpl),
            "model_class":          classify_cpl(model_cpl),
            "human_matches_engine": human_matches_engine,
            "model_matches_engine": model_matches_engine,
            "model_matches_human":  model_matches_human,
        })

    except Exception:
        skipped += 1
        continue

engine.quit()
df = pd.DataFrame(results)
df.to_csv("analysis_results.csv", index=False)

print(f"  {'#':<4} {'Turn':<6} {'Mv':<4} {'Engine':<8} {'Human':<8} "
      f"{'Model':<14} {'Human CPL':>10} {'Model CPL':>10}")
print(f"{'─'*78}")

for i, row in df.head(25).iterrows():
    marker = ""
    if row["model_matches_human"]:  marker += "✓H"
    if row["model_matches_engine"]: marker += "✓E"
    m_cpl = f"{row['model_cpl']:.0f} cp" if row["model_cpl"] is not None else "N/A"
    print(
        f"  {i+1:<4} {row['turn']:<6} {row['move_number']:<4} "
        f"{row['engine_move']:<8} {row['human_move']:<8} "
        f"{(row['model_move'] + ' ' + marker):<14} "
        f"{row['human_cpl']:>8.0f} cp "
        f"{m_cpl:>10}"
    )

print(f"{'─'*78}")
print(f"  ✓H = model matched human | ✓E = model matched engine")
human_acpl  = df["human_cpl"].mean()
model_acpl  = df["model_cpl"].dropna().mean()
human_best  = (df["human_class"] == "Best").mean()  * 100
model_best  = (df["model_class"] == "Best").mean()  * 100
human_blund = (df["human_class"] == "Blunder").mean() * 100
model_blund = (df["model_class"] == "Blunder").mean() * 100
model_match_human  = df["model_matches_human"].mean()  * 100
model_match_engine = df["model_matches_engine"].mean() * 100
human_match_engine = df["human_matches_engine"].mean() * 100

print(f"\n{'═'*62}")
print(f"  RESULTS FOR PPT  (Slides 6, 7, 8)")
print(f"{'═'*62}")
print(f"  {'Metric':<38} {'Human':>8}  {'Model':>8}")
print(f"{'─'*62}")
print(f"  {'True ACPL (Stockfish depth 12)':<38} {human_acpl:>7.1f}cp  {model_acpl:>7.1f}cp")
print(f"  {'Best move rate':<38} {human_best:>7.1f}%  {model_best:>7.1f}%")
print(f"  {'Blunder rate':<38} {human_blund:>7.1f}%  {model_blund:>7.1f}%")
print(f"  {'Matches engine move':<38} {human_match_engine:>7.1f}%  {model_match_engine:>7.1f}%")
print(f"{'─'*62}")
print(f"  Model matches human move : {model_match_human:.1f}%  ← Slide 8 key metric")
print(f"{'═'*62}")

print(f"\n  Human CPL Distribution (Slide 6):")
for label in ["Best", "Inaccuracy", "Mistake", "Blunder"]:
    pct = (df["human_class"] == label).mean() * 100
    bar = "█" * int(pct / 2)
    print(f"    {label:<12}: {pct:>5.1f}%  {bar}")

print(f"\n  Model CPL Distribution (Slide 6):")
for label in ["Best", "Inaccuracy", "Mistake", "Blunder"]:
    pct = (df["model_class"] == label).mean() * 100
    bar = "█" * int(pct / 2)
    print(f"    {label:<12}: {pct:>5.1f}%  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. Plots
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Stockfish Analysis — Human vs Model vs Engine\n"
    "NIT Andhra Pradesh | EE2621 Introduction to Machine Learning",
    fontsize=13, fontweight='bold'
)

# Plot 1 — CPL Distribution
ax   = axes[0, 0]
cats = ["Best", "Inaccuracy", "Mistake", "Blunder"]
h_pct= [(df["human_class"] == c).mean()*100 for c in cats]
m_pct= [(df["model_class"] == c).mean()*100 for c in cats]
x    = np.arange(len(cats))
w    = 0.35
ax.bar(x - w/2, h_pct, w, label="Human", color='#185FA5', alpha=0.85)
ax.bar(x + w/2, m_pct, w, label="Model", color='#1D9E75', alpha=0.85)
ax.set_title("CPL Distribution: Human vs Model (Slide 6)", fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylabel("% of moves")
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
for xi, (h, m) in enumerate(zip(h_pct, m_pct)):
    ax.text(xi - w/2, h + 0.3, f'{h:.1f}%', ha='center', fontsize=9)
    ax.text(xi + w/2, m + 0.3, f'{m:.1f}%', ha='center', fontsize=9)

# Plot 2 — ACPL comparison
ax     = axes[0, 1]
labels = ["Human\nACPL", "Model\nACPL", "Engine\n(perfect)"]
values = [human_acpl, model_acpl, 0]
colors = ['#185FA5', '#1D9E75', '#D85A30']
bars   = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.5,
            f'{val:.1f} cp', ha='center', fontweight='bold', fontsize=11)
ax.set_title("True ACPL — Stockfish Depth 12 (Slide 7)", fontweight='bold')
ax.set_ylabel("Average Centipawn Loss")
ax.grid(True, alpha=0.3, axis='y')

# Plot 3 — Move agreement
ax     = axes[1, 0]
mlabs  = ["Model\nmatches Human", "Model\nmatches Engine", "Human\nmatches Engine"]
mvals  = [model_match_human, model_match_engine, human_match_engine]
colors3= ['#534AB7', '#185FA5', '#1D9E75']
bars3  = ax.bar(mlabs, mvals, color=colors3, alpha=0.85, width=0.5)
for bar, val in zip(bars3, mvals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.3,
            f'{val:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax.set_title("Move Agreement Rates (Slide 8)", fontweight='bold')
ax.set_ylabel("Agreement (%)")
ax.grid(True, alpha=0.3, axis='y')

# Plot 4 — Human CPL vs Model CPL
ax    = axes[1, 1]
valid = df.dropna(subset=["model_cpl"])
cap   = 350
h_cpl = valid["human_cpl"].clip(0, cap)
m_cpl = valid["model_cpl"].clip(0, cap)
ax.scatter(h_cpl, m_cpl, alpha=0.35, s=10, color='#185FA5')
ax.plot([0, cap], [0, cap], 'r--', linewidth=1.5, label='Equal CPL')
ax.set_title("Human CPL vs Model CPL (Slide 8)", fontweight='bold')
ax.set_xlabel("Human Centipawn Loss")
ax.set_ylabel("Model Centipawn Loss")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("analysis_summary.png", dpi=150, bbox_inches='tight')
plt.show()
