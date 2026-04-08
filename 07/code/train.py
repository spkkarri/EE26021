import os
import math
import chess
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from Code.model import ChessGPT, compute_loss, score_to_cpl
from Code.config import ModelCFG, TrainCFG

os.makedirs("checkpoints", exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model_config = ModelCFG()
train_config = TrainCFG()
X       = np.load(train_config.train_features)
y_move  = np.load(train_config.train_moves)
y_score = np.load(train_config.train_scores)
fens    = np.load(train_config.train_fens, allow_pickle=True).tolist()

X       = torch.tensor(X,       dtype=torch.float32)
y_move  = torch.tensor(y_move,  dtype=torch.long)
y_score = torch.tensor(y_score, dtype=torch.float32).clamp(-1.0, 1.0)

print(f"  Features : {X.shape}")
print(f"  Moves    : {y_move.shape}")
print(f"  Scores   : {y_score.shape}")
print(f"  FENs     : {len(fens)}")

n_total = len(X)
n_train = int(0.80 * n_total)
n_val   = int(0.10 * n_total)
n_test  = n_total - n_train - n_val

indices   = torch.randperm(n_total)
train_idx = indices[:n_train]
val_idx   = indices[n_train:n_train + n_val]
test_idx  = indices[n_train + n_val:]

fens_arr   = np.array(fens, dtype=object)
fens_train = fens_arr[train_idx.numpy()].tolist()
fens_val   = fens_arr[val_idx.numpy()].tolist()
fens_test  = fens_arr[test_idx.numpy()].tolist()

np.save("data/test_features.npy", X[test_idx].numpy())
np.save("data/test_moves.npy",    y_move[test_idx].numpy())
np.save("data/test_scores.npy",   y_score[test_idx].numpy())
np.save("data/test_fens.npy",     fens_arr[test_idx.numpy()])
train_dataset = TensorDataset(X[train_idx], y_move[train_idx], y_score[train_idx])
val_dataset   = TensorDataset(X[val_idx],   y_move[val_idx],   y_score[val_idx])

train_loader = DataLoader(
    train_dataset,
    batch_size  = train_config.bsz,
    shuffle     = True,
    pin_memory  = train_config.pin_memory,
    num_workers = train_config.num_workers,
    drop_last   = True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size  = train_config.bsz,
    shuffle     = False,
    pin_memory  = train_config.pin_memory,
    num_workers = train_config.num_workers,
    drop_last   = False,
)

print(f"  Train : {n_train:,} positions  ({len(train_loader)} batches)")
print(f"  Val   : {n_val:,} positions  ({len(val_loader)} batches)")
print(f"  Test  : {n_test:,} positions  (held out)")

# 5. Model + Optimiser
model = ChessGPT(model_config).to(device)
model.count_params()

optimizer = optim.AdamW(
    model.parameters(),
    lr           = train_config.lr,
    weight_decay = train_config.weight_decay,
)


#  LR Schedule — cosine decay with linear warmup
def get_lr(step: int) -> float:
    if step < train_config.warmup_steps:
        return train_config.lr * (step + 1) / train_config.warmup_steps
    progress = (step - train_config.warmup_steps) / max(
        1, train_config.max_steps - train_config.warmup_steps
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return train_config.lr_min + (train_config.lr - train_config.lr_min) * cosine

def augment_batch(
    x: torch.Tensor,
    score: torch.Tensor,
    prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.rand(1).item() > prob:
        return x, score

    x_aug = x.clone()

    # Swap white ↔ black piece channels on all 64 squares
    white_pieces         = x[:, :64, :6].clone()
    black_pieces         = x[:, :64, 6:12].clone()
    x_aug[:, :64, :6]   = black_pieces
    x_aug[:, :64, 6:12] = white_pieces

    # Flip side-to-move (ch 12)
    x_aug[:, :64, 12] = 1.0 - x[:, :64, 12]

    # Swap castling rights in global token (slot 64)
    wK = x[:, 64, 0].clone()
    wQ = x[:, 64, 1].clone()
    bK = x[:, 64, 2].clone()
    bQ = x[:, 64, 3].clone()
    x_aug[:, 64, 0] = bK
    x_aug[:, 64, 1] = bQ
    x_aug[:, 64, 2] = wK
    x_aug[:, 64, 3] = wQ

    # Negate score
    score_aug = -score

    return x_aug, score_aug

def compute_move_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=-1) == targets).float().mean().item()

def compute_topk_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, k: int = 5
) -> float:
    topk    = logits.topk(k, dim=-1).indices
    correct = topk.eq(targets.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()

def compute_top3_consistency(logits: torch.Tensor, targets: torch.Tensor) -> float:
    top3    = logits.topk(3, dim=-1).indices
    correct = top3.eq(targets.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item()

def compute_acpl(
    score_preds: torch.Tensor, score_targets: torch.Tensor
) -> float:
    return ((score_preds.squeeze() - score_targets).abs() * 1000).mean().item()

def compute_cpl_distribution(
    score_preds: torch.Tensor, score_targets: torch.Tensor
) -> dict:
    cpl_values = (score_preds.squeeze() - score_targets).abs() * 1000
    n          = len(cpl_values)
    return {
        "Best (%)":       ((cpl_values < 50).sum()                            / n * 100).item(),
        "Inaccuracy (%)": (((cpl_values >= 50)  & (cpl_values < 100)).sum()   / n * 100).item(),
        "Mistake (%)":    (((cpl_values >= 100) & (cpl_values < 300)).sum()   / n * 100).item(),
        "Blunder (%)":    ((cpl_values >= 300).sum()                          / n * 100).item(),
    }

def mask_illegal_moves(
    logits: torch.Tensor, batch_fens: list[str]
) -> torch.Tensor:
    masked = torch.full_like(logits, float('-inf'))
    for i, fen in enumerate(batch_fens):
        board      = chess.Board(fen)
        legal_idxs = [m.from_square * 64 + m.to_square for m in board.legal_moves]
        masked[i, legal_idxs] = logits[i, legal_idxs]
    return masked

#  Validation Pass

def run_validation(step: int) -> dict:
    model.eval()
    val_iter    = iter(val_loader)
    xv, mv, sv = next(val_iter)
    xv, mv, sv = xv.to(device), mv.to(device), sv.to(device)
    batch_fens_v = fens_val[:xv.shape[0]]

    with torch.no_grad():
        logits_v, score_v = model(xv)
        val_loss, val_move_loss, val_score_loss = compute_loss(
            logits_v, mv, score_v, sv, model_config
        )
        masked_v = mask_illegal_moves(logits_v, batch_fens_v)
        val_acc  = compute_move_accuracy(masked_v, mv)
        val_top5 = compute_topk_accuracy(masked_v, mv, k=5)
        val_top3 = compute_top3_consistency(masked_v, mv)
        val_acpl = compute_acpl(score_v, sv)
        val_cpl  = compute_cpl_distribution(score_v, sv)

    model.train()
    return {
        "val_loss":       val_loss.item(),
        "val_move_loss":  val_move_loss.item(),
        "val_score_loss": val_score_loss.item(),
        "val_acc":        val_acc,
        "val_top5":       val_top5,
        "val_top3":       val_top3,
        "val_acpl":       val_acpl,
        "cpl_dist":       val_cpl,
    }



#  Training Loop
print("\n─── Training started ───────────────────────────────────────────")
print(f"  Augmentation : board flip prob={train_config.augment_prob}  ← Level 1")
print(f"  Weight decay : {train_config.weight_decay}                  ← Level 3")
print(f"  Dropout      : {model_config.dropout}                       ← Level 3")
print(f"  Label smooth : {model_config.label_smoothing}               ← Level 3")

best_val_loss    = float("inf")
no_improve_count = 0
step             = 0

history = {
    "train_loss": [], "val_loss":  [],
    "train_acc":  [], "val_acc":   [],
    "val_acpl":   [], "val_top3":  [],
}

training_done = False

for epoch in range(99999):
    if training_done:
        break

    for x_batch, move_batch, score_batch in train_loader:
        if step >= train_config.max_steps:
            training_done = True
            break

        x_batch     = x_batch.to(device)
        move_batch  = move_batch.to(device)
        score_batch = score_batch.to(device)

        x_batch, score_batch = augment_batch(
            x_batch, score_batch, prob=train_config.augment_prob
        )
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        move_logits, score_pred = model(x_batch)
        loss, move_loss, score_loss = compute_loss(
            move_logits, move_batch, score_pred, score_batch, model_config
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()
        with torch.no_grad():
            batch_fens_t = fens_train[:x_batch.shape[0]]
            masked_train = mask_illegal_moves(move_logits.detach(), batch_fens_t)
            train_acc    = compute_move_accuracy(masked_train, move_batch)
            train_top5   = compute_topk_accuracy(masked_train, move_batch, k=5)

        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc)

        if step > 0 and step % train_config.save_interval == 0:
            ckpt = f"checkpoints/model_step_{step}.pth"
            torch.save({
                "step":        step,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_loss":    best_val_loss,
            }, ckpt)
        if step % train_config.log_interval == 0:

            val = run_validation(step)

            history["val_loss"].append(val["val_loss"])
            history["val_acc"].append(val["val_acc"])
            history["val_acpl"].append(val["val_acpl"])
            history["val_top3"].append(val["val_top3"])

            if val["val_loss"] < best_val_loss:
                best_val_loss    = val["val_loss"]
                no_improve_count = 0
                torch.save({
                    "step":        step,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_loss":    best_val_loss,
                }, "model_best.pth")
                saved_marker = " 🏆"
            else:
                no_improve_count += 1
                saved_marker = ""

            cpl = val["cpl_dist"]
            print(
                f"\nStep {step:>6} | epoch {epoch} | lr {lr:.2e}{saved_marker}\n"
                f"  Train — loss: {loss.item():.4f}  acc: {train_acc:.4f}  top5: {train_top5:.4f}\n"
                f"  Val   — loss: {val['val_loss']:.4f}  acc: {val['val_acc']:.4f}  "
                f"top5: {val['val_top5']:.4f}\n"
                f"  ELO features — ACPL: {val['val_acpl']:.1f} cp  "
                f"Top-3 consistency: {val['val_top3']*100:.1f}%\n"
                f"  CPL dist — Best: {cpl['Best (%)']:.1f}%  "
                f"Inaccuracy: {cpl['Inaccuracy (%)']:.1f}%  "
                f"Mistake: {cpl['Mistake (%)']:.1f}%  "
                f"Blunder: {cpl['Blunder (%)']:.1f}%"
            )

            if no_improve_count >= train_config.early_stop_patience:
                print(
                    f" Early stopping at step {step} "
                    f"(no improvement for {no_improve_count} log intervals)."
                )
                training_done = True
                break

        step += 1

torch.save({
    "step":        step,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict(),
    "val_loss":    best_val_loss,
}, "model_final.pth")
print(" Final model saved → model_final.pth")
print(f"   Best val loss achieved: {best_val_loss:.4f}")

#Plots

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("ChessGPT Training Results — NIT Andhra Pradesh", fontsize=14)

val_steps = list(range(0, len(history["val_loss"]) * train_config.log_interval,
                        train_config.log_interval))

ax = axes[0, 0]
ax.plot(history["train_loss"], label="Train Loss", alpha=0.7)
ax.plot(val_steps, history["val_loss"], label="Val Loss", linewidth=2)
ax.set_title("Loss")
ax.set_xlabel("Step")
ax.legend()

ax = axes[0, 1]
ax.plot(history["train_acc"], label="Train Acc", alpha=0.7)
ax.plot(val_steps, history["val_acc"], label="Val Acc", linewidth=2)
ax.set_title("Move Accuracy — Slide 8")
ax.set_xlabel("Step")
ax.set_ylabel("Accuracy")
ax.legend()

ax = axes[1, 0]
ax.plot(val_steps, history["val_acpl"], color="orange", linewidth=2)
ax.set_title("ACPL — Slide 7")
ax.set_xlabel("Step")
ax.set_ylabel("CPL")

ax = axes[1, 1]
ax.plot(val_steps, [v * 100 for v in history["val_top3"]], color="green", linewidth=2)
ax.set_title("Top-3 Consistency (%) — Slide 7")
ax.set_xlabel("Step")
ax.set_ylabel("%")

plt.tight_layout()
plt.savefig("training_results.png", dpi=150)
plt.show()
print("📊 Training plot saved → training_results.png")

#  Final Test Evaluation  

ckpt = torch.load("model_best.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

test_features = torch.tensor(np.load("data/test_features.npy"), dtype=torch.float32)
test_moves    = torch.tensor(np.load("data/test_moves.npy"),    dtype=torch.long)
test_scores   = torch.tensor(np.load("data/test_scores.npy"),   dtype=torch.float32)
test_fens     = np.load("data/test_fens.npy", allow_pickle=True).tolist()

test_loader  = DataLoader(
    TensorDataset(test_features, test_moves, test_scores),
    batch_size=train_config.bsz
)

all_accs, all_top5, all_top3, all_acpl = [], [], [], []
cpl_counts = {"Best": 0, "Inaccuracy": 0, "Mistake": 0, "Blunder": 0}
total      = 0
fen_offset = 0

with torch.no_grad():
    for xb, mb, sb in test_loader:
        xb, mb, sb = xb.to(device), mb.to(device), sb.to(device)
        n          = xb.shape[0]
        batch_fens = test_fens[fen_offset:fen_offset + n]
        fen_offset += n

        logits, score_pred = model(xb)
        masked = mask_illegal_moves(logits, batch_fens)

        all_accs.append(compute_move_accuracy(masked, mb))
        all_top5.append(compute_topk_accuracy(masked, mb, k=5))
        all_top3.append(compute_top3_consistency(masked, mb))
        all_acpl.append(compute_acpl(score_pred, sb))

        cpl_vals = (score_pred.squeeze() - sb).abs() * 1000
        cpl_counts["Best"]       += (cpl_vals < 50).sum().item()
        cpl_counts["Inaccuracy"] += ((cpl_vals >= 50)  & (cpl_vals < 100)).sum().item()
        cpl_counts["Mistake"]    += ((cpl_vals >= 100) & (cpl_vals < 300)).sum().item()
        cpl_counts["Blunder"]    += (cpl_vals >= 300).sum().item()
        total += n

print(f"\n{'═'*55}")
print(f"  TEST RESULTS  (Slide 8 — report these numbers)")
print(f"{'═'*55}")
print(f"  Move Accuracy (exact) : {np.mean(all_accs)*100:.2f}%")
print(f"  Move Accuracy (top-5) : {np.mean(all_top5)*100:.2f}%")
print(f"  Top-3 Consistency     : {np.mean(all_top3)*100:.2f}%")
print(f"  Average CPL (ACPL)    : {np.mean(all_acpl):.1f} cp")
print(f"{'─'*55}")
print(f"  CPL Distribution (Slide 6):")
for label, count in cpl_counts.items():
    print(f"    {label:<12}: {count/total*100:.1f}%")
print(f"{'═'*55}")