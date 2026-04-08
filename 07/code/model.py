import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_legal_move_mask(
    move_logits: torch.Tensor,
    fens: list[str],
) -> torch.Tensor:
    masked = move_logits.clone()
    for i, fen in enumerate(fens):
        board = chess.Board(fen)
        legal_indices = {
            move.from_square * 64 + move.to_square
            for move in board.legal_moves
        }
        mask = torch.full((4096,), float('-inf'), device=move_logits.device)
        for idx in legal_indices:
            mask[idx] = 0.0
        masked[i] = masked[i] + mask
    return masked

# CPL Classification
def score_to_cpl(engine_score: float, model_score: float) -> tuple[float, str]:
    
    cpl = abs(engine_score - model_score) * 1000
    if cpl < 50:
        label = "Best"
    elif cpl < 100:
        label = "Inaccuracy"
    elif cpl < 300:
        label = "Mistake"
    else:
        label = "Blunder"
    return cpl, label


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float, bias: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            n_embd, n_head, dropout=dropout, batch_first=True, bias=bias
        )
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )
        self.ln1  = nn.LayerNorm(n_embd)
        self.ln2  = nn.LayerNorm(n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + self.drop(attn_out)
        x = x + self.ff(self.ln2(x))
        return x

class ChessGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        bias        = getattr(config, 'bias',    False)
        dropout     = getattr(config, 'dropout', 0.2)

        self.input_proj = nn.Linear(config.square_dim, config.n_embd, bias=bias)
        self.pos_emb    = nn.Parameter(torch.randn(1, 65, config.n_embd) * 0.02)

        self.extra_emb: nn.Parameter | None = None
        if getattr(config, 'extra_embedding', False):
            self.extra_emb = nn.Parameter(torch.randn(1, 1, config.n_embd) * 0.02)

        self.input_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(config.n_embd, config.n_head, dropout, bias=bias)
            for _ in range(config.n_layer)
        ])

        self.ln_f       = nn.LayerNorm(config.n_embd)
        self.move_head  = nn.Linear(config.n_embd, 4096, bias=bias)
        self.score_head = nn.Linear(config.n_embd, 1,    bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        fens: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x = x + self.pos_emb
        if self.extra_emb is not None:
            x = x + self.extra_emb
        x = self.input_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        x = x.mean(dim=1)

        move_logits = self.move_head(x)
        score_pred  = self.score_head(x)

        if fens is not None:
            move_logits = apply_legal_move_mask(move_logits, fens)

        return move_logits, score_pred

    def predict(
        self,
        x: torch.Tensor,
        fens: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            move_logits, score_pred = self.forward(x, fens=fens)
            move_probs = F.softmax(move_logits, dim=-1)
            score_pred = torch.tanh(score_pred)
        return move_probs, score_pred

    def count_params(self) -> None:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {n:,}  ({n/1e6:.1f}M)")


def compute_loss(
    move_logits:   torch.Tensor,
    move_targets:  torch.Tensor,
    score_pred:    torch.Tensor,
    score_targets: torch.Tensor,
    config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    # Level 3 — label smoothing stops model being overconfident
    label_smoothing = getattr(config, 'label_smoothing', 0.1)
    move_loss  = F.cross_entropy(
        move_logits, move_targets,
        label_smoothing=label_smoothing
    )
    score_loss = F.mse_loss(score_pred.squeeze(-1), score_targets)
    total_loss = (
        config.weight_loss_move  * move_loss +
        config.weight_loss_score * score_loss
    )
    return total_loss, move_loss, score_loss