"""
chess_app.py
────────────
Human-Centric Chess AI — Streamlit UI
NIT Andhra Pradesh | EE2621 Introduction to Machine Learning

Layout matches PPT Slide 4 exactly:
  Left Column   : Interactive Chessboard + Move History List
  Middle Column : Current Move Eval (Stockfish CPL) + Error Classification
  Right Column  : Model Prediction (Policy CNN) + ELO Tracker

Run:
    pip3 install streamlit
    streamlit run chess_app.py
"""

import streamlit as st
import chess
import chess.svg
import chess.engine
import torch
import numpy as np
import os
import joblib

from Code.model import ChessGPT
from Code.config import ModelCFG

# ══════════════════════════════════════════════════════════════════════════════
# Page Setup
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Human-Centric Chess AI",
    page_icon="♟",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem; font-weight: 700;
    color: #1a1a2e; margin-bottom: 2px;
}
.subtitle { font-size: 0.9rem; color: #888; margin-top: 0; }
.col-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem; font-weight: 700;
    color: #1a1a2e;
    border-bottom: 3px solid #185FA5;
    padding-bottom: 6px; margin-bottom: 14px;
}
.sub-header {
    font-size: 0.85rem; font-weight: 600;
    color: #185FA5; text-transform: uppercase;
    letter-spacing: 0.06em; margin: 14px 0 6px 0;
}
.cpl-box {
    border-radius: 10px; padding: 14px 18px;
    text-align: center; margin: 8px 0;
}
.cpl-best       { background:#e8f8f0; border:2px solid #1D9E75; color:#1D9E75; }
.cpl-inaccuracy { background:#fef9e7; border:2px solid #E6A817; color:#E6A817; }
.cpl-mistake    { background:#fef0e8; border:2px solid #D85A30; color:#D85A30; }
.cpl-blunder    { background:#fde8e8; border:2px solid #A32D2D; color:#A32D2D; }
.cpl-label { font-size:1.5rem; font-weight:700; }
.cpl-value { font-size:2.2rem; font-weight:700; }
.cpl-desc  { font-size:0.8rem; opacity:0.8; }

.model-box {
    background:#f0f4ff; border:2px solid #185FA5;
    border-radius:10px; padding:14px 18px; margin:8px 0;
}
.model-move { font-size:1.8rem; font-weight:700; color:#185FA5; }
.model-conf { font-size:0.9rem; color:#555; }

.elo-box {
    background: linear-gradient(135deg, #1a1a2e 0%, #185FA5 100%);
    border-radius:10px; padding:16px 18px; margin:8px 0; color:white;
    text-align:center;
}
.elo-value { font-size:2.4rem; font-weight:700; }
.elo-label { font-size:0.85rem; opacity:0.8; }

.move-row {
    display:flex; justify-content:space-between;
    padding:5px 8px; border-radius:6px; margin:3px 0;
    font-size:0.85rem;
}
.move-good  { background:#e8f8f0; }
.move-bad   { background:#fde8e8; }
.move-ok    { background:#f5f5f5; }

.nit-badge {
    background:#1a1a2e; color:white;
    padding:3px 10px; border-radius:20px;
    font-size:0.72rem; display:inline-block; margin-bottom:6px;
}
.legend-dot {
    display:inline-block; width:12px; height:12px;
    border-radius:50%; margin-right:5px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
DEVICE         = "mps" if torch.backends.mps.is_available() else "cpu"
DEPTH          = 10

PIECE_TO_CH = {
    (chess.PAWN,   chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2, (chess.ROOK,   chess.WHITE): 3,
    (chess.QUEEN,  chess.WHITE): 4, (chess.KING,   chess.WHITE): 5,
    (chess.PAWN,   chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8, (chess.ROOK,   chess.BLACK): 9,
    (chess.QUEEN,  chess.BLACK): 10,(chess.KING,   chess.BLACK): 11,
}

# ELO lookup from ACPL (from Slide 7 analysis)
ACPL_TO_ELO = [
    (20,  2300), (30, 2100), (40, 1950),
    (50,  1850), (70, 1750), (100, 1650),
    (150, 1550), (200, 1500),
]

def acpl_to_elo_estimate(acpl: float) -> int:
    for threshold, elo in ACPL_TO_ELO:
        if acpl <= threshold:
            return elo
    return 1500


# ══════════════════════════════════════════════════════════════════════════════
# Cached resources
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    cfg   = ModelCFG()
    model = ChessGPT(cfg).to(DEVICE)
    ckpt  = torch.load("model_best.pth", map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["step"]

@st.cache_resource
def load_engine():
    if os.path.exists(STOCKFISH_PATH):
        try:
            return chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
        except Exception:
            return None
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════
def board_to_features(board: chess.Board) -> np.ndarray:
    feat = np.zeros((65, 13), dtype=np.float32)
    stm  = 1.0 if board.turn == chess.WHITE else 0.0
    for sq in range(64):
        p = board.piece_at(sq)
        if p:
            feat[sq, PIECE_TO_CH[(p.piece_type, p.color)]] = 1.0
        feat[sq, 12] = stm
    feat[64, 0] = float(board.has_kingside_castling_rights(chess.WHITE))
    feat[64, 1] = float(board.has_queenside_castling_rights(chess.WHITE))
    feat[64, 2] = float(board.has_kingside_castling_rights(chess.BLACK))
    feat[64, 3] = float(board.has_queenside_castling_rights(chess.BLACK))
    feat[64, 4] = float(board.ep_square is not None)
    feat[64, 5] = min(board.halfmove_clock / 100.0, 1.0)
    return feat


def get_model_prediction(board: chess.Board, model) -> tuple:
    """Returns (best_move, top5_moves_with_confidence)"""
    feat = board_to_features(board)
    x    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, score_pred = model(x)

    legal = list(board.legal_moves)
    if not legal:
        return None, [], 0.0

    # Score legal moves
    move_scores = {}
    for move in legal:
        idx = move.from_square * 64 + move.to_square
        move_scores[move] = logits[0][idx].item()

    # Softmax over legal moves only
    scores_arr = np.array(list(move_scores.values()))
    probs      = np.exp(scores_arr - scores_arr.max())
    probs      = probs / probs.sum()

    moves_list = list(move_scores.keys())
    top5_idx   = np.argsort(probs)[::-1][:5]
    top5       = [(moves_list[i], probs[i]) for i in top5_idx]
    best_move  = top5[0][0]
    confidence = top5[0][1]

    return best_move, top5, float(score_pred.item())


def safe_score(score_obj):
    if score_obj is None: return None
    raw = score_obj.white().score(mate_score=10000)
    if raw is None: return None
    return max(-1000, min(1000, raw))


def classify_cpl(cpl: float) -> tuple:
    """Returns (label, css_class, description, threshold_text)"""
    if cpl < 50:
        return "Best", "cpl-best", "Near-perfect move", "< 50 cp"
    if cpl < 100:
        return "Inaccuracy", "cpl-inaccuracy", "Slightly inferior move", "50–100 cp"
    if cpl < 300:
        return "Mistake", "cpl-mistake", "Clearly inferior move", "100–300 cp"
    return "Blunder", "cpl-blunder", "Losing move", "> 300 cp"


# ══════════════════════════════════════════════════════════════════════════════
# Session state init
# ══════════════════════════════════════════════════════════════════════════════
if "board" not in st.session_state:
    st.session_state.board         = chess.Board()
    st.session_state.move_history  = []   # list of dicts
    st.session_state.last_move     = None


# ══════════════════════════════════════════════════════════════════════════════
# Load resources
# ══════════════════════════════════════════════════════════════════════════════
model, step = load_model()
engine      = load_engine()
board       = st.session_state.board


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="nit-badge">NIT Andhra Pradesh | EE2621 Introduction to Machine Learning</div>',
            unsafe_allow_html=True)
st.markdown('<p class="main-title">♟ Human-Centric Chess AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predicting human-like moves · Quantifying player skill · ELO estimation</p>',
            unsafe_allow_html=True)
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# 3-Column Layout (matches PPT Slide 4)
# ══════════════════════════════════════════════════════════════════════════════
left_col, mid_col, right_col = st.columns([1.3, 1, 1])


# ─────────────────────────────────────────────────────────────────────────────
# LEFT COLUMN — Interactive Chessboard + Move History
# ─────────────────────────────────────────────────────────────────────────────
with left_col:
    st.markdown('<p class="col-header">Left Column: The Board</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive Chessboard</p>',
                unsafe_allow_html=True)
    st.caption("Displays the current FEN state.")

    # Board controls
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.board        = chess.Board()
            st.session_state.move_history = []
            st.session_state.last_move    = None
            st.rerun()
    with c2:
        if st.button("↩️ Undo", use_container_width=True):
            if st.session_state.move_history:
                st.session_state.board.pop()
                st.session_state.move_history.pop()
                st.session_state.last_move = (
                    st.session_state.move_history[-1]["move_obj"]
                    if st.session_state.move_history else None
                )
                st.rerun()
    with c3:
        flip = st.toggle("Flip", value=False)

    # FEN input
    fen_in = st.text_input("FEN:", value=board.fen(), label_visibility="collapsed")
    if st.button("📥 Load FEN", use_container_width=True):
        try:
            st.session_state.board        = chess.Board(fen_in)
            st.session_state.move_history = []
            st.session_state.last_move    = None
            st.rerun()
        except Exception:
            st.error("Invalid FEN")

    # Render board with arrows
    arrows    = []
    last_move = st.session_state.last_move

    # Get live model prediction for current position
    if not board.is_game_over():
        live_model_move, live_top5, live_score = get_model_prediction(board, model)
        if live_model_move:
            arrows.append(chess.svg.Arrow(
                live_model_move.from_square,
                live_model_move.to_square,
                color="#D85A30"   # orange = model
            ))

        # Get engine move
        live_engine_move = None
        if engine:
            try:
                info             = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
                live_engine_move = info["pv"][0]
                arrows.append(chess.svg.Arrow(
                    live_engine_move.from_square,
                    live_engine_move.to_square,
                    color="#1D9E75"   # green = engine
                ))
            except Exception:
                pass
    else:
        live_model_move, live_top5, live_score = None, [], 0.0
        live_engine_move = None

    svg = chess.svg.board(
        board,
        arrows   = arrows,
        lastmove = last_move,
        flipped  = flip,
        size     = 380,
    )
    st.image(svg.encode(), use_container_width=True)

    # Arrow legend
    st.markdown("""
    <div style="font-size:0.8rem; margin-top:4px;">
        <span class="legend-dot" style="background:#1D9E75;"></span>Engine best &nbsp;
        <span class="legend-dot" style="background:#D85A30;"></span>Model prediction
    </div>
    """, unsafe_allow_html=True)

    # Move input
    st.markdown('<p class="sub-header">Move History List</p>', unsafe_allow_html=True)
    st.caption("Highlights selected moves for analysis.")

    move_in = st.text_input("Enter move (e2e4 or e4):", placeholder="e2e4",
                             label_visibility="collapsed")
    if st.button("▶️ Play Move", type="primary", use_container_width=True):
        if move_in.strip() and not board.is_game_over():
            try:
                try:
                    move = chess.Move.from_uci(move_in.strip())
                except Exception:
                    move = board.parse_san(move_in.strip())

                if move in board.legal_moves:
                    # Analyse before pushing
                    m_move, m_top5, m_score = get_model_prediction(board, model)

                    eval_before  = None
                    eval_after_h = None
                    eval_after_m = None
                    e_move       = None

                    if engine:
                        try:
                            info         = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
                            e_move       = info["pv"][0]
                            eval_before  = safe_score(info["score"])

                            b2 = board.copy(); b2.push(move)
                            i2 = engine.analyse(b2, chess.engine.Limit(depth=DEPTH))
                            eval_after_h = safe_score(i2["score"])

                            if m_move and m_move in board.legal_moves:
                                b3 = board.copy(); b3.push(m_move)
                                i3 = engine.analyse(b3, chess.engine.Limit(depth=DEPTH))
                                eval_after_m = safe_score(i3["score"])
                        except Exception:
                            pass

                    is_white  = board.turn == chess.WHITE
                    human_cpl = None
                    model_cpl = None

                    if eval_before is not None and eval_after_h is not None:
                        if is_white:
                            human_cpl = max(0, eval_before - eval_after_h)
                            model_cpl = max(0, eval_before - eval_after_m) if eval_after_m is not None else None
                        else:
                            human_cpl = max(0, eval_after_h - eval_before)
                            model_cpl = max(0, eval_after_m - eval_before) if eval_after_m is not None else None

                    board.push(move)
                    st.session_state.last_move = move
                    st.session_state.move_history.append({
                        "move_uci":       move.uci(),
                        "move_san":       board.san(move) if hasattr(board, 'san') else move.uci(),
                        "move_obj":       move,
                        "model_move":     m_move.uci() if m_move else "—",
                        "engine_move":    e_move.uci() if e_move else "—",
                        "human_cpl":      human_cpl,
                        "model_cpl":      model_cpl,
                        "matched_model":  m_move == move if m_move else False,
                        "matched_engine": e_move  == move if e_move  else False,
                        "turn":           "White" if is_white else "Black",
                        "move_number":    board.fullmove_number,
                    })
                    st.rerun()
                else:
                    st.error(f"Illegal move: {move_in}")
            except Exception as e:
                st.error(f"Invalid: {move_in}")

    # Move history list
    if st.session_state.move_history:
        for i, rec in enumerate(reversed(st.session_state.move_history[-8:])):
            cpl   = rec["human_cpl"]
            if cpl is None:
                css = "move-ok"
            elif cpl < 50:
                css = "move-good"
            elif cpl < 300:
                css = "move-ok"
            else:
                css = "move-bad"

            label = f"#{len(st.session_state.move_history)-i}" \
                    f" {rec['turn'][0]} {rec['move_uci']}"
            cpl_str = f"{cpl:.0f} cp" if cpl is not None else "—"
            match   = "✓M" if rec["matched_model"] else ""
            match  += " ✓E" if rec["matched_engine"] else ""

            st.markdown(
                f'<div class="move-row {css}">'
                f'<span>{label}</span>'
                f'<span style="color:#888">{cpl_str} {match}</span>'
                f'</div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# MIDDLE COLUMN — Current Move Eval (Stockfish) + Error Classification
# ─────────────────────────────────────────────────────────────────────────────
with mid_col:
    st.markdown('<p class="col-header">Middle Column: Human Metrics</p>',
                unsafe_allow_html=True)

    # ── Current Move Eval ─────────────────────────────────────────────────────
    st.markdown('<p class="sub-header">Current Move Eval (Stockfish)</p>',
                unsafe_allow_html=True)
    st.caption("Shows the Centipawn Loss (CPL) for the last move.")

    if st.session_state.move_history:
        last = st.session_state.move_history[-1]
        cpl  = last["human_cpl"]

        if cpl is not None:
            label, css, desc, thresh = classify_cpl(cpl)
            st.markdown(f"""
            <div class="cpl-box {css}">
                <div class="cpl-value">{cpl:.0f} cp</div>
                <div class="cpl-label">{label}</div>
                <div class="cpl-desc">{thresh} · {desc}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Play a move to see evaluation")

        # Game eval bar
        if st.session_state.move_history:
            valid_cpls = [r["human_cpl"] for r in st.session_state.move_history
                          if r["human_cpl"] is not None]
            if valid_cpls:
                avg_cpl = np.mean(valid_cpls)
                st.metric("Your ACPL so far", f"{avg_cpl:.1f} cp",
                          help="Average Centipawn Loss — lower is better")

                elo_est = acpl_to_elo_estimate(avg_cpl)
                st.metric("Estimated ELO (live)", f"~{elo_est}",
                          delta=f"{elo_est - 1500:+d} from baseline")
    else:
        st.info("📝 Play a move to see evaluation")
        st.markdown("""
        **How CPL works:**
        - Enter a move like `e2e4`
        - Stockfish evaluates before and after
        - CPL = how much evaluation dropped
        """)

    # ── Error Classification ──────────────────────────────────────────────────
    st.markdown('<p class="sub-header">Error Classification</p>',
                unsafe_allow_html=True)
    st.caption('Categorizes moves as "Inaccuracy," "Mistake," or "Blunder" based on CPL.')

    # Classification thresholds (Slide 6)
    thresholds = [
        ("Best",       "< 50 cp",    "cpl-best",       "1D9E75"),
        ("Inaccuracy", "50–100 cp",  "cpl-inaccuracy", "E6A817"),
        ("Mistake",    "100–300 cp", "cpl-mistake",    "D85A30"),
        ("Blunder",    "> 300 cp",   "cpl-blunder",    "A32D2D"),
    ]

    for label, range_str, css, color in thresholds:
        count = sum(1 for r in st.session_state.move_history
                    if r["human_cpl"] is not None and
                    classify_cpl(r["human_cpl"])[0] == label)
        total = len([r for r in st.session_state.move_history
                     if r["human_cpl"] is not None])
        pct   = (count / total * 100) if total > 0 else 0

        st.markdown(f"""
        <div style="display:flex; justify-content:space-between;
                    align-items:center; padding:6px 10px; margin:4px 0;
                    background:#f8f9fa; border-radius:8px;
                    border-left:4px solid #{color};">
            <span style="font-weight:600; color:#{color};">{label}</span>
            <span style="color:#888; font-size:0.82rem;">{range_str}</span>
            <span style="font-weight:700;">{count} ({pct:.0f}%)</span>
        </div>
        """, unsafe_allow_html=True)

    # Overall game stats
    if st.session_state.move_history:
        st.divider()
        total   = len(st.session_state.move_history)
        m_match = sum(1 for r in st.session_state.move_history if r["matched_model"])
        e_match = sum(1 for r in st.session_state.move_history if r["matched_engine"])

        st.markdown(f"**Moves played:** {total}")
        st.progress(m_match / total if total > 0 else 0,
                    text=f"Matched model: {m_match}/{total} ({m_match/total*100:.0f}%)" if total > 0 else "")
        st.progress(e_match / total if total > 0 else 0,
                    text=f"Matched engine: {e_match}/{total} ({e_match/total*100:.0f}%)" if total > 0 else "")


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT COLUMN — Model Prediction (Policy CNN) + ELO Tracker
# ─────────────────────────────────────────────────────────────────────────────
with right_col:
    st.markdown('<p class="col-header">Right Column: ML Insights</p>',
                unsafe_allow_html=True)

    # ── Model Prediction ──────────────────────────────────────────────────────
    st.markdown('<p class="sub-header">Model Prediction (Policy CNN)</p>',
                unsafe_allow_html=True)

    if not board.is_game_over() and live_model_move:
        confidence_pct = live_top5[0][1] * 100 if live_top5 else 0
        turn_str = "White" if board.turn == chess.WHITE else "Black"

        st.markdown(f"""
        <div class="model-box">
            <div style="font-size:0.8rem; color:#888; margin-bottom:4px;">
                The model would play:
            </div>
            <div class="model-move">{live_model_move.uci()}</div>
            <div class="model-conf">with {confidence_pct:.0f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        # Score interpretation
        score_val = live_score
        if score_val > 0.3:
            score_str = f"White is winning (+{score_val:.2f})"
            score_col = "#1D9E75"
        elif score_val < -0.3:
            score_str = f"Black is winning ({score_val:.2f})"
            score_col = "#D85A30"
        else:
            score_str = f"Position is equal ({score_val:+.2f})"
            score_col = "#888"

        st.markdown(
            f'<div style="font-size:0.85rem; color:{score_col}; '
            f'padding:6px 10px; background:#f8f9fa; '
            f'border-radius:6px; margin:6px 0;">'
            f'📊 {score_str}</div>',
            unsafe_allow_html=True
        )

        # Top 5 moves
        st.markdown("**Top-5 model moves:**")
        for i, (mv, prob) in enumerate(live_top5):
            bar_w = int(prob * 100)
            medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][i]
            st.markdown(f"""
            <div style="margin:3px 0; font-size:0.85rem;">
                {medal} <b>{mv.uci()}</b>
                <span style="color:#888;"> {prob*100:.1f}%</span>
                <div style="background:#e0e0e0; border-radius:4px;
                            height:6px; margin-top:2px;">
                    <div style="background:#185FA5; width:{bar_w}%;
                                height:6px; border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.info("Make a move to see model prediction")
        st.markdown("""
        **Model info:**
        - Architecture: Transformer (8 layers)
        - Parameters: 7.4M
        - Training: 800K positions
        - Move accuracy: 24.6%
        """)

    # ── ELO Tracker ───────────────────────────────────────────────────────────
    st.markdown('<p class="sub-header">ELO Tracker</p>',
                unsafe_allow_html=True)
    st.caption("Dynamic update of estimated ELO using the Random Forest Regressor.")

    if st.session_state.move_history:
        valid_cpls = [r["human_cpl"] for r in st.session_state.move_history
                      if r["human_cpl"] is not None]

        if valid_cpls:
            avg_cpl   = np.mean(valid_cpls)
            elo_est   = acpl_to_elo_estimate(avg_cpl)
            n_moves   = len(valid_cpls)

            st.markdown(f"""
            <div class="elo-box">
                <div class="elo-label">Estimated ELO Rating</div>
                <div class="elo-value">~{elo_est}</div>
                <div class="elo-label">Based on {n_moves} moves · ACPL = {avg_cpl:.1f} cp</div>
            </div>
            """, unsafe_allow_html=True)

            # ELO bracket
            brackets = [
                (1500, 1700, "Club beginner"),
                (1700, 1900, "Club player"),
                (1900, 2100, "Strong club"),
                (2100, 2300, "Expert"),
                (2300, 2500, "Master"),
            ]
            for lo, hi, label in brackets:
                active = lo <= elo_est < hi
                bg     = "#185FA5" if active else "#f0f0f0"
                fg     = "white"   if active else "#888"
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between;
                            padding:5px 10px; margin:3px 0;
                            background:{bg}; color:{fg};
                            border-radius:6px; font-size:0.82rem;">
                    <span>{lo}–{hi}</span>
                    <span>{label}</span>
                    {'<span>◀ You</span>' if active else ''}
                </div>
                """, unsafe_allow_html=True)

            # Reference line
            st.divider()
            st.markdown(f"""
            **Model reference (from Slide 7):**
            - Human ACPL: 44.0 cp → ~1850 ELO
            - Model ACPL: 85.2 cp → ~1750 ELO
            """)

    else:
        # Show reference stats before any moves
        st.markdown(f"""
        <div class="elo-box">
            <div class="elo-label">Play moves to estimate ELO</div>
            <div class="elo-value">—</div>
            <div class="elo-label">Random Forest Regressor (MAE=190)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        **How ELO estimation works (Slide 7):**
        1. Stockfish computes CPL for each move
        2. Features: ACPL, blunder rate, opening variety
        3. Gradient Boosting predicts ELO
        4. Updates live as you play
        """)

    # Model metrics footer
    st.divider()
    st.markdown("**Project results (Slide 8):**")
    m1, m2 = st.columns(2)
    m1.metric("Move accuracy", "24.6%")
    m2.metric("Top-5 accuracy", "57.8%")
    m1.metric("ACPL", "31.7 cp")
    m2.metric("Human match", "30.7%")


# ══════════════════════════════════════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.8rem;">
    Human-Centric Chess AI · NIT Andhra Pradesh · EE2621 Introduction to Machine Learning<br>
    Anand Pal (524109) · Rajiv Rajpoot (524166) · Raj Pandey (524165)
</div>
""", unsafe_allow_html=True)