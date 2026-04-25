import streamlit as st
import streamlit.components.v1 as components
import chess
import chess.svg
import chess.engine
import torch
import numpy as np
import os

from Code.model import ChessGPT
from Code.config import ModelCFG

st.set_page_config(
    page_title="Human-Centric Chess AI",
    page_icon="♟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global dark theme */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117;
    color: #e0e0e0;
}

/* Hide Streamlit chrome (keep header for sidebar toggle) */
#MainMenu, footer { visibility: hidden; }
header {
    visibility: visible !important;
    background: transparent !important;
}
.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 1rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}

/* Keep sidebar reopen control visible even when top chrome is hidden */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    position: fixed !important;
    top: 12px !important;
    left: 10px !important;
    z-index: 10000 !important;
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    opacity: 1 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d !important;
    width: 220px !important;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: none !important;
    color: #c9d1d9 !important;
    text-align: left !important;
    padding: 8px 12px !important;
    border-radius: 6px !important;
    font-size: 0.88rem !important;
    width: 100% !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #21262d !important;
}

/* App header */
.app-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 0 10px 0;
    border-bottom: 1px solid #30363d;
    margin-bottom: 12px;
}
.app-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
}
.app-sub {
    font-size: 0.75rem;
    color: #8b949e;
    margin-left: auto;
}

/* Dark card */
.dark-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.card-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8b949e;
    margin-bottom: 8px;
}
.section-heading {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    margin: 10px 0 12px 0;
    border-radius: 12px;
    font-size: 0.76rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #f0f6ff;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.section-heading::before {
    content: "";
    width: 4px;
    height: 18px;
    border-radius: 999px;
    flex-shrink: 0;
}
.section-heading.eval {
    background: linear-gradient(90deg, rgba(31, 111, 235, 0.18), rgba(31, 111, 235, 0.06));
    border-color: rgba(56, 139, 253, 0.28);
}
.section-heading.eval::before {
    background: #388bfd;
}
.section-heading.rating {
    background: linear-gradient(90deg, rgba(63, 185, 80, 0.18), rgba(63, 185, 80, 0.06));
    border-color: rgba(63, 185, 80, 0.28);
}
.section-heading.rating::before {
    background: #3fb950;
}
.section-heading.history {
    background: linear-gradient(90deg, rgba(139, 148, 158, 0.18), rgba(139, 148, 158, 0.06));
    border-color: rgba(139, 148, 158, 0.2);
}
.section-heading.history::before {
    background: #8b949e;
}

/* Eval big number */
.eval-value {
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1;
    color: #ffffff;
}
.eval-unit { font-size: 1rem; color: #8b949e; font-weight: 400; }
.eval-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-left: 8px;
    vertical-align: middle;
}
.badge-best       { background: #1a4731; color: #3fb950; }
.badge-inaccuracy { background: #3d2f00; color: #e3b341; }
.badge-mistake    { background: #3d1a00; color: #f0883e; }
.badge-blunder    { background: #3d0000; color: #f85149; }

/* Move quality grid */
.mq-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 4px;
}
.mq-cell {
    background: #21262d;
    border-radius: 8px;
    padding: 10px 12px;
}
.mq-count { font-size: 1.6rem; font-weight: 700; color: #ffffff; line-height: 1; }
.mq-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; margin-top: 2px; }
.mq-best       { border-left: 3px solid #3fb950; }
.mq-inaccuracy { border-left: 3px solid #e3b341; }
.mq-mistake    { border-left: 3px solid #f0883e; }
.mq-blunder    { border-left: 3px solid #f85149; }

/* Model rows */
.model-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #21262d;
    font-size: 0.85rem;
}
.model-row:last-child { border-bottom: none; }
.model-label { color: #8b949e; }
.model-value { font-weight: 600; color: #ffffff; }
.prog-bar-bg {
    background: #21262d;
    border-radius: 4px;
    height: 6px;
    margin-top: 6px;
}
.prog-bar-fill {
    background: #1f6feb;
    border-radius: 4px;
    height: 6px;
}

.cpl-eval-wrap {
    margin-top: 10px;
}
.cpl-eval-track {
    width: 100%;
    height: 12px;
    background: #21262d;
    border-radius: 999px;
    overflow: hidden;
    border: 1px solid #30363d;
}
.cpl-eval-fill {
    height: 100%;
    border-radius: inherit;
    transition: width 0.2s ease;
}
.cpl-eval-meta {
    margin-top: 6px;
    display: flex;
    justify-content: space-between;
    color: #8b949e;
    font-size: 0.74rem;
}

/* Move history rows */
.mh-row {
    display: grid;
    grid-template-columns: 30px 1fr 1fr 60px;
    gap: 4px;
    padding: 7px 8px;
    border-radius: 6px;
    font-size: 0.82rem;
    align-items: center;
    margin-bottom: 3px;
    background: #21262d;
}
.mh-num  { color: #8b949e; }
.mh-move { font-weight: 600; color: #e6edf3; }
.mh-badge {
    display: inline-block;
    padding: 1px 7px;
    border-radius: 10px;
    font-size: 0.68rem;
    font-weight: 700;
}
.history-shell {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 12px;
    max-height: 240px;
    overflow-y: auto;
}
.history-empty {
    color: #8b949e;
    font-size: 0.82rem;
    padding: 8px 2px;
}

/* ELO */
.elo-display {
    display: flex;
    align-items: baseline;
    gap: 8px;
}
.elo-main  { font-size: 2.8rem; font-weight: 700; color: #ffffff; }
.elo-delta { font-size: 1rem; color: #3fb950; font-weight: 600; }
.elo-sub   { font-size: 0.78rem; color: #8b949e; }

/* Inputs & buttons */
.stTextInput > div > input {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
    font-size: 0.88rem !important;
}
.stTextInput > div > input:focus {
    border-color: #1f6feb !important;
    box-shadow: 0 0 0 2px rgba(31,111,235,0.2) !important;
}
.stButton > button {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
}
.stButton > button:hover {
    background: #30363d !important;
    border-color: #8b949e !important;
}
[data-testid="baseButton-primary"] {
    background: #1f6feb !important;
    border-color: #1f6feb !important;
    color: #ffffff !important;
}
[data-testid="baseButton-primary"]:hover {
    background: #388bfd !important;
}
div[data-testid="stTextInput"] input {
    min-height: 58px !important;
    padding-top: 0.95rem !important;
    padding-bottom: 0.95rem !important;
    font-size: 1.04rem !important;
    border-radius: 10px !important;
}
[data-testid="baseButton-primary"] {
    min-height: 58px !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
}
div[data-testid="stVerticalBlock"] > div { gap: 0.25rem; }
</style>
""", unsafe_allow_html=True)

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

def acpl_to_elo_estimate(acpl: float) -> int:
    # Continuous mapping with lower floor so heavy blunders can fall below 1500.
    # 0 CPL -> ~2800, 200 CPL -> ~1300, and asymptotically down to 500.
    acpl = max(0.0, float(acpl))
    elo = 500 + 2300 * np.exp(-acpl / 160.0)
    return int(max(500, min(2800, round(elo))))


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
    feat = board_to_features(board)
    x    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits, score_pred = model(x)
    legal = list(board.legal_moves)
    if not legal:
        return None, [], 0.0
    move_scores = {}
    for move in legal:
        idx = move.from_square * 64 + move.to_square
        move_scores[move] = logits[0][idx].item()
    scores_arr = np.array(list(move_scores.values()))
    probs      = np.exp(scores_arr - scores_arr.max())
    probs      = probs / probs.sum()
    moves_list = list(move_scores.keys())
    top5_idx   = np.argsort(probs)[::-1][:5]
    top5       = [(moves_list[i], probs[i]) for i in top5_idx]
    return top5[0][0], top5, float(score_pred.item())


def safe_score(score_obj):
    if score_obj is None: return None
    raw = score_obj.white().score(mate_score=10000)
    if raw is None: return None
    return max(-1000, min(1000, raw))


def classify_cpl(cpl: float) -> tuple:
    # returns (label, badge_css, mq_css, color)
    if cpl < 50:
        return "BEST MOVE",   "badge-best",       "mq-best",       "#3fb950"
    if cpl < 100:
        return "INACCURACY",  "badge-inaccuracy",  "mq-inaccuracy", "#e3b341"
    if cpl < 300:
        return "MISTAKE",     "badge-mistake",     "mq-mistake",    "#f0883e"
    return     "BLUNDER",     "badge-blunder",     "mq-blunder",    "#f85149"


def estimate_rating_for_side(cpls) -> int:
    """
    Estimate a player's Elo from their own CPL list (only moves made by that side).
    This is intentionally penalized by *counts* of blunders so repeated blunders
    can drive rating down hard (e.g. >=3 blunders => <=1000).
    """
    if not cpls:
        return 1500

    cpls = [float(x) for x in cpls if x is not None]
    if not cpls:
        return 1500

    avg_cpl = float(np.mean(cpls))
    elo_base = acpl_to_elo_estimate(avg_cpl)

    counts = {"BEST MOVE": 0, "INACCURACY": 0, "MISTAKE": 0, "BLUNDER": 0}
    for cpl in cpls:
        lbl, _, _, _ = classify_cpl(cpl)
        counts[lbl] += 1

    bl = counts["BLUNDER"]
    mi = counts["MISTAKE"]
    inc = counts["INACCURACY"]
    bad = bl + mi

    # Penalty tuned so repeated blunders dominate the score.
    penalty = bl * 400 + mi * 200 + inc * 80 + max(0.0, avg_cpl - 50.0) * 3.0
    elo_adj = elo_base - penalty

    # Hard caps as requested.
    # More than 2 serious errors (mistakes/blunders) should crater rating.
    if bad >= 3:
        elo_adj = min(elo_adj, 1000)
    if bad >= 5:
        elo_adj = min(elo_adj, 850)
    if bad >= 8:
        elo_adj = min(elo_adj, 500)

    return int(max(500, min(2800, round(elo_adj))))


def try_parse_move(board: chess.Board, move_str: str):
    move_str = move_str.strip()
    if not move_str:
        return None
    try:
        move = chess.Move.from_uci(move_str)
        if move in board.legal_moves:
            return move
    except Exception:
        pass
    try:
        move = board.parse_san(move_str)
        if move in board.legal_moves:
            return move
    except Exception:
        pass
    return None



if "board" not in st.session_state:
    st.session_state.board        = chess.Board()
    st.session_state.move_history = []
    st.session_state.last_move    = None
if "flip" not in st.session_state:
    st.session_state.flip = False
if "last_component_ts" not in st.session_state:
    st.session_state.last_component_ts = 0


model, step = load_model()
engine      = load_engine()
board       = st.session_state.board


with st.sidebar:
    st.markdown("""
    <div style="padding:10px 0 16px 0; border-bottom:1px solid #30363d; margin-bottom:14px;">
        <div style="font-size:1.05rem; font-weight:700; color:#ffffff;">♟ Chess AI</div>
        <div style="font-size:0.72rem; color:#8b949e; margin-top:2px;">Human-Centric Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.7rem; color:#8b949e; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:6px;">Controls</div>', unsafe_allow_html=True)

    if st.button("🔄  New Game", use_container_width=True):
        st.session_state.board        = chess.Board()
        st.session_state.move_history = []
        st.session_state.last_move    = None
        st.rerun()

    if st.button("↩️  Undo Move", use_container_width=True):
        if st.session_state.move_history:
            st.session_state.board.pop()
            st.session_state.move_history.pop()
            st.session_state.last_move = (
                st.session_state.move_history[-1]["move_obj"]
                if st.session_state.move_history else None
            )
            st.rerun()

    flip = st.toggle("🔃  Flip Board", value=st.session_state.flip)
    st.session_state.flip = flip

    st.markdown('<div style="height:1px; background:#30363d; margin:12px 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem; color:#8b949e; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:6px;">Load Position</div>', unsafe_allow_html=True)

    fen_in = st.text_input("FEN", value=board.fen(), label_visibility="collapsed",
                           placeholder="Paste FEN here...")
    if st.button("📥  Load FEN", use_container_width=True):
        try:
            st.session_state.board        = chess.Board(fen_in)
            st.session_state.move_history = []
            st.session_state.last_move    = None
            st.rerun()
        except Exception:
            st.warning("Invalid FEN")

    st.markdown('<div style="height:1px; background:#30363d; margin:12px 0;"></div>', unsafe_allow_html=True)

    engine_status = "🟢  Stockfish ready" if engine else "🔴  Stockfish offline"
    engine_color  = "#3fb950" if engine else "#f85149"
    st.markdown(f'<div style="font-size:0.78rem; color:{engine_color};">{engine_status}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:0.7rem; color:#8b949e; margin-top:6px;">Model step: {step:,}</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:1px; background:#30363d; margin:12px 0;"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; color:#8b949e; line-height:2;">
        <span style="display:inline-block;width:9px;height:9px;background:#3fb950;border-radius:50%;margin-right:6px;vertical-align:middle;"></span>Engine best<br>
        <span style="display:inline-block;width:9px;height:9px;background:#f0883e;border-radius:50%;margin-right:6px;vertical-align:middle;"></span>Model prediction
    </div>
    """, unsafe_allow_html=True)


st.markdown("""
<div class="app-header">
    <span class="app-title">♟ Human-Centric Chess AI</span>
    <span class="app-sub">Neural Move Prediction &nbsp;·&nbsp; CPL Analysis &nbsp;·&nbsp; ELO Estimation</span>
</div>
""", unsafe_allow_html=True)



if not board.is_game_over():
    live_model_move, live_top5, live_score = get_model_prediction(board, model)
    live_engine_move = None
    live_eval_cp = None
    if engine:
        try:
            info             = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            live_engine_move = info["pv"][0]
            live_eval_cp     = safe_score(info.get("score"))
        except Exception:
            pass
else:
    live_model_move, live_top5, live_score = None, [], 0.0
    live_engine_move = None
    live_eval_cp = None


board_col, panel_host = st.columns([1.58, 0.74])
panel_col, _ = panel_host.columns([0.88, 0.12])
chess_board_component = components.declare_component(
    "chess_board_component",
    path="chess_board_component",
)


def apply_player_move(move: chess.Move):
    m_move, _, _ = get_model_prediction(board, model)
    eval_before = eval_after_h = eval_after_m = None
    e_move = None

    # Standard notation (SAN): e4, Nf3, Bc4, Rxb6, etc.
    try:
        move_san = board.san(move)
    except Exception:
        move_san = move.uci()

    model_san = "—"
    engine_san = "—"
    if m_move and m_move in board.legal_moves:
        try:
            model_san = board.san(m_move)
        except Exception:
            model_san = m_move.uci()

    if engine:
        try:
            info        = engine.analyse(board, chess.engine.Limit(depth=DEPTH))
            e_move      = info["pv"][0]
            eval_before = safe_score(info["score"])
            if e_move and e_move in board.legal_moves:
                try:
                    engine_san = board.san(e_move)
                except Exception:
                    engine_san = e_move.uci()
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
    human_cpl = model_cpl = None
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
        "move_san":       move_san,
        "move_obj":       move,
        "model_move":     m_move.uci() if m_move else "—",
        "engine_move":    e_move.uci() if e_move else "—",
        "model_san":      model_san,
        "engine_san":     engine_san,
        "human_cpl":      human_cpl,
        "model_cpl":      model_cpl,
        "matched_model":  m_move == move if m_move else False,
        "matched_engine": e_move  == move if e_move  else False,
        "turn":           "White" if is_white else "Black",
        "move_number":    board.fullmove_number,
    })


with board_col:
    # Calculate live rating estimates before component
    valid_cpls = [r["human_cpl"] for r in st.session_state.move_history if r["human_cpl"] is not None]
    white_cpls = [r["human_cpl"] for r in st.session_state.move_history if r["human_cpl"] is not None]
    black_cpls = [r["human_cpl"] for r in st.session_state.move_history if r["turn"] == "Black" and r["human_cpl"] is not None]
    
    white_rating_guess = estimate_rating_for_side(white_cpls) if white_cpls else None
    black_rating_guess = estimate_rating_for_side(black_cpls) if black_cpls else None
    if isinstance(white_rating_guess, int) and isinstance(black_rating_guess, int):
        white_rating_guess = int(white_rating_guess)
        black_rating_guess = int(black_rating_guess)
    elif isinstance(white_rating_guess, int):
        white_rating_guess = int(white_rating_guess)
    elif isinstance(black_rating_guess, int):
        black_rating_guess = int(black_rating_guess)
    
    # Prepare suggested moves for visualization as arrows
    suggested_moves = []
    if live_top5:
        for i, (move, prob) in enumerate(live_top5):
            suggested_moves.append({
                "from": move.from_square,
                "to": move.to_square,
                "uci": move.uci(),
                "rank": i + 1,
                "confidence": float(prob)
            })
    
    component_move = chess_board_component(
        fen=board.fen(),
        flipped=st.session_state.flip,
        eval_cp=live_eval_cp,
        white_rating_guess=white_rating_guess,
        black_rating_guess=black_rating_guess,
        suggested_moves=suggested_moves,
        key="live_drag_board",
        default=None,
        height=560,
    )

    if component_move and isinstance(component_move, dict):
        move_ts = int(component_move.get("timestamp", 0) or 0)
        if move_ts > st.session_state.last_component_ts:
            st.session_state.last_component_ts = move_ts
            move_uci = component_move.get("uci", "")
            try:
                move = chess.Move.from_uci(move_uci)
            except Exception:
                move = None
            if move in board.legal_moves:
                apply_player_move(move)
                st.rerun()

    # Turn indicator
    turn_icon = "⬜" if board.turn == chess.WHITE else "⬛"
    turn_text = "White to move" if board.turn == chess.WHITE else "Black to move"
    turn_col  = "#ffffff" if board.turn == chess.WHITE else "#8b949e"

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin:6px 0 8px 0;">
        <span>{turn_icon}</span>
        <span style="font-size:0.9rem;font-weight:600;color:{turn_col};">{turn_text}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:0.78rem;color:#8b949e;margin:2px 0 10px 0;">Drag and drop pieces to play moves.</div>',
        unsafe_allow_html=True
    )

    # CPL evaluation bar (alongside the board)
    if st.session_state.move_history:
        last = st.session_state.move_history[-1]
        cpl = last.get("human_cpl")
        if cpl is not None:
            cpl_pct = int(max(0, min(100, (cpl / 300.0) * 100)))
            cpl_color = "#3fb950" if cpl < 50 else ("#e3b341" if cpl < 100 else ("#f0883e" if cpl < 300 else "#f85149"))
            badge_text, badge_css, _, _ = classify_cpl(cpl)
            st.markdown(f"""
            <div class="cpl-eval-wrap">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                    <div style="font-size:0.76rem;font-weight:800;letter-spacing:0.08em;text-transform:uppercase;color:#aeb4c0;">CPL</div>
                    <div style="font-size:0.86rem;font-weight:700;color:#e6edf3;">{cpl:.0f} cp</div>
                </div>
                <div class="cpl-eval-track">
                    <div class="cpl-eval-fill" style="width:{cpl_pct}%;background:{cpl_color};"></div>
                </div>
                <div class="cpl-eval-meta">
                    <span>Best</span>
                    <span>{cpl:.0f} cp</span>
                    <span>Blunder</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ELO tracker below board
    valid_cpls = [r["human_cpl"] for r in st.session_state.move_history if r["human_cpl"] is not None]
    white_cpls = [r["human_cpl"] for r in st.session_state.move_history if r["turn"] == "White" and r["human_cpl"] is not None]
    black_cpls = [r["human_cpl"] for r in st.session_state.move_history if r["turn"] == "Black" and r["human_cpl"] is not None]

    rating_col, history_col = st.columns([0.72, 1.08])
    with rating_col:
        st.markdown('<div class="section-heading rating">Current Rating</div>', unsafe_allow_html=True)

        if not valid_cpls:
            st.markdown("""
            <div class="elo-display">
                <span class="elo-main" style="color:#8b949e;">—</span>
            </div>
            <div class="elo-sub" style="margin-top:4px;">Play moves to estimate your rating</div>
            """, unsafe_allow_html=True)
        else:
            avg_cpl = np.mean(valid_cpls)
            white_elo = estimate_rating_for_side(white_cpls) if white_cpls else "—"
            black_elo = estimate_rating_for_side(black_cpls) if black_cpls else "—"
            spread_text = "—"
            if isinstance(white_elo, int) and isinstance(black_elo, int):
                spread = white_elo - black_elo
                spread_text = f"{'+' if spread >= 0 else ''}{spread}"
                game_elo = int(round((white_elo + black_elo) / 2))
            else:
                game_elo = acpl_to_elo_estimate(avg_cpl)
            n = len(valid_cpls)

            st.markdown(f"""
            <div class="elo-display">
                <span class="elo-main">{game_elo}</span>
                <span class="elo-delta">{spread_text}</span>
            </div>
            <div class="elo-sub" style="margin-top:4px;">
                Based on {n} move(s) &nbsp;·&nbsp; ACPL = {avg_cpl:.1f} cp
            </div>
            <div style="margin-top:10px;display:flex;gap:14px;flex-wrap:wrap;">
                <div style="background:#21262d;border:1px solid #30363d;border-radius:8px;padding:8px 10px;min-width:110px;">
                    <div style="font-size:0.66rem;letter-spacing:0.08em;color:#8b949e;text-transform:uppercase;">White guess</div>
                    <div style="font-size:1.28rem;font-weight:700;color:#e6edf3;margin-top:2px;">{white_elo}</div>
                </div>
                <div style="background:#21262d;border:1px solid #30363d;border-radius:8px;padding:8px 10px;min-width:110px;">
                    <div style="font-size:0.66rem;letter-spacing:0.08em;color:#8b949e;text-transform:uppercase;">Black guess</div>
                    <div style="font-size:1.28rem;font-weight:700;color:#e6edf3;margin-top:2px;">{black_elo}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with history_col:
        st.markdown('<div class="section-heading history">Move History</div>', unsafe_allow_html=True)

        if not st.session_state.move_history:
            st.markdown('<div class="history-shell"><div class="history-empty">No moves yet.</div></div>',
                        unsafe_allow_html=True)
        else:
            history_rows = []
            for i, rec in enumerate(reversed(st.session_state.move_history[-14:])):
                cpl     = rec["human_cpl"]
                idx     = len(st.session_state.move_history) - i
                cpl_str = f"{cpl:.0f} cp" if cpl is not None else "—"
                if cpl is not None:
                    bl, _, _, color = classify_cpl(cpl)
                    short = bl.split()[0]
                    badge_html = f'<span class="mh-badge" style="background:rgba(255,255,255,0.07);color:{color};">{short}</span>'
                else:
                    badge_html = ""

                history_rows.append(
                    f'<div class="mh-row">'
                    f'<span class="mh-num">{idx}.</span>'
                    f'<span class="mh-move">{rec["turn"][0]}  {rec.get("move_san", rec.get("move_uci", "—"))}</span>'
                    f'<span style="color:#8b949e;font-size:0.8rem;">{cpl_str}</span>'
                    f'<span>{badge_html}</span>'
                    f'</div>'
                )

            st.markdown(f'<div class="history-shell">{"".join(history_rows)}</div>', unsafe_allow_html=True)

with panel_col:

    # Current Evaluation
    st.markdown('<div class="section-heading eval">Current Evaluation</div>', unsafe_allow_html=True)

    if st.session_state.move_history:
        last = st.session_state.move_history[-1]
        cpl  = last["human_cpl"]
        if cpl is not None:
            badge_text, badge_css, _, _ = classify_cpl(cpl)
            cpl_pct = int(max(0, min(100, (cpl / 300.0) * 100)))
            cpl_color = "#3fb950" if cpl < 50 else ("#e3b341" if cpl < 100 else ("#f0883e" if cpl < 300 else "#f85149"))
            st.markdown(f"""
            <div style="display:flex;align-items:baseline;gap:6px;flex-wrap:wrap;">
                <span class="eval-value">{cpl:.0f}</span>
                <span class="eval-unit">CP Loss</span>
                <span class="eval-badge {badge_css}">{badge_text}</span>
            </div>
            <div style="font-size:0.82rem;color:#8b949e;margin-top:8px;line-height:1.8;">
                Your move: <b style="color:#e6edf3;">{last.get('move_san', last.get('move_uci', '—'))}</b>
                &nbsp;·&nbsp;
                Engine best: <b style="color:#3fb950;">{last.get('engine_san', last.get('engine_move', '—'))}</b>
                &nbsp;·&nbsp;
                Model: <b style="color:#f0883e;">{last.get('model_san', last.get('model_move', '—'))}</b>
            </div>
            <div class="cpl-eval-wrap">
                <div class="cpl-eval-track">
                    <div class="cpl-eval-fill" style="width:{cpl_pct}%;background:{cpl_color};"></div>
                </div>
                <div class="cpl-eval-meta">
                    <span>Best</span>
                    <span>{cpl:.0f} cp</span>
                    <span>Blunder</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#8b949e;font-size:0.85rem;">Stockfish not available.</div>',
                        unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:16px 0;color:#8b949e;">
            <div style="font-size:1.8rem;margin-bottom:6px;">♟</div>
            <div style="font-size:0.88rem;font-weight:500;color:#c9d1d9;">Play a move to see evaluation</div>
            <div style="font-size:0.75rem;margin-top:4px;">CPL = how much worse your move is vs the engine</div>
        </div>
        """, unsafe_allow_html=True)

    # Move Quality Breakdown
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">MOVE QUALITY BREAKDOWN</div>', unsafe_allow_html=True)

    counts = {"BEST MOVE": 0, "INACCURACY": 0, "MISTAKE": 0, "BLUNDER": 0}
    for r in st.session_state.move_history:
        if r["human_cpl"] is not None:
            lbl, _, _, _ = classify_cpl(r["human_cpl"])
            counts[lbl] += 1

    st.markdown(f"""
    <div class="mq-grid">
        <div class="mq-cell mq-best">
            <div class="mq-count">{counts['BEST MOVE']}</div>
            <div class="mq-label" style="color:#3fb950;">BEST</div>
        </div>
        <div class="mq-cell mq-inaccuracy">
            <div class="mq-count">{counts['INACCURACY']}</div>
            <div class="mq-label" style="color:#e3b341;">INACCURACY</div>
        </div>
        <div class="mq-cell mq-mistake">
            <div class="mq-count">{counts['MISTAKE']}</div>
            <div class="mq-label" style="color:#f0883e;">MISTAKE</div>
        </div>
        <div class="mq-cell mq-blunder">
            <div class="mq-count">{counts['BLUNDER']}</div>
            <div class="mq-label" style="color:#f85149;">BLUNDER</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Neural Model Prediction
    st.markdown('<div class="dark-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-label">NEURAL MODEL PREDICTION</div>', unsafe_allow_html=True)

    if not board.is_game_over() and live_model_move and live_top5:
        conf      = live_top5[0][1] * 100
        score_val = live_score
        win_pct   = max(5, min(95, int(50 + score_val * 30)))
        pos_str   = "White favoured" if score_val > 0.3 else ("Black favoured" if score_val < -0.3 else "Equal position")

        st.markdown(f"""
        <div class="model-row">
            <span class="model-label">Win Probability</span>
            <span class="model-value">{win_pct}%</span>
        </div>
        <div class="prog-bar-bg">
            <div class="prog-bar-fill" style="width:{win_pct}%;"></div>
        </div>
        <div class="model-row" style="margin-top:10px;">
            <span class="model-label">Model plays</span>
            <span class="model-value" style="color:#f0883e;">{live_model_move.uci()}</span>
        </div>
        <div class="model-row">
            <span class="model-label">Confidence</span>
            <span class="model-value">{conf:.0f}%</span>
        </div>
        <div class="model-row">
            <span class="model-label">Position</span>
            <span class="model-value">{pos_str}</span>
        </div>
        <div style="margin-top:10px;font-size:0.72rem;color:#8b949e;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:6px;">Top Moves</div>
        """, unsafe_allow_html=True)

        for i, (mv, prob) in enumerate(live_top5):
            bar_w  = int(prob * 100)
            medals = ["🥇", "🥈", "🥉", "4.", "5."]
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;font-size:0.82rem;">
                <span style="width:18px;text-align:center;">{medals[i]}</span>
                <span style="font-weight:600;color:#e6edf3;width:40px;">{mv.uci()}</span>
                <div style="flex:1;background:#21262d;border-radius:3px;height:5px;">
                    <div style="background:#1f6feb;width:{bar_w}%;height:5px;border-radius:3px;"></div>
                </div>
                <span style="color:#8b949e;width:38px;text-align:right;">{prob*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

    elif board.is_game_over():
        st.markdown(f'<div style="text-align:center;padding:12px;color:#3fb950;font-weight:700;">Game Over — {board.result()}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#8b949e;font-size:0.85rem;padding:8px 0;">Loading...</div>',
                    unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
