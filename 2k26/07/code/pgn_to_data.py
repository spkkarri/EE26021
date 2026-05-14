import Code.app_ui as app_ui
import chess.pgn
import numpy as np
import os
from typing import Optional
PGN_FILE      = "lichess.pgn"   
OUT_DIR       = "data"          
MAX_GAMES     = 50_000         
MIN_ELO       = 1500            
MAX_ELO       = 2500           
MIN_MOVES     = 10              
MIN_TIME_CTRL = 60              
MAX_BLUNDER   = 3.0            
SKIP_OPENING  = 5              
MAX_POSITIONS = 800_000       

PIECE_TO_CH = {
    (app_ui.PAWN,   app_ui.WHITE): 0,
    (app_ui.KNIGHT, app_ui.WHITE): 1,
    (app_ui.BISHOP, app_ui.WHITE): 2,
    (app_ui.ROOK,   app_ui.WHITE): 3,
    (app_ui.QUEEN,  app_ui.WHITE): 4,
    (app_ui.KING,   app_ui.WHITE): 5,
    (app_ui.PAWN,   app_ui.BLACK): 6,
    (app_ui.KNIGHT, app_ui.BLACK): 7,
    (app_ui.BISHOP, app_ui.BLACK): 8,
    (app_ui.ROOK,   app_ui.BLACK): 9,
    (app_ui.QUEEN,  app_ui.BLACK): 10,
    (app_ui.KING,   app_ui.BLACK): 11,
}


def board_to_features(board: app_ui.Board) -> np.ndarray:
    features = np.zeros((65, 13), dtype=np.float32)
    side_to_move = 1.0 if board.turn == app_ui.WHITE else 0.0

    # Per-square features — 8×8×12 
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is not None:
            ch = PIECE_TO_CH[(piece.piece_type, piece.color)]
            features[sq, ch] = 1.0
        features[sq, 12] = side_to_move

    features[64, 0] = float(board.has_kingside_castling_rights(app_ui.WHITE))
    features[64, 1] = float(board.has_queenside_castling_rights(app_ui.WHITE))
    features[64, 2] = float(board.has_kingside_castling_rights(app_ui.BLACK))
    features[64, 3] = float(board.has_queenside_castling_rights(app_ui.BLACK))
    features[64, 4] = float(board.ep_square is not None)
    features[64, 5] = min(board.halfmove_clock / 100.0, 1.0)

    return features


def parse_score(comment: str) -> Optional[float]:
    
    if "%eval" not in comment:
        return None
    try:
        part = comment.split("%eval")[1].strip().split("]")[0].strip()
        if part.startswith("#"):
            return 1.0 if not part.startswith("#-") else -1.0
        val = float(part)
        val = max(-10.0, min(10.0, val))
        return val / 10.0
    except (ValueError, IndexError):
        return None


def is_bullet_game(time_control: str) -> bool:
    if not time_control or time_control == "-":
        return False
    try:
        base = int(time_control.split("+")[0])
        return base < MIN_TIME_CTRL
    except (ValueError, IndexError):
        return False


def parse_eco(game: app_ui.pgn.Game) -> str:
    return game.headers.get("ECO", "UNK").strip()[:3]  # keep only e.g. "B12"

def parse_pgn(pgn_path: str):
    features_list    = []
    moves_list       = []
    scores_list      = []
    fens_list        = []
    game_ids_list    = []   
    player_elos_list = []   
    eco_codes_list   = []   
    side_to_move_list= [] 
    seen_fens        = set()

    games_parsed   = 0
    positions_kept = 0
    skip_elo       = 0
    skip_bullet    = 0
    skip_short     = 0
    skip_blunder   = 0
    skip_duplicate = 0
    skip_opening   = 0

    print(f"Opening {pgn_path} ...")
    print(f"Filters: ELO {MIN_ELO}–{MAX_ELO} | Min moves: {MIN_MOVES} | "
          f"Min time: {MIN_TIME_CTRL}s | Max blunder swing: {MAX_BLUNDER} pawns")

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:

        while games_parsed < MAX_GAMES and positions_kept < MAX_POSITIONS:

            game = app_ui.pgn.read_game(f)
            if game is None:
                print("Reached end of PGN file.")
                break
            try:
                white_elo = int(game.headers.get("WhiteElo", "0"))
                black_elo = int(game.headers.get("BlackElo", "0"))
                if (white_elo < MIN_ELO or black_elo < MIN_ELO or
                        white_elo > MAX_ELO or black_elo > MAX_ELO):
                    skip_elo += 1
                    continue
            except ValueError:
                skip_elo += 1
                continue
            time_control = game.headers.get("TimeControl", "")
            if is_bullet_game(time_control):
                skip_bullet += 1
                continue
            moves_list_temp = list(game.mainline_moves())
            if len(moves_list_temp) < MIN_MOVES * 2:
                skip_short += 1
                continue
            eco = parse_eco(game)
            board = game.board()
            node  = game
            prev_score: Optional[float] = None
            half_move = 0
            for move in moves_list_temp:
                node = node.variation(move)
                half_move += 1
                if half_move <= SKIP_OPENING:
                    board.push(move)
                    skip_opening += 1
                    continue
                comment = node.comment or ""
                score   = parse_score(comment)
                if score is not None and prev_score is not None:
                    swing = abs(score - prev_score) * 10  # back to pawns
                    if swing > MAX_BLUNDER:
                        board.push(move)
                        prev_score = score
                        skip_blunder += 1
                        continue
                fen = board.fen()
                if fen in seen_fens:
                    board.push(move)
                    skip_duplicate += 1
                    continue
                seen_fens.add(fen)
                is_white_turn = board.turn == app_ui.WHITE
                side          = 1 if is_white_turn else 0
                player_elo    = white_elo if is_white_turn else black_elo

                feat      = board_to_features(board)
                move_idx  = move.from_square * 64 + move.to_square
                score_val = score if score is not None else 0.0
                if prev_score is None:
                    prev_score = score_val

                features_list.append(feat)
                moves_list.append(move_idx)
                scores_list.append(score_val)
                fens_list.append(fen)
                game_ids_list.append(games_parsed)       # Slide 7
                player_elos_list.append(player_elo)      # Slide 7
                eco_codes_list.append(eco)               # Slide 7
                side_to_move_list.append(side)           # Slide 7

                positions_kept += 1
                prev_score = score

                board.push(move)

                if positions_kept >= MAX_POSITIONS:
                    break

            games_parsed += 1

            if games_parsed % 500 == 0:
                print(f"  Games: {games_parsed:>6} | Positions: {positions_kept:>8,}")

    print(f"\n── Parse complete ──────────────────────────────────────────")
    print(f"  Games parsed        : {games_parsed:>8,}")
    print(f"  Positions kept      : {positions_kept:>8,}")
    print(f"  Skipped (ELO)       : {skip_elo:>8,}")
    print(f"  Skipped (bullet)    : {skip_bullet:>8,}")
    print(f"  Skipped (short game): {skip_short:>8,}")
    print(f"  Skipped (blunder)   : {skip_blunder:>8,}")
    print(f"  Skipped (duplicate) : {skip_duplicate:>8,}")
    print(f"  Skipped (opening)   : {skip_opening:>8,}")

    
    unique_ecos = len(set(eco_codes_list))
    print(f"  Unique ECO codes    : {unique_ecos:>8,}")

    return (
        np.array(features_list,    dtype=np.float32),
        np.array(moves_list,       dtype=np.int64),
        np.array(scores_list,      dtype=np.float32),
        np.array(fens_list,        dtype=object),
        np.array(game_ids_list,    dtype=np.int32),    # Slide 7
        np.array(player_elos_list, dtype=np.int32),    # Slide 7
        np.array(eco_codes_list,   dtype=object),      # Slide 7
        np.array(side_to_move_list,dtype=np.int8),     # Slide 7
    )

def save_data(features, moves, scores, fens,
              game_ids, player_elos, eco_codes, side_to_move,
              out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Core training data (Slides 4, 5)
    np.save(os.path.join(out_dir, "train_features.npy"),    features)
    np.save(os.path.join(out_dir, "train_moves.npy"),       moves)
    np.save(os.path.join(out_dir, "train_scores.npy"),      scores)
    np.save(os.path.join(out_dir, "train_fens.npy"),        fens)

    # ELO estimator features (Slide 7)
    np.save(os.path.join(out_dir, "train_game_ids.npy"),    game_ids)
    np.save(os.path.join(out_dir, "train_player_elos.npy"), player_elos)
    np.save(os.path.join(out_dir, "train_eco_codes.npy"),   eco_codes)
    np.save(os.path.join(out_dir, "train_side_to_move.npy"),side_to_move)

    print(f"Saved to '{out_dir}/'")
    print(f"  features     : {features.shape}   ({features.nbytes / 1e6:.1f} MB)")
    print(f"  moves        : {moves.shape}")
    print(f"  scores       : {scores.shape}")
    print(f"  fens         : {fens.shape}")
    print(f"  game_ids     : {game_ids.shape}       ← Slide 7: ACPL grouping")
    print(f"  player_elos  : {player_elos.shape}       ← Slide 7: ELO label")
    print(f"  eco_codes    : {eco_codes.shape}       ← Slide 7: opening breadth")
    print(f"  side_to_move : {side_to_move.shape}       ← Slide 7: per-player ACPL")


if __name__ == "__main__":
    if not os.path.exists(PGN_FILE):
        print("Pgn file not found")
        exit(1)

    features, moves, scores, fens, \
    game_ids, player_elos, eco_codes, side_to_move = parse_pgn(PGN_FILE)

    if len(features) == 0:
        print("No positions extracted")
        exit(1)

    save_data(features, moves, scores, fens,
              game_ids, player_elos, eco_codes, side_to_move,
              OUT_DIR)