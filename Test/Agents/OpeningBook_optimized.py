try:
    import Output.log_system as log_system
except Exception:
    log_system = None

import time
from Agents.foundation import *
from Agents.heuristic import get_heuristic_bb

def _ordered_moves():
    """Ưu tiên cột gần trung tâm"""
    return [3, 2, 4, 1, 5, 0, 6]
MOVE_ORDER = _ordered_moves()


def mirror_board(bb):
    """Lật bitboard qua trục dọc (đối xứng trái-phải)."""
    m = 0
    m |= (bb & 0x7F) << 42          # Col 0 -> 6
    m |= (bb & (0x7F << 7)) << 28   # Col 1 -> 5
    m |= (bb & (0x7F << 14)) << 14  # Col 2 -> 4
    m |= (bb & (0x7F << 21))        # Col 3 -> 3
    m |= (bb & (0x7F << 28)) >> 14  # Col 4 -> 2
    m |= (bb & (0x7F << 35)) >> 28  # Col 5 -> 1
    m |= (bb & (0x7F << 42)) >> 42  # Col 6 -> 0
    return m


# -------------------------
# Transposition Table (TT)
# Zobrist + bucket + bound type + bestMove
# -------------------------

TT_BUCKETS = 1 << 23  # số bucket (power-of-two để dùng bitmask)
TT_BUCKET_MASK = TT_BUCKETS - 1
TT_BUCKET_SIZE = 4

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# dict[int bucketIndex] -> list[tuple(key64, depth, value, flag, bestMove)]
tt = {}


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def _make_zobrist_tables(seed: int = 0xC0FFEE):
    # 49 bits/column representation (7*7) but we only ever use 6 bits per column.
    z_me = [0] * 49
    z_opp = [0] * 49
    x = seed & 0xFFFFFFFFFFFFFFFF
    for i in range(49):
        x = _splitmix64(x)
        z_me[i] = x
        x = _splitmix64(x)
        z_opp[i] = x
    return z_me, z_opp


_Z_ME, _Z_OPP = _make_zobrist_tables()


def zobrist_hash(me: int, opp: int) -> int:
    """Deterministic 64-bit Zobrist hash for (me, opp) bitboards."""
    h = 0
    bb = me
    while bb:
        lsb = bb & -bb
        idx = lsb.bit_length() - 1
        h ^= _Z_ME[idx]
        bb ^= lsb
    bb = opp
    while bb:
        lsb = bb & -bb
        idx = lsb.bit_length() - 1
        h ^= _Z_OPP[idx]
        bb ^= lsb
    return h & 0xFFFFFFFFFFFFFFFF


def _canonical_tt_key(me: int, opp: int):
    """Return (key64, flip) where flip=True means mirrored orientation chosen."""
    key = zobrist_hash(me, opp)
    m_me = mirror_board(me)
    m_opp = mirror_board(opp)
    m_key = zobrist_hash(m_me, m_opp)
    if (m_key < key) or (m_key == key and (m_me, m_opp) < (me, opp)):
        return m_key, True
    return key, False


def _tt_probe(key64: int, depth: int, alpha: float, beta: float):
    """Probe TT.

    Returns (hit_value_or_None, new_alpha, new_beta, bestMoveHint).
    bestMoveHint can be used for move ordering even when no cutoff/EXACT hit.
    """
    idx = key64 & TT_BUCKET_MASK
    bucket = tt.get(idx)
    if not bucket:
        return None, alpha, beta, -1

    best_hint = -1
    best_hint_depth = -1

    for k, d, v, flag, bm in bucket:
        if k != key64:
            continue
        if bm != -1 and d > best_hint_depth:
            best_hint = bm
            best_hint_depth = d
        if d < depth:
            continue

        if flag == TT_EXACT:
            return v, alpha, beta, bm
        if flag == TT_LOWER:
            if v > alpha:
                alpha = v
        elif flag == TT_UPPER:
            if v < beta:
                beta = v
        if alpha >= beta:
            return v, alpha, beta, bm

    return None, alpha, beta, best_hint


def _tt_store(key64: int, depth: int, value: float, flag: int, best_move: int):
    idx = key64 & TT_BUCKET_MASK
    bucket = tt.get(idx)
    entry = (key64, depth, value, flag, best_move)
    if bucket is None:
        tt[idx] = [entry]
        return

    # Replace same key if deeper/equal.
    for i, (k, d, _, _, _) in enumerate(bucket):
        if k == key64:
            if depth >= d:
                bucket[i] = entry
            return

    if len(bucket) < TT_BUCKET_SIZE:
        bucket.append(entry)
        return

    # Bucket full: replace the shallowest entry.
    victim_i = 0
    victim_depth = bucket[0][1]
    for i in range(1, len(bucket)):
        d = bucket[i][1]
        if d < victim_depth:
            victim_depth = d
            victim_i = i
    bucket[victim_i] = entry


import os
import json

# -------------------------
# Static Opening Book
# -------------------------
OPENING_BOOK = None


def _book_key(me, opp):
    return (me << 64) | opp


# def check_book(me, opp):
#     if OPENING_BOOK is None:
#         return None
#     key = _book_key(me, opp)
#     move = OPENING_BOOK.get(key)
#     if move is not None:
#         return move
#     m_me = mirror_board(me)
#     m_opp = mirror_board(opp)
#     m_move = OPENING_BOOK.get(_book_key(m_me, m_opp))
#     if m_move is None:
#         return None
#     return 6 - m_move

def get_book_score(me, opp):
    """Lookup score from opening book (BK02 format). Returns score or None."""
    if OPENING_BOOK is None:
        return None
    
    # Try direct lookup
    key = _book_key(me, opp)
    score = OPENING_BOOK.get(key)
    if score is not None:
        return score
    
    # Try mirrored position
    m_me = mirror_board(me)
    m_opp = mirror_board(opp)
    m_key = _book_key(m_me, m_opp)
    m_score = OPENING_BOOK.get(m_key)
    
    # Note: Score doesn't change sign for mirrored position in symmetric game
    return m_score


def load_opening_book():
    global OPENING_BOOK
    if OPENING_BOOK is not None:
        return
    OPENING_BOOK = {}
    
    # Use binary format for 10x faster loading
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try score-based book first (BK02)
    score_book_path = os.path.join(current_dir, "opening_book_score.bin")
    score_pickle_path = os.path.join(current_dir, "opening_book_score.pkl")
    
    # Fall back to old move-based book (BK01)
    book_path = os.path.join(current_dir, "opening_book.bin")
    pickle_path = os.path.join(current_dir, "opening_book.pkl")

    # Try score pickle cache first
    if os.path.exists(score_pickle_path):
        try:
            import pickle
            with open(score_pickle_path, "rb") as f:
                OPENING_BOOK = pickle.load(f)
            print(f"[Opening Book] Loaded {len(OPENING_BOOK)} positions from {score_pickle_path} (Pickle, Score-Based).")
            return
        except Exception as e:
            print(f"[Opening Book] Error loading score pickle: {e}")
    
    # Try score binary book (BK02 format)
    if os.path.exists(score_book_path):
        count = 0
        try:
            import struct
            with open(score_book_path, "rb") as f:
                magic = f.read(4)
                if magic != b"BK02":
                    print(f"[Opening Book] Warning: {score_book_path} has unexpected magic {magic}, expected BK02")
                    return
                
                raw_data = f.read()
            
            # Each entry is 17 bytes: uint64(me), uint64(opp), int8(score)
            entry_size = 17
            num_entries = len(raw_data) // entry_size
            
            for i in range(num_entries):
                # Progress reporting
                if i == num_entries // 4:
                    print("[Opening Book] Loading... 25%")
                elif i == num_entries // 2:
                    print("[Opening Book] Loading... 50%")
                elif i == (num_entries * 3) // 4:
                    print("[Opening Book] Loading... 75%")
                
                offset = i * entry_size
                me, opp, score = struct.unpack_from("<QQb", raw_data, offset)  # 'b' = signed byte
                
                OPENING_BOOK[_book_key(me, opp)] = score
                count += 1

            print("[Opening Book] Loading... 100%")
            print(f"[Opening Book] Loaded {count} positions (Score-Based BK02 format, {len(OPENING_BOOK)} entries) from {score_book_path}.")

            # Save pickle cache for next time
            try:
                import pickle
                with open(score_pickle_path, "wb") as f:
                    pickle.dump(OPENING_BOOK, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[Opening Book] Saved pickle cache to {score_pickle_path}.")
            except Exception as e:
                print(f"[Opening Book] Error saving score pickle: {e}")
            return
        except Exception as e:
            print(f"[Opening Book] Error loading score binary: {e}")
    
    # Fallback to old move-based book (BK01)
    if os.path.exists(pickle_path):
        try:
            import pickle
            with open(pickle_path, "rb") as f:
                OPENING_BOOK = pickle.load(f)
            print(f"[Opening Book] Loaded {len(OPENING_BOOK)} positions from {pickle_path} (Pickle, Move-Based).")
            return
        except Exception as e:
            print(f"[Opening Book] Error loading move pickle: {e}")
    
    if os.path.exists(book_path):
        count = 0
        try:
            import struct
            with open(book_path, "rb") as f:
                magic = f.read(4)
                if magic != b"BK01":
                    print(f"[Opening Book] Warning: {book_path} has unexpected magic, expected BK01")
                    return
                
                raw_data = f.read()
            
            # Each entry is 17 bytes: uint64(me), uint64(opp), uint8(move)
            entry_size = 17
            num_entries = len(raw_data) // entry_size
            
            for i in range(num_entries):
                if i == num_entries // 4:
                    print("[Opening Book] Loading... 25%")
                elif i == num_entries // 2:
                    print("[Opening Book] Loading... 50%")
                elif i == (num_entries * 3) // 4:
                    print("[Opening Book] Loading... 75%")
                
                offset = i * entry_size
                me, opp, move = struct.unpack_from("<QQB", raw_data, offset)
                
                OPENING_BOOK[_book_key(me, opp)] = move
                count += 1

            print("[Opening Book] Loading... 100%")
            print(f"[Opening Book] Loaded {count} positions (Move-Based BK01 format) from {book_path}.")

            try:
                import pickle
                with open(pickle_path, "wb") as f:
                    pickle.dump(OPENING_BOOK, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"[Opening Book] Saved pickle cache to {pickle_path}.")
            except Exception as e:
                print(f"[Opening Book] Error saving move pickle: {e}")
        except Exception as e:
            print(f"[Opening Book] Error loading move binary: {e}")
    else:
        print(f"[Opening Book] Warning: No opening book found at {score_book_path} or {book_path}")


# -------------------------
# Threat detection helpers
# -------------------------

BOTTOM_ROW = sum(1 << (c * 7) for c in range(7))
COLUMN_HEADERS = BOTTOM_ROW << 6
VALID_CELLS = COLUMN_HEADERS - BOTTOM_ROW

def _find_threats(b):
    """Tìm tất cả các ô trống mà nếu b đánh vào sẽ tạo thành 4 quân liên tiếp."""
    threats = 0
    # Dọc: ô phía trên 3 quân thẳng đứng
    pairs = b & (b << 1)
    triple = pairs & (pairs << 1)
    threats |= triple << 1
    
    # Ngang + 2 đường chéo
    for stride in (7, 6, 8):
        pairs = b & (b << stride)
        triple = pairs & (pairs << stride)
        threats |= (b >> stride) & (pairs << stride)       # Kiểu X_XX
        threats |= (b << stride) & (pairs >> (2 * stride)) # Kiểu XX_X
        threats |= triple << stride                        # Kiểu XXX_
        threats |= triple >> (3 * stride)                  # Kiểu _XXX
        
    return threats & VALID_CELLS


def _get_valid_moves(me, opp):
    """Return list of valid columns in center-first order."""
    moves = []
    for col in MOVE_ORDER:
        if not ((me | opp) & (1 << (col * 7 + 5))):
            moves.append(col)
    return moves


def _make_move(me, opp, col):
    """Return new_piece bitmask for playing in col. Returns 0 if col is full."""
    col_mask = 0b111111 << (col * 7)
    occupied = (me | opp) & col_mask
    if occupied & (1 << (col * 7 + 5)):
        return 0  # full
    return (occupied + (1 << (col * 7))) & col_mask


def _find_winning_move(me, opp):
    """Check if current player has a winning move. Return col or -1."""
    for col in MOVE_ORDER:
        new_piece = _make_move(me, opp, col)
        if new_piece and is_win(me | new_piece):
            return col
    return -1


def _find_forced_block(me, opp):
    """Check if opponent wins next turn. If so, return the blocking col(s).
    Returns (single_block_col, must_block) where must_block=True if we MUST block.
    If opponent has 2+ winning moves, we're dead but still return one to delay.
    Returns (-1, False) if no threat.
    """
    threat_cols = []
    for col in MOVE_ORDER:
        new_piece = _make_move(opp, me, col)  # opponent plays
        if new_piece and is_win(opp | new_piece):
            threat_cols.append(col)
    
    if not threat_cols:
        return -1, False
    if len(threat_cols) == 1:
        return threat_cols[0], True
    # Multiple threats - we're likely lost, but block one
    return threat_cols[0], True


# -------------------------
# Killer moves heuristic
# -------------------------
killer_moves = {}  # depth -> [col, col]

def _record_killer(depth, col):
    if depth not in killer_moves:
        killer_moves[depth] = [col, -1]
    elif killer_moves[depth][0] != col:
        killer_moves[depth][1] = killer_moves[depth][0]
        killer_moves[depth][0] = col

def _get_killer(depth):
    return killer_moves.get(depth, [-1, -1])


# -------------------------
# PVS Search, trả về điểm của thế cờ
# -------------------------
searching_depth = 0
def pvs(me, opp, depth, alpha, beta, deadline):
    if is_win(opp):
        ply_count = (me | opp).bit_count()
        return -(MATE_SCORE - ply_count)
    if depth == 0 or time.perf_counter() > deadline:
        return get_heuristic_bb(me, opp, (me | opp).bit_count() % 2)
    
    # Check book score first, fallback to heuristic
    book_score = get_book_score(me, opp)
    if book_score is not None:
        return book_score*200
    
    alpha0, beta0 = alpha, beta

    key64, flip = _canonical_tt_key(me, opp)
    tt_value, alpha, beta, tt_best = _tt_probe(key64, depth, alpha, beta)
    if tt_value is not None:
        return tt_value

    # bestMove từ TT (nếu lưu theo orientation canonical thì cần mirror lại).
    if tt_best != -1 and flip:
        tt_best = 6 - tt_best

    value = NNF
    best_move = -1
    first_child = True

    # Move ordering: TT best -> killer moves -> center-first order.
    ordered = []
    if tt_best != -1:
        ordered.append(tt_best)
    
    killers = _get_killer(depth)
    for k in killers:
        if k != -1 and k != tt_best:
            ordered.append(k)
    
    for c in MOVE_ORDER:
        if c not in ordered:
            ordered.append(c)

    occupied = me | opp
    playable_now = (occupied + BOTTOM_ROW) & VALID_CELLS

    # 1. Phát hiện cơ hội thắng ngay lập tức
    my_threats = _find_threats(me) & ~opp
    win_now = my_threats & playable_now
    if win_now:
        ply_count = occupied.bit_count()
        return (MATE_SCORE - ply_count - 1)

    # 2. Lọc Safe Moves và phát hiện đe dọa của đối thủ
    opp_threats = _find_threats(opp) & ~me
    opp_wins_now = opp_threats & playable_now
    safe_moves_mask = playable_now & ~(opp_threats >> 1)

    if opp_wins_now:
        if opp_wins_now & (opp_wins_now - 1): # Đối thủ có >= 2 đường thắng
            ply_count = occupied.bit_count()
            return -(MATE_SCORE - ply_count - 2)
        if not (opp_wins_now & safe_moves_mask): # Nước chặn duy nhất lại tự bóp
            ply_count = occupied.bit_count()
            return -(MATE_SCORE - ply_count - 2)
        safe_moves_mask = opp_wins_now # Bắt buộc phải đánh vào nước chặn này
    elif safe_moves_mask == 0:
        ply_count = occupied.bit_count()
        return -(MATE_SCORE - ply_count - 2) # Không có nước nào an toàn -> thua

    # Note: Book now contains scores, not moves.
    # Move ordering comes from TT hints + killer moves + center-first default.

    for col in ordered:
        col_mask = 0b111111 << (col * 7)
        occupied_col = occupied & col_mask
        if occupied_col & (1 << (col * 7 + 5)):
            continue

        new_piece = (occupied_col + (1 << (col * 7))) & col_mask
        
        # Bỏ qua các nước đi không nằm trong tập hợp an toàn
        if not (new_piece & safe_moves_mask):
            continue

        if first_child:
            res = -pvs(opp, me | new_piece, depth - 1, -beta, -alpha, deadline)
            first_child = False
        else:
            res = -pvs(opp, me | new_piece, depth - 1, -alpha - 1, -alpha, deadline)
            if alpha < res < beta:
                res = -pvs(opp, me | new_piece, depth - 1, -beta, -res, deadline)

        if res > value:
            value = res
            best_move = col
        alpha = max(alpha, value)
        if alpha >= beta:
            _record_killer(depth, col)
            break

    # Store to TT with bound type + bestMove.
    if best_move != -1:
        store_move = 6 - best_move if flip else best_move
    else:
        store_move = -1

    if value <= alpha0:
        flag = TT_UPPER
    elif value >= beta0:
        flag = TT_LOWER
    else:
        flag = TT_EXACT

    _tt_store(key64, depth, value, flag, store_move)
    return value

def agent(obs, config, timeout=2):
    print("[OpeningBookOpt] Start turn", obs.step)
    global searching_depth
    start_time = time.perf_counter()
    
    # 1. Determine thinking time budget
    # Use the passed timeout parameter if available, otherwise fallback to config.timeout
    base_timeout = timeout if timeout is not None else getattr(config, 'timeout', 2)
    overage = getattr(obs, 'remainingOverageTime', 0)
    
    # Greedy time management: Use most of the remaining overage pool if available.
    # If the opening book hits (fast path), this budget won't be consumed.
    # If it misses, we greedily search deeper using the overage pool until it runs out.
    think_time_budget = base_timeout * 0.92 + min(12, overage*3/5) # Keep 1s as absolute safety margin
    
    deadline = start_time + think_time_budget

    me, opp = encode(obs.board, obs.mark)
    
    # Nạp Opening Book nếu chưa nạp
    load_opening_book()
    
    # 2. Instant win check
    win_col = _find_winning_move(me, opp)
    if win_col != -1:
        print(f"[Instant Win] Playing column {win_col}")
        _log_move(win_col, start_time)
        return win_col
    
    # 3. Forced block check (opponent wins next turn)
    block_col, must_block = _find_forced_block(me, opp)
    if must_block:
        # Verify the block doesn't immediately lose
        # If only one blocking move, we must play it
        valid = _get_valid_moves(me, opp)
        if block_col in valid:
            print(f"[Forced Block] Must block opponent at column {block_col}")
            _log_move(block_col, start_time)
            return block_col
    
    # 4. Query Opening Book (Fast Path) - key is (me, opp) tuple
    # best_move = check_book(me, opp)
    # if best_move is not None:
    #     print(f"[Opening Book Hit] Playing precomputed move: {best_move}")
    #     _log_move(best_move, start_time)
    #     return int(best_move)
        
    # 5. Query TT for a move ordering hint
    key64, flip = _canonical_tt_key(me, opp)
    tt_hint_move = -1
    idx = key64 & TT_BUCKET_MASK
    bucket = tt.get(idx)
    if bucket:
        best_d = -1
        for k, d, v, flag, bm in bucket:
            if k == key64 and bm != -1 and d > best_d:
                tt_hint_move = bm
                best_d = d
        if tt_hint_move != -1 and flip:
            tt_hint_move = 6 - tt_hint_move

    valid_moves = _get_valid_moves(me, opp)
    if not valid_moves: return 0
    
    center_col = config.columns // 2
    # Use TT hint as initial best_move for move ordering if available, otherwise center
    best_move = tt_hint_move if (tt_hint_move != -1 and tt_hint_move in valid_moves) else min(valid_moves, key=lambda c: abs(c - center_col))
    reachedDepth = 0
    
    # First turn search goes much deeper to seed the entire early-game TT tree
    max_search_depth = min(24, 42 - (me | opp).bit_count())
    
    try:
        for depth in range(0, max_search_depth, 2):
            searching_depth = depth
            best_score = NNF
            move_at_this_depth = best_move
            scores = [NNF] * config.columns
            moves = [best_move] + [m for m in valid_moves if m != best_move]
            
            for col in moves:
                if time.perf_counter() > deadline:
                    raise TimeoutError
                
                new_piece = _make_move(me, opp, col)
                if not new_piece:
                    continue
                
                if is_win(me | new_piece):
                    _log_move(col, start_time)
                    return col
                
                if col == moves[0]:
                    score = -pvs(opp, me | new_piece, depth, NNF, INF, deadline)
                else:
                    if best_score == NNF:
                        score = -pvs(opp, me | new_piece, depth, NNF, INF, deadline)
                    else:
                        score = -pvs(opp, me | new_piece, depth, -best_score - 1, -best_score, deadline)
                        if best_score < score < INF:
                            score = -pvs(opp, me | new_piece, depth, NNF, INF, deadline)

                scores[col] = score
                if score > best_score:
                    best_score = score
                    move_at_this_depth = col
                    
            best_move = move_at_this_depth
            print("At depth:", depth, "Best move:", best_move, scores)
            reachedDepth = depth
            if best_score >= MATE_SCORE - 42:
                break  # Found forced win

    except TimeoutError:
        pass
        
    think_time = time.perf_counter() - start_time
    print(f"[OpeningBook_opt] depth {reachedDepth}, move {best_move}, time {think_time:.3f}s")
    _log_move(best_move, start_time)
    return int(best_move)


def _log_move(move, start_time):
    think_time = time.perf_counter() - start_time
    if log_system:
        try:
            log_system.log_move("OpeningBookOpt", int(move), think_time)
        except Exception:
            pass
