"""
ConnectX Agent — Layer 2 hoàn chỉnh
=====================================
Kiến trúc:
  [L1A] Thắng ngay                  (O(cols))
  [L1B] Chặn đối thủ thắng ngay     (O(cols))
  [L2]  Negamax + Alpha-Beta + ID    (iterative deepening với deadline timeout)

Fix so với phiên bản trước:
  - _Timeout được ném ngay trong lòng negamax() → không bao giờ vượt quá time_limit
  - Kiểm tra deadline mỗi N_CHECK node để giảm overhead của time.time()
  - move_order() ưu tiên cột giữa → alpha-beta cắt nhiều nhánh hơn
  - Kết quả depth hoàn chỉnh mới được chấp nhận (safe iterative deepening)
"""

import time

# ══════════════════════════════════════════════════════════════
# HẰNG SỐ & KIỂM SOÁT THỜI GIAN
# ══════════════════════════════════════════════════════════════

WIN_SCORE  = 10_000_000     # điểm thắng tuyệt đối
TIME_LIMIT = 1.75           # giây (để lại buffer 0.25s cho Kaggle overhead)
N_CHECK    = 512            # kiểm tra deadline mỗi N_CHECK node (giảm overhead)

_deadline    = 0.0          # module-level, set trước mỗi lần search
_node_count  = 0            # đếm node, reset mỗi ván

class _Timeout(Exception):
    """Ném khi hết thời gian tìm kiếm."""
    pass


# ══════════════════════════════════════════════════════════════
# PHẦN 1 — CÔNG CỤ DÙNG CHUNG
# ══════════════════════════════════════════════════════════════

def get_cell(board, row, col, config):
    return board[row * config.columns + col]


def drop_piece(board, col, mark, config):
    """Thả quân vào cột, trả về board mới (tuple để hash nhanh hơn list)."""
    next_board = list(board)
    for row in range(config.rows - 1, -1, -1):
        if next_board[row * config.columns + col] == 0:
            next_board[row * config.columns + col] = mark
            return next_board
    return None   # cột đầy


def is_winning_board(board, mark, config):
    """Kiểm tra board có phải trạng thái thắng của mark không."""
    rows, cols, k = config.rows, config.columns, config.inarow

    for r in range(rows):                        # Ngang
        for c in range(cols - k + 1):
            if all(board[r * cols + c + i] == mark for i in range(k)):
                return True

    for r in range(rows - k + 1):               # Dọc
        for c in range(cols):
            if all(board[(r + i) * cols + c] == mark for i in range(k)):
                return True

    for r in range(rows - k + 1):               # Chéo xuôi
        for c in range(cols - k + 1):
            if all(board[(r + i) * cols + c + i] == mark for i in range(k)):
                return True

    for r in range(k - 1, rows):                # Chéo ngược
        for c in range(cols - k + 1):
            if all(board[(r - i) * cols + c + i] == mark for i in range(k)):
                return True

    return False


def is_winning_move(board, col, mark, config):
    nb = drop_piece(board, col, mark, config)
    return nb is not None and is_winning_board(nb, mark, config)


# ══════════════════════════════════════════════════════════════
# PHẦN 2 — HEURISTIC (leaf node evaluation)
# ══════════════════════════════════════════════════════════════

def score_window(window, mark, opp):
    """Chấm điểm 1 cửa sổ 4 ô (zero-sum: trừ điểm đối thủ)."""
    mc = window.count(mark)
    oc = window.count(opp)
    ec = window.count(0)

    if oc > 0 and mc > 0:    # cửa sổ hỗn hợp → vô dụng cho cả hai
        return 0

    if mc == 4: return  1000
    if mc == 3: return    10  if ec == 1 else 0
    if mc == 2: return     1  if ec == 2 else 0

    if oc == 3: return  -100  if ec == 1 else 0
    if oc == 2: return    -2  if ec == 2 else 0

    return 0


def heuristic_board(board, mark, config):
    """
    Đánh giá toàn bộ board từ góc nhìn của mark.
    Dùng tại leaf node của negamax.

    Điểm dương = tốt cho mark, âm = tốt cho đối thủ.
    """
    rows, cols, k = config.rows, config.columns, config.inarow
    opp    = 3 - mark   # 1↔2
    score  = 0
    center = cols // 2

    # Center column bonus
    for r in range(rows):
        cell = board[r * cols + center]
        if cell == mark: score += 3
        elif cell == opp: score -= 3

    # Quét tất cả cửa sổ k ô
    for r in range(rows):                        # Ngang
        for c in range(cols - k + 1):
            w = [board[r * cols + c + i] for i in range(k)]
            score += score_window(w, mark, opp)

    for r in range(rows - k + 1):               # Dọc
        for c in range(cols):
            w = [board[(r + i) * cols + c] for i in range(k)]
            score += score_window(w, mark, opp)

    for r in range(rows - k + 1):               # Chéo xuôi
        for c in range(cols - k + 1):
            w = [board[(r + i) * cols + c + i] for i in range(k)]
            score += score_window(w, mark, opp)

    for r in range(k - 1, rows):                # Chéo ngược
        for c in range(cols - k + 1):
            w = [board[(r - i) * cols + c + i] for i in range(k)]
            score += score_window(w, mark, opp)

    return score


# ══════════════════════════════════════════════════════════════
# PHẦN 3 — NEGAMAX + ALPHA-BETA + DEADLINE TIMEOUT
# ══════════════════════════════════════════════════════════════

def move_order(valid_moves, config):
    """Sắp xếp cột gần trung tâm trước → alpha-beta pruning hiệu quả hơn."""
    center = config.columns // 2
    return sorted(valid_moves, key=lambda c: abs(c - center))


def negamax(board, depth, alpha, beta, mark, config):
    """
    Negamax + Alpha-Beta pruning với deadline timeout.

    ┌─ Khi đạt depth=0: trả về heuristic (leaf node)
    ├─ Khi thắng: trả về WIN_SCORE + depth (thắng sớm = điểm cao hơn)
    ├─ Khi hòa: trả về 0
    └─ Khi hết giờ: ném _Timeout (được catch ở find_best_move)

    Alpha-Beta:
      alpha = điểm tốt nhất mà MAX đảm bảo được
      beta  = điểm tốt nhất mà MIN đảm bảo được
      Khi alpha >= beta → đối thủ sẽ không cho phép nhánh này → cắt
    """
    global _node_count

    # Kiểm tra deadline mỗi N_CHECK node (tránh gọi time.time() liên tục)
    _node_count += 1
    if _node_count % N_CHECK == 0 and time.time() >= _deadline:
        raise _Timeout()

    opp         = 3 - mark
    valid_moves = [c for c in range(config.columns) if board[c] == 0]

    # Terminal states
    if not valid_moves:
        return 0                                  # Hòa (bàn đầy)

    if depth == 0:
        return heuristic_board(board, mark, config)  # Leaf: dùng heuristic

    for col in move_order(valid_moves, config):
        nb = drop_piece(board, col, mark, config)

        # Fast path: nước này thắng ngay → không cần đệ quy thêm
        if is_winning_board(nb, mark, config):
            return WIN_SCORE + depth              # Thắng sớm = điểm cao hơn

        # Đệ quy: đối thủ đi, negate điểm (zero-sum)
        score = -negamax(nb, depth - 1, -beta, -alpha, opp, config)

        if score > alpha:
            alpha = score
        if alpha >= beta:
            break                                 # Beta cutoff → pruning

    return alpha


def find_best_move(board, mark, config):
    """
    Iterative Deepening với deadline timeout an toàn.

    Thuật toán:
      depth = 1, 2, 3, ... cho đến khi hết thời gian
      - Mỗi depth hoàn chỉnh → cập nhật best_move
      - Nếu hết giờ giữa chừng → giữ kết quả depth trước (đã hoàn chỉnh)
      - _Timeout từ negamax được catch, depth bị đánh dấu không hoàn chỉnh

    Move ordering tái sử dụng:
      - Sau mỗi depth, đặt best_move lên đầu ordered[]
      - Depth sau thử nước tốt nhất trước → alpha-beta hiệu quả hơn ~30%
    """
    global _deadline, _node_count

    valid_moves = [c for c in range(config.columns) if board[c] == 0]
    ordered     = move_order(valid_moves, config)
    best_move   = ordered[0]      # fallback: cột gần trung tâm nhất
    opp         = 3 - mark
    max_depth   = config.rows * config.columns

    _deadline   = time.time() + TIME_LIMIT
    _node_count = 0

    for depth in range(1, max_depth + 1):

        if time.time() >= _deadline:
            break                             # Hết giờ trước khi bắt đầu depth mới

        d_best_score = -(WIN_SCORE + 1)
        d_best_move  = best_move
        alpha        = -(WIN_SCORE + 1)
        beta         =  (WIN_SCORE + 1)
        completed    = True

        for col in ordered:
            nb = drop_piece(board, col, mark, config)

            if is_winning_board(nb, mark, config):
                return col                    # Thắng ngay: trả về luôn

            try:
                score = -negamax(nb, depth - 1, -beta, -alpha, opp, config)
            except _Timeout:
                completed = False
                break                         # Hết giờ giữa depth → bỏ depth này

            if score > d_best_score:
                d_best_score = score
                d_best_move  = col
            if score > alpha:
                alpha = score

        if completed:
            best_move = d_best_move
            # Đặt nước tốt nhất lên đầu → depth sau pruning hiệu quả hơn
            ordered = [best_move] + [c for c in ordered if c != best_move]
            if d_best_score >= WIN_SCORE:
                break                         # Thắng chắc → không cần tìm thêm

    return best_move


# ══════════════════════════════════════════════════════════════
# PHẦN 4 — BỘ NÃO CHÍNH (hàm Kaggle gọi)
# ══════════════════════════════════════════════════════════════

def my_agent(observation, configuration):
    board  = observation.board
    mark   = observation.mark
    config = configuration
    opp    = 3 - mark

    valid_moves = [c for c in range(config.columns) if board[c] == 0]

    # ── L1A: Thắng ngay ──────────────────────────────────────────────────────
    for col in valid_moves:
        if is_winning_move(board, col, mark, config):
            return col

    # ── L1B: Chặn đối thủ thắng ngay ─────────────────────────────────────
    for col in valid_moves:
        if is_winning_move(board, col, opp, config):
            return col

    # ── L2: Negamax + Alpha-Beta + Iterative Deepening ────────────────────
    return find_best_move(board, mark, config)
