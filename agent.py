"""
ConnectX Agent — Layer 2 + Opening Book (L3)
=============================================
Kiến trúc:
  [L1A] Thắng ngay                   (O(cols), ~0.0001s)
  [L1B] Chặn đối thủ thắng ngay      (O(cols), ~0.0001s)
  [L3]  Opening Book lookup           (O(1),    ~0.00001s) ← MỚI
  [L2]  Negamax + Alpha-Beta + ID     (~1.75s)

Opening Book:
  - 267 entries, cover 6 nước đầu (3 nước mỗi bên)
  - Tính trước bằng Negamax 1.8s/position offline
  - Encode bằng bitboard (b1, b2) → compact, lookup O(1)
  - Tiết kiệm 1.75s × 6 nước = ~10.5s/ván cho giai đoạn mở

Deadline Timeout (fix từ phiên bản cũ):
  - _Timeout exception ném ra mỗi N_CHECK=512 node
  - Đảm bảo không bao giờ vượt quá TIME_LIMIT
"""

import time

# ══════════════════════════════════════════════════════════════
# HẰNG SỐ
# ══════════════════════════════════════════════════════════════

WIN_SCORE  = 10_000_000
TIME_LIMIT = 1.75
N_CHECK    = 512

_deadline   = 0.0
_node_count = 0

class _Timeout(Exception):
    pass


# ══════════════════════════════════════════════════════════════
# PHẦN 1 — OPENING BOOK (267 entries, bitboard-encoded)
# ══════════════════════════════════════════════════════════════
#
# Cấu trúc: {(bitboard_mark1, bitboard_mark2): best_col}
#   bitboard_mark1: bitmask các ô có quân mark=1 (bit i = ô thứ i trong board 1D)
#   bitboard_mark2: bitmask các ô có quân mark=2
#   best_col: cột tốt nhất cho người sắp đi
#
# Được tính bằng Negamax 1.8s/position, cov er 6 nước đầu (pieces ≤ 6):
#   pieces=0 (empty): mark=1 đi
#   pieces=1:         mark=2 đi
#   pieces=2:         mark=1 đi
#   pieces=3:         mark=2 đi
#   pieces=4:         mark=1 đi
#   pieces=6:         mark=1 đi (các đường tiếp theo)

OPENING_BOOK = {
    # 0 quân đã đi (1 entries)
    (0,0):3,

    # 1 quân đã đi (7 entries)
    (34359738368,0):3,
    (68719476736,0):3,
    (137438953472,0):3,
    (274877906944,0):3,
    (549755813888,0):3,
    (1099511627776,0):3,
    (2199023255552,0):3,

    # 2 quân đã đi (7 entries)
    (274877906944,2147483648):3,
    (274877906944,34359738368):4,
    (274877906944,68719476736):3,
    (274877906944,137438953472):3,
    (274877906944,549755813888):3,
    (274877906944,1099511627776):3,
    (274877906944,2199023255552):2,

    # 3 quân đã đi (40 entries)
    (34628173824,274877906944):3,
    (36507222016,274877906944):4,
    (69256347648,274877906944):1,
    (70866960384,274877906944):1,
    (103079215104,274877906944):3,
    (138512695296,274877906944):2,
    (139586437120,274877906944):3,
    (171798691840,274877906944):3,
    (206158430208,274877906944):3,
    (274894684160,2147483648):3,
    (277025390592,68719476736):3,
    (277025390592,137438953472):3,
    (277025390592,549755813888):3,
    (277025390592,1099511627776):3,
    (309237645312,2147483648):4,
    (343597383680,2147483648):4,
    (412316860416,2147483648):4,
    (412316860416,2199023255552):4,
    (551903297536,274877906944):3,
    (554050781184,274877906944):4,
    (584115552256,274877906944):3,
    (618475290624,274877906944):3,
    (687194767360,274877906944):3,
    (824633720832,2147483648):2,
    (824633720832,34359738368):2,
    (1101659111424,274877906944):5,
    (1108101562368,274877906944):2,
    (1133871366144,274877906944):3,
    (1168231104512,274877906944):3,
    (1236950581248,274877906944):3,
    (1374389534720,2147483648):2,
    (1649267441664,274877906944):3,
    (2201170739200,274877906944):2,
    (2216203124736,274877906944):3,
    (2233382993920,274877906944):3,
    (2267742732288,274877906944):3,
    (2336462209024,274877906944):3,
    (2473901162496,2147483648):4,
    (2748779069440,274877906944):3,
    (3298534883328,274877906944):3,

    # 4 quân đã đi (43 entries)
    (274894684160,2147614720):1,
    (274894684160,36507222016):3,
    (274894684160,70866960384):3,
    (274894684160,139586437120):3,
    (274894684160,551903297536):3,
    (274894684160,1101659111424):1,
    (274894684160,2201170739200):1,
    (277025390592,68736253952):3,
    (277025390592,69256347648):4,
    (277025390592,103079215104):4,
    (277025390592,137455730688):2,
    (277025390592,138512695296):2,
    (277025390592,171798691840):2,
    (277025390592,206158430208):2,
    (277025390592,549772591104):4,
    (277025390592,554050781184):4,
    (277025390592,584115552256):3,
    (277025390592,618475290624):3,
    (277025390592,687194767360):3,
    (277025390592,1099528404992):3,
    (277025390592,1108101562368):2,
    (277025390592,1133871366144):1,
    (277025390592,1168231104512):1,
    (277025390592,1236950581248):3,
    (277025390592,1649267441664):4,
    (277025390592,2267742732288):5,
    (277025390592,2336462209024):3,
    (277025390592,2748779069440):4,
    (277025390592,3298534883328):2,
    (412316860416,2200096997376):1,
    (412316860416,2201170739200):4,
    (412316860416,2216203124736):4,
    (412316860416,2233382993920):4,
    (412316860416,2267742732288):3,
    (412316860416,2748779069440):3,
    (412316860416,3298534883328):1,
    (824633720832,34628173824):2,
    (824633720832,36507222016):2,
    (824633720832,38654705664):5,
    (824633720832,103079215104):5,
    (824633720832,171798691840):3,
    (824633720832,1133871366144):3,
    (824633720832,2233382993920):2,

    # 6 quân đã đi (169 entries)
    (274894815232,36507223040):4,
    (274894815232,36775657472):5,
    (274894815232,70866961408):1,
    (274894815232,71403831296):5,
    (274894815232,105226698752):5,
    (274894815232,139586438144):2,
    (274894815232,140660178944):2,
    (274894815232,173946175488):2,
    (274894815232,208305913856):1,
    (274894815232,551903298560):4,
    (274894815232,556198264832):4,
    (274894815232,586263035904):5,
    (274894815232,620622774272):5,
    (274894815232,689342251008):2,
    (274894815232,1136018849792):1,
    (274894815232,1170378588160):1,
    (274894815232,1239098064896):1,
    (274894815232,1651414925312):5,
    (274894815232,2235530477568):2,
    (274894815232,2269890215936):5,
    (274894815232,2338609692672):1,
    (274894815232,2750926553088):4,
    (277025521664,68736254976):3,
    (277025521664,69273124864):1,
    (277025521664,103095992320):4,
    (277025521664,206175207424):2,
    (277025521664,618492067840):4,
    (277025521664,1168247881728):3,
    (277025521664,2267759509504):3,
    (277033779200,138512760832):4,
    (277033779200,138529472512):4,
    (277033779200,172872433664):3,
    (277033779200,207232172032):3,
    (277033779200,688268509184):3,
    (277033779200,1238024323072):3,
    (277033779200,2337535950848):3,
    (277042167808,584115683328):3,
    (277042167808,584383987712):3,
    (277042167808,588410519552):3,
    (277042167808,618475421696):1,
    (277042167808,619012161536):3,
    (277042167808,622770257920):3,
    (277042167808,652835028992):3,
    (277042167808,687194898432):2,
    (277042167808,688268509184):3,
    (277042167808,691489734656):3,
    (277042167808,721554505728):3,
    (277042167808,755914244096):3,
    (277042167808,1236950712320):5,
    (277042167808,1238024323072):3,
    (277042167808,1245540515840):3,
    (277042167808,1271310319616):3,
    (277042167808,1305670057984):3,
    (277042167808,1683627180032):3,
    (277042167808,1717986918400):3,
    (277042167808,1786706395136):3,
    (277042167808,2336462340096):3,
    (277042167808,2337535950848):3,
    (277042167808,2353642078208):3,
    (277042167808,2370821947392):3,
    (277042167808,2405181685760):3,
    (277042167808,2817498546176):3,
    (277042167808,2886218022912):3,
    (277042167808,3435973836800):3,
    (277562261504,1168235298816):2,
    (277562261504,1168247881728):3,
    (277562261504,1176821039104):0,
    (277562261504,1202590842880):3,
    (277562261504,1305670057984):1,
    (277562261504,1717986918400):3,
    (277562261504,3367254360064):0,
    (278099132416,137455861760):6,
    (278099132416,137464119296):5,
    (278099132416,171807080448):2,
    (278099132416,171815469056):3,
    (278099132416,172067127296):4,
    (278099132416,206166818816):1,
    (278099132416,206175207424):1,
    (278099132416,206695301120):3,
    (278099132416,240518168576):3,
    (278099132416,687211544576):4,
    (278099132416,721554505728):3,
    (278099132416,755914244096):4,
    (278099132416,1236967358464):3,
    (278099132416,1271310319616):3,
    (278099132416,1305670057984):1,
    (278099132416,2336478986240):3,
    (278099132416,2370821947392):4,
    (278099132416,2405181685760):3,
    (343614160896,2147615744):2,
    (343614160896,2684485632):2,
    (343614160896,36507353088):1,
    (343614160896,139586568192):1,
    (343614160896,551903428608):1,
    (343614160896,1101659242496):2,
    (343614160896,1102195982336):2,
    (343614160896,1110249046016):2,
    (343614160896,1136018849792):4,
    (343614160896,1239098064896):3,
    (343614160896,1651414925312):4,
    (343614160896,2201170870272):2,
    (343614160896,2201707610112):2,
    (343614160896,2218350608384):2,
    (343614160896,2235530477568):4,
    (343614160896,2338609692672):1,
    (343614160896,2750926553088):4,
    (343614160896,3300682366976):2,
    (826781204480,69260541952):1,
    (826781204480,69273124864):5,
    (826781204480,73551314944):5,
    (826781204480,103095992320):5,
    (826781204480,103347650560):5,
    (826781204480,103616086016):5,
    (826781204480,107374182400):5,
    (826781204480,171815469056):2,
    (826781204480,172067127296):3,
    (826781204480,172872433664):3,
    (826781204480,176093659136):3,
    (826781204480,206695301120):1,
    (826781204480,240518168576):4,
    (826781204480,1133888143360):4,
    (826781204480,1134139801600):2,
    (826781204480,1138166333440):3,
    (826781204480,1142461300736):2,
    (826781204480,1168767975424):4,
    (826781204480,1202590842880):4,
    (826781204480,1271310319616):4,
    (826781204480,2268279603200):4,
    (826781204480,2302102470656):4,
    (826781204480,2370821947392):4,
    (826781204480,3332894621696):4,
    (962072674304,34630270976):1,
    (962072674304,35701915648):1,
    (962072674304,36523999232):1,
    (962072674304,36775657472):1,
    (962072674304,37580963840):1,
    (962072674304,38923141120):1,
    (962072674304,40802189312):1,
    (962072674304,103347650560):5,
    (962072674304,105226698752):5,
    (962072674304,1134139801600):1,
    (962072674304,1136018849792):1,
    (962072674304,2233651429376):1,
    (962072674304,2234456735744):1,
    (962072674304,2235530477568):1,
    (962072674304,2237677961216):1,
    (962072674304,2250562863104):1,
    (962072674304,2302102470656):5,
    (962072674304,3332894621696):1,
    (1376537018368,2267759509504):3,
    (1376537018368,2268279603200):4,
    (1376537018368,2276332666880):1,
    (1376537018368,2284922601472):4,
    (1376537018368,2302102470656):4,
    (1376537018368,2405181685760):2,
    (1376537018368,2817498546176):3,
    (1924145348608,38688260096):2,
    (1924145348608,38923141120):2,
    (1924145348608,40802189312):2,
    (1924145348608,47244640256):2,
    (1924145348608,103347650560):2,
    (1924145348608,103616086016):2,
    (1924145348608,105226698752):2,
    (1924145348608,107374182400):2,
    (1924145348608,111669149696):2,
    (1924145348608,176093659136):6,
    (1924145348608,240518168576):6,
    (1924145348608,2237677961216):2,
    (1924145348608,2302102470656):2,
}


def _board_key(board):
    """
    Encode board thành (bitboard_mark1, bitboard_mark2).
    Mỗi bit i tương ứng với ô thứ i trong mảng 1D của Kaggle.
    O(n) nhưng n=42 → cực nhanh.
    """
    b1 = b2 = 0
    for i, c in enumerate(board):
        if   c == 1: b1 |= (1 << i)
        elif c == 2: b2 |= (1 << i)
    return (b1, b2)


def get_book_move(board):
    """
    Tra cứu Opening Book. Trả về col hoặc None nếu không có trong sách.
    O(1) lookup — cực kỳ nhanh (< 0.00001s).

    Lưu ý: book lưu best_col cho người SẮP đi tại vị thế đó.
    Trong ConnectX, người sắp đi được xác định bởi số quân trên bàn:
      - pieces chẵn → mark=1 đi
      - pieces lẻ  → mark=2 đi
    Không cần truyền mark vào — book đã mã hóa đúng.
    """
    key = _board_key(board)
    col = OPENING_BOOK.get(key)
    # Kiểm tra nước đi hợp lệ (cột chưa đầy)
    if col is not None and board[col] == 0:
        return col
    return None


# ══════════════════════════════════════════════════════════════
# PHẦN 2 — CÔNG CỤ DÙNG CHUNG
# ══════════════════════════════════════════════════════════════

def get_cell(board, row, col, config):
    return board[row * config.columns + col]


def drop_piece(board, col, mark, config):
    """Thả quân vào cột. Trả về board mới hoặc None nếu cột đầy."""
    next_board = list(board)
    for row in range(config.rows - 1, -1, -1):
        if next_board[row * config.columns + col] == 0:
            next_board[row * config.columns + col] = mark
            return next_board
    return None


def is_winning_board(board, mark, config):
    """Kiểm tra board có phải trạng thái thắng của mark không."""
    rows, cols, k = config.rows, config.columns, config.inarow

    for r in range(rows):                           # Ngang
        for c in range(cols - k + 1):
            if all(board[r*cols+c+i] == mark for i in range(k)):
                return True

    for r in range(rows - k + 1):                  # Dọc
        for c in range(cols):
            if all(board[(r+i)*cols+c] == mark for i in range(k)):
                return True

    for r in range(rows - k + 1):                  # Chéo xuôi
        for c in range(cols - k + 1):
            if all(board[(r+i)*cols+c+i] == mark for i in range(k)):
                return True

    for r in range(k - 1, rows):                   # Chéo ngược
        for c in range(cols - k + 1):
            if all(board[(r-i)*cols+c+i] == mark for i in range(k)):
                return True

    return False


def is_winning_move(board, col, mark, config):
    nb = drop_piece(board, col, mark, config)
    return nb is not None and is_winning_board(nb, mark, config)


# ══════════════════════════════════════════════════════════════
# PHẦN 3 — HEURISTIC (leaf node evaluation)
# ══════════════════════════════════════════════════════════════

def score_window(window, mark, opp):
    mc = window.count(mark)
    oc = window.count(opp)
    ec = window.count(0)

    if oc > 0 and mc > 0: return 0   # cửa sổ hỗn hợp

    if mc == 4: return  1000
    if mc == 3: return    10 if ec == 1 else 0
    if mc == 2: return     1 if ec == 2 else 0

    if oc == 3: return  -100 if ec == 1 else 0
    if oc == 2: return    -2 if ec == 2 else 0

    return 0


def heuristic_board(board, mark, config):
    """Đánh giá board từ góc nhìn mark. Dùng tại leaf node của negamax."""
    rows, cols, k = config.rows, config.columns, config.inarow
    opp   = 3 - mark
    score = 0
    ctr   = cols // 2

    for r in range(rows):
        cell = board[r * cols + ctr]
        if cell == mark: score += 3
        elif cell == opp: score -= 3

    for r in range(rows):
        for c in range(cols - k + 1):
            w = [board[r*cols+c+i] for i in range(k)]
            score += score_window(w, mark, opp)

    for r in range(rows - k + 1):
        for c in range(cols):
            w = [board[(r+i)*cols+c] for i in range(k)]
            score += score_window(w, mark, opp)

    for r in range(rows - k + 1):
        for c in range(cols - k + 1):
            w = [board[(r+i)*cols+c+i] for i in range(k)]
            score += score_window(w, mark, opp)

    for r in range(k - 1, rows):
        for c in range(cols - k + 1):
            w = [board[(r-i)*cols+c+i] for i in range(k)]
            score += score_window(w, mark, opp)

    return score


# ══════════════════════════════════════════════════════════════
# PHẦN 4 — NEGAMAX + ALPHA-BETA + DEADLINE TIMEOUT
# ══════════════════════════════════════════════════════════════

def move_order(valid_moves, config):
    """Ưu tiên cột gần trung tâm → alpha-beta pruning hiệu quả hơn."""
    center = config.columns // 2
    return sorted(valid_moves, key=lambda c: abs(c - center))


def negamax(board, depth, alpha, beta, mark, config):
    """
    Negamax + Alpha-Beta + Deadline timeout.

    - Kiểm tra deadline mỗi N_CHECK=512 node (giảm overhead time.time())
    - Ném _Timeout khi hết giờ → thoát đệ quy ngay lập tức
    - WIN_SCORE + depth: thắng sớm hơn được ưu tiên hơn
    """
    global _node_count

    _node_count += 1
    if _node_count % N_CHECK == 0 and time.time() >= _deadline:
        raise _Timeout()

    opp         = 3 - mark
    valid_moves = [c for c in range(config.columns) if board[c] == 0]

    if not valid_moves:
        return 0

    if depth == 0:
        return heuristic_board(board, mark, config)

    for col in move_order(valid_moves, config):
        nb = drop_piece(board, col, mark, config)

        if is_winning_board(nb, mark, config):
            return WIN_SCORE + depth

        score = -negamax(nb, depth - 1, -beta, -alpha, opp, config)

        if score > alpha:
            alpha = score
        if alpha >= beta:
            break

    return alpha


def find_best_move(board, mark, config):
    """
    Iterative Deepening với deadline timeout an toàn.
    Chỉ cập nhật best_move khi một depth được hoàn chỉnh 100%.
    Tái sử dụng best_move của depth trước để tối ưu move ordering.
    """
    global _deadline, _node_count

    valid_moves = [c for c in range(config.columns) if board[c] == 0]
    ordered     = move_order(valid_moves, config)
    best_move   = ordered[0]
    opp         = 3 - mark
    max_depth   = config.rows * config.columns

    _deadline   = time.time() + TIME_LIMIT
    _node_count = 0

    for depth in range(1, max_depth + 1):

        if time.time() >= _deadline:
            break

        d_best_score = -(WIN_SCORE + 1)
        d_best_move  = best_move
        alpha        = -(WIN_SCORE + 1)
        beta         =  (WIN_SCORE + 1)
        completed    = True

        for col in ordered:
            nb = drop_piece(board, col, mark, config)

            if is_winning_board(nb, mark, config):
                return col

            try:
                score = -negamax(nb, depth - 1, -beta, -alpha, opp, config)
            except _Timeout:
                completed = False
                break

            if score > d_best_score:
                d_best_score = score
                d_best_move  = col
            if score > alpha:
                alpha = score

        if completed:
            best_move = d_best_move
            ordered   = [best_move] + [c for c in ordered if c != best_move]
            if d_best_score >= WIN_SCORE:
                break

    return best_move


# ══════════════════════════════════════════════════════════════
# PHẦN 5 — BỘ NÃO CHÍNH (hàm Kaggle gọi)
# ══════════════════════════════════════════════════════════════

def my_agent(observation, configuration):
    board  = observation.board
    mark   = observation.mark
    config = configuration
    opp    = 3 - mark

    valid_moves = [c for c in range(config.columns) if board[c] == 0]

    # ── L1A: Thắng ngay ─────────────────────────────────────────────────────
    for col in valid_moves:
        if is_winning_move(board, col, mark, config):
            return col

    # ── L1B: Chặn đối thủ thắng ngay ────────────────────────────────────
    for col in valid_moves:
        if is_winning_move(board, col, opp, config):
            return col

    # ── L3: Opening Book (O(1) lookup, ~0.00001s) ────────────────────────
    book_col = get_book_move(board)
    if book_col is not None:
        return book_col

    # ── L2: Negamax + Alpha-Beta + Iterative Deepening (~1.75s) ─────────
    return find_best_move(board, mark, config)