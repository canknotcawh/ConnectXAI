"""
FOUNDATION LAYER - Các hàm cơ bản không phụ thuộc
================================================
Những hàm này là nền tảng, không gọi hàm khác.
Học những hàm này trước để hiểu game mechanics.
"""

import numpy as np
import time

# class SearchTimeout(Exception):
#     """Raised when minimax search exceeds the global time budget."""


MAX_THINK_TIME = 1.85
INF = 1000000
NNF = -1000000
MATE_SCORE = 100000


class Config:
    def __init__(self, row, column, inarrow):
        self.rows = row
        self.columns = column
        self.inarow = inarrow
        
config = Config(6, 7, 4)

def drop_piece(grid, col, mark):
    """
    Simulate: Thả 1 quân vào cột col và trả về board mới.
    
    Ý nghĩa:
    - Tìm hàng trống thấp nhất trong cột (vật lý trò chơi)
    - Đặt quân (mark=1 hoặc 2) vào vị trí đó
    - Không thay đổi board gốc (trả về board copy)
    
    Phụ thuộc: Không gọi hàm khác
    Được gọi bởi: score_move_a(), score_move_b()
    
    Ví dụ:
    >>> grid = np.array([[0,0,0],[0,0,0],[0,0,0]])
    >>> result = drop_piece(grid, 1, 1)
    >>> result[2,1] == 1  # Quân rơi xuống hàng cuối
    True
    """
    next_grid = grid.copy()
    for row in range(config.rows-1, -1, -1):
        if next_grid[row][col] == 0:
            break
    next_grid[row][col] = mark
    return next_grid


def check_window(window, piece, config):
    if window.count((piece%2)+1)==0:
        return window.count(piece)
    else:
        return -1
    
# def check_timeout():
#     if time.perf_counter() >= SEARCH_DEADLINE:
#         raise SearchTimeout()
# def is_timeout():
#     return time.perf_counter() >= SEARCH_DEADLINE


def encode(board, mark):
    """Convert 1D board list to two bitboards: current player and opponent."""
    me = 0
    opp = 0
    for c in range(config.columns):
        for r in range(config.rows):
            val = board[r * config.columns + c]
            if val == 0:
                continue
            bit = 1 << (c * 7 + (config.rows - 1 - r))
            if val == mark:
                me |= bit
            else:
                opp |= bit
    return me, opp


def is_win(b):
    """Check win on bitboard using bitwise pattern matching."""
    m = b & (b << 7)
    if m & (m << 14):
        return True

    m = b & (b << 1)
    if m & (m << 2):
        return True

    m = b & (b << 8)
    if m & (m << 16):
        return True

    m = b & (b << 6)
    if m & (m << 12):
        return True

    return False