# ============================================================================
# HEURISTIC EVALUATION LAYER - Đánh giá trạng thái board
# ============================================================================
from Agents.foundation import *

_BB_WINDOW_MASKS_CACHE = {}


def get_heuristic(grid, mark, config):
    """
    Tính điểm đánh giá (heuristic score) cho một trạng thái board.
    
    Ý nghĩa:
    - Cộng điểm nếu bạn có nhiều quân liên tiếp (cơ hội thắng)
    - Trừ điểm nếu đối thủ có nhiều quân liên tiếp (nguy hiểm)
    - Sử dụng cấp số nhân: 2 quân = 4^2, 3 quân = 4^3, 4 quân = 4^4
    
    Phụ thuộc: count_windows()
    """
    score = 0
    num = count_windows(grid,mark,config)
    for i in range(config.inarow):
        if (i==(config.inarow-1) and num[i+1] >= 1):
            return INF
        score += (4**(i))*num[i+1]
    num_opp = count_windows (grid,mark%2+1,config)
    for i in range(config.inarow):
        if (i==(config.inarow-1) and num_opp[i+1] >= 1):
            return float ("-inf")
        score -= (2**((2*i)+1))*num_opp[i+1]
    return score


def count_windows(grid, piece, config):
    num_windows = [0] * (config.inarow + 1)
    # horizontal
    for row in range(config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[row, col:col+config.inarow])
            type_window = check_window(window, piece, config)
            if type_window != -1:
                num_windows[type_window] += 1
    # vertical
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns):
            window = list(grid[row:row+config.inarow, col])
            type_window = check_window(window, piece, config)
            if type_window != -1:
                num_windows[type_window] += 1
    # positive diagonal
    for row in range(config.rows-(config.inarow-1)):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
            type_window = check_window(window, piece, config)
            if type_window != -1:
                num_windows[type_window] += 1
    # negative diagonal
    for row in range(config.inarow-1, config.rows):
        for col in range(config.columns-(config.inarow-1)):
            window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
            type_window = check_window(window, piece, config)
            if type_window != -1:
                num_windows[type_window] += 1
    return num_windows


# Giá trị cơ bản của từng loại window.
# 1 quân gần như vô hại.
# 3 quân rất mạnh nhưng vẫn chưa phải thắng chắc.
# Giá trị cơ bản của từng loại window.
WINDOW_WEIGHTS = (0, 1, 4, 30)

def get_heuristic_bb(me, opp, parity):
    """
    Hàm heuristic nâng cao sử dụng Bitboard, có tính đến:
    - Parity (Nhịp độ): Người đi đầu (0) kiểm soát hàng 0,2,4. Người đi sau (1) kiểm soát hàng 1,3,5.
    - Threat Analysis: Đánh giá cửa thắng 3 quân dựa trên tính Immediate và Parity.
    - Coi như 'opp' là người đi tiếp theo (để phòng thủ chặt chẽ).
    """
    occupied = me | opp
    ply_count = occupied.bit_count()
    
    # Parity của mỗi bên
    my_parity = parity
    opp_parity = 1 - parity

    score = 0
    
    # Duyệt toàn bộ window masks
    for mask in _get_bb_window_masks():
        me_count = (mask & me).bit_count()
        opp_count = (mask & opp).bit_count()
        
        if me_count > 0 and opp_count > 0:
            continue # Window bị chặn
        
        if me_count == 4: return MATE_SCORE - ply_count
        if opp_count == 4: return -(MATE_SCORE - ply_count)
        
        if me_count > 0:
            # Phân tích Threat của mình
            empty_mask = mask & ~me
            bit_idx = empty_mask.bit_length() - 1
            row = bit_idx % 7
            # Parity bonus: Nếu hàng này thuộc quyền kiểm soát của mình
            parity_bonus = 1
            if me_count == 3 and row % 2 == my_parity:
                parity_bonus += (5 - row)
            # Immediate check: Không có vì opp đi tiếp
            # is_immediate = (row == 0) or (occupied & (1 << (bit_idx - 1)))
            # if is_immediate:
            #     score += 10 * me_count
            score += WINDOW_WEIGHTS[me_count] * parity_bonus
                
        elif opp_count > 0:
            # Phân tích Threat của đối thủ (đang giả định opp đi tiếp)
            empty_mask = mask & ~opp
            bit_idx = empty_mask.bit_length() - 1
            row = bit_idx % 7
            # Parity penalty
            parity_penalty = 1
            if opp_count == 3 and row % 2 == opp_parity:
                parity_penalty += (5 - row)
            # Immediate penalty: CỰC KỲ NGUY HIỂM nếu opp đi tiếp
            is_immediate = (row == 0) or (occupied & (1 << (bit_idx - 1)))
            if is_immediate:
                score -= 20 * opp_count
            score -= WINDOW_WEIGHTS[opp_count] * parity_penalty * 2

    return score

def _get_bb_window_masks():
    """
    Sinh toàn bộ mask của các window 4 ô trên board bitboard.

    Vì connectX ở đây luôn là board cố định 6x7, số window hợp lệ là cố định.
    Do đó hàm này:
    - tạo mask một lần
    - lưu vào cache `_BB_WINDOW_MASKS_CACHE`
    - các lần gọi sau chỉ lấy lại từ cache để tiết kiệm thời gian

    Mỗi mask là một window 4 ô theo một trong 4 hướng:
    - ngang
    - dọc
    - chéo xuôi
    - chéo ngược

    `count_windows_bb()` sẽ dùng các mask này để đếm:
    - window nào chỉ có quân của `me`
    - window nào có quân của đối thủ thì bỏ qua
    """

    # Dùng tuple (rows, columns, inarow) làm key để cache mask theo cấu hình.
    key = (config.rows, config.columns, config.inarow)
    if key in _BB_WINDOW_MASKS_CACHE:
        return _BB_WINDOW_MASKS_CACHE[key]

    masks = []

    # Window ngang: cùng một hàng, tăng dần theo cột.
    for row in range(config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            mask = 0
            for k in range(config.inarow):
                mask |= 1 << ((col + k) * 7 + row)
            masks.append(mask)

    # Window dọc: cùng một cột, tăng dần theo hàng.
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns):
            mask = 0
            for k in range(config.inarow):
                mask |= 1 << (col * 7 + (row + k))
            masks.append(mask)

    # Chéo xuôi: đi từ trái-trên xuống phải-dưới.
    for row in range(config.rows - (config.inarow - 1)):
        for col in range(config.columns - (config.inarow - 1)):
            mask = 0
            for k in range(config.inarow):
                mask |= 1 << ((col + k) * 7 + (row + k))
            masks.append(mask)

    # Chéo ngược: đi từ trái-dưới lên phải-trên.
    for row in range(config.inarow - 1, config.rows):
        for col in range(config.columns - (config.inarow - 1)):
            mask = 0
            for k in range(config.inarow):
                mask |= 1 << ((col + k) * 7 + (row - k))
            masks.append(mask)

    # Lưu cache để các lần sau không phải tạo lại toàn bộ mask.
    _BB_WINDOW_MASKS_CACHE[key] = masks
    return masks


def count_windows_bb(me, opp):
    """ Đếm số lượng window chứa quân mình và không chứa quân địch. Vì nếu chứa quân địch thì không thể kết nối được nữa"""
    num_windows = [0] * (config.inarow + 1)
    for mask in _get_bb_window_masks():
        if (mask & opp) != 0:
            continue
        num_windows[(mask & me).bit_count()] += 1
    return num_windows


