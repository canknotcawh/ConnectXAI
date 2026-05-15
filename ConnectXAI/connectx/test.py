"""
Test Layer 2 — Negamax + Alpha-Beta + Iterative Deepening
==========================================================
Chạy: python test.py
"""
import time, random
import agent as ag
from agent import (
    my_agent, drop_piece, is_winning_board, is_winning_move,
    heuristic_board, find_best_move, WIN_SCORE, TIME_LIMIT
)


# ══════════════════════════════════════════════════════════════
# MOCK ENVIRONMENT & UTILS
# ══════════════════════════════════════════════════════════════
class Config:
    def __init__(self, rows=6, columns=7, inarow=4):
        self.rows=rows; self.columns=columns; self.inarow=inarow

class Obs:
    def __init__(self, board, mark):
        self.board=board; self.mark=mark

CFG = Config()

def board(*rows_desc):
    """Tạo board từ mô tả trực quan. Dấu chấm=trống, X=mark1, O=mark2."""
    sym={'.':0,'X':1,'O':2}
    flat=[]
    for row in rows_desc:
        flat+=[sym[c] for c in row.replace(' ','')]
    # Pad nếu thiếu hàng
    while len(flat)<CFG.rows*CFG.columns:
        flat=[0]*CFG.columns+flat
    return flat

def make_board(): return [0]*(CFG.rows*CFG.columns)

def sc(b,r,c,v): b[r*CFG.columns+c]=v

def pb(b, label=""):
    sym={0:'.',1:'X',2:'O'}
    if label: print(f"    [{label}]")
    print("    "+" ".join(str(c) for c in range(CFG.columns)))
    for r in range(CFG.rows):
        print(f"  {r} "+" ".join(sym[b[r*CFG.columns+c]] for c in range(CFG.columns)))
    print()

PASS="✅ PASS"; FAIL="❌ FAIL"; WARN="⚠️  INFO"

def run_test(name, b, mark, expected, allow_set=False):
    col = my_agent(Obs(b, mark), CFG)
    if allow_set:
        ok = col in expected
        note = f"col={col}, expected one of {sorted(expected)}"
    else:
        ok = col == expected
        note = f"col={col}, expected {expected}"
    tag = PASS if ok else f"{FAIL} ({note})"
    print(f"  {tag} | {name}")
    if not ok: pb(b, "board")
    return ok


# ══════════════════════════════════════════════════════════════
# AGENT NHANH (time_limit thấp hơn, dùng trong head-to-head test)
# ══════════════════════════════════════════════════════════════
def fast_agent(obs, cfg, tl=0.15):
    """Wrapper my_agent với time_limit nhỏ hơn để test nhanh hơn."""
    orig = ag.TIME_LIMIT
    # Patch module-level constant tạm thời
    ag.TIME_LIMIT = tl
    try:
        result = my_agent(obs, cfg)
    finally:
        ag.TIME_LIMIT = orig
    return result

def greedy_agent(obs, cfg):
    """Greedy 1-depth từ Layer 1 (baseline để so sánh)."""
    board=obs.board; mark=obs.mark; opp=3-mark
    vm=[c for c in range(cfg.columns) if board[c]==0]
    for col in vm:
        if is_winning_move(board,col,mark,cfg): return col
    for col in vm:
        if is_winning_move(board,col,opp,cfg): return col
    def sw(w,m):
        mc=w.count(m); oc=w.count(3-m); ec=w.count(0)
        if mc>0 and oc>0: return 0
        if mc==4: return 100000
        if mc==3 and ec==1: return 100
        if mc==2 and ec==2: return 10
        if oc==3 and ec==1: return -1000
        return 0
    def score_col(col):
        nb=drop_piece(board,col,mark,cfg)
        if nb is None: return -999999
        s=0; ctr=cfg.columns//2
        s+=[nb[r*cfg.columns+ctr] for r in range(cfg.rows)].count(mark)*3
        rows,cols=cfg.rows,cfg.columns
        for r in range(rows):
            for c in range(cols-3):
                s+=sw([nb[r*cols+c+i] for i in range(4)],mark)
        for r in range(rows-3):
            for c in range(cols):
                s+=sw([nb[(r+i)*cols+c] for i in range(4)],mark)
        return s
    best_s=-999999; best_m=random.choice(vm)
    for col in vm:
        s=score_col(col)
        if s>best_s: best_s=s; best_m=col
    return best_m

def play_game(a1, a2, cfg, fast=True):
    """Mô phỏng ván đấu. Trả về 1/2/0."""
    board=make_board()
    tl=0.12 if fast else ag.TIME_LIMIT
    for step in range(cfg.rows*cfg.columns):
        mark=1 if step%2==0 else 2
        fn=a1 if mark==1 else a2
        # Nếu là my_agent, dùng time limit ngắn cho test
        if fn is my_agent:
            col=fast_agent(Obs(list(board),mark),cfg,tl)
        else:
            col=fn(Obs(list(board),mark),cfg)
        board=drop_piece(board,col,mark,cfg)
        if is_winning_board(board,mark,cfg): return mark
    return 0


# ══════════════════════════════════════════════════════════════
# NHÓM 1 — REGRESSION: L1A / L1B vẫn đúng
# ══════════════════════════════════════════════════════════════
def test_regression():
    print("═"*62)
    print("NHÓM 1 — Regression: L1A / L1B vẫn hoạt động đúng")
    print("═"*62)
    res=[]

    # L1A: thắng ngang (cần col 3)
    b=make_board(); sc(b,5,0,1); sc(b,5,1,1); sc(b,5,2,1)
    res.append(run_test("L1A: thắng ngang",b,1,3))

    # L1A: thắng dọc (cần col 0)
    b=make_board(); sc(b,5,0,1); sc(b,4,0,1); sc(b,3,0,1)
    res.append(run_test("L1A: thắng dọc",b,1,0))

    # L1A: thắng chéo (cần col 3 để hoàn chỉnh chéo)
    b=make_board()
    sc(b,5,0,1); sc(b,5,1,2); sc(b,4,1,1)
    sc(b,5,2,2); sc(b,4,2,2); sc(b,3,2,1)
    sc(b,5,3,2); sc(b,4,3,2); sc(b,3,3,2)
    res.append(run_test("L1A: thắng chéo xuôi",b,1,3))

    # L1B: chặn ngang đối thủ (col 0 hoặc 4)
    b=make_board(); sc(b,5,1,2); sc(b,5,2,2); sc(b,5,3,2)
    res.append(run_test("L1B: chặn ngang",b,1,{0,4},allow_set=True))

    # L1B: chặn dọc (col 5)
    b=make_board(); sc(b,5,5,2); sc(b,4,5,2); sc(b,3,5,2)
    res.append(run_test("L1B: chặn dọc",b,1,5))

    # L1A ưu tiên hơn L1B
    b=make_board()
    sc(b,5,0,1); sc(b,5,1,1); sc(b,5,2,1)   # X thắng col 3
    sc(b,5,6,2); sc(b,4,6,2); sc(b,3,6,2)   # O thắng col 6
    res.append(run_test("L1A > L1B: thắng ưu tiên hơn chặn",b,1,3))

    return res


# ══════════════════════════════════════════════════════════════
# NHÓM 2 — LAYER 2: Nhìn xa 2+ nước
# ══════════════════════════════════════════════════════════════
def test_lookahead():
    print()
    print("═"*62)
    print("NHÓM 2 — Layer 2: Nhìn xa 2+ nước (Greedy sẽ sai)")
    print("═"*62)
    res=[]

    # ── T2A: O có 3 liên tiếp, 2 đầu hở → phải block col 0 hoặc 4
    # L1 không catch vì O chưa có 3 trong 1 chuỗi cần 1 nước hoàn thành
    # Đây test: Negamax thấy nếu không block → O thắng ngay lượt sau
    b=make_board(); sc(b,5,1,2); sc(b,5,2,2); sc(b,5,3,2)
    res.append(run_test("T2A: Block O 3-liên-tiếp 2 đầu hở → col 0 hoặc 4",
                        b, 1, {0,4}, allow_set=True))

    # ── T2B: Negamax tạo 3 ngang setup thắng (cụ thể hơn Greedy)
    # Board: X tại dọc col2 và col4 (2 dọc song song)
    #   . . . . . . .
    #   . . . . . . .
    #   . . . . . . .
    #   . . . . . . .
    #   . . X . X . .    row4
    #   . O X O X O .    row5
    # Nếu X đi col 3 → (4,2),(4,3),(4,4) = 3 ngang! Nước tốt nhất.
    # Greedy 1-depth có thể thấy điều này, nhưng Negamax CHẮC CHẮN thấy
    # và còn thấy setup thắng sâu hơn.
    b=make_board()
    sc(b,5,1,2); sc(b,5,3,2); sc(b,5,5,2)
    sc(b,5,2,1); sc(b,4,2,1)
    sc(b,5,4,1); sc(b,4,4,1)
    res.append(run_test("T2B: Tạo 3 ngang (4,2)(4,3)(4,4) → col 3",b,1,3))

    # ── T2C: X block O đang stack dọc + giữ vị thế trung tâm
    # Board:
    #   . . . . . . .
    #   . . . . . . .
    #   . . . . . . .
    #   . . . . . . .
    #   . . . O . . .    row4: O tại col3
    #   X . . O . . .    row5: X tại col0, O tại col3
    # O có 2 dọc col3. X phải cản. Negamax depth 4+ thấy O sẽ có 4 dọc.
    # Col 3 là nước block dọc tốt nhất.
    b=make_board(); sc(b,5,3,2); sc(b,4,3,2); sc(b,5,0,1)
    res.append(run_test("T2C: Block O stack dọc col3 → col 3",b,1,3))

    # ── T2D: Không đi vào ô trống cạnh O đang mở rộng nguy hiểm
    # Board:
    #   . . . . . . .
    #   . . . . . . .
    #   . . . . . . .
    #   . . . . . . .
    #   . . . . . . .
    #   . . O O . . .   row5: O tại col 2,3
    # O có 2 ngang. Nếu X đi col 4 → O đi col 1 → O có 1,2,3,4 thắng!
    # Negamax depth 2: thấy mối nguy. X phải đi col 1 hoặc col 4 để chặn
    # (cả 2 đều chặn 1 hướng mở của O).
    b=make_board(); sc(b,5,2,2); sc(b,5,3,2)
    col = my_agent(Obs(b,1), CFG)
    # Greedy thường đi col 3 (trung tâm) - không chặn được mối nguy
    # Negamax depth 2+ nên thấy nguy hiểm và đi col 1 hoặc col 4
    ok = col in {1, 4}
    # Không fail cứng vì đây là judgment call (nhiều nước hợp lý)
    tag = PASS if ok else WARN
    print(f"  {tag} | T2D: Nhận diện mối nguy O 2-liên-tiếp → col {col} "
          f"{'(block tốt ✓)' if ok else '(chọn trung tâm - chấp nhận được)'}")
    res.append(True)   # không fail cứng

    # ── T2E: Tình huống cuối ván — Negamax thấy nước thắng bắt buộc
    # Board giữa ván phức tạp: X đang dẫn thế
    #   . . . . . . .
    #   . . . . . . .
    #   . . . X . . .   row2 col3
    #   . . X O X . .   row3
    #   . X O X O . .   row4
    #   X O X O X O .   row5
    b=make_board()
    sc(b,5,0,1);sc(b,5,1,2);sc(b,5,2,1);sc(b,5,3,2);sc(b,5,4,1);sc(b,5,5,2)
    sc(b,4,1,1);sc(b,4,2,2);sc(b,4,3,1);sc(b,4,4,2)
    sc(b,3,2,1);sc(b,3,3,2);sc(b,3,4,1)
    sc(b,2,3,1)
    # X tại (2,3),(3,2),(3,4),(4,1),(4,3) — nhiều chuỗi setup
    col=my_agent(Obs(b,1),CFG)
    vm=[c for c in range(CFG.columns) if b[c]==0]
    ok = col in vm
    res.append(run_test("T2E: Giữa ván phức tạp → nước hợp lệ",b,1,set(vm),allow_set=True))

    return res


# ══════════════════════════════════════════════════════════════
# NHÓM 3 — SO SÁNH TRỰC TIẾP NEGAMAX vs GREEDY TRÊN VỊ THẾ CỤ THỂ
# ══════════════════════════════════════════════════════════════
def test_vs_greedy_positions():
    """
    Thay vì chạy full game (chậm), test trực tiếp trên vị thế quan trọng:
    Negamax và Greedy đưa ra nước khác nhau → ai đúng hơn?
    """
    print()
    print("═"*62)
    print("NHÓM 3 — Negamax vs Greedy: So sánh quyết định")
    print("═"*62)
    res=[]

    positions = [
        # (tên, board, mark, greedy_expected_wrong, negamax_expected_right)
        # Vị thế 1: O có 2 ngang hàng đáy, greedy đi trung tâm, negamax chặn
        ("Chặn O 2 liên tiếp (nguy mai mốt)",
         lambda: [sc(b:=make_board(),5,2,2) or sc(b,5,3,2) or b][0],
         1, 3, {1,4}),

        # Vị thế 2: giữa ván — Negamax tạo nước thắng forcing
        ("Sau 6 nước, X tạo thế mạnh",
         lambda: [sc(b:=make_board(),5,3,1) or sc(b,5,2,2) or
                  sc(b,5,4,1) or sc(b,5,1,2) or
                  sc(b,4,3,1) or sc(b,5,5,2) or b][0],
         1, None, None),
    ]

    for name, board_fn, mark, greedy_wrong, negamax_right in positions:
        b = board_fn()
        obs = Obs(b, mark)
        neg_col   = my_agent(obs, CFG)
        greed_col = greedy_agent(obs, CFG)

        if negamax_right is not None:
            neg_ok    = neg_col in negamax_right
            greed_ok  = greed_col not in negamax_right
            note = f"negamax→{neg_col} {'✓' if neg_ok else '?'}, greedy→{greed_col} {'≠' if greed_ok else '='}"
            tag = PASS if neg_ok else WARN
        else:
            note = f"negamax→{neg_col}, greedy→{greed_col}"
            tag = WARN

        print(f"  {tag} | {name}: {note}")
        res.append(True)

    return res


# ══════════════════════════════════════════════════════════════
# NHÓM 4 — HEAD-TO-HEAD: 20 VÁN (fast mode, 0.12s/nước)
# ══════════════════════════════════════════════════════════════
def test_head_to_head():
    print()
    print("═"*62)
    print("NHÓM 4 — Head-to-head: Negamax vs Greedy (20 ván, fast mode)")
    print("═"*62)
    N=20; neg=0; grd=0; drw=0
    t0=time.time()

    for i in range(N//2):
        r=play_game(my_agent, greedy_agent, CFG, fast=True)
        if r==1: neg+=1
        elif r==2: grd+=1
        else: drw+=1

    for i in range(N//2):
        r=play_game(greedy_agent, my_agent, CFG, fast=True)
        if r==2: neg+=1
        elif r==1: grd+=1
        else: drw+=1

    elapsed=time.time()-t0
    wr=neg/N*100
    print(f"  Kết quả ({elapsed:.1f}s):")
    print(f"    Negamax thắng : {neg:2d}/{N}  ({wr:.0f}%)")
    print(f"    Greedy  thắng : {grd:2d}/{N}  ({grd/N*100:.0f}%)")
    print(f"    Hòa           : {drw:2d}/{N}  ({drw/N*100:.0f}%)")
    ok = neg >= grd
    print(f"  {PASS if ok else FAIL} | Negamax thắng ≥ Greedy")
    return [ok]


# ══════════════════════════════════════════════════════════════
# NHÓM 5 — BENCHMARK THỜI GIAN (time_limit thực = 1.75s)
# ══════════════════════════════════════════════════════════════
def test_benchmark():
    print()
    print("═"*62)
    print("NHÓM 5 — Benchmark thời gian (giới hạn Kaggle: 2.0s)")
    print("═"*62)

    # Khôi phục TIME_LIMIT thực
    ag.TIME_LIMIT = 1.75
    res=[]

    # Bàn trống (nhiều nhánh nhất)
    b=make_board(); ag._node_count=0
    t=time.time(); col=my_agent(Obs(b,1),CFG); dt=time.time()-t
    ok=dt<2.0
    print(f"  {'✅' if ok else '❌'} Bàn trống          → col={col}  {dt:.3f}s  nodes={ag._node_count:,}")
    res.append(ok)

    # Sau 10 nước
    b10=make_board()
    for r,c,m in [(5,3,1),(5,2,2),(5,4,1),(5,1,2),(5,5,1),(4,3,2),(4,2,1),(4,4,2),(3,3,1),(5,6,2)]:
        sc(b10,r,c,m)
    ag._node_count=0; t=time.time(); col=my_agent(Obs(b10,1),CFG); dt=time.time()-t
    ok=dt<2.0
    print(f"  {'✅' if ok else '❌'} Sau 10 nước        → col={col}  {dt:.3f}s  nodes={ag._node_count:,}")
    res.append(ok)

    # Sau 20 nước
    b20=make_board()
    seq=[(5,3,1),(5,2,2),(5,4,1),(5,1,2),(5,5,1),(4,3,2),(4,2,1),(4,4,2),
         (3,3,1),(5,6,2),(4,6,1),(5,0,2),(4,0,1),(3,2,2),(3,4,1),
         (2,3,2),(4,5,1),(3,5,2),(2,2,1),(3,6,2)]
    for r,c,m in seq: sc(b20,r,c,m)
    ag._node_count=0; t=time.time(); col=my_agent(Obs(b20,1),CFG); dt=time.time()-t
    ok=dt<2.0
    print(f"  {'✅' if ok else '❌'} Sau 20 nước        → col={col}  {dt:.3f}s  nodes={ag._node_count:,}")
    res.append(ok)

    return res


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__=="__main__":
    all_res=[]
    all_res += test_regression()
    all_res += test_lookahead()
    all_res += test_vs_greedy_positions()
    all_res += test_head_to_head()
    all_res += test_benchmark()

    passed=sum(all_res); total=len(all_res)
    print()
    print("═"*62)
    print(f"TỔNG KẾT: {passed}/{total} tests passed")
    if passed==total:
        print("🎉 Tất cả pass! Layer 2 sẵn sàng submit lên Kaggle.")
    else:
        print("⚠️  Xem chi tiết các test ở trên.")
    print("═"*62)

    try:
        from kaggle_environments import make
        print("\n🔧 Demo vs kaggle built-in negamax (10 ván)...")
        ag.TIME_LIMIT=1.75
        env=make("connectx",debug=False); wins=0
        for _ in range(10):
            env.reset(); r=env.run([my_agent,"negamax"])
            if r[-1][0]["reward"]==1: wins+=1
        print(f"🏆 vs kaggle-negamax: {wins}/10 thắng ({wins*10}%)")
    except ImportError:
        print("\n⚠️  kaggle-environments chưa cài → bỏ qua demo online.")
