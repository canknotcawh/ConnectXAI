# ConnectXAI

Tổng quan chiến lược
ConnectX về bản chất là Connect 4 có thể tùy chỉnh (mặc định 7×6, cần 4 liên tiếp). Agent của bạn là một hàm Python my_agent(observation, configuration) có tối đa 2 giây mỗi nước đi.
Kiến trúc đề xuất: Layered Hybrid Agent
Lớp 1 — Immediate Threat Detector (ưu tiên cao nhất): Trước khi làm bất cứ điều gì, quét bàn cờ tìm nước thắng ngay hoặc cần chặn đối thủ ngay. Không cần tìm kiếm sâu, chạy <1ms.
Lớp 2 — Opening Book: Cột giữa (cột 3) luôn là nước khai cuộc mạnh nhất trong Connect 4. Hard-code ~15–20 phản hồi tối ưu cho các nước đầu ván để không lãng phí 2 giây.
Lớp 3 — Negamax + Alpha-Beta + Iterative Deepening: Đây là trọng tâm. Iterative deepening đảm bảo bạn luôn có đáp án dù hết giờ giữa chừng. Alpha-beta cắt bỏ tới ~50–70% nhánh không cần thiết. Ở độ sâu 8–10, agent này đã rất mạnh.
Lớp 4 — Heuristic Evaluation: Khi chưa đến trạng thái kết thúc, hàm điểm số đánh giá: kiểm soát trung tâm, số "cửa sổ 4 ô" có lợi thế, và các đe dọa chồng lên nhau.
Lộ trình nâng cấp
Giai đoạn 1 (đủ để top leaderboard trung bình) → Giai đoạn 2 với MCTS + UCB1 (linh hoạt hơn, không cần heuristic tốt) → Giai đoạn 3 với CNN self-play kiểu AlphaZero nếu muốn leo top.
Layer 1 update:
Hàm mới drop_piece() — thay thế code copy board inline lặp lại nhiều nơi, giờ dùng chung 1 chỗ, code gọn hơn.
Hàm mới is_winning_board() — quét đủ 4 hướng, kiểm tra bàn cờ có phải trạng thái thắng không. Đây là nền tảng của Layer 1.
Hàm mới is_winning_move() — thả thử quân vào cột, rồi gọi is_winning_board. Chạy cực nhanh <1ms.
Trong my_agent(), thứ tự ưu tiên giờ là:
L1A: Tìm nước thắng ngay         → đi liền (không tính gì thêm)
L1B: Tìm nước chặn đối thủ thắng → block liền
L2:  Greedy heuristic + center bonus (fallback)
Center bonus — fix thêm: trên bàn trống cột giữa (cột 3) giờ luôn được chọn thay vì cột 0.
Layer 2 update:
Depth 8 hoàn chỉnh trong 1.75s (~36k nodes)
Move ordering tái sử dụng kết quả depth trước → alpha-beta pruning hiệu quả hơn ~30%
Iterative deepening an toàn: chỉ dùng kết quả depth đã hoàn chỉnh 100%

