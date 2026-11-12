# Hướng dẫn viết agent Wrongway

Sinh viên được yêu cầu lập trình chiến thuật cho một quân cờ: đi từ hàng xuất phát tới hàng đối diện trước đối thủ. Arena đã cài đặt luật chơi, kiểm tra hành động, quản lý lượt và tổ chức thi đấu. Bài nộp là **một file Python `.py`**, có hàm `choose_action(observation)`.

## 1. Chạy thử

Cần Python 3.9 trở lên. Các ví dụ chỉ sử dụng thư viện chuẩn của Python. Mở terminal tại thư mục `wrongway`. Trên macOS/Linux, dùng `python3`; trên Windows, thay `python3` bằng `py -3` trong lệnh dưới đây:

```bash
python3 arena.py --player1 submissions/greedy.py --player2 submissions/wall_builder.py --render --log logs/demo.json
```

Hai tham số người chơi cũng có thể nhận tên file không có đuôi `.py`, chẳng hạn `--player1 greedy --player2 wall_builder`.

Mùa giải của lớp do giảng viên tổ chức. Sinh viên dùng Arena trong bộ tài liệu này để tự chạy các trận đấu thử.

Để thử bài của mình, sao chép `submissions/TEMPLATE.py` thành `submissions/MSSV.py` (ví dụ, `submissions/26127000.py`) rồi chỉnh sửa hàm `choose_action`. Trong lệnh chạy trận đấu, dùng `--player1 submissions/MSSV.py` để đấu với bot mẫu; đổi sang `--player2 submissions/MSSV.py` để thử bên xuất phát còn lại.

## 2. Bàn cờ và luật chơi

Bàn cờ có 9 hàng và 9 cột. Một ô có tọa độ `[row, col]`:

- Hàng (`row`) tăng từ trên xuống dưới: `0` đến `8`.
- Cột (`col`) tăng từ trái sang phải: `0` đến `8`.
- Người chơi `0` xuất phát ở `[8, 4]`, cần tới một ô bất kỳ của hàng `0`.
- Người chơi `1` xuất phát ở `[0, 4]`, cần tới một ô bất kỳ của hàng `8`.

Tọa độ luôn theo bàn cờ cố định; không lật tọa độ khi đổi bên. Đọc `you["goal_row"]` thay vì mặc định rằng mình luôn đi lên.

Mỗi lượt chọn một hành động: di chuyển, đặt tường hoặc đứng yên. Bình thường, quân đi một ô ngang hoặc dọc qua cạnh không bị tường chặn. Nếu đối thủ đứng ở ô kề, có thể nhảy qua đối thủ khi cạnh phía sau còn mở. Chỉ khi phía sau bị tường hoặc biên bàn cờ chặn mới có thể đi chéo vòng qua đối thủ theo cạnh còn mở. Arena tính sẵn tất cả các nước này trong `legal_moves`.

Mỗi người bắt đầu với **10 tường**, số còn lại xem `you["walls_remaining"]`. Mỗi tường chặn hai cạnh kề nhau:

| Hướng | Vị trí neo `(row, col)` | Hai cạnh bị chặn |
| --- | --- | --- |
| `H` (nằm ngang) | `(r, c)` | `[r,c] - [r+1,c]` và `[r,c+1] - [r+1,c+1]` |
| `V` (thẳng đứng) | `(r, c)` | `[r,c] - [r,c+1]` và `[r+1,c] - [r+1,c+1]` |

Vị trí neo của tường có hàng và cột từ `0` đến `7`. Ví dụ, tường `H` tại `(3, 4)` chặn đường từ `[3,4]` xuống `[4,4]` và từ `[3,5]` xuống `[4,5]`. Tường không được chồng lấn, cắt nhau hoặc chặn hết mọi đường tới đích của bất kỳ người chơi nào. `legal_walls` chỉ chứa các tường bạn được phép đặt ở lượt hiện tại; danh sách này rỗng khi đã hết tường.

Người đầu tiên tới hàng đích thắng ngay. Mỗi hành động của một người tính là một lượt; mặc định sau tối đa **500 lượt** mà chưa có người thắng, ván đấu hòa. Giảng viên có thể cấu hình giới hạn này bằng `--max-turns`.

## 3. Bài nộp

Arena gọi hàm này mỗi khi tới lượt bạn:

```python
def choose_action(observation):
    # Đọc trạng thái, lựa chọn chiến thuật rồi trả về một dict hành động.
    return {"type": "move", "to": observation["legal_moves"][0]}
```

`observation` là một `dict` gồm dữ liệu Python thông thường, tương thích JSON. Đây là bản chụp trạng thái; sửa dữ liệu trong đó sẽ không sửa bàn cờ thật.

| Khóa | Ý nghĩa |
| --- | --- |
| `board_size` | Kích thước bàn cờ: `9` |
| `player_id` | Số hiệu của bạn: `0` hoặc `1` |
| `current_player` | Người đang tới lượt |
| `turn` | Số hành động đã diễn ra, bắt đầu từ `0` |
| `max_turns` | Giới hạn số lượt được cấu hình, mặc định `500` |
| `you` | `dict` chứa `position: [row, col]`, `goal_row`, `walls_remaining` của bạn |
| `opponent` | Các trường tương tự cho đối thủ |
| `walls` | `dict` có `horizontal` và `vertical`, mỗi danh sách chứa các vị trí neo `[row, col]` đã đặt |
| `legal_moves` | Danh sách các ô `[row, col]` bạn có thể đi tới ngay |
| `legal_walls` | Danh sách các `dict` `{ "orientation": "H" hoặc "V", "row": r, "col": c }` có thể đặt ngay |
| `distances` | Ma trận `9 × 9`: khoảng cách ngắn nhất từ từng ô tới hàng đích của bạn, tính theo tường và bỏ qua hai quân cờ; `-1` nếu không tới được |
| `done` | Ván đấu đã kết thúc hay chưa |
| `winner` | Số hiệu người thắng, hoặc `None` khi chưa có người thắng/hòa |

Trả về đúng một trong ba dạng sau:

```python
# Di chuyển: chọn một ô trong observation["legal_moves"].
return {"type": "move", "to": [row, col]}

# Đặt tường: chọn một cấu hình trong observation["legal_walls"].
return {"type": "wall", "orientation": "H", "row": row, "col": col}

# Đứng yên, nhường lượt cho đối thủ :v
return {"type": "pass"}
```

Các câu lệnh `return` này dùng bên trong hàm; mỗi lần gọi chỉ chạy một nhánh. Sinh viên sẽ không cần tự viết vòng lặp của trận đấu, đọc bàn phím hoặc cài đặt luật kiểm tra hành động.

**Giới hạn thực thi:** mỗi lần chọn hành động có tối đa **1 giây**; việc nạp module có giới hạn riêng **5 giây**. Quá giờ, lỗi khi chạy hoặc hành động không hợp lệ khiến quân đứng yên trong lượt đó, rồi chuyển lượt cho đối thủ. Hãy tránh công việc nặng khi Python nạp file.

Module thường tồn tại xuyên suốt một ván nên có thể dùng biến toàn cục để ghi nhớ. Sau khi quá giờ hoặc tiến trình tác tử bị chết, arena khởi động lại tác tử và bộ nhớ này có thể mất. Chiến thuật phải vẫn hoạt động khi chỉ có `observation` hiện tại. Arena khởi tạo seed cho thư viện `random` chuẩn; có thể dùng `random.choice(...)` như ví dụ. Không bảo đảm kết quả giống từng bit giữa các máy hoặc khi agent sát giới hạn thời gian.

## 4. Hãy bắt đầu từ vòng lặp, hàm, list và dict

Mẫu `TEMPLATE.py` đã là một agent chạy được: duyệt các nước hợp lệ rồi chọn ô gần hàng đích nhất.

```python
def choose_action(observation):
    moves = observation["legal_moves"]  # Truy cập dict bằng khóa.
    distances = observation["distances"]
    if not moves:
        return {"type": "pass"}

    best_move = moves[0]                # Truy cập phần tử đầu của list.
    best_distance = float("inf")
    for move in moves:                 # Duyệt list bằng vòng lặp.
        row, col = move
        distance = distances[row][col] # Ma trận là list chứa các list.
        if distance >= 0 and distance < best_distance:
            best_move = move
            best_distance = distance

    return {"type": "move", "to": list(best_move)}
```

Hãy thử thay đổi từng ý tưởng nhỏ, chạy đấu thử, rồi quan sát log:

1. Khi nhiều nước có cùng khoảng cách, ưu tiên cột gần giữa bàn cờ.
2. Tách cách tính điểm cho một nước đi thành hàm riêng để dễ đọc và chỉnh sửa.
3. Thử chọn ngẫu nhiên giữa các nước đồng điểm bằng `random.choice`.
4. Xem `wall_builder.py` để học cách mô hình hóa cạnh bị chặn và dùng BFS tìm đường ngắn nhất. So sánh độ dài đường của cả hai người trước/sau khi đặt một tường hợp lệ.
5. Tự xây dựng chiến thuật kết hợp di chuyển và đặt tường, rồi đánh giá qua nhiều đối thủ, nhiều seed và cả hai bên xuất phát.

`distances` và BFS mẫu bỏ qua vị trí quân đối thủ nên chỉ là ước lượng đường đi. Dùng `legal_moves` để chọn nước đi thực tế, nhất là khi hai quân đứng cạnh nhau. `wall_builder.py` minh họa một chiến thuật đơn giản, không phải lời giải tối ưu.

## 5. Các ví dụ và cách nộp

| File | Mục đích |
| --- | --- |
| `submissions/TEMPLATE.py` | Mẫu dễ sửa, dùng vòng lặp và điều kiện |
| `submissions/greedy.py` | Di chuyển tham lam theo khoảng cách tới đích |
| `submissions/random_agent.py` | Chọn nước ngẫu nhiên, đôi khi đặt tường |
| `submissions/wall_builder.py` | Ước lượng lợi ích của tường bằng BFS |

Nộp **một file `.py` độc lập**, ví dụ `MSSV.py`. File phải định nghĩa `choose_action(observation)` và trả về `dict` theo giao thức trên. Dùng thư viện chuẩn; không phụ thuộc file riêng, thư viện của arena hoặc dữ liệu trên máy cá nhân. Chạy thử cả khi là người chơi `0` và người chơi `1` trước khi nộp.
