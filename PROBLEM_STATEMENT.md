# Wrongway Arena

**Đồ án môn Tư duy tính toán**

## 1. Giới thiệu

Wrongway Arena giúp sinh viên làm quen với Python thông qua việc xây dựng một chương trình điều khiển quân cờ, gọi là **agent**. Mỗi agent cần đưa quân cờ của mình tới bờ đối diện trước đối thủ, đồng thời có thể đặt tường để làm chậm đường đi của đối thủ.

Đây là đồ án **cá nhân**. Mỗi sinh viên phát triển một agent trong **một file Python duy nhất**, sau đó tham gia mùa giải giữa các bài nộp của lớp. Arena đã cung cấp luật chơi, bàn cờ, danh sách hành động hợp lệ và cơ chế tổ chức trận đấu. Sinh viên tập trung vào cách đọc trạng thái, lựa chọn hành động và cải thiện chiến thuật.

Thông qua đồ án, sinh viên luyện tập:

- Sử dụng biến, điều kiện, vòng lặp, hàm, `list` và `dict` trong Python.
- Chia một nhiệm vụ thành các bước nhỏ và biểu diễn cách ra quyết định bằng chương trình.
- Quan sát kết quả chạy thử, tìm nguyên nhân hành vi chưa tốt và điều chỉnh chiến thuật.

## 2. Arena

Arena là chương trình Python tự động điều hành trận đấu giữa hai agent. Mỗi agent cung cấp hàm `choose_action(observation)`: nhận trạng thái bàn cờ và trả về hành động muốn thực hiện.

### 2.1. Vòng lặp tương tác

Hai người chơi **thay phiên nhau**, mỗi lượt thực hiện một hành động. Arena vận hành theo quy trình:

```text
Khởi tạo bàn cờ và hai agent.
Trong khi trận đấu chưa kết thúc:
    1. Cung cấp trạng thái hiện tại cho agent đang tới lượt.
    2. Gọi choose_action(observation), giới hạn 1 giây.
    3. Nếu hành động hợp lệ, cập nhật bàn cờ.
       Nếu quá giờ, xảy ra lỗi hoặc hành động sai, giữ nguyên bàn cờ.
    4. Tăng số lượt đã diễn ra.
    5. Kiểm tra người thắng và giới hạn tổng số lượt.
    6. Nếu trận đấu tiếp tục, chuyển lượt cho đối thủ.
```

Trạng thái được cung cấp là bản sao. Việc sửa `observation` trong bài làm không làm thay đổi bàn cờ thật.

### 2.2. Bàn cờ và điều kiện kết thúc

Bàn cờ có kích thước **9 × 9**. Một ô được xác định bằng `[row, col]`, trong đó hàng tăng từ trên xuống dưới, cột tăng từ trái sang phải; cả hai chỉ số đều thuộc khoảng `0` đến `8`.

```text
       0 1 2 3 4 5 6 7 8
    0  . . . . 1 . . . .    Hàng đích của người chơi 0
    1  . . . . . . . . .
    2  . . . . . . . . .
    3  . . . . . . . . .
    4  . . . . . . . . .
    5  . . . . . . . . .
    6  . . . . . . . . .
    7  . . . . . . . . .
    8  . . . . 0 . . . .    Hàng đích của người chơi 1
```

Ký hiệu `0`, `1` là hai quân cờ; `.` là ô trống. Người chơi 0 bắt đầu tại `[8, 4]` và cần tới một ô bất kỳ của hàng 0. Người chơi 1 bắt đầu tại `[0, 4]` và cần tới một ô bất kỳ của hàng 8. Người chơi 0 đi trước.

**Người đầu tiên tới hàng đích của mình thắng ngay.** Mặc định, trận đấu kết thúc hòa nếu sau **500 lượt đơn của cả hai người cộng lại** vẫn chưa có người thắng. Lượt đứng yên, quá giờ hoặc hành động sai đều được tính vào giới hạn này. Nếu tới đích ở lượt cuối cùng được phép, agent vẫn được tính thắng.

Tọa độ bàn cờ không thay đổi khi đổi bên. Agent cần đọc `you["goal_row"]` để xác định đích của mình.

### 2.3. Các hành động

Mỗi lượt, agent chọn một trong ba hành động sau.

**Di chuyển.** Bình thường, quân cờ đi một ô lên, xuống, trái hoặc phải, qua cạnh không có tường. Không được đi ra ngoài bàn cờ hoặc đứng cùng ô với đối thủ.

Khi đối thủ ở ô kề và giữa hai quân không có tường, có thể nhảy thẳng qua đối thủ nếu ô phía sau nằm trong bàn cờ và cạnh phía sau không bị chặn. Chỉ khi phía sau bị tường hoặc biên bàn cờ chặn mới được đi chéo vòng qua đối thủ, theo cạnh bên còn mở. Arena tính sẵn các đích đến hợp lệ trong `legal_moves`.

**Đặt tường.** Mỗi người bắt đầu với **10 tường**. Một tường chắn hai cạnh ô liền nhau và có hướng ngang `H` hoặc dọc `V`. Đặt một tường tiêu tốn một lượt; tường đã đặt không được thu hồi hoặc di chuyển.

Tường được xác định bằng hướng và vị trí neo `(row, col)`, với hai chỉ số từ `0` đến `7`. Cách quy đổi vị trí neo sang các cạnh bị chặn được mô tả trong [STUDENT_GUIDE.md](STUDENT_GUIDE.md).

Tường không được chồng lấn hoặc cắt ngang nhau. Sau khi đặt tường, **cả hai người vẫn phải còn ít nhất một đường tới hàng đích của mình**, khi xét các cạnh bị tường chặn và bỏ qua vị trí quân cờ. Arena tính sẵn những tường được phép đặt trong `legal_walls`.

**Đứng yên.** Agent có thể trả về `{"type": "pass"}` để giữ nguyên vị trí và nhường lượt cho đối thủ.

### 2.4. Dữ liệu đầu vào và hành động trả về

Bài làm phải định nghĩa hàm `choose_action(observation)`. Đầu vào `observation` là một `dict`, gồm thông tin hai người chơi (`you`, `opponent`), các tường đã đặt và số lượt. Arena còn cung cấp sẵn:

- `legal_moves`: các ô `[row, col]` được phép di chuyển tới.
- `legal_walls`: các tường được phép đặt, gồm `orientation`, `row`, `col`.
- `distances`: ma trận khoảng cách ngắn nhất tới đích của mình, xét tường và bỏ qua hai quân cờ; `-1` nghĩa là không có đường tới đích.

Hàm phải trả về một `dict` thuộc đúng một trong ba dạng:

```python
{"type": "move", "to": [row, col]}
{"type": "wall", "orientation": "H", "row": row, "col": col}
{"type": "pass"}
```

Với hành động đặt tường, `orientation` là `"H"` hoặc `"V"`. Các tọa độ phải là số nguyên. Phần tử của `legal_walls` chưa có khóa `"type"`; cần bổ sung `"type": "wall"` khi tạo hành động.

Sinh viên được sử dụng trực tiếp các dữ liệu tính sẵn để xây dựng chiến thuật bằng điều kiện, vòng lặp và hàm; không bắt buộc cài đặt thuật toán tìm kiếm nâng cao. Bảng đầy đủ các trường dữ liệu và ví dụ viết agent nằm trong [STUDENT_GUIDE.md](STUDENT_GUIDE.md).

### 2.5. Bộ mã được cung cấp và cách chạy thử

Giảng viên cung cấp `wrongway-student.zip`, gồm mã Arena để chạy trận đấu và xem replay, mẫu bài nộp, ba bot ví dụ (`greedy`, `random_agent`, `wall_builder`) cùng tài liệu hướng dẫn.

Cần **Python 3.9 trở lên**, chỉ sử dụng thư viện chuẩn. Sao chép `submissions/TEMPLATE.py` thành `submissions/MSSV.py` rồi viết chiến thuật trong `choose_action`. Chạy thử ở cả hai bên xuất phát và với các bot ví dụ theo hướng dẫn trong [STUDENT_GUIDE.md](STUDENT_GUIDE.md).

Bài làm được chấm trên Arena do giảng viên quản lý; việc sửa mã Arena trên máy cá nhân không thay đổi luật chấm.

## 3. Mùa giải và đánh giá

### 3.1. Thể thức thi đấu

Giảng viên tổ chức một mùa giải cho mỗi giai đoạn nộp bài. Các agent thi đấu vòng tròn; mặc định mỗi cặp gặp nhau **hai ván đổi bên**, để mỗi agent có một ván đi trước và một ván đi sau.

Với cấu hình hai ván mỗi cặp và `n` bài tham gia, mỗi agent thi đấu `2 × (n − 1)` ván; toàn bộ mùa giải có `n × (n − 1)` ván. Điểm thi đấu được tính như sau:

| Kết quả một ván | Điểm thi đấu |
| --- | ---: |
| Thắng | 3 |
| Hòa | 1 |
| Thua | 0 |

Tổng điểm thi đấu bằng `3 × số ván thắng + số ván hòa`. Xếp hạng theo tổng điểm giảm dần; nếu bằng điểm, xét số ván thắng. Nếu cả hai chỉ số bằng nhau, các agent được đồng hạng.

Ví dụ minh họa một mùa giải có ba agent, mỗi agent chơi bốn ván:

| Agent | Thắng | Hòa | Thua | Điểm thi đấu | Hạng |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 2 | 1 | 1 | 7 | 1 |
| B | 2 | 1 | 1 | 7 | 1 |
| C | 0 | 2 | 2 | 2 | 3 |

### 3.2. Điểm đồ án

Đồ án được đánh giá trên thang **10 điểm**:

| Tiêu chí | Điểm tối đa |
| --- | ---: |
| Mức độ hoàn thiện file Python theo yêu cầu đề bài | 8 |
| Thứ hạng trong mùa giải của lần nộp đầu | 1 |
| Thứ hạng trong mùa giải của lần nộp sau | 1 |
| **Tổng cộng** | **10** |

Điểm thi đấu trong từng mùa giải dùng để xác định thứ hạng. Phần điểm tối đa 1 điểm của mỗi mùa giải được quy đổi theo tỷ lệ điểm thi đấu đạt được trên tổng điểm tối đa có thể đạt:

```text
Điểm mùa giải = (3 × số ván thắng + số ván hòa) / (3 × số ván đã đấu)

Điểm đồ án = điểm hoàn thiện file Python
             + điểm mùa giải giai đoạn 1
             + điểm mùa giải giai đoạn 2
```

Ví dụ, agent A trong bảng trên thắng 2, hòa 1, thua 1 sau 4 ván, nên nhận `7 / 12 ≈ 0,58` điểm cho mùa giải đó. Agent B có cùng kết quả nên nhận cùng số điểm.

Trong một mùa giải vòng tròn, các agent có cùng số ván, nên cùng tổng điểm thi đấu sẽ nhận cùng phần điểm quy đổi. Tiêu chí phụ số ván thắng dùng để sắp thứ hạng khi bằng tổng điểm, không cộng thêm điểm đồ án. Thắng tất cả các ván tương ứng với 1 điểm cho mùa giải.

Sinh viên cần ưu tiên hoàn thiện một file Python chạy đúng giao thức và xử lý được trạng thái do Arena cung cấp, sau đó cải thiện hiệu quả chiến thuật.

## 4. Nộp bài

Đây là bài làm **cá nhân**. Ở mỗi giai đoạn, sinh viên nộp trực tiếp **đúng một file `MSSV.py` trên Moodle**. Không nộp báo cáo, file ZIP, thư mục Arena, log hoặc các file phụ.

Ví dụ, sinh viên có mã số `26127000` nộp:

```text
26127000.py
```

Bài nộp phải chứa hàm `choose_action(observation)` và tự hoạt động trong một file. Sinh viên có thể viết thêm các hàm hỗ trợ trong cùng file này; không phụ thuộc file riêng hoặc đường dẫn trên máy cá nhân.

Đồ án có **hai giai đoạn liên tiếp, mỗi giai đoạn kéo dài một tuần**:

| Giai đoạn | Thời gian | Nội dung |
| --- | --- | --- |
| Giai đoạn 1 | Tuần thứ nhất | Hoàn thiện phiên bản đầu, chạy thử và nộp `MSSV.py` để tham gia mùa giải đầu tiên. |
| Giai đoạn 2 | Tuần thứ hai | Cải thiện chiến thuật và nộp lại `MSSV.py` để tham gia mùa giải tiếp theo. |

Ngày và giờ mở/đóng từng giai đoạn được giảng viên công bố trên Moodle. **Chỉ sinh viên đã nộp bài ở giai đoạn 1 mới được nộp bài ở giai đoạn 2. Không chấp nhận bài nộp trễ trong bất kỳ trường hợp nào.**

Trước khi nộp, sinh viên cần kiểm tra tên file, tên hàm, kiểu hành động trả về và chạy thử khi ở cả hai bên xuất phát.

## 5. Lưu ý

- **Giới hạn thời gian:** mỗi lần chọn hành động có tối đa **1 giây**, bao gồm thời gian xử lý và giao tiếp với agent. Việc khởi động và nạp file có giới hạn riêng **5 giây**. Thời gian Arena tính các nước hợp lệ và ma trận khoảng cách không trừ vào một giây chọn hành động.
- **Xử lý lỗi:** quá thời gian, lỗi chương trình hoặc hành động không hợp lệ làm agent mất lượt đó; bàn cờ giữ nguyên và chuyển lượt cho đối thủ. Trận đấu vẫn tiếp tục.
- **Thư viện:** sử dụng thư viện chuẩn đi kèm Python. Không yêu cầu cài thêm thư viện và không nộp `requirements.txt`. Hàm chọn hành động phải tự trả về kết quả, không chờ nhập bàn phím hoặc thao tác thủ công.
- **Trạng thái giữa các lượt:** có thể dùng biến toàn cục để ghi nhớ trong một ván. Khi agent bị quá giờ hoặc tiến trình bị dừng, Arena có thể khởi động lại và bộ nhớ này sẽ mất. Agent cần vẫn hoạt động với trạng thái hiện tại.
- **Công cụ AI:** không hạn chế sử dụng công cụ AI. Sinh viên cần sử dụng có trách nhiệm, hiểu mã nguồn mình nộp và có khả năng giải thích chiến thuật. Giảng viên có thể tổ chức vấn đáp bổ sung để đánh giá mức độ hiểu bài.
- **Liêm chính học thuật:** không sao chép bài của sinh viên khác hoặc gian lận kết quả thi đấu. Sinh viên chịu trách nhiệm về bài làm mình nộp và tuân thủ quy định của học phần.
