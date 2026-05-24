# TÀI LIỆU ĐẶC TẢ YÊU CẦU PHẦN MỀM (SRS)

**Tên dự án:** Web Dashboard Quản trị An ninh mạng IoT & MLOps (IoT-SOC Platform)

**Phiên bản:** 1.0

**Ngăn xếp công nghệ cốt lõi:** React.js (Frontend), Python/FastAPI (Backend API), Celery (Background Tasks), WebSocket (Real-time).

---

## 1. GIỚI THIỆU (INTRODUCTION)

### 1.1. Mục đích của tài liệu

Tài liệu này đặc tả các yêu cầu chức năng (Functional) và phi chức năng (Non-functional) cho hệ thống Web Application IoT-SOC. Hệ thống đóng vai trò là Trung tâm điều hành (Cloud Server), giao tiếp với các thiết bị biên (Edge Nodes), trực quan hóa dữ liệu mạng, quản lý vòng đời mô hình học máy (MLOps) và xử lý cảnh báo xâm nhập.

### 1.2. Phân quyền người dùng (User Roles)

Hệ thống được thiết kế tinh gọn với 2 cấp độ phân quyền chính để đảm bảo tính dễ sử dụng:

- **Admin (Quản trị viên / ML Engineer):** Toàn quyền hệ thống. Có quyền truy cập cấu hình hệ thống, module bảo mật cốt lõi, quản lý Model Health (MLOps), kích hoạt tiến trình Retraining và Deploy mô hình xuống Edge.
- **Operator (Người dùng phổ thông / SOC):** Chỉ xem/thao tác những thông tin cần thiết. Quyền giám sát Live Monitor, tiếp nhận và xác nhận/hủy bỏ cảnh báo, xem phân tích xu hướng (Analytics) cơ bản.

---

## 2. ĐẶC TẢ YÊU CẦU CHỨC NĂNG (FUNCTIONAL REQUIREMENTS)

### MODULE 1: LIVE MONITORING (XỬ LÝ LUỒNG THỜI GIAN THỰC)

**FR 1.1: Tiếp nhận và Phân phối luồng dữ liệu (Data Streaming)**

- **Mô tả:** Hệ thống Frontend phải duy trì một kết nối WebSocket liên tục với Backend để nhận dữ liệu mà không cần tải lại trang.
- **Đầu vào (Input):** Các payload JSON được đẩy từ Backend qua WebSocket (bao gồm 2 loại: telemetry_log và alert_log).
- **Xử lý (Logic):**
    - Giới hạn bộ đệm (Buffer Limit) trên RAM của trình duyệt: React State chỉ lưu tối đa 1000 logs gần nhất để tránh tràn bộ nhớ (Memory Leak). Các log cũ hơn sẽ bị đẩy khỏi mảng.
- **Đầu ra (Output):** Render lên giao diện các bảng log cuộn liên tục và biểu đồ chuỗi thời gian (Time-series line chart) với độ trễ (latency) < 500ms.

**FR 1.2: Phân tích chéo trên Cloud (Cloud Verification)**

- **Mô tả:** Cho phép người dùng chọn một cảnh báo từ mô hình Edge (LightGBM) và yêu cầu mô hình Cloud (CNN-LSTM) xác minh lại.
- **Đầu vào:** alert_id và src_ip.
- **Xử lý:**
    - Frontend gọi HTTP POST request.
    - Backend Python truy vấn Database lấy 10 luồng mạng gần nhất của src_ip đó, đưa qua pipeline tiền xử lý và chạy hàm model.predict() bằng mô hình CNN-LSTM.
- **Đầu ra:** Trả về Frontend một chuỗi JSON chứa xác suất (Probabilities) của 5 lớp tấn công và cập nhật trạng thái cảnh báo trên UI.

### MODULE 2: THREAT ANALYTICS (TRUY VẤN VÀ GÁN NHÃN LẠI)

**FR 2.1: Máy tìm kiếm Log (Query Engine)**

- **Mô tả:** Cung cấp tính năng truy vấn log mạng lịch sử dựa trên các trường đặc trưng.
- **Đầu vào:** Chuỗi truy vấn (Ví dụ: dst_port == 80 AND label == "PortScan") và Khoảng thời gian (Time range).
- **Xử lý:** Backend Python sẽ parse (dịch) chuỗi truy vấn này thành ngôn ngữ truy vấn của Database (ví dụ Elasticsearch Query DSL hoặc SQL) và thực thi.
- **Đầu ra:** Mảng JSON chứa các bản ghi thỏa mãn điều kiện và các thông số tổng hợp (Aggregation) để React.js vẽ biểu đồ (Pie chart, Bar chart).

**FR 2.2: Phản hồi và Gán nhãn lại (Human-in-the-Loop Feedback)**

- **Mô tả:** Chức năng cốt lõi để tạo dữ liệu cho việc Retrain. Người dùng có thể sửa nhãn sai do AI dự đoán.
- **Đầu vào:** ID của luồng mạng (flow_id) và Nhãn thực tế (true_label - do chuyên gia chọn).
- **Xử lý:** Backend nhận request, cập nhật cột true_label và is_verified = True trong Database.
- **Đầu ra:** Thông báo cập nhật thành công (Toast notification).

### MODULE 3: MLOPS & MODEL HEALTH (GIÁM SÁT MÔ HÌNH)

**FR 3.1: Đánh giá Hiệu năng Động (Dynamic Metrics Calculation)**

- **Mô tả:** Hệ thống tự động tính toán lại các chỉ số của mô hình dựa trên dữ liệu đã được người dùng gán nhãn lại (is_verified = True).
- **Xử lý:**
    - Backend sử dụng scikit-learn chạy một tác vụ ngầm (Cronjob) mỗi giờ một lần: kéo dữ liệu predicted_label và true_label từ DB.
    - Tính toán Confusion Matrix, Precision, Recall, F1-Score.
- **Đầu ra:** REST API cung cấp dữ liệu số liệu để React.js render thành Gauge Charts và Heatmap.

**FR 3.2: Phát hiện Lệch Dữ liệu (Data Drift Detection)**

- **Xử lý:** Tác vụ ngầm của Backend sẽ so sánh phân phối thống kê (Mean, Variance) của các đặc trưng quan trọng (VD: bidirectional_duration_ms) giữa tập dữ liệu training_set gốc và tập dữ liệu trong 7 ngày gần nhất.
- **Đầu ra:** Nếu độ lệch vượt ngưỡng (Threshold), kích hoạt cờ Drift_Alert = True để Frontend hiển thị cảnh báo yêu cầu Retrain.

### MODULE 4: RETRAINING PIPELINE (VÒNG LẶP HUẤN LUYỆN LẠI)

Đây là module phức tạp nhất, yêu cầu xử lý bất đồng bộ (Asynchronous) vì việc train mô hình mất nhiều thời gian.

**FR 4.1: Khởi chạy Tiến trình Huấn luyện (Trigger Retrain)**

- **Mô tả:** Kích hoạt quá trình huấn luyện lại mô hình trên server.
- **Đầu vào:** Lệnh POST request từ Frontend chứa config (VD: Lấy dữ liệu 30 ngày qua).
- **Xử lý:**
    - API FastAPI nhận request và đẩy một tác vụ (Task) vào Message Queue (Celery/RabbitMQ). FastAPI trả về mã task_id ngay lập tức (không chờ train xong).
    - **Celery Worker** (chạy ngầm bằng Python) thực thi tuần tự script: Data Extraction -> Preprocessing -> Retrain CNN-LSTM -> Knowledge Distillation -> Export to .onnx.

**FR 4.2: Cập nhật Trạng thái Tiến trình (Task Progress Tracking)**

- **Mô tả:** Frontend phải hiển thị được tiến trình train cho người dùng xem.
- **Xử lý:** React.js sử dụng kỹ thuật Polling (gọi API liên tục mỗi 3 giây) hoặc WebSocket, gửi task_id lên Backend để hỏi "Train đến đâu rồi?".
- **Đầu ra:** Tỷ lệ % hoàn thành, Epoch hiện tại, và Loss/Accuracy logs.

**FR 4.3: Triển khai Mô hình (OTA Deployment)**

- **Đầu vào:** ID của mô hình .onnx mới nhất và danh sách các Edge Node đích.
- **Xử lý:** Backend sẽ gửi một lệnh "Update Model" kèm theo URL tải file .onnx qua MQTT đến các Raspberry Pi. Các Pi sẽ tải file về và tự động restart service phân tích.

---

## 3. KIẾN TRÚC HỆ THỐNG VÀ NGĂN XẾP CÔNG NGHỆ (TECH STACK DETAILS)

Để các module trên hoạt động đúng logic, kiến trúc giữa React.js và Python phải được thiết kế như sau:

### 3.1. Tầng Frontend (Client-side)

- **Core:** React.js (Sử dụng Vite hoặc Create React App).
- **State Management:** Sử dụng **Zustand** hoặc **Redux Toolkit** để quản lý trạng thái kết nối WebSocket và lưu trữ tạm thời các log đang trôi trên màn hình.
- **Data Fetching:** Sử dụng **Axios** hoặc **React Query** để gọi REST APIs (cho các chức năng Analytics và Retraining).
- **Data Visualization:** Sử dụng **Apache ECharts** (khuyên dùng vì hiệu năng xử lý hàng ngàn điểm dữ liệu rất tốt) hoặc **Chart.js** cho các biểu đồ tĩnh.

### 3.2. Tầng Backend API (Server-side)

- **Core Web Framework:** **FastAPI** (Python). Lý do: Xử lý WebSocket cực tốt, bất đồng bộ (Async/Await) bản địa, tốc độ cao, hỗ trợ tích hợp với hệ sinh thái Data Science (Pandas, Numpy, Keras) rất mượt.
- **Xử lý Tác vụ Nặng (Background Tasks):** **Celery** + **Redis** làm Message Broker. Bắt buộc phải có để quá trình model.fit() (Retrain) không làm treo Web API.

### 3.3. Tầng Database & Streaming

- **Time-series/Log DB:** **Elasticsearch**. Nơi lưu trữ hàng triệu dòng log mạng để đáp ứng tốc độ truy vấn (Search) dưới <1 giây cho tab Analytics.
- **Relational DB:** **PostgreSQL**. Nơi lưu trữ thông tin User, Thông tin cấu hình Raspberry Pi, Danh sách các phiên bản Model.
- **Realtime Broker:** **MQTT Broker** (Mosquitto) để nhận dữ liệu từ Raspberry Pi đẩy lên, sau đó Backend Python sẽ subscribe vào MQTT này để xử lý và đẩy tiếp ra React.js qua WebSocket.

---

## 4. YÊU CẦU PHI CHỨC NĂNG (NON-FUNCTIONAL REQUIREMENTS)

1. **Hiệu năng (Performance):**
    - Trình duyệt React.js không được crash hoặc giật lag khi nhận >100 sự kiện/giây từ WebSocket. (Yêu cầu kỹ thuật: Tối ưu Virtual DOM rendering).
    - Các truy vấn log trong khoảng thời gian 7 ngày phải trả về kết quả < 2 giây.
2. **Tính sẵn sàng (Availability):**
    - Quá trình huấn luyện lại (Retrain) mô hình ngốn 100% GPU/CPU không được làm gián đoạn khả năng tiếp nhận Log và Cảnh báo của các API chính. (Yêu cầu kỹ thuật: Tách biệt Worker process và API process).
3. **Khả năng Tái tạo (Reproducibility):**
    - Mỗi khi Retrain, hệ thống phải tự động lưu lại phiên bản bộ tiền xử lý (Scaler, Encoder) cùng với file .onnx tương ứng. Không được có tình trạng mô hình mới nhưng dùng Scaler cũ.