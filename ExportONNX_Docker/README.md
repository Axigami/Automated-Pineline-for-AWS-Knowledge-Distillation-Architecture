# Triển khai Lambda ExportONNX sử dụng Docker Container

Vì các thư viện như `numpy`, `scikit-learn`, `lightgbm`, và `onnxmltools` có dung lượng rất lớn, vượt quá giới hạn 250MB (unzipped size) của AWS Lambda (kể cả khi chia thành nhiều Layers hay upload qua S3), giải pháp tối ưu và "chuẩn chỉnh" nhất là **đóng gói tất cả vào một Docker Container Image**.

AWS Lambda hỗ trợ Container Image với giới hạn lên tới **10 GB**, dư sức để chạy các thư viện Machine Learning nặng.

## 📂 Cấu trúc thư mục

- `Dockerfile`: File cấu hình để build Docker Image dựa trên base image Lambda Python 3.12 của AWS.
- `requirements.txt`: Danh sách các thư viện gộp từ layer 1, 2, 3 (boto3, numpy, sklearn, onnx, lightgbm...).
- `ExportONNX.py`: Code lambda function (được tự động copy từ thư mục `Lambda code`).
- `build_and_push.ps1`: Script PowerShell tự động build và push Image lên Amazon ECR.

## 🚀 Hướng dẫn

### Bước 1: Điều kiện tiên quyết
1. Máy tính đã cài đặt **Docker Desktop** và đang chạy.
2. Đã cài đặt **AWS CLI** và cấu hình tài khoản (có quyền ECR & Lambda).

### Bước 2: Chạy script Build & Push
Mở **PowerShell**, di chuyển vào đúng thư mục `ExportONNX_Docker` và chạy lệnh sau:

```powershell
./build_and_push.ps1
```

Script này sẽ tự động:
1. Đăng nhập vào Amazon ECR (Elastic Container Registry).
2. Tạo ECR Repository có tên `export-onnx-lambda` (nếu chưa có).
3. Build Docker container chứa code và toàn bộ các thư viện nặng.
4. Push container lên AWS ECR.

### Bước 3: Tạo/Cập nhật AWS Lambda Function lấy nguồn từ Container
1. Truy cập [AWS Lambda Console](https://console.aws.amazon.com/lambda/).
2. Chọn **Create function**.
3. Chọn tùy chọn **Container image** (thay vì "Author from scratch").
4. Đặt tên Hàm (ví dụ: `ExportONNXFunction`).
5. Ở mục **Container image URI**, nhấn **Browse images**, chọn repo `export-onnx-lambda` vừa push và chọn tag `latest`.
6. Nhấn **Create function**.

*Lưu ý: Nếu Lambda cần thêm quyền S3, hãy vào tab **Configuration** > **Permissions** để thêm IAM role cho nó. Cấp Memory cho function lớn một chút (ví dụ 1024MB - 2048MB) để quá trình convert model không bị lỗi Out of Memory.*

### Bước 4 (Optional): Cập nhật function khi thay đổi code
Nếu bạn sửa code trong `ExportONNX.py`, bạn chỉ cần chạy lại `./build_and_push.ps1`. Sau đó vào AWS Lambda Console:
1. Vào function của bạn.
2. Tại tab **Image configuration**, nhấn **Deploy new image**.
3. Chọn lại image name: `export-onnx-lambda:latest` và nhấn **Save**.
