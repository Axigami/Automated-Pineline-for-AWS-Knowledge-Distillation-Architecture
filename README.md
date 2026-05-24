# Pipeline Phát Hiện Tấn Công Mạng - AWS

Hệ thống tự động phát hiện tấn công mạng sử dụng Knowledge Distillation: Model Teacher 7 lớp (cloud) → Model Student nhị phân (edge).

## 🎯 Kiến Trúc

**Teacher (Cloud)**: CNN-LSTM 7 lớp → Benign, Botnet, BruteForce, DDoS, DoS, PortScan, WebAttack  
**Student (Edge)**: LightGBM nhị phân → Benign vs Attack  
**Chiến lược**: Chuyển kiến thức 7 lớp sang model nhị phân đơn giản cho IoT

---

## 🚀 Triển Khai Nhanh

### 1. Deploy Model Teacher

```bash
# Export model (tương thích CPU)
python export_model_cpu_compatible.py

# Đóng gói cho SageMaker
tar -czf teacher_model.tar.gz -C teacher_savedmodel .

# Upload lên S3
aws s3 cp teacher_model.tar.gz \
    s3://anomalytraffic/models/cloud/teacher_interleaved_multitask_best.tar.gz

# Cập nhật endpoint
aws sagemaker update-endpoint --endpoint-name tf-endpoint --endpoint-config-name <config-mới>
```

### 2. Deploy Lambda Functions

Tất cả Lambda trong `Lambda code/` đã sẵn sàng:
- `IOT-PROJECT.py` - Inference chính (xử lý 7 lớp)
- `Relabel.py` - Gán nhãn lại conflicts
- `PrepareDistillationData.py` - Chuẩn bị dữ liệu training
- `Triggerfinetuning.py` - Kích hoạt fine-tune teacher
- `TriggerDistillation.py` - Kích hoạt training student

---

## ⚠️ Lỗi Thường Gặp & Cách Sửa

### Lỗi CudnnRNN
**Vấn đề**: Model train trên GPU không chạy được trên CPU instance  
**Giải pháp**: Dùng `export_model_cpu_compatible.py` - force CPU mode khi export

### Lỗi Shape Sai
**Vấn đề**: 
- `convolution input must be 4-dimensional: [1,1,4]`
- Model CNN cần input 4D: `(batch, 10, n_features, 1)`

**Giải pháp**: 
- ✅ Lambda đã sửa - replicate single flow 10 lần tạo sequence
- ✅ Add batch dimension: `sequence[np.newaxis, ...]` → shape `(1, 10, n_features, 1)`
- ✅ Ensure float32: `vec_scaled.astype(np.float32)` tránh lỗi "half" precision

### Lỗi Data Type (half precision)
**Vấn đề**: `Failed to write JSON value for tensor type: half`  
**Giải pháp**: Force `float32` trong `build_payload()` trước khi serialize JSON

### Lỗi Feature Count Mismatch (66 vs 75)
**Vấn đề**: Scaler có 66 features nhưng code hardcode 75 features  
**Giải pháp**: 
- ✅ Tất cả scripts tự động load từ `scaler_stats.json`
- Lambda: `build_payload()` dùng `len(vec_scaled)` (dynamic)
- Training: `N_FEATURES = get_n_features()` load từ S3
- Đảm bảo `scaler_stats.json` đã upload: `s3://anomalytraffic/data/raw/log/scaler_stats.json`

### Dự Đoán Sai
**Giải pháp**: 
1. Kiểm tra `scaler_stats.json` khớp với training data
2. Đảm bảo feature engineering giống training code
3. Verify đủ 75 features

---

## 📁 Cấu Trúc Project

```
.
├── Lambda code/              # Lambda functions (tương thích 7 lớp)
├── finetune/                 # Fine-tune teacher
├── distill/                  # Training student
├── export_model_cpu_compatible.py  # Script export CPU-compatible
└── README.md
```

---

## 🔧 Chi Tiết Model

### Teacher (7 lớp)
- Input: `(batch, 10, n_features, 1)` - chuỗi 10 flows (n_features từ scaler)
- Output: `(batch, 7)` - xác suất 7 lớp
- Kiến trúc: CNN-LSTM xen kẽ
- Training: Focal loss với class weights

### Student (2 lớp)
- Input: `(batch, n_features)` - single flow (n_features từ scaler)
- Output: `(batch,)` - xác suất Attack
- Training: LightGBM nhị phân với soft labels từ teacher
- Công thức: P(Attack) = tổng P(6 lớp attack)

### Lambda Inference
- Tạo sequence bằng cách nhân đôi single flow 10 lần
- Xử lý multi-output model (lấy window_output)
- Validate 7 lớp
- Phát hiện conflicts dựa trên nguồn dữ liệu

---

## 📊 Luồng Dữ Liệu

```
Edge Device → S3 (JSON) → Lambda
                            ↓
                  Feature Engineering (n_features từ scaler)
                            ↓
                  Chuẩn hóa (scaler_stats.json)
                            ↓
                  Tạo sequence (10 flows)
                            ↓
                  Teacher dự đoán (7 lớp)
                            ↓
                  Phát hiện conflicts
                            ↓
                  DynamoDB + S3
```

---

## 🔑 Tính Năng Chính

✅ Teacher 7 lớp phân loại chi tiết  
✅ Student nhị phân nhanh cho edge  
✅ Tương thích CPU (không cần GPU)  
✅ Pipeline tự động học từ conflicts  
✅ Phát hiện conflicts theo route  
✅ Xử lý chuỗi thời gian  

---

## 📝 Cấu Hình

### Biến Môi Trường (Lambda)
```bash
SAGEMAKER_ENDPOINT=tf-endpoint
OUTPUT_BUCKET=anomalytraffic
REGION=ap-southeast-2
CONFLICTS_TABLE=AnomalyConflicts
```

### Scaler Configuration
**File**: `s3://anomalytraffic/data/raw/log/scaler_stats.json`

Format:
```json
{
  "n_features": 66,
  "feature_names": ["bidirectional_packets", "bidirectional_bytes", ...],
  "mean": [0.123, 0.456, ...],
  "scale": [1.234, 2.345, ...]
}
```

**⚠️ QUAN TRỌNG**: 
- Tất cả scripts tự động load `n_features` từ file này
- Không cần hardcode số lượng features
- Nếu không tìm thấy, mặc định dùng 75 features

### Cấu Trúc S3
```
s3://anomalytraffic/
├── data/
│   ├── raw/log/              # Traffic bình thường
│   ├── anomalies/anomaly/    # Traffic tấn công
│   └── distillation/train/   # Dữ liệu training
├── models/
│   ├── cloud/                # Model teacher
│   └── edge/lightgbm/        # Model student
└── predictions/              # Kết quả dự đoán
```

### DynamoDB Tables
- `AnomalyPredictions` - Dự đoán từ nguồn anomaly
- `LogPredictions` - Dự đoán từ nguồn log
- `AnomalyConflicts` - Conflicts phát hiện được

---

## 🧪 Test

```bash
# Test endpoint (với 66 features từ scaler)
python -c "
import boto3, json, numpy as np
sm = boto3.client('sagemaker-runtime')
# Load scaler để lấy n_features
s3 = boto3.client('s3')
obj = s3.get_object(Bucket='anomalytraffic', Key='data/raw/log/scaler_stats.json')
scaler = json.loads(obj['Body'].read())
n_feat = scaler['n_features']
print(f'Using {n_feat} features')
data = np.random.randn(1, 10, n_feat, 1).tolist()
resp = sm.invoke_endpoint(
    EndpointName='tf-endpoint',
    ContentType='application/json',
    Body=json.dumps({'instances': data})
)
print(json.loads(resp['Body'].read()))
"

# Test Lambda
aws lambda invoke --function-name IOT-PROJECT --payload file://test.json response.json
```

---

## 📚 7 Lớp Phân Loại

- **Benign** (0): Traffic bình thường
- **Botnet** (1): Hoạt động botnet
- **BruteForce** (2): Tấn công brute force
- **DDoS** (3): Tấn công từ chối dịch vụ phân tán
- **DoS** (4): Tấn công từ chối dịch vụ
- **PortScan** (5): Quét cổng
- **WebAttack** (6): Tấn công web

---

## 🔄 Vòng Lặp Cải Tiến

1. Edge gửi flow data lên S3
2. Lambda xử lý với teacher (7 lớp)
3. Phát hiện conflicts (theo route)
4. Conflicts tích lũy trong DynamoDB
5. Khi đạt ngưỡng (100 conflicts):
   - Gán nhãn lại conflicts
   - Chuẩn bị CSV training
   - Fine-tune teacher (nếu cải thiện ≥ 0.1%)
   - Distill sang student nhị phân
   - Deploy models mới
6. Lặp lại

---

## � Lưu Ý

- **CPU vs GPU**: Luôn dùng `export_model_cpu_compatible.py` để deploy CPU
- **Sequences**: Lambda nhân đôi single flow 10 lần - chấp nhận được cho inference
- **Conflicts**: Chỉ dùng conflicts confidence cao (≥0.85) để training
- **Distillation**: Temperature=3.0 cho soft labels
- **Chi phí**: CPU instance rẻ hơn GPU ~10 lần

---

## 📞 Debug

Xem logs CloudWatch:
```bash
# Lambda logs
aws logs tail /aws/lambda/IOT-PROJECT --follow

# SageMaker endpoint logs
aws logs tail /aws/sagemaker/Endpoints/tf-endpoint --follow
```

---

**Cập nhật**: 2026-04-16  
**Phiên bản**: Teacher 7 lớp + Student nhị phân  
**Trạng thái**: Sẵn sàng production ✅
