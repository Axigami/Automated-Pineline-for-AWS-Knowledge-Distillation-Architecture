# Pipeline Chưng Cất Mô Hình IoT — AWS Architecture

> **Tổng quan:** Hệ thống tự động phát hiện bất thường mạng IoT, thu thập mâu thuẫn dự đoán, tự gán nhãn lại và chưng cất mô hình nhỏ hơn (LightGBM) từ mô hình thầy (Teacher — SageMaker) để cuối cùng xuất ra file ONNX nhẹ, sẵn sàng triển khai trên thiết bị biên.

---

## Tài nguyên AWS sử dụng

| Tài nguyên | Tên / ID | Vai trò |
|---|---|---|
| **S3 Bucket** | `anomalytraffic` | Lưu trữ trung tâm cho data thô, dự đoán, models |
| **DynamoDB** | `AnomalyPredictions` | Kết quả dự đoán của mô hình cloud từ luồng anomaly |
| **DynamoDB** | `LogPredictions` | Kết quả dự đoán của mô hình cloud từ luồng log thông thường |
| **DynamoDB** | `AnomalyConflicts` | Các bản ghi mâu thuẫn cần xử lý |
| **DynamoDB** | `distillationjobs` | Trạng thái các job chưng cất |
| **SageMaker Endpoint** | `tf-endpoint` | Mô hình Teacher chạy inference |
| **Script** | `DeployCloudmodel.py` | Tạo & deploy SageMaker Endpoint từ model đã train |
| **Lambda** | `IOT-PROJECT` | Inference của mô hình cloud + phát hiện conflict |
| **Lambda** | `Distillation` | Scheduler kiểm tra threshold |
| **Lambda** | `Relabel` | Tự động gán nhãn lại conflict |
| **Lambda** | `PrepareDistillationData` | Xuất CSV training data |
| **Lambda** | `ExportONNX` | Convert LightGBM → ONNX |
| **EventBridge** | CloudWatch Events | Trigger Distillation theo lịch |

---

## Cấu trúc S3 Bucket `anomalytraffic`

```
anomalytraffic/
├── data/
│   ├── anomalies/anomaly/      ← Input: file JSON luồng bất thường (từ thiết bị)
│   ├── raw/log/                ← Input: file JSON luồng log thông thường (từ thiết bị)
│   │   └── scaler_stats.json   ← Thống kê scaler (mean, scale, feature_names)
│   └── distillation/
│       └── train/              ← Output CSV dùng để train Student model
├── predictions/
│   ├── anomalyprediction/      ← Kết quả dự đoán của mô hình cloud từ luồng anomaly
│   └── logprediction/          ← Kết quả dự đoán của mô hình cloud từ luồng log
├── models/
│   ├── lightgbm/               ← Student model sau khi train (file .txt)
│   └── onnx/                   ← Model đã convert sang ONNX
└── scripts/                    ← Lambda deployment packages
```

---

## Luồng Hoạt Động Toàn Hệ Thống

```
┌─────────────┐       S3 Event       ┌──────────────────┐
│  IoT Device │ ──── upload JSON ───► │  Lambda          │
│  / Sensor   │                       │  IOT-PROJECT.py  │
└─────────────┘                       └────────┬─────────┘
                                               │
                              ┌────────────────┼─────────────────┐
                              │                │                 │
                              ▼                ▼                 ▼
                      SageMaker          DynamoDB            S3 (predictions/)
                      Endpoint           AnomalyPredictions  _pred.json
                      (inference)        LogPredictions
                              │
                        Conflict?
                              │ YES
                              ▼
                       DynamoDB
                       AnomalyConflicts
                       (status: pending)

                              │
                    ┌─────────┴──────────┐
                    │  EventBridge       │ (mỗi N giờ / ngày)
                    │  → Lambda          │
                    │  Distillation.py   │
                    └─────────┬──────────┘
                              │ count >= THRESHOLD?
                              │ YES
                              ▼
                       Lambda: Relabel.py
                       (async invoke)
                              │
                    ┌─────────┴──────────┐
                    │  Gán nhãn tự động  │
                    │  → status:         │
                    │    relabeled       │
                    │    (high/low conf) │
                    └─────────┬──────────┘
                              │ (trigger tiếp theo — manual hoặc EventBridge khác)
                              ▼
                  Lambda: PrepareDistillationData.py
                  Query AnomalyConflicts WHERE
                  status='relabeled' AND relabel_confidence='high'
                              │
                              ▼
                   Xuất CSV → S3:
                   data/distillation/train/distillation_train_*.csv
                              │
                              ▼
                  [Train LightGBM Student Model]
                  (SageMaker Training Job hoặc EC2)
                              │
                              ▼
                   Model .txt → S3: models/lightgbm/
                              │
                              ▼
                  Lambda: ExportONNX.py
                  Convert LightGBM → ONNX
                              │
                              ▼
                   ONNX Model → S3: models/onnx/
                   ✅ Sẵn sàng deploy lên Edge Device
```

---

## Chi Tiết Từng Bước

### Bước 0 — Deploy Cloud Model Real Time (`DeployCloudmodel.py`)

**Mục đích:** Khởi tạo SageMaker Endpoint `tf-endpoint` — đây là **điều kiện tiên quyết** để toàn bộ pipeline hoạt động. Script này cần chạy **một lần duy nhất** (hoặc mỗi khi cần redeploy model mới) trước khi các Lambda bắt đầu gọi inference.

**Chạy tại:** SageMaker Notebook Instance hoặc môi trường có `sagemaker` SDK và IAM Role hợp lệ.

**Code:**

```python
import sagemaker
from sagemaker.tensorflow import TensorFlowModel

role = sagemaker.get_execution_role()

model = TensorFlowModel(
    model_data="s3://anomalytraffic/models/cloud/model.tar.gz",
    role=role,
    framework_version="2.12"
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="tf-endpoint"
)
![alt text](image.png)
```

**Quy trình:**
1. Lấy IAM Execution Role từ SageMaker environment (`get_execution_role()`)
2. Khởi tạo `TensorFlowModel` trỏ đến file model đã đóng gói tại `s3://anomalytraffic/models/cloud/model.tar.gz`
3. Deploy model lên một instance `ml.m5.large` (1 instance) với tên endpoint `tf-endpoint`
4. Sau khi deploy xong, `tf-endpoint` sẵn sàng nhận request inference từ Lambda `IOT-PROJECT`

**Cấu hình quan trọng:**

| Tham số | Giá trị | Mô tả |
|---|---|---|
| `model_data` | `s3://anomalytraffic/models/cloud/model.tar.gz` | S3 URI của model artifact (TF SavedModel nén) |
| `framework_version` | `2.12` | Phiên bản TensorFlow của serving container |
| `instance_type` | `ml.m5.large` | Loại instance cho inference |
| `initial_instance_count` | `1` | Số instance khởi đầu |
| `endpoint_name` | `tf-endpoint` | Tên endpoint — **phải khớp** với biến `SAGEMAKER_ENDPOINT` trong Lambda |

> **Lưu ý quan trọng:**
> - Model artifact phải được đóng gói đúng định dạng TensorFlow SavedModel: `model.tar.gz` chứa thư mục `1/` (hoặc `00001/`) bên trong
> - `framework_version="2.12"` phải tương thích với TensorFlow version dùng khi training
> - Quá trình deploy thường mất **3–5 phút** — script sẽ block và chờ đến khi endpoint `InService`
> - IAM Role cần có permission: `sagemaker:CreateModel`, `sagemaker:CreateEndpointConfig`, `sagemaker:CreateEndpoint`, và `s3:GetObject` trên bucket `anomalytraffic`

---

### Bước 1 — Inference & Phát hiện Conflict (`IOT-PROJECT.py`)

**Trigger:** S3 Event khi có file JSON mới upload vào `data/`

**Logic routing theo đường dẫn S3:**

| S3 Prefix | Loại nguồn | Expected Label | DynamoDB Target |
|---|---|---|---|
| `data/anomalies/anomaly/` | Luồng bất thường (attack) | `attack` | `AnomalyPredictions` |
| `data/raw/log/` | Luồng log thường | `benign` | `LogPredictions` |

**Quy trình xử lý mỗi file:**
1. Đọc file JSON từ S3 (hỗ trợ multi-line NDJSON)
2. **Feature Engineering** cho mỗi flow:
   - Trích xuất 28 đặc trưng số từ thống kê gói tin
   - Tính toán các tỉ lệ dẫn xuất: `pkt_per_byte_ratio`, `flow_symmetry`, `byte_symmetry`, flag ratios (SYN, ACK, PSH,...), port bucket
   - Mã hóa `application_category_name` (28 categories → số nguyên)
3. **Chuẩn hóa** vector đặc trưng với `scaler_stats.json` (z-score: `(x - mean) / scale`)
4. **Gọi SageMaker Endpoint** → nhận xác suất 5 lớp: `Benign / Botnet / DDoS / DoS / PortScan`
5. **Phát hiện Conflict** theo quy tắc:

```
Nguồn anomaly  → Dự đoán Benign     CONFLICT (model bỏ sót tấn công)
Nguồn log      → Dự đoán Attack     CONFLICT (model báo nhầm lưu lượng bình thường)
```

6. Ghi **tất cả kết quả** vào DynamoDB (AnomalyPredictions / LogPredictions)
7. Ghi **conflict** riêng vào `AnomalyConflicts` với `status: "pending"`
8. Lưu file kết quả tổng hợp `*_pred.json` vào S3 `predictions/`

---

### Bước 2 — Scheduler Kiểm tra Ngưỡng (`Distillation.py`)

**Trigger:** EventBridge (CloudWatch Events) — chạy định kỳ (ví dụ: mỗi 6 giờ hoặc 1 ngày)

**Biến môi trường cần thiết:**

| Tên biến | Ví dụ | Mô tả |
|---|---|---|
| `CONFLICTS_TABLE` | `AnomalyConflicts` | Tên bảng DynamoDB |
| `THRESHOLD` | `100` | Số conflict tối thiểu để kích hoạt |
| `RELABEL_FUNCTION` | `Relabel` | Tên Lambda cần invoke |

**Quy trình:**
1. Query DynamoDB `AnomalyConflicts` qua GSI `status-index` với `status = "pending"`
2. Đếm tổng số conflict (có xử lý pagination)
3. Nếu `count >= THRESHOLD` → **async invoke** Lambda `Relabel`
4. Nếu chưa đủ → log và kết thúc

>  **Lý do dùng async invoke:** Relabel có thể xử lý hàng trăm records, tránh timeout Lambda 15 phút

---

### Bước 3 — Tự Động Gán Nhãn Lại (`Relabel.py`)

**Trigger:** Được invoke async từ `Distillation.py`

**Quy tắc gán nhãn theo Route:**

| Expected | Actual (Model dự đoán) | Kết quả | Confidence |
|---|---|---|---|
| `attack` | `Benign` | `attack_needs_review` | `low`  Cần review thủ công |
| `Benign` | Bất kỳ attack | `Benign` | `high`  Tự động sửa |
| Khác | Khác | Giữ expected | `low`  Fallback |

**Lý do thiết kế:**
- Khi model dự đoán Benign nhưng nguồn là attack → **không biết đây là loại tấn công nào** → phải để con người xem lại
- Khi model dự đoán attack nhưng nguồn là log thường → **tin tưởng nguồn dữ liệu gốc** → sửa thành Benign với confidence cao

**Output DynamoDB (update item):**
```json
{
  "status": "relabeled",
  "correct_label": "Benign",
  "relabel_confidence": "high",
  "relabel_reason": "Log source incorrectly predicted as DDoS - corrected to Benign",
  "needs_manual_review": false,
  "relabeled_at": "2026-04-11T08:00:00+00:00"
}
```

---

### Bước 4 — Tổng Hợp Data Training (`PrepareDistillationData.py`)

**Trigger:** Thủ công hoặc EventBridge sau khi Relabel xong

**Biến môi trường cần thiết:**

| Tên biến | Ví dụ | Mô tả |
|---|---|---|
| `CONFLICTS_TABLE` | `AnomalyConflicts` | Tên bảng DynamoDB |
| `BUCKET` | `anomalytraffic` | S3 bucket đích |

**Quy trình:**
1. Query `AnomalyConflicts` với `status = "relabeled"` **VÀ** `relabel_confidence = "high"`
2. Kiểm tra: cần ít nhất **10 samples** để bắt đầu
3. Với mỗi conflict:
   - Parse `flow_data` (JSON stored as string)
   - Map label text → số: `Benign=0, Botnet=1, DDoS=2, DoS=3, PortScan=4`
   - Tạo row CSV: `{label: N, ...flow_features}`
4. Ghi CSV vào `/tmp/` rồi upload lên S3: `data/distillation/train/distillation_train_YYYYMMDD_HHMMSS.csv`
5. **Mark conflicts đã dùng**: cập nhật `status = "used"` để không train lại

> **Lưu ý quan trọng về data chưng cất:**
> - Chỉ lấy `relabel_confidence = "high"` → loại bỏ các case cần review thủ công
> - Data chưng cất **KHÔNG** phải mọi prediction, mà chỉ là các case **mà Teacher model bị nhầm** và đã được sửa lại
> - Mục đích: dạy Student model không lặp lại lỗi của Teacher

**Output:** `s3://anomalytraffic/data/distillation/train/distillation_train_20260411_080000.csv`

---

### Bước 5 — Chưng Cất Mô Hình (`Distillation.py` + SageMaker Training)

> **Lưu ý:** File `Distillation.py` hiện tại đóng vai trò **scheduler/orchestrator**, không phải phần training model thực sự. Phần training LightGBM Student model được chạy bởi một SageMaker Training Job riêng

**Flow hoàn chỉnh của Knowledge Distillation:**

```
Teacher Model (SageMaker tf-endpoint)
    │ Soft labels (probabilities) từ inference
    │
    ▼
AnomalyConflicts (DynamoDB)
    │ Cases model Teacher bị sai
    │
    ▼
Relabeled Data (high confidence only)
    │
    ▼
CSV Training Data (S3: data/distillation/train/)
    │
    ▼
LightGBM Student Model Training
(SageMaker Training Job / EC2)
    │
    ▼
model.txt (S3: models/lightgbm/)
```

---

### Bước 6 — Xuất ONNX (`ExportONNX.py`)

**Trigger:** Manual hoặc auto sau khi SageMaker Training Job hoàn thành

**Quy trình:**
1. Tìm model LightGBM mới nhất trong `s3://anomalytraffic/models/lightgbm/`
2. Download về `/tmp/model.txt`
3. Convert sang ONNX format (opset 12)
4. Upload lên `s3://anomalytraffic/models/onnx/model_{timestamp}.onnx`

> **Lưu ý triển khai ExportONNX:**
> - `pip install` trong Lambda function sẽ **chậm** và có nguy cơ timeout
> - **Khuyến nghị:** đóng gói thư viện vào Lambda Layer (lightgbm, onnxmltools, skl2onnx)
> - `FloatTensorType([None, 15])` — con số **15 features** phải khớp với số feature thực tế trong model của bạn

---

## 🔧 Biến Môi Trường Lambda Cần Cấu Hình

### Script: `DeployCloudmodel.py` (tham số hardcode trong script)
| Tham số | Giá trị | Mô tả |
|---|---|---|
| `model_data` | `s3://anomalytraffic/models/cloud/model.tar.gz` | S3 path tới model artifact |
| `framework_version` | `2.12` | TensorFlow version |
| `instance_type` | `ml.m5.large` | Loại instance SageMaker |
| `endpoint_name` | `tf-endpoint` | Tên endpoint sau khi deploy |

### Lambda: `IOT-PROJECT`
| Biến | Giá trị ví dụ | Bắt buộc |
|---|---|---|
| `SAGEMAKER_ENDPOINT` | `tf-endpoint` | ✅ |
| `OUTPUT_BUCKET` | `anomalytraffic` | ✅ |
| `REGION` | `ap-southeast-2` | ✅ |
| `CONFLICTS_TABLE` | `AnomalyConflicts` | ✅ |

### Lambda: `Distillation`
| Biến | Giá trị ví dụ | Bắt buộc |
|---|---|---|
| `CONFLICTS_TABLE` | `AnomalyConflicts` | ✅ |
| `THRESHOLD` | `100` | ✅ |
| `RELABEL_FUNCTION` | `Relabel` | ✅ |

### Lambda: `Relabel` & `PrepareDistillationData`
| Biến | Giá trị ví dụ | Bắt buộc |
|---|---|---|
| `CONFLICTS_TABLE` | `AnomalyConflicts` | ✅ |
| `BUCKET` | `anomalytraffic` | ✅ |

---

## 📊 DynamoDB Schema Chi Tiết

### Bảng `AnomalyConflicts`

| Attribute | Type | Mô tả |
|---|---|---|
| `conflict_id` | String (PK) | UUID duy nhất |
| `created_at` | Number (SK) | Unix timestamp lúc tạo |
| `status` | String | `pending` → `relabeled` → `used` |
| `flow_data` | String | JSON serialized của toàn bộ flow features |
| `expected_label` | String | `attack` hoặc `Benign` |
| `actual_prediction` | String | JSON: `{label, confidence, probabilities}` |
| `conflict_reason` | String | Mô tả lý do conflict |
| `conflict_rule` | String | Rule name từ ROUTE_MAP |
| `source_key` | String | S3 key của file nguồn |
| `device_id` | String | ID thiết bị IoT |
| `correct_label` | String | Sau relabel: nhãn đúng |
| `relabel_confidence` | String | `high` hoặc `low` |
| `needs_manual_review` | Boolean | True nếu cần xem xét thủ công |

> **GSI cần tạo:** `status-index` với partition key là `status` — bắt buộc cho Relabel.py và Distillation.py

### Bảng `AnomalyPredictions` / `LogPredictions`

| Attribute | Type | Mô tả |
|---|---|---|
| `device_id` | String (PK) | ID thiết bị |
| `sk` | String (SK) | `{timestamp}#{flow_id}#{uuid8}` |
| `label` | String | Nhãn dự đoán |
| `confidence` | Decimal | Xác suất cao nhất |
| `probabilities` | Map | Xác suất từng lớp |

---

## Lưu Ý Quan Trọng Khi Vận Hành

1. **`DeployCloudmodel.py` phải chạy trước** khi bất kỳ Lambda nào gọi SageMaker — nếu endpoint `tf-endpoint` chưa tồn tại, Lambda `IOT-PROJECT` sẽ throw `EndpointNotFound` exception.

2. **GSI `status-index`** trên bảng `AnomalyConflicts` **PHẢI** được tạo để các query hoạt động — DynamoDB không tự tạo index.

3. **ExportONNX** nên dùng **Lambda Layer** thay vì `pip install` runtime — tránh cold start chậm và timeout.

4. **Số feature trong ONNX** (`FloatTensorType([None, 15])`) phải khớp chính xác với số features thực tế trong Student model.

5. **PrepareDistillationData** chỉ query `relabel_confidence = 'high'` thông qua **FilterExpression**, không phải KeyCondition — DynamoDB sẽ scan toàn bộ `relabeled` items trước khi filter. Nếu dataset lớn, nên tạo thêm GSI compound: `(relabel_confidence, status)`.

6. **Distillation.py** chỉ invoke Relabel async — bạn cần tự kích hoạt `PrepareDistillationData` sau khi Relabel xong (qua EventBridge hoặc Step Functions).

7. **Chi phí SageMaker Endpoint:** Instance `ml.m5.large` tính phí theo giờ ngay cả khi không có request — nên **delete endpoint** khi không cần dùng và redeploy bằng `DeployCloudmodel.py` khi cần.

---

## Vòng Đời Một Conflict Record

```
[S3 upload] → IOT-PROJECT dự đoán sai
                    │
                    ▼
          AnomalyConflicts
          status: "pending"
                    │
          (Distillation.py đủ threshold)
                    │
                    ▼
          Relabel.py xử lý
          status: "relabeled"
          relabel_confidence: "high" / "low"
                    │
          PrepareDistillationData lấy chỉ "high"
                    │
                    ▼
          status: "used"
          → Đã xuất vào CSV training
          → Không được dùng lại
```

---
