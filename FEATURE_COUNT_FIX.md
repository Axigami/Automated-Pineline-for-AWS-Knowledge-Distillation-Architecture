# Fix: Dynamic Feature Count (66 vs 75)

## Vấn Đề

Scaler có **66 features** nhưng code hardcode **75 features** → Shape mismatch error

## Giải Pháp

### ✅ Lambda Code (IOT-PROJECT.py)
- `build_payload()`: Dùng `len(vec_scaled)` thay vì hardcode 75
- `predict()`: Validate `len(LABEL_MAP)` thay vì hardcode 7
- **Dynamic**: Tự động adapt với số features từ scaler

### ✅ Training Scripts
**finetune/FineTuneTeacher.py**:
```python
# Trước: N_FEATURES = 75  # Hardcoded
# Sau:
def get_n_features():
    obj = s3_client.get_object(Bucket=BUCKET, Key=SCALER_KEY)
    scaler = json.loads(obj['Body'].read())
    return scaler['n_features']

N_FEATURES = get_n_features()  # Dynamic: 66 hoặc 75
```

**distill/IOT-PROJECT.py**:
- Tương tự load `N_FEATURES` từ S3
- Validate và pad/truncate nếu CSV không khớp

### ✅ README.md
- Thêm section "Scaler Configuration"
- Cập nhật troubleshooting cho feature mismatch
- Sửa test script để load n_features từ scaler

## Kiểm Tra

```bash
# 1. Verify scaler đã upload
aws s3 ls s3://anomalytraffic/data/raw/log/scaler_stats.json

# 2. Xem nội dung
aws s3 cp s3://anomalytraffic/data/raw/log/scaler_stats.json - | jq '.n_features'

# 3. Test Lambda với flow data thật
aws lambda invoke \
  --function-name IOT-PROJECT \
  --payload file://test_flow.json \
  response.json
```

## Kết Quả

✅ Lambda tự động adapt với 66 features  
✅ Training scripts load từ S3  
✅ Không cần hardcode  
✅ Dễ thay đổi số features sau này  

## Files Đã Sửa

1. `Lambda code/IOT-PROJECT.py` - Dynamic payload building
2. `finetune/FineTuneTeacher.py` - Load N_FEATURES từ S3
3. `distill/IOT-PROJECT.py` - Load N_FEATURES từ S3
4. `README.md` - Documentation update

---

**Ngày**: 2026-04-16  
**Status**: ✅ Complete
