"""
Kaggle Notebook: Test Model Input Shape
Mục đích: Tìm shape chính xác model cần để fix Lambda
"""

import numpy as np
import tensorflow as tf
import keras

# ============================================================
# 1. LOAD MODEL
# ============================================================
print("=" * 60)
print("STEP 1: Load Model")
print("=" * 60)

# Thay path này bằng path model của bạn trên Kaggle
MODEL_PATH = "/kaggle/input/your-model-path/teacher_interleaved_multitask_best.keras"

model = keras.models.load_model(MODEL_PATH, compile=False)

print(f"✅ Model loaded successfully")
print(f"\n📋 Model Summary:")
model.summary()

# ============================================================
# 2. KIỂM TRA INPUT/OUTPUT SHAPE
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Inspect Model Shapes")
print("=" * 60)

print(f"\n🔍 Input Layer:")
for i, inp in enumerate(model.inputs):
    print(f"  Input {i}: {inp.name}")
    print(f"    Shape: {inp.shape}")
    print(f"    Dtype: {inp.dtype}")

print(f"\n🔍 Output Layer(s):")
for i, out in enumerate(model.outputs):
    print(f"  Output {i}: {out.name}")
    print(f"    Shape: {out.shape}")
    print(f"    Dtype: {out.dtype}")

# ============================================================
# 3. TEST DIFFERENT SHAPES
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Test Different Input Shapes")
print("=" * 60)

# Giả sử model cần (batch, time, features, channels)
# Thử các shape khác nhau

test_shapes = [
    # Format: (description, shape)
    ("Shape 1: (1, 10, 66, 1)", (1, 10, 66, 1)),
    ("Shape 2: (1, 10, 75, 1)", (1, 10, 75, 1)),
    ("Shape 3: (10, 66, 1)", (10, 66, 1)),
    ("Shape 4: (10, 75, 1)", (10, 75, 1)),
    ("Shape 5: (1, 66, 1)", (1, 66, 1)),
    ("Shape 6: (1, 75, 1)", (1, 75, 1)),
]

for desc, shape in test_shapes:
    print(f"\n🧪 Testing {desc}")
    print(f"   Shape: {shape}")
    
    try:
        # Tạo random data
        test_data = np.random.randn(*shape).astype(np.float32)
        
        # Thử predict
        output = model.predict(test_data, verbose=0)
        
        # Thành công!
        print(f"   ✅ SUCCESS!")
        print(f"   Input shape: {test_data.shape}")
        
        if isinstance(output, list):
            print(f"   Output shapes: {[o.shape for o in output]}")
            for i, o in enumerate(output):
                print(f"     Output {i}: {o.shape}")
        else:
            print(f"   Output shape: {output.shape}")
        
        print(f"\n   🎯 THIS IS THE CORRECT SHAPE!")
        print(f"   Use this in Lambda: {shape}")
        
        # In ra sample output
        if isinstance(output, list):
            print(f"\n   Sample predictions:")
            for i, o in enumerate(output):
                print(f"     Output {i}: {o[0][:5]}...")  # First 5 values
        else:
            print(f"\n   Sample prediction: {output[0][:5]}...")
        
        break  # Tìm thấy shape đúng rồi, dừng lại
        
    except Exception as e:
        print(f"   ❌ FAILED: {str(e)[:100]}")

# ============================================================
# 4. TEST VỚI SEQUENCE DATA THẬT
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Test with Real Sequence Data")
print("=" * 60)

# Giả lập data thật: 1 flow với 66 features
single_flow = np.random.randn(66).astype(np.float32)
print(f"Single flow shape: {single_flow.shape}")

# Cách 1: Replicate 10 lần
print(f"\n📝 Method 1: Replicate single flow 10 times")
flow_2d = single_flow.reshape(66, 1)  # (66, 1)
sequence = np.tile(flow_2d, (10, 1, 1))  # (10, 66, 1)
batch_sequence = sequence[np.newaxis, ...]  # (1, 10, 66, 1)

print(f"  Flow 2D: {flow_2d.shape}")
print(f"  Sequence: {sequence.shape}")
print(f"  Batch sequence: {batch_sequence.shape}")

try:
    output = model.predict(batch_sequence, verbose=0)
    print(f"  ✅ SUCCESS with shape {batch_sequence.shape}")
    
    if isinstance(output, list):
        print(f"  Output shapes: {[o.shape for o in output]}")
    else:
        print(f"  Output shape: {output.shape}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

# Cách 2: Thử với 75 features
print(f"\n📝 Method 2: With 75 features")
single_flow_75 = np.random.randn(75).astype(np.float32)
flow_2d_75 = single_flow_75.reshape(75, 1)
sequence_75 = np.tile(flow_2d_75, (10, 1, 1))
batch_sequence_75 = sequence_75[np.newaxis, ...]

print(f"  Batch sequence: {batch_sequence_75.shape}")

try:
    output = model.predict(batch_sequence_75, verbose=0)
    print(f"  ✅ SUCCESS with shape {batch_sequence_75.shape}")
    
    if isinstance(output, list):
        print(f"  Output shapes: {[o.shape for o in output]}")
    else:
        print(f"  Output shape: {output.shape}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")

# ============================================================
# 5. KẾT LUẬN
# ============================================================
print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

print("""
Dựa trên kết quả test trên:

1. Model input shape chính xác là: _______________
2. Model output shape là: _______________
3. Lambda cần build payload với shape: _______________

Code Lambda đúng:
```python
def build_payload(vec_scaled: np.ndarray) -> str:
    n_features = len(vec_scaled)
    vec_scaled = vec_scaled.astype(np.float32)
    
    # Reshape và replicate
    flow_2d = vec_scaled.reshape(n_features, 1)
    sequence = np.tile(flow_2d, (10, 1, 1))
    batch_sequence = sequence[np.newaxis, ...]
    
    # Shape: (1, 10, n_features, 1)
    return json.dumps({"instances": batch_sequence.tolist()})
```
""")

print("\n✅ Test hoàn tất! Kiểm tra output phía trên để biết shape chính xác.")
