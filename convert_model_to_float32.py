"""
Convert Model from Float16 to Float32
Chạy script này trên Kaggle để fix lỗi "half" precision
"""

import tensorflow as tf
import keras
import numpy as np
import os
import shutil

print("=" * 60)
print("CONVERT MODEL: FLOAT16 → FLOAT32")
print("=" * 60)

# ============================================================
# 1. LOAD MODEL
# ============================================================
print("\n📥 Step 1: Load model...")

# Thay path này
MODEL_PATH = "/kaggle/input/your-model/teacher_interleaved_multitask_best.keras"

model = keras.models.load_model(MODEL_PATH, compile=False)
print(f"✅ Model loaded")

# ============================================================
# 2. KIỂM TRA DTYPE HIỆN TẠI
# ============================================================
print("\n🔍 Step 2: Check current dtypes...")

print("\nLayer dtypes:")
for layer in model.layers[:5]:  # Show first 5 layers
    if hasattr(layer, 'dtype'):
        print(f"  {layer.name}: {layer.dtype}")

# ============================================================
# 3. CONVERT SANG FLOAT32
# ============================================================
print("\n🔄 Step 3: Convert to float32...")

# Method 1: Clone model với float32
try:
    # Get model config
    config = model.get_config()
    
    # Create new model with float32
    with tf.keras.mixed_precision.Policy('float32'):
        model_float32 = keras.Model.from_config(config)
    
    # Copy weights
    for layer, layer_float32 in zip(model.layers, model_float32.layers):
        if layer.get_weights():
            weights = [w.astype(np.float32) for w in layer.get_weights()]
            layer_float32.set_weights(weights)
    
    print("✅ Converted using clone method")
    
except Exception as e:
    print(f"⚠️ Clone method failed: {e}")
    print("Trying alternative method...")
    
    # Method 2: Cast weights directly
    model_float32 = model
    for layer in model_float32.layers:
        if layer.get_weights():
            weights = [w.astype(np.float32) for w in layer.get_weights()]
            layer.set_weights(weights)
    
    print("✅ Converted using direct cast")

# ============================================================
# 4. VERIFY CONVERSION
# ============================================================
print("\n✅ Step 4: Verify conversion...")

print("\nNew layer dtypes:")
for layer in model_float32.layers[:5]:
    if hasattr(layer, 'dtype'):
        print(f"  {layer.name}: {layer.dtype}")

# Test prediction
print("\n🧪 Testing prediction...")
test_input = np.random.randn(1, 10, 66, 1).astype(np.float32)
output = model_float32.predict(test_input, verbose=0)

if isinstance(output, list):
    print(f"✅ Prediction successful! Output shapes: {[o.shape for o in output]}")
    print(f"   Output dtypes: {[o.dtype for o in output]}")
else:
    print(f"✅ Prediction successful! Output shape: {output.shape}")
    print(f"   Output dtype: {output.dtype}")

# ============================================================
# 5. SAVE MODEL
# ============================================================
print("\n💾 Step 5: Save float32 model...")

# Save as .keras
output_keras = "/kaggle/working/teacher_float32.keras"
model_float32.save(output_keras)
print(f"✅ Saved .keras: {output_keras}")

# Save as SavedModel for TensorFlow Serving
output_savedmodel = "/kaggle/working/teacher_savedmodel"
if os.path.exists(output_savedmodel):
    shutil.rmtree(output_savedmodel)

tf.saved_model.save(model_float32, output_savedmodel)
print(f"✅ Saved SavedModel: {output_savedmodel}")

# ============================================================
# 6. CREATE TAR.GZ FOR SAGEMAKER
# ============================================================
print("\n📦 Step 6: Create tar.gz for SageMaker...")

import tarfile

tar_path = "/kaggle/working/teacher_model_float32.tar.gz"

with tarfile.open(tar_path, "w:gz") as tar:
    # Add SavedModel directory
    tar.add(output_savedmodel, arcname=".")

print(f"✅ Created: {tar_path}")

# ============================================================
# 7. VERIFY TAR CONTENTS
# ============================================================
print("\n📋 Step 7: Verify tar contents...")

with tarfile.open(tar_path, "r:gz") as tar:
    print("Files in tar.gz:")
    for member in tar.getmembers()[:10]:  # Show first 10 files
        print(f"  {member.name}")

# ============================================================
# 8. TEST LOADED MODEL
# ============================================================
print("\n🧪 Step 8: Test reloaded model...")

# Reload from SavedModel
reloaded = tf.saved_model.load(output_savedmodel)
print(f"✅ Reloaded SavedModel")

# Get serving function
infer = reloaded.signatures["serving_default"]
print(f"✅ Got serving signature")

# Test with correct input
test_input_dict = {
    list(infer.structured_input_signature[1].keys())[0]: 
    tf.constant(test_input, dtype=tf.float32)
}

output_reloaded = infer(**test_input_dict)
print(f"✅ Prediction successful!")
print(f"   Output keys: {list(output_reloaded.keys())}")

for key, val in output_reloaded.items():
    print(f"   {key}: shape={val.shape}, dtype={val.dtype}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("✅ CONVERSION COMPLETE!")
print("=" * 60)

print(f"""
📦 Files created:
1. {output_keras}
2. {output_savedmodel}/
3. {tar_path}

📤 Next steps:
1. Download teacher_model_float32.tar.gz
2. Upload to S3:
   aws s3 cp teacher_model_float32.tar.gz \\
       s3://anomalytraffic/models/cloud/teacher_interleaved_multitask_best.tar.gz

3. Update SageMaker endpoint (hoặc tạo mới)

4. Test Lambda - lỗi "half" sẽ biến mất!
""")

print("=" * 60)
