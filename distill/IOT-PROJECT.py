# =============================================================
# Knowledge Distillation - BINARY STUDENT FROM 7-CLASS TEACHER
# Teacher : CNN-LSTM 7-class Sequence Model via SageMaker Endpoint
# Student : LightGBM Binary (Benign vs Attack) → saved as student_lgbm.txt
# Output  : s3://anomalytraffic/models/edge/lightgbm/
# 
# Strategy: Collapse 7-class teacher predictions into binary
#   - Benign (class 0) → Benign
#   - All attacks (classes 1-6) → Attack
# =============================================================

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import traceback
from datetime import datetime

import boto3

# ── Clients (initialised once at import time) ────────────────────────────────
s3_client      = boto3.client('s3')
sm_runtime     = boto3.client('sagemaker-runtime')
lambda_client  = boto3.client('lambda')

BUCKET          = os.environ.get('BUCKET', 'anomalytraffic')
TEACHER_ENDPOINT = os.environ.get('TEACHER_ENDPOINT', 'tf-endpoint')
EXPORT_FUNCTION  = os.environ.get('EXPORT_FUNCTION', 'ExportONNX')
MODEL_S3_PREFIX  = 'models/edge/lightgbm/'
SCALER_KEY      = 'data/raw/log/scaler_stats.json'

# Teacher 7-class configuration
TEACHER_CLASSES = ['Benign', 'Botnet', 'BruteForce', 'DDoS', 'DoS', 'PortScan', 'WebAttack']
N_TEACHER_CLASSES = len(TEACHER_CLASSES)
WINDOW_SIZE = 10  # Teacher expects sequences of 10 flows

# Load feature count dynamically from scaler
_scaler_cache = None

def get_n_features():
    """Load feature count from scaler_stats.json"""
    global _scaler_cache
    if _scaler_cache is None:
        try:
            obj = s3_client.get_object(Bucket=BUCKET, Key=SCALER_KEY)
            _scaler_cache = json.loads(obj['Body'].read())
            print(f"✅ Scaler loaded: {_scaler_cache['n_features']} features")
        except Exception as e:
            print(f"⚠️ Failed to load scaler, defaulting to 75 features: {e}")
            _scaler_cache = {'n_features': 75}
    return _scaler_cache['n_features']

N_FEATURES = get_n_features()

# Student binary configuration
STUDENT_CLASSES = ['Benign', 'Attack']
N_STUDENT_CLASSES = 2

# ── Debug header ─────────────────────────────────────────────────────────────
print("🔥 Knowledge Distillation - Binary Student from 7-Class Teacher")
print("=" * 60)
print("Python version:", sys.version)
print(f"Teacher: {N_TEACHER_CLASSES} classes - {TEACHER_CLASSES}")
print(f"Student: {N_STUDENT_CLASSES} classes - {STUDENT_CLASSES}")
print(f"Teacher input: ({WINDOW_SIZE}, {N_FEATURES}, 1)")
print("Command-line args:", sys.argv)
print("SageMaker env vars:")
for k, v in os.environ.items():
    if k.startswith("SM_"):
        print(f"  {k} = {v}")
print("=" * 60)


# ── Args ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--num-leaves",    "--num_leaves",    type=int,   default=127,  dest="num_leaves")
    p.add_argument("--max-depth",     "--max_depth",     type=int,   default=6,    dest="max_depth")
    p.add_argument("--learning-rate", "--learning_rate", type=float, default=0.05, dest="learning_rate")
    p.add_argument("--n-estimators",  "--n_estimators",  type=int,   default=400,  dest="n_estimators")
    p.add_argument("--temperature",   type=float, default=3.0)
    p.add_argument("--use-soft-labels","--use_soft_labels", type=str, default="true", dest="use_soft_labels")

    p.add_argument("--train",        type=str,
                   default=os.environ.get("SM_CHANNEL_TRAINING", "./data"))
    p.add_argument("--model-dir",    "--model_dir", type=str,
                   default=os.environ.get("SM_MODEL_DIR", "./model"), dest="model_dir")
    p.add_argument("--output-data-dir","--output_data_dir", type=str,
                   default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"), dest="output_data_dir")

    args, unknown = p.parse_known_args()
    if unknown:
        print("⚠️ Ignoring unknown args:", unknown)

    args.use_soft_labels = str(args.use_soft_labels).lower() == "true"
    print("✅ Parsed args:", vars(args))

    if not os.path.exists(args.train):
        raise RuntimeError(f"❌ Training path not found: {args.train}")

    return args


# ── Load data ────────────────────────────────────────────────────────────────
def load_data(data_dir):
    """
    Load CSV data prepared by PrepareDistillationData.
    Expected format: 'label' column (0-6 for 7 classes) + 75 feature columns
    
    Converts 7-class labels to binary:
    - 0 (Benign) → 0 (Benign)
    - 1-6 (Attacks) → 1 (Attack)
    """
    print(f"\n📂 Looking for CSV in: {data_dir}")

    csvs = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csvs:
        raise RuntimeError(f"❌ No CSV found in {data_dir}")

    # Load all CSVs
    dfs = []
    for csv_file in csvs:
        csv_path = os.path.join(data_dir, csv_file)
        print(f"📄 Loading: {csv_path}")
        df = pd.read_csv(csv_path)
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"📋 Total shape: {df.shape} | Columns: {list(df.columns)[:10]}...")

    # Check for label column
    if 'label' not in df.columns:
        raise RuntimeError(f"❌ 'label' column not found. Available: {df.columns.tolist()}")
    
    # Extract features (all columns except 'label')
    feat_cols = [c for c in df.columns if c != 'label']
    
    if not feat_cols:
        raise RuntimeError("❌ No feature columns found")

    X = df[feat_cols].astype("float32").values
    y_multiclass = df['label'].values.astype(int)
    
    # Validate feature count
    n_feat = X.shape[1]
    print(f"📐 CSV features: {n_feat}, scaler expects: {N_FEATURES}")
    
    if n_feat != N_FEATURES:
        print(f"⚠️ Feature count mismatch!")
        if n_feat < N_FEATURES:
            pad_width = N_FEATURES - n_feat
            print(f"  ➕ Padding with {pad_width} zeros")
            pad = np.zeros((X.shape[0], pad_width), dtype='float32')
            X = np.hstack([X, pad])
        else:
            print(f"  ✂️ Truncating from {n_feat} to {N_FEATURES}")
            X = X[:, :N_FEATURES]
    else:
        print(f"✅ Feature count matches scaler")
    
    # Convert to binary: 0=Benign, 1-6=Attack
    y_binary = (y_multiclass > 0).astype(int)
    
    print(f"✅ Loaded X: {X.shape}, y_multiclass: {y_multiclass.shape}")
    print(f"📊 Original 7-class distribution:")
    for label, count in zip(*np.unique(y_multiclass, return_counts=True)):
        if label < len(TEACHER_CLASSES):
            print(f"  {label} ({TEACHER_CLASSES[label]}): {count}")
    
    print(f"\n📊 Binary distribution:")
    for label, count in zip(*np.unique(y_binary, return_counts=True)):
        print(f"  {label} ({STUDENT_CLASSES[label]}): {count}")
    
    return X, y_binary, feat_cols


# ── Soft labels via SageMaker Endpoint ──────────────────────────────────────
def get_soft_labels(X_flat, temperature):
    """
    Call tf-endpoint to get teacher soft labels (7-class), then collapse to binary.
    
    Teacher expects sequences: (batch, 10, 75, 1)
    Teacher outputs: (batch, 7) probabilities
    
    Collapse strategy:
    - P(Benign) = teacher_probs[0]
    - P(Attack) = sum(teacher_probs[1:6])
    
    Returns:
        soft_probs_binary: (N_windows,) - probability of Attack class
    """
    print(f"\n🍯 Getting soft labels from 7-class teacher: {TEACHER_ENDPOINT}")
    print(f"📐 Input shape: {X_flat.shape}")

    # Pad to 75 features if needed
    n_feat = X_flat.shape[1]
    if n_feat < N_FEATURES:
        pad = np.zeros((X_flat.shape[0], N_FEATURES - n_feat), dtype='float32')
        X_flat = np.hstack([X_flat, pad])
        print(f"📐 Padded to: {X_flat.shape}")
    elif n_feat > N_FEATURES:
        X_flat = X_flat[:, :N_FEATURES]
        print(f"📐 Truncated to: {X_flat.shape}")

    # Create sequences (tumbling window)
    n_samples = len(X_flat)
    n_windows = n_samples // WINDOW_SIZE
    
    if n_samples < WINDOW_SIZE:
        print(f"⚠️ Only {n_samples} samples, padding to create 1 window")
        pad_len = WINDOW_SIZE - n_samples
        X_flat = np.vstack([
            np.zeros((pad_len, N_FEATURES), dtype='float32'),
            X_flat
        ])
        n_windows = 1
    else:
        # Drop remainder to fit window size
        remainder = n_samples % WINDOW_SIZE
        if remainder > 0:
            print(f"  ℹ️ Dropping last {remainder} samples to fit window size")
            X_flat = X_flat[:n_samples - remainder]
            n_samples = len(X_flat)
            n_windows = n_samples // WINDOW_SIZE
    
    # Reshape to sequences: (N_windows, 10, 75, 1)
    X_seq = X_flat.reshape(n_windows, WINDOW_SIZE, N_FEATURES, 1)
    print(f"✅ Created {n_windows} sequences of shape {X_seq.shape}")

    # Call endpoint in batches
    batch_size = 32  # Smaller batch for sequence model
    all_probs_7class  = []

    for i in range(0, len(X_seq), batch_size):
        batch   = X_seq[i:i + batch_size].tolist()
        payload = json.dumps({"instances": batch})

        response = sm_runtime.invoke_endpoint(
            EndpointName=TEACHER_ENDPOINT,
            ContentType='application/json',
            Body=payload
        )
        result = json.loads(response['Body'].read())
        
        # Handle both single and multi-output models
        preds = result.get('predictions', result.get('outputs', []))
        
        # If multi-output, take window_output (last output)
        if isinstance(preds[0], list) and len(preds[0]) > 1:
            probs = np.array([p[-1] if isinstance(p, list) else p for p in preds])
        else:
            probs = np.array(preds)
        
        all_probs_7class.append(probs)

        if (i // batch_size) % 10 == 0:
            print(f"  batch {i // batch_size + 1}: {probs.shape}")

    probs_7class = np.vstack(all_probs_7class)   # (N_windows, 7)
    print(f"✅ Teacher 7-class output shape: {probs_7class.shape}")

    # Temperature scaling on 7-class logits
    if temperature != 1.0:
        logits = np.log(probs_7class + 1e-9) / temperature
        probs_7class = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        print(f"🌡️ Temperature scaling applied (T={temperature})")

    # Collapse to binary: P(Attack) = sum of all attack classes
    prob_benign = probs_7class[:, 0]  # Class 0
    prob_attack = np.sum(probs_7class[:, 1:], axis=1)  # Classes 1-6
    
    print(f"\n📊 Binary soft label statistics:")
    print(f"  P(Benign): min={prob_benign.min():.4f}, max={prob_benign.max():.4f}, mean={prob_benign.mean():.4f}")
    print(f"  P(Attack): min={prob_attack.min():.4f}, max={prob_attack.max():.4f}, mean={prob_attack.mean():.4f}")
    
    # Return probability of positive class (Attack)
    return prob_attack


# ── Train LightGBM ───────────────────────────────────────────────────────────
def train_lgbm_binary(X_tr, y_soft_tr, X_val, y_soft_val, feat_names, args):
    """
    Train LightGBM for binary classification using soft labels from teacher.
    
    Args:
        X_tr, X_val: Feature matrices
        y_soft_tr, y_soft_val: Soft probabilities for Attack class (N,)
        feat_names: Feature column names
        args: Training arguments
    """
    print("\n🌲 Training LightGBM student (binary)")

    params = {
        "objective":      "binary",
        "metric":         ["binary_logloss", "auc"],
        "learning_rate":  args.learning_rate,
        "num_leaves":     args.num_leaves,
        "max_depth":      args.max_depth,
        "verbosity":      1,
        "force_row_wise": True
    }

    dtr  = lgb.Dataset(X_tr,  y_soft_tr,  feature_name=feat_names)
    dval = lgb.Dataset(X_val, y_soft_val, feature_name=feat_names)

    model = lgb.train(
        params,
        dtr,
        num_boost_round=args.n_estimators,
        valid_sets=[dtr, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(30, verbose=True),
            lgb.log_evaluation(20)
        ]
    )

    print(f"✅ Training done. Best iteration: {model.best_iteration}")
    return model


# ── Save locally + upload directly to S3 ────────────────────────────────────
def save_and_upload(model, feat_names, metrics, args):
    print("\n💾 Saving model files")

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)

    # ── Local saves ──────────────────────────────────────────────────────────
    txt_local  = os.path.join(args.model_dir, "student_lgbm.txt")
    pkl_local  = os.path.join(args.model_dir, "student_lgbm.pkl")
    meta_local = os.path.join(args.model_dir, "metadata.json")

    model.save_model(txt_local)
    joblib.dump(model, pkl_local)

    meta = {
        "model_type":       "lightgbm_binary",
        "classes":          STUDENT_CLASSES,
        "n_classes":        N_STUDENT_CLASSES,
        "feature_names":    feat_names,
        "n_features":       len(feat_names),
        "decision_threshold": 0.5,
        "best_iteration":   model.best_iteration,
        "accuracy":         metrics.get("accuracy"),
        "roc_auc":          metrics.get("roc_auc"),
        "onnx_available":   False,   # will be True after ExportONNX runs
        "trained_at":       datetime.utcnow().isoformat(),
        "teacher_model":    "CNN-LSTM Sequence Model (7-class)",
        "teacher_classes":  TEACHER_CLASSES,
        "distillation_method": "knowledge_distillation_7to2"
    }

    with open(meta_local, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Local saves done: {args.model_dir}")

    # ── Upload directly to s3://anomalytraffic/models/edge/lightgbm/ ─────────
    uploads = [
        (txt_local,  f"{MODEL_S3_PREFIX}student_lgbm.txt"),
        (pkl_local,  f"{MODEL_S3_PREFIX}student_lgbm.pkl"),
        (meta_local, f"{MODEL_S3_PREFIX}metadata.json"),
    ]

    for local_path, s3_key in uploads:
        s3_client.upload_file(local_path, BUCKET, s3_key)
        print(f"✅ Uploaded s3://{BUCKET}/{s3_key}")

    # ── Metrics to output dir ────────────────────────────────────────────────
    metrics_path = os.path.join(args.output_data_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


# ── Trigger ExportONNX Lambda ────────────────────────────────────────────────
def trigger_export_onnx():
    print(f"\n🚀 Triggering {EXPORT_FUNCTION}...")
    try:
        lambda_client.invoke(
            FunctionName=EXPORT_FUNCTION,
            InvocationType='Event',   # async
            Payload=json.dumps({
                'model_key': f"{MODEL_S3_PREFIX}student_lgbm.txt"
            })
        )
        print(f"✅ {EXPORT_FUNCTION} triggered")
    except Exception as e:
        print(f"❌ Failed to trigger {EXPORT_FUNCTION}: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    try:
        args = parse_args()

        # 1. Load tabular data
        X, y_true, feat_cols = load_data(args.train)

        # 2. Get soft labels from tf-endpoint (7-class probabilities)
        y_soft = get_soft_labels(X, args.temperature)
        
        # Align y_true with y_soft (may have dropped samples due to windowing)
        if len(y_soft) < len(y_true):
            print(f"⚠️ Windowing reduced samples from {len(y_true)} to {len(y_soft)}")
            # Reconstruct y_true for windows (use last label in each window)
            n_windows = len(y_soft)
            y_true_windowed = y_true.reshape(n_windows, WINDOW_SIZE)[:, -1]
            y_true = y_true_windowed

        # 3. Train/val split 80/20
        split = int(0.8 * len(X))
        X_tr, X_val = X[:split],      X[split:]
        y_soft_tr, y_soft_val = y_soft[:split], y_soft[split:]
        y_true_tr, y_true_val = y_true[:split], y_true[split:]
        
        print(f"\n📐 Train/Val: {len(X_tr)}/{len(X_val)}")

        # 4. Train
        model = train_lgbm(X_tr, y_soft_tr, X_val, y_soft_val, feat_cols, args)

        # 5. Evaluate
        from sklearn.metrics import accuracy_score, classification_report
        
        val_pred_probs = model.predict(X_val)  # (N, 7)
        val_pred_class = np.argmax(val_pred_probs, axis=1)
        
        accuracy = accuracy_score(y_true_val, val_pred_class)
        
        print(f"\n📊 Validation Results:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\n📋 Classification Report:")
        print(classification_report(y_true_val, val_pred_class, 
                                   target_names=CLASS_NAMES, 
                                   zero_division=0))

        metrics = {
            "accuracy":   float(accuracy),
            "val_size":   len(X_val),
            "train_size": len(X_tr),
            "n_classes":  N_CLASSES,
            "classes":    CLASS_NAMES
        }

        # 6. Save locally + push to S3
        save_and_upload(model, feat_cols, metrics, args)

        # 7. Trigger ExportONNX
        trigger_export_onnx()

        print("\n" + "=" * 60)
        print("✅ DISTILLATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ FATAL ERROR: {type(e).__name__}: {e}")
        print("=" * 60)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
