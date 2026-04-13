# =============================================================
# Knowledge Distillation
# Teacher : CNN via SageMaker Endpoint (tf-endpoint)
# Student : LightGBM → saved as student_lgbm.txt
# Output  : s3://anomalytraffic/models/edge/lightgbm/
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

# ── Debug header ─────────────────────────────────────────────────────────────
print("🔥 Script started")
print("=" * 60)
print("Python version:", sys.version)
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
    p.add_argument("--positive-class","--positive_class",type=int,   default=1,    dest="positive_class")
    p.add_argument("--task",          type=str,   choices=["binary", "multiclass"], default="binary")
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
    print(f"\n📂 Looking for CSV in: {data_dir}")

    csvs = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csvs:
        raise RuntimeError(f"❌ No CSV found in {data_dir}")

    df = pd.read_csv(os.path.join(data_dir, csvs[0]))
    print(f"📋 Shape: {df.shape} | Columns: {df.columns.tolist()}")

    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feat_cols:
        raise RuntimeError("❌ No 'feature_*' columns found in CSV")

    X = df[feat_cols].astype("float32").values
    print(f"✅ Loaded {X.shape} from {len(feat_cols)} features")
    return X, feat_cols


# ── Soft labels via SageMaker Endpoint ──────────────────────────────────────
def get_soft_labels(X_flat, temperature, class_idx):
    """
    Call tf-endpoint to get teacher soft labels.
    Teacher expects shape (N, 75, 1) — pad if needed.
    """
    print(f"\n🍯 Getting soft labels from endpoint: {TEACHER_ENDPOINT}")
    print(f"📐 Input shape: {X_flat.shape}")

    # Pad to 75 features
    n_feat = X_flat.shape[1]
    if n_feat < 75:
        pad = np.zeros((X_flat.shape[0], 75 - n_feat), dtype='float32')
        X_flat = np.hstack([X_flat, pad])
        print(f"📐 Padded to: {X_flat.shape}")
    elif n_feat > 75:
        X_flat = X_flat[:, :75]

    # Reshape → (N, 75, 1)
    X_reshaped = X_flat.reshape(X_flat.shape[0], 75, 1)

    # Call endpoint in batches of 100
    batch_size = 100
    all_probs  = []

    for i in range(0, len(X_reshaped), batch_size):
        batch   = X_reshaped[i:i + batch_size].tolist()
        payload = json.dumps({"instances": batch})

        response = sm_runtime.invoke_endpoint(
            EndpointName=TEACHER_ENDPOINT,
            ContentType='application/json',
            Body=payload
        )
        result = json.loads(response['Body'].read())
        probs  = np.array(result['predictions'])   # (batch, n_classes)
        all_probs.append(probs)

        if (i // batch_size) % 10 == 0:
            print(f"  batch {i // batch_size + 1}: {probs.shape}")

    probs = np.vstack(all_probs)   # (N, n_classes)
    print(f"✅ Teacher output shape: {probs.shape}")

    # Temperature scaling
    if temperature != 1.0:
        logits = np.log(probs + 1e-9) / temperature
        probs  = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        print(f"🌡️ Temperature scaling applied (T={temperature})")

    if class_idx >= probs.shape[1]:
        raise ValueError(f"❌ class_idx={class_idx} but model has {probs.shape[1]} classes")

    y_soft = probs[:, class_idx]
    print(f"✅ Soft labels: min={y_soft.min():.4f} max={y_soft.max():.4f} mean={y_soft.mean():.4f}")
    return y_soft, probs


# ── Train LightGBM ───────────────────────────────────────────────────────────
def train_lgbm(X_tr, y_tr, X_val, y_val, feat_names, args):
    print("\n🌲 Training LightGBM student")

    params = {
        "objective":      "binary",
        "metric":         ["binary_logloss", "auc"],
        "learning_rate":  args.learning_rate,
        "num_leaves":     args.num_leaves,
        "max_depth":      args.max_depth,
        "verbosity":      1,
        "force_row_wise": True
    }

    dtr  = lgb.Dataset(X_tr,  y_tr,  feature_name=feat_names)
    dval = lgb.Dataset(X_val, y_val, feature_name=feat_names)

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
        "classes":          ["Benign", "Attack"],
        "feature_names":    feat_names,
        "n_features":       len(feat_names),
        "decision_threshold": 0.9,
        "best_iteration":   model.best_iteration,
        "accuracy":         metrics.get("accuracy"),
        "roc_auc":          metrics.get("auc"),
        "onnx_available":   False,   # will be True after ExportONNX runs
        "trained_at":       datetime.utcnow().isoformat(),
        "feature_set":      "top15_correlation"
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
        X, feat_cols = load_data(args.train)

        # 2. Get soft labels from tf-endpoint
        y_soft, _ = get_soft_labels(X, args.temperature, args.positive_class)

        # 3. Train/val split 80/20
        split = int(0.8 * len(X))
        X_tr, X_val = X[:split],      X[split:]
        y_tr, y_val = y_soft[:split], y_soft[split:]
        print(f"\n📐 Train/Val: {len(X_tr)}/{len(X_val)}")

        # 4. Train
        model = train_lgbm(X_tr, y_tr, X_val, y_val, feat_cols, args)

        # 5. Evaluate
        from sklearn.metrics import roc_auc_score, accuracy_score
        val_pred        = model.predict(X_val)
        y_val_binary    = (y_val > 0.5).astype(int)
        val_pred_binary = (val_pred > 0.5).astype(int)

        metrics = {
            "auc":        float(roc_auc_score(y_val_binary, val_pred)),
            "accuracy":   float(accuracy_score(y_val_binary, val_pred_binary)),
            "val_size":   len(X_val),
            "train_size": len(X_tr)
        }
        print(f"✅ AUC: {metrics['auc']:.4f} | Accuracy: {metrics['accuracy']:.4f}")

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
