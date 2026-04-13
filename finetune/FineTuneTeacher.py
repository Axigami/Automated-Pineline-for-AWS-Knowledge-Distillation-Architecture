# =============================================================
# Fine-Tune Teacher CNN
# - Load model từ s3://anomalytraffic/models/cloud/model.tar.gz
# - Fine-tune với conflict data mới
# - So sánh accuracy cũ vs mới
# - Nếu tốt hơn: upload model.tar.gz mới + update tf-endpoint
# - Trigger IOT-PROJECT (Distillation) nếu update thành công
# =============================================================

import os
import sys
import json
import tarfile
import argparse
import traceback
import numpy as np
import pandas as pd
import tensorflow as tf
import boto3
from datetime import datetime

s3_client     = boto3.client('s3')
sm_client     = boto3.client('sagemaker')
lambda_client = boto3.client('lambda')

BUCKET           = os.environ.get('BUCKET', 'anomalytraffic')
TEACHER_S3_KEY   = 'models/cloud/model.tar.gz'
ENDPOINT_NAME    = os.environ.get('TEACHER_ENDPOINT', 'tf-endpoint')
DISTILL_FUNCTION = os.environ.get('DISTILL_FUNCTION', 'TriggerDistillation')

# Multiclass — 5 classes
CLASS_NAMES = ['Benign', 'Botnet', 'DDoS', 'DoS', 'PortScan']
N_CLASSES   = len(CLASS_NAMES)
INPUT_SHAPE = (75, 1)

print("🔥 FineTuneTeacher started")
print("=" * 60)
print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("=" * 60)


# ── Args ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",        type=int,   default=10)
    p.add_argument("--batch-size",    "--batch_size", type=int, default=32, dest="batch_size")
    p.add_argument("--learning-rate", "--learning_rate", type=float, default=1e-4, dest="learning_rate")
    p.add_argument("--min-improvement", "--min_improvement", type=float, default=0.001,
                   dest="min_improvement",
                   help="Minimum accuracy improvement to accept new model (default: 0.001 = 0.1%)")
    p.add_argument("--train", type=str,
                   default=os.environ.get("SM_CHANNEL_TRAINING", "./data"))
    p.add_argument("--model-dir", "--model_dir", type=str,
                   default=os.environ.get("SM_MODEL_DIR", "./model"), dest="model_dir")
    p.add_argument("--output-data-dir", "--output_data_dir", type=str,
                   default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"), dest="output_data_dir")

    args, unknown = p.parse_known_args()
    if unknown:
        print("⚠️ Ignoring unknown args:", unknown)

    print("✅ Args:", vars(args))
    return args


# ── Load & prepare data ───────────────────────────────────────────────────────
def load_data(data_dir):
    print(f"\n📂 Loading data from: {data_dir}")

    csvs = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csvs:
        raise RuntimeError(f"❌ No CSV found in {data_dir}")

    df = pd.read_csv(os.path.join(data_dir, csvs[0]))
    print(f"📋 Shape: {df.shape}")

    # Feature columns
    feat_cols = [c for c in df.columns if c.startswith("feature_")]
    if not feat_cols:
        raise RuntimeError("❌ No 'feature_*' columns found")

    X = df[feat_cols].astype("float32").values   # (N, n_feat)

    # Pad / truncate to 75
    n_feat = X.shape[1]
    if n_feat < 75:
        pad = np.zeros((X.shape[0], 75 - n_feat), dtype='float32')
        X = np.hstack([X, pad])
    elif n_feat > 75:
        X = X[:, :75]

    # Reshape → (N, 75, 1)
    X = X.reshape(X.shape[0], 75, 1)

    # Label column
    if 'label' not in df.columns:
        raise RuntimeError("❌ No 'label' column found in CSV")

    y_raw = df['label'].values.astype(int)

    # One-hot encode for 5 classes
    y = tf.keras.utils.to_categorical(y_raw, num_classes=N_CLASSES)

    print(f"✅ X: {X.shape} | y: {y.shape}")
    print(f"📊 Class distribution: {dict(zip(*np.unique(y_raw, return_counts=True)))}")

    return X, y, y_raw


# ── Download & load teacher model ────────────────────────────────────────────
def load_teacher_model():
    print(f"\n📥 Downloading teacher model from s3://{BUCKET}/{TEACHER_S3_KEY}")

    local_tar   = "/tmp/teacher_model.tar.gz"
    extract_dir = "/tmp/teacher_model"

    s3_client.download_file(BUCKET, TEACHER_S3_KEY, local_tar)
    print("✅ Downloaded")

    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(local_tar, 'r:gz') as tar:
        tar.extractall(extract_dir)
    print("✅ Extracted")

    # Find saved_model.pb
    savedmodel_dir = None
    if os.path.exists(os.path.join(extract_dir, "saved_model.pb")):
        savedmodel_dir = extract_dir
    else:
        for root, dirs, files in os.walk(extract_dir):
            if "saved_model.pb" in files:
                savedmodel_dir = root
                break

    if not savedmodel_dir:
        raise RuntimeError(f"❌ saved_model.pb not found in {extract_dir}")

    print(f"📂 Loading SavedModel from: {savedmodel_dir}")
    model = tf.keras.models.load_model(savedmodel_dir)
    print("✅ Teacher model loaded")
    print(model.summary())

    return model


# ── Evaluate model accuracy ───────────────────────────────────────────────────
def evaluate(model, X, y_raw, label=""):
    y_pred = model.predict(X, verbose=0)
    y_pred_class = np.argmax(y_pred, axis=1)

    accuracy = np.mean(y_pred_class == y_raw)
    print(f"📊 {label} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Per-class accuracy
    for i, cls in enumerate(CLASS_NAMES):
        mask = y_raw == i
        if mask.sum() > 0:
            cls_acc = np.mean(y_pred_class[mask] == i)
            print(f"   {cls}: {cls_acc:.4f} (n={mask.sum()})")

    return accuracy


# ── Fine-tune ─────────────────────────────────────────────────────────────────
def fine_tune(model, X_tr, y_tr, X_val, y_val, args):
    print(f"\n🔧 Fine-tuning with lr={args.learning_rate}, epochs={args.epochs}")

    # Unfreeze all layers for fine-tuning
    model.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1
    )

    print("✅ Fine-tuning complete")
    return model, history


# ── Package & upload new model.tar.gz ────────────────────────────────────────
def upload_new_model(model, args):
    print("\n📦 Packaging new model.tar.gz")

    save_dir  = "/tmp/new_teacher_model"
    tar_path  = "/tmp/new_teacher_model.tar.gz"

    # Save as SavedModel format
    tf.saved_model.save(model, save_dir)
    print(f"✅ Saved to: {save_dir}")

    # Backup old model before overwriting
    timestamp  = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    backup_key = f"models/cloud/backups/model_{timestamp}.tar.gz"

    print(f"💾 Backing up old model to s3://{BUCKET}/{backup_key}")
    s3_client.copy_object(
        Bucket=BUCKET,
        CopySource={'Bucket': BUCKET, 'Key': TEACHER_S3_KEY},
        Key=backup_key
    )
    print("✅ Backup done")

    # Pack new model
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(save_dir, arcname='.')
    print(f"✅ Packed: {tar_path}")

    # Upload new model.tar.gz
    s3_client.upload_file(tar_path, BUCKET, TEACHER_S3_KEY)
    print(f"✅ Uploaded new model to s3://{BUCKET}/{TEACHER_S3_KEY}")

    return backup_key


# ── Update SageMaker endpoint ─────────────────────────────────────────────────
def update_endpoint(args):
    print(f"\n🔄 Updating endpoint: {ENDPOINT_NAME}")

    timestamp    = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    model_name   = f"tf-teacher-{timestamp}"
    config_name  = f"tf-teacher-config-{timestamp}"

    # Get current endpoint config to reuse instance type
    current = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    current_config = sm_client.describe_endpoint_config(
        EndpointConfigName=current['EndpointConfigName']
    )
    instance_type = current_config['ProductionVariants'][0]['InstanceType']
    role_arn      = os.environ['SAGEMAKER_ROLE']

    print(f"   Instance type: {instance_type}")

    # Create new SageMaker Model
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': current_config['ProductionVariants'][0].get(
                'ModelDataDownloadTimeoutInSeconds', None
            ),
            'ModelDataUrl': f's3://{BUCKET}/{TEACHER_S3_KEY}',
            'Environment': {
                'SAGEMAKER_TFS_DEFAULT_MODEL_NAME': 'model'
            }
        },
        ExecutionRoleArn=role_arn
    )

    # Create new endpoint config
    sm_client.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            'VariantName':    'AllTraffic',
            'ModelName':      model_name,
            'InstanceType':   instance_type,
            'InitialInstanceCount': 1
        }]
    )

    # Update endpoint (blue/green, zero downtime)
    sm_client.update_endpoint(
        EndpointName=ENDPOINT_NAME,
        EndpointConfigName=config_name
    )

    print(f"✅ Endpoint update triggered — may take 5-10 min to complete")
    print(f"   New model:  {model_name}")
    print(f"   New config: {config_name}")


# ── Trigger Distillation ──────────────────────────────────────────────────────
def trigger_distillation():
    print(f"\n🚀 Triggering distillation: {DISTILL_FUNCTION}")
    try:
        lambda_client.invoke(
            FunctionName=DISTILL_FUNCTION,
            InvocationType='Event',
            Payload=json.dumps({'triggered_by': 'FineTuneTeacher'})
        )
        print(f"✅ {DISTILL_FUNCTION} triggered")
    except Exception as e:
        print(f"❌ Failed to trigger distillation: {e}")


# ── Save report ───────────────────────────────────────────────────────────────
def save_report(report, args):
    os.makedirs(args.output_data_dir, exist_ok=True)
    path = os.path.join(args.output_data_dir, "finetune_report.json")
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✅ Report saved: {path}")

    # Also upload to S3 for visibility
    s3_client.upload_file(
        path, BUCKET,
        f"models/cloud/reports/finetune_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    )


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    try:
        args = parse_args()

        # 1. Load data
        X, y, y_raw = load_data(args.train)

        # Train/val split 80/20
        split  = int(0.8 * len(X))
        X_tr,  X_val  = X[:split],     X[split:]
        y_tr,  y_val  = y[:split],     y[split:]
        yr_tr, yr_val = y_raw[:split], y_raw[split:]
        print(f"📐 Train/Val: {len(X_tr)}/{len(X_val)}")

        # 2. Load teacher
        model = load_teacher_model()

        # 3. Baseline accuracy BEFORE fine-tune
        print("\n📊 Evaluating BEFORE fine-tune...")
        acc_before = evaluate(model, X_val, yr_val, label="BEFORE")

        # 4. Fine-tune
        model, history = fine_tune(model, X_tr, y_tr, X_val, y_val, args)

        # 5. Accuracy AFTER fine-tune
        print("\n📊 Evaluating AFTER fine-tune...")
        acc_after = evaluate(model, X_val, yr_val, label="AFTER")

        improvement = acc_after - acc_before
        print(f"\n📈 Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")

        report = {
            "acc_before":    float(acc_before),
            "acc_after":     float(acc_after),
            "improvement":   float(improvement),
            "min_required":  args.min_improvement,
            "accepted":      improvement >= args.min_improvement,
            "timestamp":     datetime.utcnow().isoformat(),
            "epochs_ran":    len(history.history['loss'])
        }

        # 6. Accept or reject
        if improvement >= args.min_improvement:
            print(f"\n✅ Model improved by {improvement*100:.2f}% — ACCEPTED")

            backup_key = upload_new_model(model, args)
            report['backup_key'] = backup_key

            update_endpoint(args)
            trigger_distillation()

        else:
            print(f"\n⚠️ Improvement {improvement*100:.2f}% < threshold {args.min_improvement*100:.2f}%")
            print("🔄 Keeping old model — pipeline STOPPED")
            report['reason'] = f"Improvement {improvement:.4f} below threshold {args.min_improvement}"

        # 7. Save report
        save_report(report, args)

        print("\n" + "=" * 60)
        print(f"{'✅ FINE-TUNE ACCEPTED' if report['accepted'] else '⚠️ FINE-TUNE REJECTED — old model kept'}")
        print("=" * 60)
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ FATAL: {type(e).__name__}: {e}")
        print("=" * 60)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
