#!/usr/bin/env python3
# =============================================================
# Fine-Tune Teacher CNN - FIXED VERSION
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

CLASS_NAMES = ['Benign', 'Botnet', 'DDoS', 'DoS', 'PortScan']
N_CLASSES   = len(CLASS_NAMES)
INPUT_SHAPE = (75, 1)

print("=" * 70)
print("🔥 FineTuneTeacher - FIXED VERSION")
print("=" * 70)
print(f"Python: {sys.version}")
print(f"TensorFlow: {tf.__version__}")
print(f"Bucket: {BUCKET}")
print(f"Teacher endpoint: {ENDPOINT_NAME}")
print("=" * 70)


def parse_args():
    """Parse command line arguments"""
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", "--batch_size", type=int, default=32, dest="batch_size")
    p.add_argument("--learning-rate", "--learning_rate", type=float, default=1e-4, dest="learning_rate")
    p.add_argument("--min-improvement", "--min_improvement", type=float, default=0.001,
                   dest="min_improvement",
                   help="Minimum accuracy improvement (default: 0.001 = 0.1%)")
    p.add_argument("--train", type=str,
                   default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    p.add_argument("--model-dir", "--model_dir", type=str,
                   default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"), dest="model_dir")
    p.add_argument("--output-data-dir", "--output_data_dir", type=str,
                   default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"), dest="output_data_dir")

    args, unknown = p.parse_known_args()
    if unknown:
        print(f"⚠️ Ignoring unknown args: {unknown}")

    print(f"\n✅ Parsed arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    
    return args


def load_data(data_dir):
    """Load and prepare training data from CSV"""
    print(f"\n📂 Loading data from: {data_dir}")
    
    # Check directory exists
    if not os.path.exists(data_dir):
        raise RuntimeError(f"❌ Directory not found: {data_dir}")
    
    # Find CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    
    if not csv_files:
        raise RuntimeError(f"❌ No CSV files found in {data_dir}")
    
    print(f"📋 Found {len(csv_files)} CSV file(s)")
    
    # Load first CSV
    csv_path = os.path.join(data_dir, csv_files[0])
    print(f"📄 Loading: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded shape: {df.shape}")
    print(f"📊 Columns: {list(df.columns)[:10]}...")  # Show first 10 cols
    
    # Check for label column
    if 'label' not in df.columns:
        raise RuntimeError(f"❌ 'label' column not found in CSV. Available: {list(df.columns)}")
    
    # Extract features - all columns except 'label'
    feature_cols = [c for c in df.columns if c != 'label']
    
    if not feature_cols:
        raise RuntimeError("❌ No feature columns found")
    
    print(f"🔢 Feature columns: {len(feature_cols)}")
    
    # Extract features
    X = df[feature_cols].astype('float32').values
    
    # Pad or truncate to 75 features
    n_feat = X.shape[1]
    print(f"📐 Original features: {n_feat}, target: 75")
    
    if n_feat < 75:
        pad_width = 75 - n_feat
        print(f"  ➕ Padding with {pad_width} zeros")
        pad = np.zeros((X.shape[0], pad_width), dtype='float32')
        X = np.hstack([X, pad])
    elif n_feat > 75:
        print(f"  ✂️ Truncating from {n_feat} to 75")
        X = X[:, :75]
    
    # Reshape to (N, 75, 1)
    X = X.reshape(X.shape[0], 75, 1)
    
    # Extract labels
    y_raw = df['label'].values.astype(int)
    
    # Validate label range
    unique_labels = np.unique(y_raw)
    print(f"📊 Unique labels: {unique_labels}")
    
    if np.max(y_raw) >= N_CLASSES:
        raise RuntimeError(f"❌ Invalid label {np.max(y_raw)} (max should be {N_CLASSES-1})")
    
    # One-hot encode
    y = tf.keras.utils.to_categorical(y_raw, num_classes=N_CLASSES)
    
    print(f"✅ X: {X.shape}, y: {y.shape}")
    print(f"📊 Label distribution:")
    for label, count in zip(*np.unique(y_raw, return_counts=True)):
        print(f"  Class {label} ({CLASS_NAMES[label]}): {count} samples")
    
    return X, y, y_raw


def load_teacher_model():
    """Download and load teacher model from S3"""
    print(f"\n📥 Downloading teacher model...")
    print(f"  Source: s3://{BUCKET}/{TEACHER_S3_KEY}")
    
    local_tar = "/tmp/teacher_model.tar.gz"
    extract_dir = "/tmp/teacher_model"
    
    try:
        s3_client.download_file(BUCKET, TEACHER_S3_KEY, local_tar)
        print(f"✅ Downloaded to: {local_tar}")
        
        # Extract tar.gz
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(local_tar, 'r:gz') as tar:
            tar.extractall(extract_dir)
        print(f"✅ Extracted to: {extract_dir}")
        
        # Find saved_model.pb
        savedmodel_dir = None
        
        # Check root directory first
        if os.path.exists(os.path.join(extract_dir, "saved_model.pb")):
            savedmodel_dir = extract_dir
        else:
            # Search subdirectories
            for root, dirs, files in os.walk(extract_dir):
                if "saved_model.pb" in files:
                    savedmodel_dir = root
                    break
        
        if not savedmodel_dir:
            # Debug: show directory structure
            print("❌ saved_model.pb not found!")
            print("📂 Directory structure:")
            for root, dirs, files in os.walk(extract_dir):
                level = root.replace(extract_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files
                    print(f"{subindent}{file}")
            
            raise RuntimeError(f"saved_model.pb not found in {extract_dir}")
        
        print(f"📂 Loading SavedModel from: {savedmodel_dir}")
        model = tf.keras.models.load_model(savedmodel_dir)
        print(f"✅ Model loaded successfully")
        
        # Print model summary
        print("\n📋 Model Summary:")
        model.summary()
        
        return model
        
    except Exception as e:
        print(f"❌ Failed to load teacher model: {e}")
        raise


def evaluate(model, X, y_raw, label=""):
    """Evaluate model accuracy"""
    print(f"\n📊 Evaluating {label}...")
    
    y_pred = model.predict(X, verbose=0)
    y_pred_class = np.argmax(y_pred, axis=1)
    
    accuracy = np.mean(y_pred_class == y_raw)
    print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Per-class accuracy
    print(f"  Per-class Accuracy:")
    for i, cls in enumerate(CLASS_NAMES):
        mask = y_raw == i
        if mask.sum() > 0:
            cls_acc = np.mean(y_pred_class[mask] == i)
            print(f"    {cls:12} : {cls_acc:.4f} (n={mask.sum()})")
    
    return accuracy


def fine_tune(model, X_tr, y_tr, X_val, y_val, args):
    """Fine-tune the model"""
    print(f"\n🔧 Fine-tuning...")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    
    # Unfreeze all layers
    model.trainable = True
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
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
    
    # Train
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


def upload_new_model(model, args):
    """Package and upload new model to S3"""
    print("\n📦 Packaging new model...")
    
    save_dir = "/tmp/new_teacher_model"
    tar_path = "/tmp/new_teacher_model.tar.gz"
    
    # Save model as SavedModel format
    tf.saved_model.save(model, save_dir)
    print(f"✅ Saved to: {save_dir}")
    
    # Backup old model
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    backup_key = f"models/cloud/backups/model_{timestamp}.tar.gz"
    
    print(f"💾 Backing up old model to: s3://{BUCKET}/{backup_key}")
    
    try:
        s3_client.copy_object(
            Bucket=BUCKET,
            CopySource={'Bucket': BUCKET, 'Key': TEACHER_S3_KEY},
            Key=backup_key
        )
        print("✅ Backup complete")
    except Exception as e:
        print(f"⚠️ Backup failed (continuing anyway): {e}")
    
    # Create tar.gz
    with tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(save_dir, arcname='.')
    print(f"✅ Created tar.gz: {tar_path}")
    
    # Upload
    s3_client.upload_file(tar_path, BUCKET, TEACHER_S3_KEY)
    print(f"✅ Uploaded to: s3://{BUCKET}/{TEACHER_S3_KEY}")
    
    return backup_key


def update_endpoint(args):
    """Update SageMaker endpoint with new model"""
    print(f"\n🔄 Updating endpoint: {ENDPOINT_NAME}")
    
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    model_name = f"tf-teacher-{timestamp}"
    config_name = f"tf-teacher-config-{timestamp}"
    
    try:
        # Get current endpoint config
        current_endpoint = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        current_config_name = current_endpoint['EndpointConfigName']
        
        current_config = sm_client.describe_endpoint_config(
            EndpointConfigName=current_config_name
        )
        
        # Get instance type and container image from current config
        variant = current_config['ProductionVariants'][0]
        instance_type = variant['InstanceType']
        container_image = variant.get('Image')
        
        print(f"  Current config: {current_config_name}")
        print(f"  Instance type: {instance_type}")
        
        # Get execution role from environment or current config
        role_arn = os.environ.get('SAGEMAKER_ROLE')
        
        if not role_arn:
            raise RuntimeError("SAGEMAKER_ROLE environment variable not set!")
        
        print(f"  Execution role: {role_arn}")
        
        # Determine container image for TensorFlow
        # Use AWS Deep Learning Container for TensorFlow Serving
        region = boto3.Session().region_name
        account_id = '763104351884'  # AWS DLC account ID
        
        if not container_image:
            container_image = f"{account_id}.dkr.ecr.{region}.amazonaws.com/tensorflow-inference:2.13-cpu"
        
        print(f"  Container image: {container_image}")
        
        # Create new SageMaker Model
        print(f"  Creating model: {model_name}")
        
        sm_client.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': container_image,
                'ModelDataUrl': f's3://{BUCKET}/{TEACHER_S3_KEY}',
                'Environment': {
                    'SAGEMAKER_TFS_DEFAULT_MODEL_NAME': 'model',
                    'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                    'SAGEMAKER_REGION': region
                }
            },
            ExecutionRoleArn=role_arn
        )
        
        print(f"✅ Model created: {model_name}")
        
        # Create new endpoint config
        print(f"  Creating endpoint config: {config_name}")
        
        sm_client.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[{
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InstanceType': instance_type,
                'InitialInstanceCount': 1
            }]
        )
        
        print(f"✅ Endpoint config created: {config_name}")
        
        # Update endpoint
        print(f"  Updating endpoint (this may take 5-10 minutes)...")
        
        sm_client.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name
        )
        
        print(f"✅ Endpoint update initiated")
        print(f"  Monitor status with: aws sagemaker describe-endpoint --endpoint-name {ENDPOINT_NAME}")
        
    except Exception as e:
        print(f"❌ Failed to update endpoint: {e}")
        raise


def trigger_distillation():
    """Trigger distillation Lambda"""
    print(f"\n🚀 Triggering distillation: {DISTILL_FUNCTION}")
    
    try:
        lambda_client.invoke(
            FunctionName=DISTILL_FUNCTION,
            InvocationType='Event',
            Payload=json.dumps({'triggered_by': 'FineTuneTeacher'})
        )
        print(f"✅ {DISTILL_FUNCTION} triggered")
        
    except Exception as e:
        print(f"⚠️ Failed to trigger distillation: {e}")


def save_report(report, args):
    """Save training report"""
    print("\n💾 Saving report...")
    
    os.makedirs(args.output_data_dir, exist_ok=True)
    
    report_path = os.path.join(args.output_data_dir, "finetune_report.json")
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ Report saved: {report_path}")
    
    # Upload to S3
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    s3_key = f"models/cloud/reports/finetune_{timestamp}.json"
    
    try:
        s3_client.upload_file(report_path, BUCKET, s3_key)
        print(f"✅ Report uploaded: s3://{BUCKET}/{s3_key}")
    except Exception as e:
        print(f"⚠️ Failed to upload report: {e}")


def main():
    """Main training function"""
    
    try:
        # Parse arguments
        args = parse_args()
        
        # Load data
        X, y, y_raw = load_data(args.train)
        
        # Split train/val
        split = int(0.8 * len(X))
        X_tr, X_val = X[:split], X[split:]
        y_tr, y_val = y[:split], y[split:]
        yr_tr, yr_val = y_raw[:split], y_raw[split:]
        
        print(f"\n📐 Train/Val split: {len(X_tr)}/{len(X_val)}")
        
        # Load teacher model
        model = load_teacher_model()
        
        # Baseline accuracy
        acc_before = evaluate(model, X_val, yr_val, label="BEFORE fine-tuning")
        
        # Fine-tune
        model, history = fine_tune(model, X_tr, y_tr, X_val, y_val, args)
        
        # Accuracy after fine-tuning
        acc_after = evaluate(model, X_val, yr_val, label="AFTER fine-tuning")
        
        # Calculate improvement
        improvement = acc_after - acc_before
        
        print(f"\n📈 Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        print(f"   Threshold: {args.min_improvement} ({args.min_improvement*100:.2f}%)")
        
        # Create report
        report = {
            "acc_before": float(acc_before),
            "acc_after": float(acc_after),
            "improvement": float(improvement),
            "min_required": args.min_improvement,
            "accepted": improvement >= args.min_improvement,
            "timestamp": datetime.utcnow().isoformat(),
            "epochs_ran": len(history.history['loss']),
            "samples": {
                "train": len(X_tr),
                "val": len(X_val),
                "total": len(X)
            }
        }
        
        # Accept or reject
        if improvement >= args.min_improvement:
            print(f"\n✅ MODEL ACCEPTED (improvement: {improvement*100:.2f}%)")
            
            backup_key = upload_new_model(model, args)
            report['backup_key'] = backup_key
            
            update_endpoint(args)
            trigger_distillation()
            
        else:
            print(f"\n⚠️ MODEL REJECTED")
            print(f"   Improvement {improvement*100:.2f}% < threshold {args.min_improvement*100:.2f}%")
            print(f"   Keeping old model")
            
            report['reason'] = f"Improvement {improvement:.4f} below threshold"
        
        # Save report
        save_report(report, args)
        
        print("\n" + "=" * 70)
        if report['accepted']:
            print("✅ FINE-TUNING COMPLETE - MODEL UPDATED")
        else:
            print("⚠️ FINE-TUNING COMPLETE - OLD MODEL KEPT")
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"❌ FATAL ERROR: {type(e).__name__}: {e}")
        print("=" * 70)
        traceback.print_exc()
        
        return 1


if __name__ == "__main__":
    sys.exit(main())