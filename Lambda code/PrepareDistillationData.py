import boto3
import json
import os
import csv
import time
from datetime import datetime

dynamodb = boto3.resource('dynamodb')
s3       = boto3.client('s3')

CONFLICTS_TABLE = os.environ.get('CONFLICTS_TABLE', 'AnomalyConflicts')
BUCKET          = os.environ.get('BUCKET', 'anomalytraffic')
OUTPUT_PREFIX   = 'data/distillation/train/'
FINETUNING_FUNCTION = os.environ.get('FINETUNING_FUNCTION', 'Triggerfinetuning')

lambda_client = boto3.client('lambda')

# ==============================================================================
# Feature Engineering Helpers (Same logic as Edge Inference)
# ==============================================================================
APP_CAT_MAP = {
    "Advertisement":  0,  "Chat":           1,  "Cloud":          2,
    "Collaborative":  3,  "ConnCheck":      4,  "Cybersecurity":  5,
    "DataTransfer":   6,  "Database":       7,  "Download":       8,
    "Email":          9,  "Game":          10,  "IoT-Scada":     11,
    "Media":         12,  "Mining":        13,  "Music":         14,
    "Network":       15,  "RPC":           16,  "RemoteAccess":  17,
    "Shopping":      18,  "SocialNetwork": 19,  "SoftwareUpdate":20,
    "Streaming":     21,  "System":        22,  "Unspecified":   23,
    "VPN":           24,  "Video":         25,  "VoIP":          26,
    "Web":           27,
}

def _g(flow: dict, key: str, default=0.0) -> float:
    try:
        v = float(flow.get(key, default))
        return 0.0 if (v != v) else v
    except (TypeError, ValueError):
        return float(default)

def _port_bucket(p) -> int:
    try:
        p = int(p)
    except (TypeError, ValueError):
        return 2
    if p <= 1023:  return 0
    if p <= 49151: return 1
    return 2

def engineer_features(flow: dict) -> dict:
    bi_pkts   = _g(flow, "bidirectional_packets")
    bi_bytes  = _g(flow, "bidirectional_bytes")
    s2d_pkts  = _g(flow, "src2dst_packets")
    d2s_pkts  = _g(flow, "dst2src_packets")
    s2d_bytes = _g(flow, "src2dst_bytes")
    d2s_bytes = _g(flow, "dst2src_bytes")

    feat = {}
    for k, v in flow.items():
        if isinstance(v, (int, float)):
            feat[k] = float(v)

    feat["pkt_per_byte_ratio"]  = bi_pkts  / (bi_bytes  + 1e-9)
    feat["flow_symmetry"]       = s2d_pkts / (s2d_pkts + d2s_pkts + 1e-9)
    feat["byte_symmetry"]       = s2d_bytes/ (s2d_bytes + d2s_bytes + 1e-9)

    feat["bidirectional_syn_ratio"] = _g(flow, "bidirectional_syn_packets") / (bi_pkts + 1e-9)
    feat["src2dst_syn_ratio"]       = _g(flow, "src2dst_syn_packets")       / (s2d_pkts + 1e-9)
    feat["dst2src_syn_ratio"]       = _g(flow, "dst2src_syn_packets")       / (d2s_pkts + 1e-9)

    for flag in ["ack", "psh", "rst", "fin", "cwr", "ece", "urg"]:
        feat[f"bidirectional_{flag}_ratio"] = \
            _g(flow, f"bidirectional_{flag}_packets") / (bi_pkts + 1e-9)

    dp = int(_g(flow, "dst_port"))
    feat["dst_port_bucket"]        = float(_port_bucket(dp))
    feat["dst_port_is_well_known"] = float(feat["dst_port_bucket"] == 0)

    cat_raw = flow.get("application_category_name", "Unspecified")
    feat["application_category_name"] = float(APP_CAT_MAP.get(str(cat_raw), -1))
    feat["application_confidence"]    = _g(flow, "application_confidence")

    return feat

def get_scaler(s3_client, bucket):
    obj = s3_client.get_object(Bucket=bucket, Key='data/raw/log/scaler_stats.json')
    return json.loads(obj["Body"].read())


def lambda_handler(event, context):
    """
    Prepare distillation training data from relabeled conflicts.
    Export to CSV for SageMaker training.
    
    SIMPLIFIED: Only export CSV, don't trigger SageMaker job.
    """
    
    print("📊 Preparing distillation training data...")
    
    # If triggered by Relabel, wait for DynamoDB consistency
    if event.get('triggered_by') == 'Relabel':
        print("⏳ Waiting 5s for DynamoDB GSI consistency...")
        time.sleep(5)
    
    table = dynamodb.Table(CONFLICTS_TABLE)
    
    # ========================================
    # Query high-confidence relabeled conflicts
    # ========================================
    
    try:
        # Use Scan to avoid GSI eventual consistency
        response = table.scan(
            FilterExpression='#status = :relabeled AND relabel_confidence = :high',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':relabeled': 'relabeled',
                ':high': 'high'
            }
        )
        
        conflicts = response.get('Items', [])
        
        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression='#status = :relabeled AND relabel_confidence = :high',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':relabeled': 'relabeled',
                    ':high': 'high'
                },
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            conflicts.extend(response.get('Items', []))
        
    except Exception as e:
        print(f"❌ DynamoDB scan failed: {e}")
        return {'statusCode': 500, 'error': str(e)}
    
    print(f"📋 Found {len(conflicts)} high-confidence relabeled conflicts")
    
    if len(conflicts) < 10:
        print(f"⚠️ Not enough data for training (need at least 10)")
        return {
            'statusCode': 400,
            'message': f'Only {len(conflicts)} samples, need at least 10'
        }
    
    # ========================================
    # Build CSV rows
    # ========================================
    
    label_map = {
        'Benign': 0,
        'Botnet': 1,
        'DDoS': 2,
        'DoS': 3,
        'PortScan': 4
    }
    
    # Load Scaler from S3 to know exactly what features are required and to standardize them
    try:
        scaler = get_scaler(s3, BUCKET)
        feature_names = scaler["feature_names"]
        means = scaler["mean"]
        scales = scaler["scale"]
    except Exception as e:
        print(f"❌ Failed to load scaler_stats.json: {e}")
        return {'statusCode': 500, 'error': f"Failed to load scaler_stats: {e}"}
        
    csv_rows = []
    
    for conflict in conflicts:
        try:
            flow_data = json.loads(conflict.get('flow_data', '{}'))
            correct_label = conflict.get('correct_label', '')
            
            # Map label to numeric
            label_num = label_map.get(correct_label, 0)
            
            # Engineer features from raw JSON
            feat_dict = engineer_features(flow_data)
            
            # Standardize and map to exactly match SageMaker input format
            row = {'label': label_num}
            for i, feat_name in enumerate(feature_names):
                val = feat_dict.get(feat_name, 0.0)
                scaled_val = (val - float(means[i])) / (float(scales[i]) + 1e-9)
                row[feat_name] = round(scaled_val, 6) # Round to save space
            
            csv_rows.append(row)
            
        except Exception as e:
            print(f"⚠️ Skip conflict {conflict.get('conflict_id')}: {e}")
            continue
    
    print(f"Prepared {len(csv_rows)} training samples")
    
    if not csv_rows:
        return {
            'statusCode': 400,
            'message': 'No valid training samples prepared'
        }
    
    # ========================================
    # Write CSV to /tmp
    # ========================================
    
    timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    csv_filename = f'/tmp/distillation_train_{timestamp}.csv'
    
    # Exact field names
    fieldnames = ['label'] + feature_names
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(csv_rows)
    
    # ========================================
    # Upload to S3
    # ========================================
    
    s3_key = f'{OUTPUT_PREFIX}distillation_train_{timestamp}.csv'
    
    s3.upload_file(csv_filename, BUCKET, s3_key)
    
    print(f"✅ Uploaded to s3://{BUCKET}/{s3_key}")
    
    # ========================================
    # Mark conflicts as 'used'
    # ========================================
    
    marked_count = 0
    
    for conflict in conflicts[:len(csv_rows)]:
        try:
            table.update_item(
                Key={
                    'conflict_id': conflict['conflict_id'],
                    'created_at': conflict['created_at']
                },
                UpdateExpression='SET #status = :used',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':used': 'used'}
            )
            marked_count += 1
        except Exception as e:
            print(f"⚠️ Could not mark {conflict.get('conflict_id')} as used: {e}")
    
    print(f"✅ Marked {marked_count}/{len(csv_rows)} conflicts as used")
    
    # ========================================
    # Trigger Fine-Tuning Lambda (Automated Trigger)
    # ========================================
    
    print(f"🚀 Triggering Fine-Tuning Lambda: {FINETUNING_FUNCTION}")
    try:
        lambda_client.invoke(
            FunctionName=FINETUNING_FUNCTION,
            InvocationType='Event',
            Payload=json.dumps({
                'triggered_by': 'PrepareDistillationData',
                'csv_s3_path': f's3://{BUCKET}/{s3_key}',
                'sample_count': len(csv_rows)
            })
        )
        print(f"✅ Triggered {FINETUNING_FUNCTION}")
    except Exception as e:
        print(f"❌ Failed to trigger {FINETUNING_FUNCTION}: {e}")

    # ========================================
    # Return success
    # ========================================
    
    return {
        'statusCode': 200,
        'csv_s3_path': f's3://{BUCKET}/{s3_key}',
        'sample_count': len(csv_rows),
        'timestamp': timestamp,
        'message': f'Successfully prepared {len(csv_rows)} samples for distillation training'
    }