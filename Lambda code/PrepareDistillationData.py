import boto3
import json
import os
from datetime import datetime
import csv

dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')

CONFLICTS_TABLE = os.environ.get('CONFLICTS_TABLE', 'AnomalyConflicts')
BUCKET = os.environ.get('BUCKET', 'anomalytraffic')
OUTPUT_PREFIX = 'data/distillation/train/'

def lambda_handler(event, context):
    """
    Prepare training data from relabeled conflicts
    Export to CSV for SageMaker training
    """
    
    print("📊 Preparing distillation training data...")
    
    table = dynamodb.Table(CONFLICTS_TABLE)
    
    # Query relabeled conflicts with HIGH confidence
    try:
        # BUG FIX: ExpressionAttributeValues must be a single dict —
        # merging KeyCondition value + FilterExpression value together.
        response = table.query(
            IndexName='status-index',
            KeyConditionExpression='#status = :relabeled',
            FilterExpression='relabel_confidence = :high',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={
                ':relabeled': 'relabeled',
                ':high': 'high'
            }
        )
        
        conflicts = response.get('Items', [])
        
        # Pagination
        while 'LastEvaluatedKey' in response:
            response = table.query(
                IndexName='status-index',
                KeyConditionExpression='#status = :relabeled',
                FilterExpression='relabel_confidence = :high',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':relabeled': 'relabeled',
                    ':high': 'high'
                },
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            conflicts.extend(response.get('Items', []))
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return {'statusCode': 500, 'error': str(e)}
    
    print(f"Found {len(conflicts)} high-confidence relabeled conflicts")
    
    if len(conflicts) < 10:
        print("⚠️ Not enough data for training (need at least 10)")
        return {
            'statusCode': 400,
            'message': f'Only {len(conflicts)} conflicts, need at least 10'
        }
    
    # Prepare CSV data
    csv_rows = []
    
    for conflict in conflicts:
        try:
            flow_data = json.loads(conflict.get('flow_data', '{}'))
            correct_label = conflict.get('correct_label', '')
            
            # Map label to numeric
            label_map = {
                'Benign': 0,
                'Botnet': 1,
                'DDoS': 2,
                'DoS': 3,
                'PortScan': 4
            }
            
            label_num = label_map.get(correct_label, 0)
            
            # Add flow features + label
            row = {
                'label': label_num,
                **flow_data
            }
            
            csv_rows.append(row)
            
        except Exception as e:
            print(f"⚠️ Skip conflict {conflict.get('conflict_id')}: {e}")
            continue
    
    print(f"Prepared {len(csv_rows)} training samples")
    
    # Write CSV to /tmp
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    csv_filename = f'/tmp/distillation_train_{timestamp}.csv'
    
    if csv_rows:
        # BUG FIX: dict.keys() - {'label'} returns a KeysView, not stable;
        # collect all feature columns seen across ALL rows (union), sorted for
        # reproducibility, then put 'label' first.
        all_keys = set()
        for row in csv_rows:
            all_keys.update(row.keys())
        all_keys.discard('label')
        fieldnames = ['label'] + sorted(all_keys)
        
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(csv_rows)
        
        # Upload to S3
        s3_key = f'{OUTPUT_PREFIX}distillation_train_{timestamp}.csv'
        
        s3.upload_file(csv_filename, BUCKET, s3_key)
        
        print(f"✅ Uploaded to s3://{BUCKET}/{s3_key}")
        
        # Update conflicts status to "used"
        for conflict in conflicts[:len(csv_rows)]:
            table.update_item(
                Key={
                    'conflict_id': conflict['conflict_id'],
                    'created_at': conflict['created_at']
                },
                UpdateExpression='SET #status = :used',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':used': 'used'}
            )
        
        return {
            'statusCode': 200,
            'csv_s3_path': f's3://{BUCKET}/{s3_key}',
            'sample_count': len(csv_rows),
            'timestamp': timestamp
        }
    
    else:
        print("❌ No valid training samples")
        return {
            'statusCode': 400,
            'message': 'No valid training samples prepared'
        }