import boto3
import json
import os
import csv
import time
from datetime import datetime

dynamodb = boto3.resource('dynamodb')
s3       = boto3.client('s3')
sm       = boto3.client('sagemaker')

CONFLICTS_TABLE     = os.environ.get('CONFLICTS_TABLE', 'AnomalyConflicts')
BUCKET              = os.environ.get('BUCKET', 'anomalytraffic')
OUTPUT_PREFIX       = 'data/distillation/train/'
ECR_IMAGE_FINETUNE  = os.environ.get('ECR_IMAGE_FINETUNE', '')
ECR_IMAGE_DISTILL   = os.environ.get('ECR_IMAGE_DISTILL', '')
SAGEMAKER_ROLE      = os.environ.get('SAGEMAKER_ROLE', '')
DISTILL_FUNCTION    = os.environ.get('DISTILL_FUNCTION', 'TriggerDistillation')


def lambda_handler(event, context):
    print("📊 Preparing distillation training data...")

    # Nếu trigger từ Relabel thì chờ GSI sync
    if event.get('triggered_by') == 'Relabel':
        print("⏳ Waiting 5s for DynamoDB GSI consistency...")
        time.sleep(5)

    table = dynamodb.Table(CONFLICTS_TABLE)

    # ── Dùng Scan thay Query để tránh GSI eventual consistency ───────────────
    try:
        response  = table.scan(
            FilterExpression='#status = :relabeled AND relabel_confidence = :high',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={':relabeled': 'relabeled', ':high': 'high'}
        )
        conflicts = response.get('Items', [])

        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression='#status = :relabeled AND relabel_confidence = :high',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':relabeled': 'relabeled', ':high': 'high'},
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            conflicts.extend(response.get('Items', []))

    except Exception as e:
        print(f"❌ DynamoDB scan failed: {e}")
        return {'statusCode': 500, 'error': str(e)}

    print(f"📋 Found {len(conflicts)} high-confidence conflicts")

    if len(conflicts) < 10:
        return {'statusCode': 400, 'message': f'Only {len(conflicts)} samples, need at least 10'}

    # ── Build CSV ─────────────────────────────────────────────────────────────
    label_map = {'Benign': 0, 'Botnet': 1, 'DDoS': 2, 'DoS': 3, 'PortScan': 4}
    csv_rows  = []

    for conflict in conflicts:
        try:
            flow_data = json.loads(conflict.get('flow_data', '{}'))
            label_num = label_map.get(conflict.get('correct_label', ''), 0)
            csv_rows.append({'label': label_num, **flow_data})
        except Exception as e:
            print(f"⚠️ Skipping {conflict.get('conflict_id')}: {e}")

    if not csv_rows:
        return {'statusCode': 400, 'message': 'No valid training samples'}

    # ── Write & upload CSV ────────────────────────────────────────────────────
    timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    local_csv = f'/tmp/distillation_train_{timestamp}.csv'

    all_keys = set()
    for row in csv_rows:
        all_keys.update(row.keys())
    all_keys.discard('label')
    fieldnames = ['label'] + sorted(all_keys)

    with open(local_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(csv_rows)

    s3_key = f'{OUTPUT_PREFIX}distillation_train_{timestamp}.csv'
    s3.upload_file(local_csv, BUCKET, s3_key)
    print(f"✅ CSV uploaded: s3://{BUCKET}/{s3_key}")

    # ── Kick SageMaker job ────────────────────────────────────────────────────
    job_name = f"finetune-teacher-{timestamp}"
    print(f"🚀 Starting SageMaker job: {job_name}")

    try:
        sm.create_training_job(
            TrainingJobName=job_name,
            AlgorithmSpecification={
                'TrainingImage':     ECR_IMAGE_FINETUNE,
                'TrainingInputMode': 'File'
            },
            RoleArn=SAGEMAKER_ROLE,
            InputDataConfig=[{
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType':             'S3Prefix',
                        'S3Uri':                  f's3://{BUCKET}/{OUTPUT_PREFIX}',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'text/csv'
            }],
            OutputDataConfig={'S3OutputPath': f's3://{BUCKET}/sagemaker/finetune-output/'},
            ResourceConfig={'InstanceType': 'ml.m5.xlarge', 'InstanceCount': 1, 'VolumeSizeInGB': 20},
            StoppingCondition={'MaxRuntimeInSeconds': 7200},
            Environment={
                'BUCKET':            BUCKET,
                'TEACHER_ENDPOINT':  'tf-endpoint',
                'DISTILL_FUNCTION':  DISTILL_FUNCTION,
                'SAGEMAKER_ROLE':    SAGEMAKER_ROLE,
                'ECR_IMAGE_DISTILL': ECR_IMAGE_DISTILL
            }
        )
        print(f"✅ SageMaker job started: {job_name}")

    except Exception as e:
        print(f"❌ Failed to start SageMaker job: {e}")
        # KHÔNG mark used khi fail — để có thể retry
        return {'statusCode': 500, 'error': str(e)}

    # ── Mark used CHỈ sau khi SageMaker job tạo thành công ───────────────────
    marked = 0
    for conflict in conflicts[:len(csv_rows)]:
        try:
            table.update_item(
                Key={
                    'conflict_id': conflict['conflict_id'],
                    'created_at':  conflict['created_at']
                },
                UpdateExpression='SET #status = :used',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':used': 'used'}
            )
            marked += 1
        except Exception as e:
            print(f"⚠️ Could not mark {conflict.get('conflict_id')} as used: {e}")

    print(f"✅ Marked {marked}/{len(csv_rows)} conflicts as used")

    return {
        'statusCode':   200,
        'csv_s3_path':  f's3://{BUCKET}/{s3_key}',
        'sample_count': len(csv_rows),
        'finetune_job': job_name,
        'timestamp':    timestamp
    }