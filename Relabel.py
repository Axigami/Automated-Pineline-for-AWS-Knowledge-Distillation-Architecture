import boto3
import json
import os
from datetime import datetime, timezone

dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')

CONFLICTS_TABLE = os.environ.get('CONFLICTS_TABLE', 'AnomalyConflicts')
BUCKET = os.environ.get('BUCKET', 'anomalytraffic')


def lambda_handler(event, context):
    """
    Relabel conflicts based on route validation
    
    Rules:
    1. Expected "attack" + Actual "Benign" → Mark for manual review
    2. Expected "Benign" + Actual attack → Correct to "Benign" (high confidence)
    """
    
    print("🔄 Starting conflict relabeling (route-based)...")
    
    table = dynamodb.Table(CONFLICTS_TABLE)
    
    # Query all pending conflicts
    try:
        response = table.query(
            IndexName='status-index',
            KeyConditionExpression='#status = :pending',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={':pending': 'pending'}
        )
        
        conflicts = response.get('Items', [])
        
        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = table.query(
                IndexName='status-index',
                KeyConditionExpression='#status = :pending',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':pending': 'pending'},
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            conflicts.extend(response.get('Items', []))
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return {'statusCode': 500, 'error': str(e)}
    
    print(f"📊 Found {len(conflicts)} pending conflicts")
    
    if len(conflicts) == 0:
        return {
            'statusCode': 200,
            'relabeled_count': 0,
            'message': 'No pending conflicts'
        }
    
    # Relabel each
    relabeled_count = 0
    high_conf_count = 0
    low_conf_count = 0
    manual_review_count = 0
    
    for conflict in conflicts:
        try:
            conflict_id = conflict['conflict_id']
            created_at = conflict['created_at']
            
            expected_label = conflict.get('expected_label', '')
            actual_pred = json.loads(conflict.get('actual_prediction', '{}'))
            actual_label = actual_pred.get('label', '')
            
            print(f"🔍 {conflict_id}: expected={expected_label}, actual={actual_label}")
            
            # Apply relabeling logic
            result = determine_correct_label_route_based(
                expected_label, 
                actual_label,
                conflict.get('conflict_rule', ''),
                actual_pred.get('confidence', 0)
            )
            
            correct_label = result['correct_label']
            confidence = result['confidence']
            reason = result['reason']
            needs_review = result.get('needs_review', False)
            
            # Update DynamoDB
            table.update_item(
                Key={
                    'conflict_id': conflict_id,
                    'created_at': created_at
                },
                UpdateExpression='''
                    SET #status = :status,
                        correct_label = :label,
                        relabel_confidence = :conf,
                        relabel_reason = :reason,
                        relabeled_at = :now,
                        needs_manual_review = :review
                ''',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={
                    ':status': 'relabeled',
                    ':label': correct_label,
                    ':conf': confidence,
                    ':reason': reason,
                    ':now': datetime.now(timezone.utc).isoformat(),
                    ':review': needs_review
                }
            )
            
            relabeled_count += 1
            
            if confidence == 'high':
                high_conf_count += 1
            else:
                low_conf_count += 1
            
            if needs_review:
                manual_review_count += 1
            
            print(f"✅ {expected_label} → {correct_label} ({confidence})")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue
    
    print("=" * 60)
    print(f"✅ Relabeled: {relabeled_count}")
    print(f"🎯 High confidence: {high_conf_count}")
    print(f"⚠️ Low confidence: {low_conf_count}")
    print(f"👁️ Manual review needed: {manual_review_count}")
    print("=" * 60)
    
    return {
        'statusCode': 200,
        'relabeled_count': relabeled_count,
        'breakdown': {
            'high_confidence': high_conf_count,
            'low_confidence': low_conf_count,
            'manual_review': manual_review_count
        }
    }


def determine_correct_label_route_based(expected_label, actual_label, 
                                         conflict_rule, actual_confidence):
    """
    Determine correct label based on route expectations
    
    Rules:
    1. Expected "attack" + Actual "Benign"
       → Cannot auto-correct (don't know which attack type)
       → Mark for manual review
    
    2. Expected "Benign" + Actual attack (any type)
       → Trust expected (source is ground truth)
       → Correct to "Benign"
       → High confidence
    """
    
    # Rule 1: Anomaly source predicted as Benign
    if expected_label == "attack" and actual_label == "Benign":
        return {
            'correct_label': 'attack_needs_review',
            'confidence': 'low',
            'reason': f'Anomaly source predicted as Benign - needs manual labeling to determine attack type',
            'needs_review': True
        }
    
    # Rule 2: Log source predicted as attack
    if expected_label == "Benign" and actual_label != "Benign":
        return {
            'correct_label': 'Benign',
            'confidence': 'high',
            'reason': f'Log source incorrectly predicted as {actual_label} - corrected to Benign',
            'needs_review': False
        }
    
    # Fallback
    return {
        'correct_label': expected_label,
        'confidence': 'low',
        'reason': 'Unknown conflict pattern',
        'needs_review': True
    }