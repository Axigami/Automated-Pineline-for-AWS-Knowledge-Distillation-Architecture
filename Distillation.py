import boto3
import os

# Initialize clients
dynamodb = boto3.resource('dynamodb')
lambda_client = boto3.client('lambda')

def lambda_handler(event, context):
    CONFLICTS_TABLE = os.environ['CONFLICTS_TABLE']
    THRESHOLD = int(os.environ.get('THRESHOLD', '100'))
    RELABEL_FUNCTION = os.environ['RELABEL_FUNCTION']
    
    print(f"🔍 Checking conflicts (threshold: {THRESHOLD})")
    
    table = dynamodb.Table(CONFLICTS_TABLE)
    
    # Count pending conflicts using GSI
    try:
        response = table.query(
            IndexName='status-index',
            KeyConditionExpression='#status = :pending',
            ExpressionAttributeNames={'#status': 'status'},
            ExpressionAttributeValues={':pending': 'pending'},
            Select='COUNT'
        )
        
        count = response.get('Count', 0)
        
        # Handle pagination for accurate count
        while 'LastEvaluatedKey' in response:
            response = table.query(
                IndexName='status-index',
                KeyConditionExpression='#status = :pending',
                ExpressionAttributeNames={'#status': 'status'},
                ExpressionAttributeValues={':pending': 'pending'},
                ExclusiveStartKey=response['LastEvaluatedKey'],
                Select='COUNT'
            )
            count += response.get('Count', 0)
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        return {
            'statusCode': 500,
            'error': str(e)
        }
    
    print(f"📊 Pending conflicts: {count}/{THRESHOLD}")
    
    if count >= THRESHOLD:
        print("🚨 Threshold reached! Triggering relabeling...")
        
        try:
            # Invoke Relabel Lambda asynchronously
            lambda_client.invoke(
                FunctionName=RELABEL_FUNCTION,
                InvocationType='Event'  # Async
            )
            
            print(f"✅ Triggered {RELABEL_FUNCTION}")
            
            return {
                'statusCode': 200,
                'triggered': True,
                'conflict_count': count,
                'threshold': THRESHOLD,
                'message': f'Relabeling triggered for {count} conflicts'
            }
            
        except Exception as e:
            print(f"❌ Failed to trigger relabel: {e}")
            return {
                'statusCode': 500,
                'error': str(e)
            }
    
    else:
        print(f"✅ Below threshold ({count}/{THRESHOLD}), no action needed")
        
        return {
            'statusCode': 200,
            'triggered': False,
            'conflict_count': count,
            'threshold': THRESHOLD,
            'message': f'Only {count} conflicts, need {THRESHOLD - count} more to trigger'
        }