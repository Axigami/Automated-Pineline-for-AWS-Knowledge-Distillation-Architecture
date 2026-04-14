import boto3
import os
import json
from datetime import datetime

s3 = boto3.client('s3')
sm = boto3.client('sagemaker')

BUCKET              = os.environ.get('BUCKET', 'anomalytraffic')
ECR_IMAGE_FINETUNE  = os.environ.get('ECR_IMAGE_FINETUNE', '')
SAGEMAKER_ROLE      = os.environ.get('SAGEMAKER_ROLE', '')
INPUT_DATA_PREFIX   = 'data/distillation/train/'


def lambda_handler(event, context):
    print("🔥 TriggerFineTuning Lambda")
    print("=" * 60)
    
    # ========================================
    # Validate environment variables
    # ========================================
    
    if not ECR_IMAGE_FINETUNE:
        return {
            'statusCode': 400,
            'error': 'ECR_IMAGE_FINETUNE environment variable not set'
        }
    
    if not SAGEMAKER_ROLE:
        return {
            'statusCode': 400,
            'error': 'SAGEMAKER_ROLE environment variable not set'
        }
    
    print(f"📦 ECR Image: {ECR_IMAGE_FINETUNE}")
    print(f"🔐 SageMaker Role: {SAGEMAKER_ROLE}")
    print("")
    
    # ========================================
    # Find training data
    # ========================================
    
    csv_path = event.get('csv_s3_path', '')
    
    if csv_path:
        # Use provided CSV path
        print(f"📄 Using provided CSV: {csv_path}")
        training_data = csv_path.replace(f's3://{BUCKET}/', '')
        training_data = '/'.join(training_data.split('/')[:-1]) + '/'
    else:
        # Find latest CSV
        print(f"🔍 Finding latest CSV in s3://{BUCKET}/{INPUT_DATA_PREFIX}")
        
        try:
            response = s3.list_objects_v2(
                Bucket=BUCKET,
                Prefix=INPUT_DATA_PREFIX
            )
            
            if 'Contents' not in response:
                return {
                    'statusCode': 404,
                    'error': f'No training data found in s3://{BUCKET}/{INPUT_DATA_PREFIX}'
                }
            
            # Get latest CSV
            csvs = [obj for obj in response['Contents'] if obj['Key'].endswith('.csv')]
            
            if not csvs:
                return {
                    'statusCode': 404,
                    'error': 'No CSV files found'
                }
            
            # Sort by LastModified
            csvs.sort(key=lambda x: x['LastModified'], reverse=True)
            latest_csv = csvs[0]
            
            print(f"✅ Found latest CSV: {latest_csv['Key']}")
            print(f"   Size: {latest_csv['Size'] / 1024:.1f} KB")
            print(f"   Modified: {latest_csv['LastModified']}")
            
            training_data = INPUT_DATA_PREFIX
            
        except Exception as e:
            print(f"❌ Failed to find training data: {e}")
            return {
                'statusCode': 500,
                'error': str(e)
            }
    
    print(f"📂 Training data: s3://{BUCKET}/{training_data}")
    print("")
    
    # ========================================
    # Create SageMaker training job
    # ========================================
    
    timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    job_name = f"finetune-teacher-{timestamp}"
    
    print(f"🚀 Starting SageMaker job: {job_name}")
    
    try:
        sm.create_training_job(
            TrainingJobName=job_name,
            
            # Algorithm specification
            AlgorithmSpecification={
                'TrainingImage': ECR_IMAGE_FINETUNE,
                'TrainingInputMode': 'File'
            },
            
            # IAM role
            RoleArn=SAGEMAKER_ROLE,
            
            # Input data
            InputDataConfig=[{
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': f's3://{BUCKET}/{training_data}',
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'text/csv',
                'CompressionType': 'None'
            }],
            
            # Output
            OutputDataConfig={
                'S3OutputPath': f's3://{BUCKET}/sagemaker/finetune-output/'
            },
            
            # Resources
            ResourceConfig={
                'InstanceType': 'ml.m5.xlarge',
                'InstanceCount': 1,
                'VolumeSizeInGB': 30
            },
            
            # Stopping condition
            StoppingCondition={
                'MaxRuntimeInSeconds': 7200  # 2 hours max
            },
            
            # Hyperparameters (passed to script as args)
            HyperParameters={
                'epochs': '10',
                'batch-size': '32',
                'learning-rate': '0.0001',
                'min-improvement': '0.001'
            },
            
            # Environment variables for script
            Environment={
                'BUCKET': BUCKET,
                'TEACHER_ENDPOINT': 'tf-endpoint',
                'DISTILL_FUNCTION': 'TriggerDistillation',
                'SAGEMAKER_ROLE': SAGEMAKER_ROLE,
                'AWS_DEFAULT_REGION': os.environ.get('AWS_REGION', 'ap-southeast-2')
            },
            
            # Tags
            Tags=[
                {'Key': 'Project', 'Value': 'Distillation'},
                {'Key': 'Stage', 'Value': 'FineTuning'},
                {'Key': 'Timestamp', 'Value': timestamp}
            ]
        )
        
        print(f"✅ SageMaker job created: {job_name}")
        print(f"   Instance: ml.m5.xlarge")
        print(f"   Max runtime: 2 hours")
        print("")
        print("📊 Monitor progress:")
        print(f"   aws sagemaker describe-training-job --training-job-name {job_name}")
        print("")
        print("📋 View logs:")
        print(f"   aws logs tail /aws/sagemaker/TrainingJobs --follow")
        print("")
        
        return {
            'statusCode': 200,
            'job_name': job_name,
            'training_data': f's3://{BUCKET}/{training_data}',
            'timestamp': timestamp,
            'message': f'Fine-tuning job {job_name} started successfully'
        }
        
    except Exception as e:
        print(f"❌ Failed to create training job: {e}")
        
        return {
            'statusCode': 500,
            'error': str(e),
            'job_name': job_name
        }