import boto3
import os
import json
from datetime import datetime

s3 = boto3.client('s3')
sm = boto3.client('sagemaker')

BUCKET               = os.environ.get('BUCKET', 'anomalytraffic')
ECR_IMAGE_DISTILL    = os.environ.get('ECR_IMAGE_DISTILL', '')
SAGEMAKER_ROLE       = os.environ.get('SAGEMAKER_ROLE', '')
TEACHER_ENDPOINT     = os.environ.get('TEACHER_ENDPOINT', 'tf-endpoint')
EXPORT_FUNCTION      = os.environ.get('EXPORT_FUNCTION', 'ExportONNX')
INPUT_DATA_PREFIX    = 'data/distillation/train/'


def lambda_handler(event, context):
    print("🔥 TriggerDistillation Lambda")
    print("=" * 60)
    
    # ========================================
    # Validate environment variables
    # ========================================
    
    if not ECR_IMAGE_DISTILL:
        return {
            'statusCode': 400,
            'error': 'ECR_IMAGE_DISTILL environment variable not set'
        }
    
    if not SAGEMAKER_ROLE:
        return {
            'statusCode': 400,
            'error': 'SAGEMAKER_ROLE environment variable not set'
        }
    
    print(f"📦 ECR Image: {ECR_IMAGE_DISTILL}")
    print(f"🔐 SageMaker Role: {SAGEMAKER_ROLE}")
    print(f"🎓 Teacher Endpoint: {TEACHER_ENDPOINT}")
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
        # Find latest CSV since FineTuneTeacher just passes {'triggered_by': 'FineTuneTeacher'}
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
            
            # SageMaker expects the prefix containing CSV chunks
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
    job_name = f"distill-student-{timestamp}"
    
    print(f"🚀 Starting SageMaker Distillation job: {job_name}")
    
    try:
        sm.create_training_job(
            TrainingJobName=job_name,
            
            # Algorithm specification
            AlgorithmSpecification={
                'TrainingImage': ECR_IMAGE_DISTILL,
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
                'S3OutputPath': f's3://{BUCKET}/sagemaker/distill-output/'
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
            
            # Hyperparameters (passed to LightGBM script as args)
            HyperParameters={
                'num-leaves': '127',
                'max-depth': '6',
                'learning-rate': '0.05',
                'n-estimators': '400',
                'temperature': '3.0'
            },
            
            # Environment variables for the Distillation image
            Environment={
                'BUCKET': BUCKET,
                'TEACHER_ENDPOINT': TEACHER_ENDPOINT,
                'EXPORT_FUNCTION': EXPORT_FUNCTION,
                'AWS_DEFAULT_REGION': os.environ.get('AWS_REGION', 'ap-southeast-2')
            },
            
            # Tags
            Tags=[
                {'Key': 'Project', 'Value': 'KnowledgeDistillation'},
                {'Key': 'Stage', 'Value': 'StudentTraining'},
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
        
        return {
            'statusCode': 200,
            'job_name': job_name,
            'training_data': f's3://{BUCKET}/{training_data}',
            'timestamp': timestamp,
            'message': f'Distillation job {job_name} started successfully'
        }
        
    except Exception as e:
        print(f"❌ Failed to create distillation training job: {e}")
        
        return {
            'statusCode': 500,
            'error': str(e),
            'job_name': job_name
        }
