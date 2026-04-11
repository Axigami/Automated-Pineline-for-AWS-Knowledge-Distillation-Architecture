import boto3
import os
import subprocess
import json

s3 = boto3.client('s3')

BUCKET = os.environ.get('BUCKET', 'anomalytraffic')
MODEL_PREFIX = 'models/lightgbm/'
ONNX_PREFIX = 'models/onnx/'

def lambda_handler(event, context):
    """
    Convert LightGBM model to ONNX format
    """
    
    print("🔄 Converting LightGBM to ONNX...")
    
    # Get model path from event or use latest
    model_key = event.get('model_key')
    
    if not model_key:
        # Find latest model
        response = s3.list_objects_v2(
            Bucket=BUCKET,
            Prefix=MODEL_PREFIX
        )
        
        if 'Contents' not in response:
            return {
                'statusCode': 404,
                'message': 'No models found'
            }
        
        # Get latest
        models = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
        model_key = models[0]['Key']
    
    print(f"Converting model: s3://{BUCKET}/{model_key}")
    
    # Download model
    local_model = '/tmp/model.txt'
    s3.download_file(BUCKET, model_key, local_model)
    
    # Install dependencies (if not in layer)
    subprocess.run([
        'pip', 'install', 
        'lightgbm', 'onnxmltools', 'skl2onnx', 
        '--target', '/tmp/libs',
        '--quiet'
    ])
    
    import sys
    sys.path.insert(0, '/tmp/libs')
    
    import lightgbm as lgb
    from onnxmltools.convert import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType
    
    # Load LightGBM model
    booster = lgb.Booster(model_file=local_model)
    
    # Convert to ONNX
    initial_types = [('input', FloatTensorType([None, 15]))]  # 15 features
    
    onnx_model = convert_lightgbm(
        booster,
        initial_types=initial_types,
        target_opset=12
    )
    
    # Save ONNX
    local_onnx = '/tmp/model.onnx'
    
    with open(local_onnx, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    
    # Upload to S3
    timestamp = model_key.split('/')[-1].replace('.txt', '')
    onnx_key = f'{ONNX_PREFIX}model_{timestamp}.onnx'
    
    s3.upload_file(local_onnx, BUCKET, onnx_key)
    
    print(f"✅ ONNX model saved to s3://{BUCKET}/{onnx_key}")
    
    return {
        'statusCode': 200,
        'onnx_s3_path': f's3://{BUCKET}/{onnx_key}',
        'model_key': model_key
    }