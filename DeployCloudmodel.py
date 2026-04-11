import sagemaker
from sagemaker.tensorflow import TensorFlowModel

role = sagemaker.get_execution_role()

model = TensorFlowModel(
    model_data="s3://anomalytraffic/models/cloud/model.tar.gz",
    role=role,
    framework_version="2.12"
)

predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="tf-endpoint"
)