FROM public.ecr.aws/lambda/python:3.12

RUN dnf install -y libgomp gcc gcc-c++ && dnf clean all

COPY requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir -r requirements.txt -t ${LAMBDA_TASK_ROOT} && \
    pip install --no-cache-dir "setuptools<71" -t ${LAMBDA_TASK_ROOT} --upgrade

COPY ExportONNX.py ${LAMBDA_TASK_ROOT}

CMD ["ExportONNX.lambda_handler"]