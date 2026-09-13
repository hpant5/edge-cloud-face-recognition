FROM public.ecr.aws/lambda/python:3.11

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Keep torch's cache inside the writable Lambda filesystem
ENV TORCH_HOME=/tmp

COPY fr_lambda.py ${LAMBDA_TASK_ROOT}/
# Supply your own embedding store at build time:
# COPY data.pt ${LAMBDA_TASK_ROOT}/

CMD ["fr_lambda.lambda_handler"]
