FROM 314815235551.dkr.ecr.us-east-2.amazonaws.com/sagemaker-spark-processing:3.5-cpu-py312-v1.0

# Copy requirements first (for better caching)
COPY requirements.txt /tmp/requirements.txt

# Install dependencies with persistent pip cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# Set working directory (code will be mounted from S3 at runtime)
WORKDIR /opt/ml/processing

# SageMaker Spark containers expect spark-submit to be overridden via ContainerEntrypoint
