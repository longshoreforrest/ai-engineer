ARG TARGETPLATFORM=linux/amd64
FROM --platform=${TARGETPLATFORM} python:3.9-slim

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      --extra-index-url https://pypi.ngc.nvidia.com/simple \
      "pytriton>=0.6,<0.7" \
      pillow \
      torch==2.2.0 \
      torchvision==0.17.0

WORKDIR /adapter
COPY inference_adapter.py .
      
CMD ["python", "inference_adapter.py"]
