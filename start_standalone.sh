#!/usr/bin/env bash

# Export env vars
export_env_vars() {
    echo "Exporting environment variables..."
    printenv | grep -E '^RUNPOD_|^PATH=|^_=' | awk -F = '{ print "export " $1 "=\"" $2 "\"" }' >> /etc/rp_environment
    echo 'source /etc/rp_environment' >> ~/.bashrc
}

echo "Worker Initiated"

echo "Symlinking files from Network Volume"
ln -s /runpod-volume /workspace
rm -rf /root/.cache
rm -rf /root/.ifnude
rm -rf /root/.insightface
ln -s /runpod-volume/.cache /root/.cache
ln -s /runpod-volume/.ifnude /root/.ifnude
ln -s /runpod-volume/.insightface /root/.insightface

echo "Starting RunPod Handler"
export PYTHONUNBUFFERED=1
cd /workspace/runpod

# export_env_vars

export BUCKET_ENDPOINT_URL
export AWS_S3_REGION
export AWS_S3_ACCESS_KEY_ID
export AWS_S3_SECRET_ACCESS_KEY
export AWS_S3_BUCKET_NAME
export BUCKET_ENDPOINT_URL
export BUCKET_ACCESS_KEY_ID
export BUCKET_SECRET_ACCESS_KEY

python3 -u rp_handler.py
