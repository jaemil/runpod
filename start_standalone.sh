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

export_env_vars

python3 -u rp_handler.py
