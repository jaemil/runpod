#!/usr/bin/env bash

TORCH_VERSION="2.2.0"

echo "Deleting Inswapper Serverless Worker"
rm -rf /workspace/runpod-worker-inswapper

echo "Deleting venv"
rm -rf /workspace/venv

echo "Cloning Inswapper Serverless Worker repo to /workspace"
cd /workspace
git clone https://github.com/ashleykleynhans/runpod-worker-inswapper.git
cd runpod-worker-inswapper

echo "Installing Ubuntu updates"
apt update
apt -y upgrade

echo "Installing git-lfs and unzip Ubuntu packages"
apt -y install git-lfs unzip

echo "Creating and activating venv"
python3 -m venv /workspace/venv
source /workspace/venv/bin/activate

echo "Installing Torch"
pip3 install --no-cache-dir torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Installing Inswapper Serverless Worker"
pip3 install -r requirements.txt
pip3 uninstall -y onnxruntime
pip3 install onnxruntime-gpu

echo "Downloading insightface checkpoints"
mkdir -p checkpoints/models
cd checkpoints
wget -O inswapper_128.onnx https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx
cd models
wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
mkdir buffalo_l
cd buffalo_l
unzip ../buffalo_l.zip

echo "Installing CodeFormer"
cd /workspace/runpod-worker-inswapper
git lfs install
git clone https://huggingface.co/spaces/sczhou/CodeFormer

echo "Downloading CodeFormer weights"
cd /workspace/runpod-worker-inswapper
mkdir -p CodeFormer/CodeFormer/weights/CodeFormer
wget -O CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
mkdir -p CodeFormer/CodeFormer/weights/facelib
wget -O CodeFormer/CodeFormer/weights/facelib/detection_Resnet50_Final.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth"
wget -O CodeFormer/CodeFormer/weights/facelib/parsing_parsenet.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth"
mkdir -p CodeFormer/CodeFormer/weights/realesrgan
wget -O CodeFormer/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth"

echo "Creating logs directory"
mkdir -p /workspace/logs
