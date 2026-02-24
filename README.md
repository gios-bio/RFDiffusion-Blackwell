# RFDiffusion-Blackwell
Dockerfile and patch for installation of RFDiffusion under GPUs with Blackwell architecture

Code is based on https://github.com/JMB-Scripts/RFdiffusion-dockerfile-nvidia-RTX5090

# Usage
- clone the repo
- Go inside and assemble docker image. I prefer to use Podman: `podman build --device nvidia.com/gpu=all --format docker -t rfdiffusion .` For docker use `docker buildx build --gpus all -t rfdiffusion .`