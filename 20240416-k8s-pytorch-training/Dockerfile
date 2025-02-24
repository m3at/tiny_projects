# Pin to specific build to avoid re-pulling daily
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime@sha256:11691e035a3651d25a87116b4f6adc113a27a29d8f5a6a583f8569e0ee5ff897

ADD requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Download the toy dataset once ahead of time
# RUN python3 -c "from torchvision.datasets import MNIST as D; D('.', train=True, download=True); D('.', train=False, download=True)"
RUN python3 -c "from torchvision.datasets import CIFAR10 as D; D('.', train=True, download=True); D('.', train=False, download=True)"

ADD payload.py payload.py

ENTRYPOINT ["python3", "payload.py"]
