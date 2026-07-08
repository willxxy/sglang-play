# sglang-play
Playing with SGLANG on H100 NVL GPUs

## Installation
1. `uv venv`
2. `uv pip install --upgrade pip`
3. `uv pip install --prerelease=allow sglang`
4. `uv pip install --force-reinstall  torch==2.11.0 torchaudio==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu129`
5. `uv pip install --force-reinstall sglang-kernel --index-url https://docs.sglang.ai/whl/cu129/`
6. `uv pip install --force-reinstall sgl-deep-gemm --index-url https://docs.sglang.ai/whl/cu129/ --no-deps`



### check_environment.py output

```
(sglang-play) bash-4.4$ uv run python3 src/check_environment.py 
Python: 3.11.14 (main, Dec 17 2025, 21:07:37) [Clang 21.1.4 ]
CUDA available: True
GPU 0,1,2,3,4,5,6,7: NVIDIA H100 NVL
GPU 0,1,2,3,4,5,6,7 Compute Capability: 9.0
CUDA_HOME: /home/whan/cuda-12.8
NVCC: Cuda compilation tools, release 12.8, V12.8.61
CUDA Driver Version: 570.158.01
PyTorch: 2.11.0+cu129
sglang: 0.5.10.post1
sglang-kernel: 0.4.4+cu129
flashinfer_python: 0.6.7.post3
flashinfer_cubin: 0.6.7.post3
flashinfer_jit_cache: Module Not Found
triton: 3.6.0
transformers: 5.3.0
torchao: 0.9.0
numpy: 2.4.4
aiohttp: 3.14.1
fastapi: 0.139.0
huggingface_hub: 1.22.0
interegular: 0.3.3
modelscope: 1.38.1
orjson: 3.11.9
outlines: 0.1.11
packaging: 26.2
psutil: 7.2.2
pydantic: 2.14.0a1
python-multipart: 0.0.32
pyzmq: 27.1.0
uvicorn: 0.51.0
uvloop: 0.22.1
vllm: Module Not Found
xgrammar: 0.1.32
openai: 2.6.1
tiktoken: 0.13.0
anthropic: 0.116.0
litellm: Module Not Found
torchcodec: 0.9.1
NVIDIA Topology: 
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    NIC0    NIC1    CPU Affinity     NUMA Affinity   GPU NUMA ID
GPU0     X      NV12    NODE    NODE    SYS     SYS     SYS     SYS     NODE    NODE    0-63,128-191     0               N/A
GPU1    NV12     X      NODE    NODE    SYS     SYS     SYS     SYS     PHB     PHB     0-63,128-191     0               N/A
GPU2    NODE    NODE     X      NV12    SYS     SYS     SYS     SYS     NODE    NODE    0-63,128-191     0               N/A
GPU3    NODE    NODE    NV12     X      SYS     SYS     SYS     SYS     NODE    NODE    0-63,128-191     0               N/A
GPU4    SYS     SYS     SYS     SYS      X      NV12    NODE    NODE    SYS     SYS     64-127,192-255   1               N/A
GPU5    SYS     SYS     SYS     SYS     NV12     X      NODE    NODE    SYS     SYS     64-127,192-255   1               N/A
GPU6    SYS     SYS     SYS     SYS     NODE    NODE     X      NV12    SYS     SYS     64-127,192-255   1               N/A
GPU7    SYS     SYS     SYS     SYS     NODE    NODE    NV12     X      SYS     SYS     64-127,192-255   1               N/A
NIC0    NODE    PHB     NODE    NODE    SYS     SYS     SYS     SYS      X      PIX
NIC1    NODE    PHB     NODE    NODE    SYS     SYS     SYS     SYS     PIX      X 

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges withina NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_0
  NIC1: mlx5_1


ulimit soft: 8192
(sglang-play) bash-4.4$
```