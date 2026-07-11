# 1. What CUDA version does the driver actually support? (needs >= 13000 for the cute kernels)
python3 -c "import ctypes; l=ctypes.CDLL('libcuda.so.1'); v=ctypes.c_int(); l.cuDriverGetVersion(ctypes.byref(v)); print(v.value)"

# 2. Is the fallback already on without you setting it?
env | grep -i FLASHINFER
python3 -c "import flashinfer.norm as n; print('CUDA-JIT fallback active:', n._USE_CUDA_NORM)"

# 3. Driver version of this node
nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1