
# CUDA Programming Examples



```sh
nvcc add.cu -o /tmp/add_cuda && /tmp/add_cuda

nvprof /tmp/add_cuda
nvprof --unified-memory-profiling off /tmp/add_cuda
```

## TODO

- https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/
- https://devblogs.nvidia.com/parallelforall/unified-memory-cuda-beginners/