# Gravity

This is an attempt to compare computation time for different implementation of a (triangular) matrix multiplication task.

The task is a brute force simulation of the nbody problem, thus `O(n^n)` in time complexity.

4 (+1) implementations are compared:
- `nbody_c`: implemention in C.
- `nbody_cuda`: implementation in Cuda (disclaimer: I'm not a Cuda expert, don't expect this to be well optimized).
- `nbody_pytorch`: implementation in Pytorch. GPU acceleration can be deactivated by adding the flag `CUDA_VISIBLE_DEVICES=` to the commandline.
- `nbody_triton`: implementation in Pytorch with Triton, to fuse kernel operation and acceleration computation time.

Rendering is also available.

## Install

```
python setup.py install
```

## Run

```
python run.py --backend [nbody_c,nbody_cuda,nbody_pytorch,nbody_triton]
```

## Benchmark

TODO

## Rendering

Computed trajectories are stored in `result.data`. They can be rendered by adding the flag `--render` then rendered.

To render previously saved trajectories, run:

```
python run.py --render path/to/another.data 
```
