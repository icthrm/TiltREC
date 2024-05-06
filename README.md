# TiltRec

## Introduction

**TiltRec** is an efficient software tool for cryo-electron tomography reconstruction using MPI and CUDA.

### System Requirements and Dependencies
- CMake: Version 2.6 or newer.
- MPICH 
- CUDA: A minimum of version 11 is required to leverage GPU acceleration for computations.
- OpenCV4
- Docker（optional）
## Compilation Instructions 
### Without Docker

```bash
mkdir build
cd build
cmake ..
make -j8
```
### Using Docker
```bash
docker build -t tiltrec:v1 .
docker run -it --rm --gpus all tiltrec:v1 /bin/bash
mkdir build
cd build
cmake ..
make -j8
```

## Command-Line Options
TiltRec provides a variety of command-line options to allow users more flexibility when running cryo-ET reconstructions. The basic command-line structure is as follows:

```bash
mpirun -n p ./ycuda [options]
```

## Parameters

- **`-INPUT(-i) <input filename>`**:The name of the input MRC file used for reconstruction.

- **`-OUTPUT(-o) <output filename>`**: Designates the name of the resulting MRC file.

- **`-TILTFILE(-t) <tilt angle filename>`**: The file containing tilt angles for the reconstruction.

- **`-INITIAL <initial reconstruction filename>`**: Provides an initial MRC file to be used as the model for iterative reconstruction methods (optional).

- **`-GEOMETRY(-g) <4 integers>`**: Defines geometric information: offset, pitch angle, z-axis offset, and thickness.

- **`-AXIS <reconstruction axis>`**: Sets the axis for reconstruction, either y (default) or z.

- **`-METHOD(-m) <method name>`**: Selects the reconstruction method:
  - Back Projection ：BPT
  - Filtered Back Projection: FBP
  - Weighted Back Projection: WBP
  - SART: SART, number of iterations, relaxation parameter
  - SIRT: SIRT, number of iterations, relaxation parameter
  - ADMM: ADMM, number of iterations, number of conjugate gradient iterations, relaxation parameter, threshold

- **`-help(-h)`**: Displays help information.



## Example
The following example demonstrates how to perform a reconstruction using the Weighted Back Projection (WBP) method along the y-axis with two threads:
```bash
mpirun -n 2 ./ycuda --input ../../data/BBb/BBb_fin.mrc --output ../../data/BBb/BBb_SIRT_y.mrc --tiltfile ../../data/BBb/BBb.rawtlt --geometry 0,0,0,300 --method WBP
```