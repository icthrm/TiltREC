#ifndef BASIC_HY__
#define BASIC_HY__

#include <cuda.h>
#include <cuda_runtime.h>
#include "mrcmx/mrcstack.h"

#define PI 3.14159265358979323846
#define PI_180 0.01745329252f
#define D2R(__ANGLE__) ((__ANGLE__)*PI_180)
#define CHECK_CUDA(func)                                         \
  {                                                              \
    cudaError_t status = (func);                                 \
    if (status != cudaSuccess)                                   \
    {                                                            \
      printf("CUDA API failed at line %d with error: %s (%d)\n", \
             __LINE__, cudaGetErrorString(status), status);      \
    }                                                            \
  }

struct SimCoeff
{
	float a[4];
	float b[4];
};

struct Bodyinfo
{
  float steplength;
  float slice_steplength;
  float Y_add_left;
  float Y_add_right;
  size_t volsize;
  size_t volslicesize;
  size_t projsize;
};

struct CuTaskDataY
{
	float *slice;
	float *projs;
	float *s;
  float *ax;
  float *w;
	int nproj;
	int x;
	int z;
	int y;
};
struct SysInfo
{
	int id;
	int procs;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int namelen;
};

size_t GetCuTaskDataSizePerPartition(int nproj, int x, int z);

// Allocate device memory
bool CuMallocBPTTaskData(CuTaskDataY &cudev, int nproj, int x, int z, int steplength);
bool CuMallocSIRTTaskData(CuTaskDataY &cudev, int nproj, int x, int z, int steplength);
bool CuMallocSARTTaskData(CuTaskDataY &cudev, int nproj, int x, int z, int steplength);

// Free device memory
void CuFreeTaskData(CuTaskDataY &cudev);
void CuFreeTaskDataADMM(CuTaskDataY &cudev);

// Calculate the coefficient values of the projected point
__device__ void CuValCoef(const Point3D &coord, int angidx, float *x, float *y);

// Bilinear interpolation
__device__ void CuBilinearValue(int width, int height, float X, float Y, float *vwt);
__device__ void CuBilinearValue(float *data, int width, int height, float X, float Y, float *val, float *vwt);

// Calculate the difference between projections
__global__ void CuCalcProjectionDiffKernel(float *projs, float *reproj_val, float *reproj_wt, int x, int y);
__global__ void CuCalcProjectionDiffKernel_SART(float *projs, float *reproj_val, float *reproj_wt, int x, int y, int projIdxStart);

// Select the desired volume
__global__ void CuSelectKernel(float *slice, int n, int steplength, int left);

// Reprojection
__device__ void DeviceReprojectKernel(float X, float Y, float x, float y, float *rval, float volval);
__global__ void CuReprojectKernel(float *slice, float *reproj_val, int x, int z, int y);
__global__ void CuReprojectKernel_SART(float *slice, float *reproj_val, int x, int z, int y, int projIdxStart);

// Backprojection
__global__ void CuBackProjKernel(float *slice, float *projs, int x, int z, int y);
__global__ void CuBackProjWeightAndValueKernel(float *valvol, cudaTextureObject_t projtex, int volcor_x, int volcor_y, int nproj, int volcor_z);
__global__ void CuBackProjWeightAndValueKernel_SART(float *valvol, cudaTextureObject_t projtex, int volcor_x, int volcor_y, int volcor_z, int angidx);

// Compute the weights
__global__ void CuVolWeightKernel(float *wtvol, int volcor_x, int volcor_y, int volcor_z, int nproj);
__global__ void CuReWeightKernel(float *reproj_wt, int volcor_x, int volcor_y, int volcor_z);

// Update the 3D volume
__global__ void CuUpdateVolumeByWeightsKernel(float *slice, float *valvol, float *wtvol, float gamma, size_t maxN);
__global__ void CuUpdateVolumeByProjDiffKernel_SART(float *proj, float *vol, int volcor_x, int volcor_y, int volcor_z, int angidx, float gamma);
__global__ void CufloatToByteKernel(float* d_floatData, float mean, float std, float scale_factor, size_t maxN) ;
#endif