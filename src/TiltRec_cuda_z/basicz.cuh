#ifndef BASIC_HZ__
#define BASIC_HZ__

#include "mrcmx/mrcstack.h"
#include "../TiltRec_cuda_y/basic.cuh"
#define DATATYPE

#define CUERR                                                                                \
	{                                                                                        \
		cudaError_t err;                                                                     \
		if ((err = cudaGetLastError()) != cudaSuccess)                                       \
		{                                                                                    \
			printf("CUDA error: %s : %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
			exit(1);                                                                         \
		}                                                                                    \
	}


struct Coeff
{
	union
	{
		double p[20];
		struct
		{
			double a[10];
			double b[10];
		};
	};
};

struct CuTaskDataZ
{
	Point3DF *origin;
	SimCoeff *coeffs;
	float *slice;
	float *projs;
	float *s;
	float *c;
	int nproj;
	int x;
	int z;
	int y;
};

//Allocate device memory
bool CuMallocBPTTaskDataZ(CuTaskDataZ &cudev, int nproj, int x, int y, int steplength);
bool CuMallocSIRTTaskDataZ(CuTaskDataZ &cudev, int nproj, int x, int y, int thickness, int batchSize);
bool CuMallocADMMTaskDataZ(CuTaskDataZ &cudev, int nproj, int x, int y, int thickness, int batchSize);
//Free device memory
void CuFreeTaskDataZ(CuTaskDataZ &cudev);

//Calculate the coefficient values of the projected point
__device__ void CuValCoefZ(const Point3DF &origin, const Point3D &coord, const SimCoeff &coeff, Weight *wt);

//Bilinear interpolation
__device__ void CuBilinearValueZ(const float *data, int width, int height, const Weight &wt, float *val, float *vwt);

//Calculate the difference between projections
__global__ void CuCalcProjectionDiffKernelZ(float *projs, float *reproj_val, float *reproj_wt, int x, int y);
__global__ void CuCalcReProjectionKernelZ(float *reproj_val, float *reproj_wt, int x, int y);

//Reprojection
__global__ void CuReprojectKernelZ(const Point3DF *origin, SimCoeff *coeffs, float *slice,
								   float *reproj_val, float *reproj_wt, int x, int y, int coordz_offset, int angIdxStart);

//Backprojection
__global__ void CuBackProjKernelZ(Point3DF *origin, SimCoeff *coeffs, float *slice, float *projs, int x, int y, int coordz_offset);
__global__ void CuBackProjWeightAndValueKernelZ(Point3DF *origin, SimCoeff *coeffs, float *valvol, float *wtvol, float *diffs, int x, int y, 
                                                int coordz_offset, int angIdxStart);

//Update the 3D volume
__global__ void CuUpdateVolumeByWeightsKernelZ(float *slice, float *valvol, float *wtvol, float gamma, size_t maxN);
__global__ void CuUpdateVolumeByProjDiffKernelZ(Point3DF *origin, SimCoeff *coeffs, float *vol, float *diffs,
												float gamma, int x, int y, int coordz_offset, int angIdxStart);

// ADMM
__global__ void CuAtb_ADMM_Z(Point3DF *origin, SimCoeff *coeffs,float *htb, int cudev_x, int cudev_y, 
                             float *projsdata, int coordz_offset, int angIdxStart);
__global__ void CuATbGammaIt_ADMM_Z(float *htb, float *u_k, float *d_k, size_t volsize, float gamma);


__global__ void CuAtA_ADMM1_Z(Point3DF *origin, SimCoeff *coeffs, float *ax, float *weight, float *voldata, 
                              int cudev_x, int cudev_y, int coordz_offset, int angIdxStart);
__global__ void CuAtA_ADMM2_Z(Point3DF *origin, SimCoeff *coeffs, float *ax, float *weight, 
                    float *atax, int cudev_x, int cudev_y, int coordz_offset, int angIdxStart);
__global__ void CuAtA_ADMM3_Z(float *x_0, float *vol_data, float gamma, int volsize);
void CuATaGammaI_ADMM_Z(Point3DF *origin, SimCoeff *coeffs, float *a_x, float *w, float *x0, int cudev_x, 
                      int cudev_y, int volsize, float *voldata, float gamma, dim3 dim_1grid, dim3 dim_3grid, 
					  dim3 dim_block, int coordz_offset, int angIdxStart);

void CuApplycg_ADMM_Z(CuTaskDataZ &cudevice, float *voldata, float *x0, float *htb, int numberIteration, 
                    float gamma, size_t volsize, dim3 dim_1grid, dim3 dim_3grid, dim3 dim_block);

__global__ void CuSoft_ADMM_Z(float *u_k, float *d_k, float soft, int cudev_x, int volsize);

#endif