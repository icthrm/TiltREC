#include "basic.cuh"

__device__ __constant__ struct SimCoeff c_coeff[180];
__device__ __constant__ struct Point3DF c_origin;
size_t GetCuTaskDataSizePerPartition(int nproj, int x, int z)
{
	return sizeof(Point3DF) + sizeof(SimCoeff) * nproj + sizeof(float) * x * z * 4 + sizeof(float) * x * nproj;
}

bool CuMallocBPTTaskData(CuTaskDataY &cudev, int nproj, int x, int z, int steplength)
{
	CHECK_CUDA(cudaMalloc((void **)&cudev.slice, sizeof(float) * x * z * steplength))
	CHECK_CUDA(cudaMalloc((void **)&cudev.projs, sizeof(float) * x * nproj * steplength))
	cudev.x = x;
	cudev.z = z;
	cudev.y = steplength;
	cudev.nproj = nproj;
	return true;
}

bool CuMallocSIRTTaskData(CuTaskDataY &cudev, int nproj, int x, int z, int steplength)
{

	CHECK_CUDA(cudaMalloc((void **)&cudev.slice, sizeof(float) * x * z * steplength))
	CHECK_CUDA(cudaMalloc((void **)&cudev.projs, sizeof(float) * x * nproj * steplength))
	CHECK_CUDA(cudaMalloc((void **)&cudev.s, sizeof(float) * x * nproj * steplength))
	cudev.x = x;
	cudev.z = z;
	cudev.y = steplength;
	cudev.nproj = nproj;

	return true;
}
bool CuMallocSARTTaskData(CuTaskDataY &cudev, int nproj, int x, int z, int steplength)
{

	CHECK_CUDA(cudaMalloc((void **)&cudev.slice, sizeof(float) * x * z * steplength))
	CHECK_CUDA(cudaMalloc((void **)&cudev.projs, sizeof(float) * x * nproj * steplength))
	CHECK_CUDA(cudaMalloc((void **)&cudev.s, sizeof(float) * x * steplength))
	cudev.x = x;
	cudev.z = z;
	cudev.y = steplength;
	cudev.nproj = nproj;

	return true;
}

void CuFreeTaskData(CuTaskDataY &cudev){
	CHECK_CUDA(cudaFree(cudev.slice))
	CHECK_CUDA(cudaFree(cudev.projs))
	}

void CuFreeTaskDataADMM(CuTaskDataY &cudev){
	CHECK_CUDA(cudaFree(cudev.slice))
	CHECK_CUDA(cudaFree(cudev.projs))
	CHECK_CUDA(cudaFree(cudev.ax))
	CHECK_CUDA(cudaFree(cudev.w))
	}

__device__ void CuValCoef(const Point3D &coord, int angidx, float *x, float *y)
{

	float X, Y, Z;

	X = coord.x - c_origin.x;
	Y = coord.y - c_origin.y;
	Z = coord.z - c_origin.z;

	*x = c_coeff[angidx].a[0] + c_coeff[angidx].a[1] * X + c_coeff[angidx].a[2] * Y + c_coeff[angidx].a[3] * Z + c_origin.x;
	*y = c_coeff[angidx].b[0] + c_coeff[angidx].b[1] * X + c_coeff[angidx].b[2] * Y + c_coeff[angidx].b[3] * Z + c_origin.y;
}

__device__ void CuBilinearValue(int width, int height, float X, float Y, float *vwt)
{
	float c;
	Weight wt;
	wt.x_min = floor(X); //
	wt.y_min = floor(Y);

	wt.x_min_del = X - wt.x_min;
	wt.y_min_del = Y - wt.y_min;
	if (wt.x_min >= 0 && wt.x_min < width && wt.y_min >= 0 && wt.y_min < height)
	{ //(x_min, y_min)
		c = (1 - wt.x_min_del) * (1 - wt.y_min_del);
		*vwt += c;
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < width && wt.y_min >= 0 && wt.y_min < height)
	{ //(x_min+1, y_min)
		c = wt.x_min_del * (1 - wt.y_min_del);
		*vwt += c;
	}
	if (wt.x_min >= 0 && wt.x_min < width && (wt.y_min + 1) >= 0 && (wt.y_min + 1) < height)
	{ //(x_min, y_min+1)
		c = (1 - wt.x_min_del) * wt.y_min_del;
		*vwt += c;
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < width && (wt.y_min + 1) >= 0 && (wt.y_min + 1) < height)
	{ //(x_min+1, y_min+1)
		c = wt.x_min_del * wt.y_min_del;
		*vwt += c;
	}
}
__device__ void CuBilinearValue(float *data, int width, int height, float X, float Y, float *val, float *vwt)
{
	int n;
	float c;
	Weight wt;
	wt.x_min = floor(X); //
	wt.y_min = floor(Y);

	wt.x_min_del = X - wt.x_min;
	wt.y_min_del = Y - wt.y_min;
	if (wt.x_min >= 0 && wt.x_min < width && wt.y_min >= 0 && wt.y_min < height)
	{ //(x_min, y_min)
		n = wt.x_min + wt.y_min * width;
		c = (1 - wt.x_min_del) * (1 - wt.y_min_del);
		*vwt += c;
		*val += c * data[n];
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < width && wt.y_min >= 0 && wt.y_min < height)
	{ //(x_min+1, y_min)
		n = wt.x_min + 1 + wt.y_min * width;
		c = wt.x_min_del * (1 - wt.y_min_del);
		*vwt += c;
		*val += c * data[n];
	}
	if (wt.x_min >= 0 && wt.x_min < width && (wt.y_min + 1) >= 0 && (wt.y_min + 1) < height)
	{ //(x_min, y_min+1)
		n = wt.x_min + (wt.y_min + 1) * width;
		c = (1 - wt.x_min_del) * wt.y_min_del;
		*vwt += c;
		*val += c * data[n];
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < width && (wt.y_min + 1) >= 0 && (wt.y_min + 1) < height)
	{ //(x_min+1, y_min+1)
		n = wt.x_min + 1 + (wt.y_min + 1) * width;
		c = wt.x_min_del * wt.y_min_del;
		*vwt += c;
		*val += c * data[n];
	}
}
__global__ void CuCalcProjectionDiffKernel(float *projs, float *reproj_val, float *reproj_wt, int x, int y)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= x * y)
	{
		return;
	}
	size_t offset = blockIdx.y * x * y + idx;

	float diff = 0;
	if (reproj_wt[offset])
	{
		diff = reproj_val[offset] / reproj_wt[offset];
	}
	reproj_val[offset] = projs[offset] - diff;
}
__global__ void CuCalcProjectionDiffKernel_SART(float *projs, float *reproj_val, float *reproj_wt, int x, int y, int projIdxStart)
{
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= x * y)
	{
		return;
	}
	size_t offset = projIdxStart * x * y + idx;

	float diff = 0;
	if (reproj_wt[offset] > 1e-7)
	{
		diff = reproj_val[idx] / reproj_wt[offset];
	}
	reproj_val[idx] = projs[idx] - diff;
}

__global__ void CuUpdateVolumeByWeightsKernel(float *slice, float *valvol, float *wtvol, float gamma, size_t maxN)
{
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (i < maxN)
	{
		float local_wtvol = wtvol[i];
		float local_newSlice;
		if (local_wtvol > 10e-6)
		{
			local_newSlice = valvol[i] / local_wtvol * gamma;
			slice[i] += local_newSlice;
		}
	}
}

__global__ void CuSelectKernel(float *slice, int n, int steplength, int left)
{
	size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

	if (i < n * steplength)
	{
		slice[i] = slice[n * left + i];
	}
}
__device__ void DeviceReprojectKernel(float X, float Y, float x, float y, float *rval, float volval)
{
	int n;
	float c;
	Weight wt;
	wt.x_min = floor(X); //
	wt.y_min = floor(Y);

	wt.x_min_del = X - wt.x_min;
	wt.y_min_del = Y - wt.y_min;
	if (wt.x_min >= 0 && wt.x_min < x && wt.y_min >= 0 && wt.y_min < y)
	{								 //(x_min, y_min)
		n = wt.x_min + wt.y_min * x; // index in reproj
		c = (1 - wt.x_min_del) * (1 - wt.y_min_del);
		atomicAdd(&rval[n], c * (volval));
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < x && wt.y_min >= 0 && wt.y_min < y)
	{									 //(x_min+1, y_min)
		n = wt.x_min + 1 + wt.y_min * x; // index in reproj
		c = wt.x_min_del * (1 - wt.y_min_del);
		atomicAdd(&rval[n], c * (volval));
	}

	if (wt.x_min >= 0 && wt.x_min < x && (wt.y_min + 1) >= 0 && (wt.y_min + 1) < y)
	{ //(x_min, y_min+1)
		n = wt.x_min + (wt.y_min + 1) * x;
		c = (1 - wt.x_min_del) * (wt.y_min_del);
		atomicAdd(&rval[n], c * (volval));
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < x && (wt.y_min + 1) >= 0 && (wt.y_min + 1) < y)
	{ //(x_min+1, y_min+1)
		n = (wt.x_min + 1) + (wt.y_min + 1) * x;
		c = wt.x_min_del * (wt.y_min_del);
		atomicAdd(&rval[n], c * (volval));
	}
}
__global__ void CuReprojectKernel(float *slice, float *reproj_val, int x, int z, int y)
{
	size_t xzId = blockIdx.x * blockDim.x + threadIdx.x;
	if (xzId >= x * z)
	{
		return;
	}

	Point3D coord;
	coord.z = xzId / x;
	coord.x = xzId - coord.z * x;
	coord.y = 0;
	int angidx = blockIdx.y;
	float X, Y;

	CuValCoef(coord, angidx, &X, &Y);

	float volval;
	float *rval = reproj_val + angidx * x * y;

	for ((coord.y) = 0; (coord.y) < y; (coord.y)++)
	{
		volval = slice[coord.x + x * z * coord.y + coord.z * x];
		DeviceReprojectKernel(X, Y, x, y, rval, volval);
		X += c_coeff[angidx].a[2];
		Y += c_coeff[angidx].b[2];
	}
}

__global__ void CuReprojectKernel_SART(float *slice, float *reproj_val, int x, int z, int y, int projIdxStart)
{
	size_t xzId = blockIdx.x * blockDim.x + threadIdx.x;
	if (xzId >= x * z)
	{
		return;
	}

	Point3D coord;
	coord.z = xzId / x;
	coord.x = xzId - coord.z * x;
	coord.y = 0;
	int angidx = projIdxStart;
	float X, Y;

	CuValCoef(coord, angidx, &X, &Y);

	float volval = slice[(size_t)x * z * coord.y + xzId];
	float *rval = reproj_val;

	for ((coord.y) = 0; (coord.y) < y; (coord.y)++)
	{
		volval = slice[coord.x + x * z * coord.y + coord.z * x];
		DeviceReprojectKernel(X, Y, x, y, rval, volval);
		X += c_coeff[angidx].a[2];
		Y += c_coeff[angidx].b[2];
	}
}

__global__ void CuBackProjWeightAndValueKernel(float *valvol, cudaTextureObject_t projtex, int volcor_x, int volcor_y, int nproj, int volcor_z)
{
	size_t xId = blockIdx.x * blockDim.x + threadIdx.x;

	if (xId >= volcor_x)
	{
		return;
	}

	int angidx = 0;
	float n1, n2;
	float s1 = 0;
	size_t validx = xId + blockIdx.y * volcor_x + volcor_x * volcor_z * blockIdx.z;

	for ((angidx) = 0; (angidx) < nproj; (angidx)++)
	{
		n1 = c_coeff[angidx].a[0] + c_coeff[angidx].a[1] * (xId - c_origin.x) + c_coeff[angidx].a[2] * (blockIdx.z - c_origin.y) + c_coeff[angidx].a[3] * (blockIdx.y - c_origin.z) + c_origin.x;
		n2 = c_coeff[angidx].b[0] + c_coeff[angidx].b[1] * (xId - c_origin.x) + c_coeff[angidx].b[2] * (blockIdx.z - c_origin.y) + c_coeff[angidx].b[3] * (blockIdx.y - c_origin.z) + c_origin.y;
		s1 += tex3D<float>(projtex, n1 + 0.5f, n2 + 0.5f, angidx + 0.5f);
	}

	valvol[validx] = s1;
}
__global__ void CuBackProjWeightAndValueKernel_SART(float *valvol, cudaTextureObject_t projtex, int volcor_x, int volcor_y, int volcor_z, int angidx)
{
	size_t xId = blockIdx.x * blockDim.x + threadIdx.x;

	if (xId >= volcor_x)
	{
		return;
	}

	float n1, n2;
	size_t validx = xId + blockIdx.y * volcor_x + volcor_x * volcor_z * blockIdx.z;

	n1 = c_coeff[angidx].a[0] + c_coeff[angidx].a[1] * (xId - c_origin.x) + c_coeff[angidx].a[2] * (blockIdx.z - c_origin.y) + c_coeff[angidx].a[3] * (blockIdx.y - c_origin.z) + c_origin.x;
	n2 = c_coeff[angidx].b[0] + c_coeff[angidx].b[1] * (xId - c_origin.x) + c_coeff[angidx].b[2] * (blockIdx.z - c_origin.y) + c_coeff[angidx].b[3] * (blockIdx.y - c_origin.z) + c_origin.y;
	valvol[validx] = tex2D<float>(projtex, n1 + 0.5f, n2 + 0.5f);
}
__global__ void CuVolWeightKernel(float *wtvol, int volcor_x, int volcor_y, int volcor_z, int nproj)
{

	size_t xzId = blockIdx.x * blockDim.x + threadIdx.x;
	if (xzId >= volcor_x)
	{
		return;
	}

	int angidx = 0;
	float n1, n2;
	size_t validx = xzId + blockIdx.y * volcor_x + volcor_x * volcor_z * blockIdx.z;
	float c = 0;

	for ((angidx) = 0; (angidx) < nproj; (angidx)++)
	{
		n1 = c_coeff[angidx].a[0] + c_coeff[angidx].a[1] * (xzId - c_origin.x) + c_coeff[angidx].a[2] * (blockIdx.z - c_origin.y) + c_coeff[angidx].a[3] * (blockIdx.y - c_origin.z) + c_origin.x;
		n2 = c_coeff[angidx].b[0] + c_coeff[angidx].b[1] * (xzId - c_origin.x) + c_coeff[angidx].b[2] * (blockIdx.z - c_origin.y) + c_coeff[angidx].b[3] * (blockIdx.y - c_origin.z) + c_origin.y;
		CuBilinearValue(volcor_x, volcor_y, n1, n2, &c);
	}

	wtvol[validx] = c;
}
__global__ void CuReWeightKernel(float *reproj_wt, int volcor_x, int volcor_y, int volcor_z)
{
	size_t xzId = blockIdx.x * blockDim.x + threadIdx.x;
	if (xzId >= volcor_x * volcor_z)
	{
		return;
	}

	Point3D coord;
	coord.z = xzId / volcor_x;
	coord.x = xzId - coord.z * volcor_x;
	coord.y = 0;
	int angidx = blockIdx.y;
	float X, Y;

	CuValCoef(coord, angidx, &X, &Y);

	float *rwt = reproj_wt + angidx * volcor_x * volcor_y;

	for ((coord.y) = 0; (coord.y) < volcor_y; (coord.y)++)
	{
		DeviceReprojectKernel(X, Y, volcor_x, volcor_y, rwt, 1);
		X += c_coeff[angidx].a[2];
		Y += c_coeff[angidx].b[2];
	}
}
__global__ void CuBackProjKernel(float *slice, float *projs, int x, int z, int y)
{
	size_t xzId = blockIdx.x * blockDim.x + threadIdx.x;

	if (xzId >= x * z)
	{
		return;
	}

	Point3D coord;
	coord.z = xzId / x;
	coord.x = xzId - coord.z * x;
	coord.y = blockIdx.y;

	int angidx = blockIdx.z;

	float n1, n2;
	CuValCoef(coord, angidx, &n1, &n2);//对

	float *proj = projs + angidx * x * y;

	float s = 0;
	float c = 0;

	CuBilinearValue(proj, x, y, n1, n2, &s, &c);

	if (c > 1e-5)
	{
		size_t validx = (size_t)x * z * coord.y + xzId;//对

		atomicAdd(&slice[validx], s / c);
	}
}

__global__ void CuUpdateVolumeByProjDiffKernel_SART(float *proj, float *vol, int volcor_x, int volcor_y, int volcor_z, int angidx, float gamma)
{
	size_t xzId = blockIdx.x * blockDim.x + threadIdx.x;

	if (xzId >= volcor_x)
	{
		return;
	}

	float n1, n2;
	size_t validx = xzId + blockIdx.y * volcor_x + volcor_x * volcor_z * blockIdx.z;
	float s = 0;
	float c = 0;
	n1 = c_coeff[angidx].a[0] + c_coeff[angidx].a[1] * (xzId - c_origin.x) + c_coeff[angidx].a[2] * (blockIdx.z - c_origin.y) + c_coeff[angidx].a[3] * (blockIdx.y - c_origin.z) + c_origin.x;
	n2 = c_coeff[angidx].b[0] + c_coeff[angidx].b[1] * (xzId - c_origin.x) + c_coeff[angidx].b[2] * (blockIdx.z - c_origin.y) + c_coeff[angidx].b[3] * (blockIdx.y - c_origin.z) + c_origin.y;

	CuBilinearValue(proj, volcor_x, volcor_y, n1, n2, &s, &c);
	if (c > 10e-7)
	{
		vol[validx] += (s / c) * gamma;
	}
}
