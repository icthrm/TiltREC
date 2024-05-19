#include "basicz.cuh"

bool CuMallocBPTTaskDataZ(CuTaskDataZ &cudev, int nproj, int x, int y, int steplength)
{
	cudaMalloc((void **)&cudev.origin, sizeof(Point3DF));
	cudaMalloc((void **)&cudev.coeffs, sizeof(SimCoeff) * nproj);
	cudev.x = x;
	cudev.z = steplength;
	cudev.y = y;
	cudev.nproj = nproj;
	return true;
}

bool CuMallocSIRTTaskDataZ(CuTaskDataZ &cudev, int nproj, int x, int y, int thickness, int batchSize)
{
	cudaMalloc((void **)&cudev.origin, sizeof(Point3DF));
	cudaMalloc((void **)&cudev.coeffs, sizeof(SimCoeff) * nproj);
	cudaMalloc((void **)&cudev.s, sizeof(float) * x * y * batchSize);
	cudaMalloc((void **)&cudev.c, sizeof(float) * x * y * batchSize);
	CUERR
	cudev.x = x;
	cudev.y = y;
	cudev.z = thickness;
	cudev.nproj = nproj;
	return true;
}

void CuFreeTaskDataZ(CuTaskDataZ &cudev)
{
	cudaFree(cudev.coeffs);
	cudaFree(cudev.slice);
	cudaFree(cudev.projs);
	cudaFree(cudev.origin);
	cudaFree(cudev.s);
	cudaFree(cudev.c);
}

__device__ void CuValCoefZ(const Point3DF &origin, const Point3D &coord, const SimCoeff &coeff, Weight *wt)
{
	float x, y;

	float X, Y, Z, n[2];
	X = coord.x - origin.x;
	Y = coord.y - origin.y;
	Z = coord.z - origin.z;

	n[0] = coeff.a[0] + coeff.a[1] * X + coeff.a[2] * Y + coeff.a[3] * Z;
	n[1] = coeff.b[0] + coeff.b[1] * X + coeff.b[2] * Y + coeff.b[3] * Z;

	x = n[0] + origin.x;
	y = n[1] + origin.y;

	wt->x_min = floor(x);
	wt->y_min = floor(y);

	wt->x_min_del = x - wt->x_min;
	wt->y_min_del = y - wt->y_min;
}

__device__ void CuBilinearValueZ(const float *data, int width, int height, const Weight &wt, float *val, float *vwt)
{
	size_t n;
	float c;
	if (wt.x_min >= 0 && wt.x_min < width && wt.y_min >= 0 && wt.y_min < height)
	{ //(x_min, y_min)
		n = wt.x_min + wt.y_min * width;
		c = (1 - wt.x_min_del) * (1 - wt.y_min_del);
		*val += c * data[n];
		*vwt += c;
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < width && wt.y_min >= 0 && wt.y_min < height)
	{ //(x_min+1, y_min)
		n = wt.x_min + 1 + wt.y_min * width;
		c = wt.x_min_del * (1 - wt.y_min_del);
		*val += c * data[n];
		*vwt += c;
	}
	if (wt.x_min >= 0 && wt.x_min < width && (wt.y_min + 1) >= 0 && (wt.y_min + 1) < height)
	{ //(x_min, y_min+1)
		n = wt.x_min + (wt.y_min + 1) * width;
		c = (1 - wt.x_min_del) * wt.y_min_del;
		*val += c * data[n];
		*vwt += c;
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < width && (wt.y_min + 1) >= 0 && (wt.y_min + 1) < height)
	{ //(x_min+1, y_min+1)
		n = wt.x_min + 1 + (wt.y_min + 1) * width;
		c = wt.x_min_del * wt.y_min_del;
		*val += c * data[n];
		*vwt += c;
	}
}

__global__ void CuCalcProjectionDiffKernelZ(float *projs, float *reproj_val, float *reproj_wt, int x, int y)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= x * y)
	{
		return;
	}
	size_t offset = (size_t)blockIdx.y * x * y + idx;
	float diff = 0;
	if (reproj_wt[offset])
	{
		diff = reproj_val[offset] / reproj_wt[offset];
	}
	reproj_val[offset] = projs[offset] - diff;
}

__global__ void CuCalcReProjectionKernelZ(float *reproj_val, float *reproj_wt, int x, int y)
{
	size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= x * y)
	{
		return;
	}
	size_t offset = (size_t)blockIdx.y * x * y + idx;
	double local_reproj_wt = reproj_wt[offset];
	double local_reproj_val = reproj_val[offset];
	if (local_reproj_wt)
	{
		local_reproj_val /= local_reproj_wt;
	}
	reproj_val[offset] = local_reproj_val;
}

__global__ void CuUpdateVolumeByWeightsKernelZ(float *slice, float *valvol, float *wtvol, float gamma, size_t maxN)
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

__global__ void CuReprojectKernelZ(const Point3DF *origin, SimCoeff *coeffs, float *slice,
								   float *reproj_val, float *reproj_wt, int x, int y, int coordz_offset, int angIdxStart)
{
	size_t xyId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

	if (xyId >= x * y)
	{
		return;
	}

	Point3D coord;
	coord.z = coordz_offset + blockIdx.y; 
	coord.y = xyId / x;					 
	coord.x = xyId - coord.y * x;		 
	int angidx = angIdxStart + blockIdx.z; 

	Weight wt;
	CuValCoefZ(*origin, coord, coeffs[angidx], &wt);

	float volval = slice[(size_t)x * y * blockIdx.y + xyId];
	float *rval = reproj_val + blockIdx.z * x * y;
	float *rwt = reproj_wt + blockIdx.z * x * y;

	size_t n;
	float c;
	if (wt.x_min >= 0 && wt.x_min < x && wt.y_min >= 0 && wt.y_min < y)
	{								 //(x_min, y_min)
		n = wt.x_min + wt.y_min * x; // index in reproj
		c = (1 - wt.x_min_del) * (1 - wt.y_min_del);
		atomicAdd(&rval[n], c * (volval));
		atomicAdd(&rwt[n], c);
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < x && wt.y_min >= 0 && wt.y_min < y)
	{									 //(x_min+1, y_min)
		n = wt.x_min + 1 + wt.y_min * x; // index in reproj
		c = wt.x_min_del * (1 - wt.y_min_del);
		atomicAdd(&rval[n], c * (volval));
		atomicAdd(&rwt[n], c);
	}
	if (wt.x_min >= 0 && wt.x_min < x && (wt.y_min + 1) >= 0 && (wt.y_min + 1) < y)
	{									   //(x_min, y_min+1)
		n = wt.x_min + (wt.y_min + 1) * x; // index in reproj
		c = (1 - wt.x_min_del) * wt.y_min_del;
		atomicAdd(&rval[n], c * (volval));
		atomicAdd(&rwt[n], c);
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < x && (wt.y_min + 1) >= 0 && (wt.y_min + 1) < y)
	{											 //(x_min+1, y_min+1)
		n = (wt.x_min + 1) + (wt.y_min + 1) * x; // index in reproj
		c = wt.x_min_del * wt.y_min_del;
		atomicAdd(&rval[n], c * (volval));
		atomicAdd(&rwt[n], c);
	}
}

__global__ void CuBackProjKernelZ(Point3DF *origin, SimCoeff *coeffs, float *slice, float *projs, int x, int y, int coordz_offset)
{
	size_t xyId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

	if (xyId < x * y)
	{
		Point3D coord;
		coord.z = coordz_offset + blockIdx.y; 
		coord.y = xyId / x;			 
		coord.x = xyId - coord.y * x; 
		int angidx = blockIdx.z;	  

		Weight wt;
		CuValCoefZ(*origin, coord, coeffs[angidx], &wt);

		float *proj = projs + angidx * x * y;
		float s = 0;
		float c = 0;
		CuBilinearValueZ(proj, x, y, wt, &s, &c);

		if (c)
		{
			size_t validx = (size_t)x * y * blockIdx.y + xyId;
			atomicAdd(&slice[validx], s / c);
		}
	}
}

__global__ void CuBackProjWeightAndValueKernelZ(Point3DF *origin, SimCoeff *coeffs, float *valvol, float *wtvol, float *diffs, int x, int y, int coordz_offset, int angIdxStart)
{
	size_t xyId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

	if (xyId >= x * y)
	{
		return;
	}

	Point3D coord;
	coord.z = coordz_offset + blockIdx.y; 
	coord.y = xyId / x;					   
	coord.x = xyId - coord.y * x;		 
	int angidx = angIdxStart + blockIdx.z; 

	Weight wt;
	CuValCoefZ(*origin, coord, coeffs[angidx], &wt);

	float *diff = diffs + (size_t)blockIdx.z * x * y;
	float s = 0;
	float c = 0;
	CuBilinearValueZ(diff, x, y, wt, &s, &c);

	size_t validx = (size_t)x * y * blockIdx.y + xyId;
	atomicAdd(&valvol[validx], s);
	atomicAdd(&wtvol[validx], c);
}

__global__ void CuUpdateVolumeByProjDiffKernelZ(Point3DF *origin, SimCoeff *coeffs, float *vol, float *diffs,
												float gamma, int x, int y, int coordz_offset, int angIdxStart)
{
	size_t xyId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

	if (xyId >= x * y)
	{
		return;
	}

	Point3D coord;
	coord.z = coordz_offset + blockIdx.y; 
	coord.y = xyId / x;					
	coord.x = xyId - coord.y * x;		   
	int angidx = angIdxStart + blockIdx.z; 

	Weight wt;
	CuValCoefZ(*origin, coord, coeffs[angidx], &wt);

	float *diff = diffs + blockIdx.z * x * y;
	float s = 0;
	float c = 0;
	CuBilinearValueZ(diff, x, y, wt, &s, &c);

	size_t validx = (size_t)x * y * blockIdx.y + xyId;
	if (c)
		atomicAdd(&vol[validx], (s / c) * gamma);
}

bool CuMallocADMMTaskDataZ(CuTaskDataZ &cudev, int nproj, int x, int y, int thickness, int batchSize)
{
	cudaMalloc((void **)&cudev.origin, sizeof(Point3DF));
	cudaMalloc((void **)&cudev.coeffs, sizeof(SimCoeff) * nproj);
	cudaMalloc((void **)&cudev.s, sizeof(float) * x * y);
	cudaMalloc((void **)&cudev.c, sizeof(float) * x * y);
	CUERR
	cudev.x = x;
	cudev.y = y;
	cudev.z = thickness;
	cudev.nproj = nproj;
	return true;
}

__device__ void CuAtbKernel_ADMM_Z(Weight &wt, int cudev_x, int cudev_y, 
                                    float *projsdata, float *voldata)
{
	int n;
	float w;
	
	if (wt.x_min >= 0 && wt.x_min < cudev_x && wt.y_min >= 0 &&wt.y_min < cudev_y)
	{																	
		n = wt.x_min + wt.y_min * cudev_x;	 //(x_min, y_min)	 
		w = (1 - wt.x_min_del) * (1 - wt.y_min_del);
		*voldata += w * projsdata[n];		
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < cudev_x &&wt.y_min >= 0 && wt.y_min < cudev_y)
	{											 
		n = wt.x_min + 1 + wt.y_min * cudev_x; //(x_min+1, y_min)
		w = wt.x_min_del * (1 - wt.y_min_del);
		*voldata += w * projsdata[n];
	}
	if (wt.x_min >= 0 && wt.x_min < cudev_x && (wt.y_min + 1) >= 0 &&(wt.y_min + 1) < cudev_y)
	{											   
		n = wt.x_min + (wt.y_min + 1) * cudev_x; //(x_min, y_min+1)
		w = (1 - wt.x_min_del) * wt.y_min_del;
		*voldata += w * projsdata[n];		
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < cudev_x &&(wt.y_min + 1) >= 0 &&(wt.y_min + 1) < cudev_y)
	{													 
		n = (wt.x_min + 1) + (wt.y_min + 1) * cudev_x; //(x_min+1, y_min+1)
		w = wt.x_min_del * wt.y_min_del;
		*voldata += w * projsdata[n];	
	}
}

__global__ void CuAtb_ADMM_Z(Point3DF *origin, SimCoeff *coeffs, float *htb, int cudev_x, int cudev_y, 
                            float *projsdata, int coordz_offset, int angIdxStart)
{
	size_t xyId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
	if (xyId >= cudev_x * cudev_y)
	{
		return;
	}
	size_t validx = (size_t)cudev_x * cudev_y * blockIdx.y + xyId;

	Point3D coord;
	coord.z = coordz_offset + blockIdx.y; 
	coord.y = xyId / cudev_x;					 
	coord.x = xyId - coord.y * cudev_x;		 
	int angidx = angIdxStart + blockIdx.z; 

	Weight wt;
	CuValCoefZ(*origin, coord, coeffs[angidx], &wt);

	float data = 0;
	CuAtbKernel_ADMM_Z(wt, cudev_x, cudev_y, projsdata, &data);

	atomicAdd(&htb[validx], data);
}

__global__ void CuATbGammaIt_ADMM_Z(float *htb, float *u_k, float *d_k, size_t volsize, float gamma)
{
	size_t xyId = blockIdx.x * blockDim.x + threadIdx.x;
	if (xyId >= volsize)
	{
		return;
	}

	size_t validx = xyId;
	htb[validx] +=  gamma * (u_k[validx] - d_k[validx]);
}

__global__ void CuAtA_ADMM1_Z(Point3DF *origin, SimCoeff *coeffs, float *ax, float *weight, float *voldata, 
                              int cudev_x, int cudev_y, int coordz_offset, int angIdxStart)
{
	size_t xyId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

	if (xyId >= cudev_x * cudev_y)
	{
		return;
	}

	Point3D coord;
	coord.z = coordz_offset + blockIdx.y; 
	coord.y = xyId / cudev_x;					 
	coord.x = xyId - coord.y * cudev_x;		 
	int angidx = angIdxStart + blockIdx.z; 

	Weight wt;
	CuValCoefZ(*origin, coord, coeffs[angidx], &wt);

	int n;
	float w;

	size_t volid = (size_t)cudev_x * cudev_y * blockIdx.y + xyId;

	if (wt.x_min >= 0 && wt.x_min < cudev_x && wt.y_min >= 0 &&wt.y_min < cudev_y)
	{	//(x_min, y_min)
		n = wt.x_min + wt.y_min * cudev_x; 
		w = (1 - wt.x_min_del) * (1 - wt.y_min_del);
		atomicAdd(&ax[n], w * voldata[volid]);
		atomicAdd(&weight[n], w);
		}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < cudev_x && wt.y_min >= 0 && wt.y_min < cudev_y)
	{	//(x_min+1, y_min)
		n = wt.x_min + 1 + wt.y_min * cudev_x; 
		w = wt.x_min_del * (1 - wt.y_min_del);
		atomicAdd(&ax[n], w * voldata[volid]);
		atomicAdd(&weight[n], w);
	}
	if (wt.x_min >= 0 && wt.x_min < cudev_x && (wt.y_min + 1) >= 0 &&(wt.y_min + 1) < cudev_y)
	{	//(x_min, y_min+1)
		n = wt.x_min + (wt.y_min + 1) * cudev_x; 
		w = (1 - wt.x_min_del) * wt.y_min_del;
		atomicAdd(&ax[n], w * voldata[volid]);
		atomicAdd(&weight[n], w);
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < cudev_x && (wt.y_min + 1) >= 0 && (wt.y_min + 1) < cudev_y)
	{   //(x_min+1, y_min+1)
		n = (wt.x_min + 1) + (wt.y_min + 1) * cudev_x; 
		w = wt.x_min_del * wt.y_min_del;
		atomicAdd(&ax[n], w * voldata[volid]);
		atomicAdd(&weight[n], w * wt.y_min_del);
	}
}

__global__ void CuAtA_ADMM2_Z(Point3DF *origin, SimCoeff *coeffs, float *ax, float *weight, 
                    float *atax, int cudev_x, int cudev_y, int coordz_offset, int angIdxStart)
{
	size_t xyId = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

	if (xyId >= cudev_x * cudev_y)
	{
		return;
	}

	Point3D coord;
	coord.z = coordz_offset + blockIdx.y; 
	coord.y = xyId / cudev_x;					 
	coord.x = xyId - coord.y * cudev_x;		 
	int angidx = angIdxStart + blockIdx.z; 

	Weight wt;
	CuValCoefZ(*origin, coord, coeffs[angidx], &wt);

	int n;
	float w;

	size_t volid = (size_t)cudev_x * cudev_y * blockIdx.y + xyId;

	if (wt.x_min >= 0 && wt.x_min < cudev_x 
	 	&& wt.y_min >= 0 &&wt.y_min < cudev_y)
	{	//(x_min, y_min)
		n = wt.x_min + wt.y_min * cudev_x;
		w = (1 - wt.x_min_del) * (1 - wt.y_min_del) ;
		if (fabs(weight[n]) > 10e-6)
		{
			atomicAdd(&atax[volid], w * ax[n] / weight[n]);
		}
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < cudev_x 
	 	&&wt.y_min >= 0 && wt.y_min < cudev_y)
	{	//(x_min+1, y_min)
		n = wt.x_min + 1 + wt.y_min * cudev_x;
		w = wt.x_min_del * (1 - wt.y_min_del);
		if (fabs(weight[n]) > 10e-6)
		{
			atomicAdd(&atax[volid], w * ax[n] / weight[n]);
		}
	}
	if (wt.x_min >= 0 && wt.x_min < cudev_x
	    &&(wt.y_min + 1) >= 0 &&(wt.y_min + 1) < cudev_y)
	{	//(x_min, y_min+1)
		n = wt.x_min + (wt.y_min + 1) * cudev_x;
		w = (1 - wt.x_min_del) * wt.y_min_del;
		if (fabs(weight[n]) > 10e-6)
		{
			atomicAdd(&atax[volid], w * ax[n] / weight[n]);
		}
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < cudev_x 
	    &&(wt.y_min + 1) >= 0 &&(wt.y_min + 1) < cudev_y)
	{   //(x_min+1, y_min+1)
		n = (wt.x_min + 1) + (wt.y_min + 1) * cudev_x; 
		w = wt.x_min_del * wt.y_min_del;
		if (fabs(weight[n]) > 10e-6)
		{
			atomicAdd(&atax[volid], w * ax[n] / weight[n]);
		}
	}
}

__global__ void CuAtA_ADMM3_Z(float *x_0, float *vol_data, float gamma, int volsize)
{
	size_t xyId = blockIdx.x * blockDim.x + threadIdx.x;
	if (xyId >= volsize)
	{
		return;
	}
	size_t volidx = xyId;
	
	x_0[volidx] += gamma * vol_data[volidx];
}

void CuATaGammaI_ADMM_Z(Point3DF *origin, SimCoeff *coeffs, float *a_x, float *w, float *x0, int cudev_x, 
                      int cudev_y, int volsize, float *voldata, float gamma, dim3 dim_1grid, dim3 dim_3grid, 
					  dim3 dim_block, int coordz_offset, int angIdxStart)
{
	cudaMemset(a_x, 0, sizeof(float) * cudev_x * cudev_y);
	cudaMemset(w, 0, sizeof(float) * cudev_x * cudev_y);
	CuAtA_ADMM1_Z<<<dim_3grid, dim_block>>>(origin, coeffs, a_x, w, voldata, cudev_x, 
	                                        cudev_y, coordz_offset, angIdxStart); 
	cudaDeviceSynchronize();
	CuAtA_ADMM2_Z<<<dim_3grid, dim_block>>>(origin, coeffs, a_x, w, x0, cudev_x, 
	                                        cudev_y, coordz_offset, angIdxStart);
	cudaDeviceSynchronize();
	CuAtA_ADMM3_Z<<<dim_1grid, dim_block>>>(x0, voldata, gamma, volsize); 
	cudaDeviceSynchronize();
}

__global__ void CuCG_r(float *r_0, float *x_0, float *atb, int volsize)
{
	size_t xyId = blockIdx.x * blockDim.x + threadIdx.x;
	if (xyId >= volsize)
	{
		return;
	}
	size_t validx = xyId;
	r_0[validx] = atb[validx] - x_0[validx];
}

__global__ void CuCG_p(float *r_0, float *p_0, int volsize)
{
	size_t xyId = blockIdx.x * blockDim.x + threadIdx.x;
	if (xyId >= volsize)
	{
		return;
	}
	size_t validx = xyId;
	p_0[validx] = r_0[validx];
}

__global__ void CuCG_d0(float *r_0, float *d_0, int volsize)
{
	size_t xyId = blockIdx.x * blockDim.x + threadIdx.x;
	if (xyId >= volsize)
	{
		return;
	}
	size_t validx = xyId;
	atomicAdd(d_0, r_0[validx] * r_0[validx]); 
}

__global__ void CuCG_d2(float *p_0, float *h_0, float *d_2, int volsize)
{
	size_t xyId = blockIdx.x * blockDim.x + threadIdx.x;
	if (xyId >= volsize)
	{
		return;
	}
	size_t validx = xyId;
	atomicAdd(d_2, h_0[validx] * p_0[validx]);
}

__global__ void CuCG_voldata(float *voldata, float *p_0, float alpha, int volsize)
{
	size_t xyId = blockIdx.x * blockDim.x + threadIdx.x;
	if (xyId >= volsize)
	{
		return;
	}
	size_t validx = xyId;
	voldata[validx] += alpha * p_0[validx];
}

__global__ void CuCG_r_(float *r_0, float *a_x, float *atb_lt, int volsize)
{
	size_t xyId = blockIdx.x * blockDim.x + threadIdx.x;
	if (xyId >= volsize)
	{
		return;
	}
	size_t validx = xyId;
	r_0[validx] = atb_lt[validx] - a_x[validx];
}

__global__ void CuCG_d1(float *d_1, float *r_0, float *h_0, int volsize)
{
	size_t xyId = blockIdx.x * blockDim.x + threadIdx.x;
	if (xyId >= volsize)
	{
		return;
	}
	size_t validx = xyId;
	atomicAdd(d_1, r_0[validx] * h_0[validx]);
}

__global__ void CuCG_p_(float beta, float *p_0, float *r_0, int volsize)
{
	size_t xyId = blockIdx.x * blockDim.x + threadIdx.x;
	if (xyId >= volsize)
	{
		return;
	}
	size_t validx = xyId;
	p_0[validx] = beta * (p_0[validx]) + r_0[validx];
}

void CuApplycg_ADMM_Z(CuTaskDataZ &cudevice, float *voldata, float *x0, float *htb, int numberIteration, 
                    float gamma, size_t volsize, dim3 dim_1grid, dim3 dim_3grid, dim3 dim_block)
{
	float *r0, *p0, *Ax, *h0;
	cudaMallocManaged((void **)&r0, sizeof(float) * volsize);
	cudaMallocManaged((void **)&p0, sizeof(float) * volsize);
	cudaMallocManaged((void **)&Ax, sizeof(float) * volsize);
	cudaMallocManaged((void **)&h0, sizeof(float) * volsize);
	CUERR
	cudaMemset(r0, 0, sizeof(float) * volsize);
	cudaMemset(p0, 0, sizeof(float) * volsize);
	cudaMemset(Ax, 0, sizeof(float) * volsize);
	CUERR

	float *d0, *d1, *d2;
	cudaMallocManaged(&d0, sizeof(float));
	cudaMallocManaged(&d1, sizeof(float));
	cudaMallocManaged(&d2, sizeof(float));
	CUERR
	float beta = 0;
	float d0_value = 0;
	float d1_value = 0;
	float d2_value = 0;

	CuCG_r<<<dim_1grid, dim_block>>>(r0, x0, htb, volsize);
	cudaDeviceSynchronize();

	CuCG_p<<<dim_1grid, dim_block>>>(r0, p0, volsize);
	cudaDeviceSynchronize();

	for (int i = 0; i < numberIteration; i++)
	{
		cudaMemset(h0, 0, sizeof(float) * volsize);
		CuATaGammaI_ADMM_Z(cudevice.origin, cudevice.coeffs, cudevice.s, cudevice.c, h0, cudevice.x, cudevice.y, 
                           volsize, p0, gamma, dim_1grid, dim_3grid, dim_block, 0, 0);
		float alpha = 0;

		cudaMemset(d0, 0, sizeof(float));
		cudaMemset(d1, 0, sizeof(float));
		cudaMemset(d2, 0, sizeof(float));
		CUERR
		CuCG_d0<<<dim_1grid, dim_block>>>(r0, d0, volsize);
		cudaDeviceSynchronize();
		cudaMemcpy(&d0_value, d0, sizeof(float), cudaMemcpyDeviceToHost);
		CUERR

		CuCG_d2<<<dim_1grid, dim_block>>>(p0, h0, d2, volsize); 
		cudaDeviceSynchronize();
		cudaMemcpy(&d2_value, d2, sizeof(float), cudaMemcpyDeviceToHost);
		CUERR
		if (fabs(d2_value)> 10e-6)
		{
			alpha = d0_value / d2_value; 
		}
		else
		{
			alpha = 0;
		}
		CuCG_voldata<<<dim_1grid, dim_block>>>(voldata, p0, alpha, volsize); 
		cudaDeviceSynchronize();

		CuATaGammaI_ADMM_Z(cudevice.origin, cudevice.coeffs, cudevice.s, cudevice.c, Ax, cudevice.x, cudevice.y, 
                           volsize, voldata, gamma, dim_1grid, dim_3grid, dim_block, 0, 0);
		cudaDeviceSynchronize();
		CuCG_r_<<<dim_1grid, dim_block>>>(r0, Ax, htb, volsize);
		cudaDeviceSynchronize();
		CuCG_d1<<<dim_1grid, dim_block>>>(d1, r0, h0, volsize);
		cudaDeviceSynchronize();
		
		cudaMemcpy(&d1_value, d1, sizeof(float), cudaMemcpyDeviceToHost);
		CUERR

		if (fabs(d2_value) > 10e-6)
		{
			beta = d1_value / d2_value;
		}
		else
		{
			beta = 0;
		}
		CuCG_p_<<<dim_1grid, dim_block>>>(beta, p0, r0, volsize);
		cudaDeviceSynchronize();
	}
	cudaFree(r0);
	cudaFree(p0);
	cudaFree(Ax);
	cudaFree(h0);
	cudaFree(d0);
	cudaFree(d1);
	cudaFree(d2);
}

__global__ void CuSoft_ADMM_Z(float *u_k, float *d_k, float soft, int cudev_x, int volsize)
{
	size_t xyId = blockIdx.x * blockDim.x + threadIdx.x;
	if (xyId >= volsize)
	{
		return;
	}

	size_t validx = xyId;

	float u;
	if (u_k[validx] + d_k[validx] < -soft)
	{
		u = u_k[validx] + d_k[validx] + soft;
		d_k[validx] = d_k[validx] + u_k[validx] - u;
		u_k[validx] = u;
	}
	else if (u_k[validx] + d_k[validx] > soft)
	{
		u = u_k[validx] + d_k[validx] - soft;
		d_k[validx] = d_k[validx] + u_k[validx] - u;
		u_k[validx] = u;
	}
	else
	{
		u = 0;
		d_k[validx] = d_k[validx] + u_k[validx] - u;
		u_k[validx] = u;
	}
}

