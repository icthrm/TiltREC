#include "recon_y.cuh"
#include "basic.cu"
extern cudaDeviceProp deviceProps;

void ReadSliceBlock(Volume &vol, Bodyinfo &volinfo, MrcStackM &projs, float *tmp, int y, int thickness, int gridYMax, float pitch_angle)
{
  volinfo.Y_add_right = ceil(fabsf(tan(D2R(pitch_angle))) * thickness);
  volinfo.Y_add_left = ceil(fabsf(tan(D2R(pitch_angle))) * thickness);
  volinfo.slice_steplength = volinfo.Y_add_left + volinfo.steplength + volinfo.Y_add_right;
  //volinfo.steplength = max(64, gridYMax);
  if (y < volinfo.Y_add_left)
  {
    volinfo.Y_add_left = y;
    volinfo.slice_steplength = volinfo.Y_add_left + volinfo.steplength + volinfo.Y_add_right;
  }
  else if (y + volinfo.steplength + volinfo.Y_add_right > vol.height && y + volinfo.steplength <= vol.height)
  {
    volinfo.Y_add_right = vol.height - y - volinfo.steplength;
    volinfo.slice_steplength = volinfo.Y_add_left + volinfo.steplength + volinfo.Y_add_right;
  }
  else if (y + volinfo.steplength > vol.height)
  {
    volinfo.slice_steplength = volinfo.Y_add_left + vol.height - y;
    volinfo.Y_add_right = 0;
    volinfo.steplength = vol.height - y;
  }

  projs.ReadBlock(y - volinfo.Y_add_left, y + volinfo.steplength + volinfo.Y_add_right, 'y', tmp);
  volinfo.volsize = (size_t)projs.X() * thickness * volinfo.slice_steplength;
  volinfo.projsize = (size_t)projs.X() * volinfo.slice_steplength * projs.Z();
}

void ReadSliceBlock(Volume &vol, Bodyinfo &volinfo, MrcStackM &projs, int y, int thickness, int gridYMax, float pitch_angle, int start, int length)
{
  volinfo.Y_add_right = ceil(fabsf(tan(D2R(pitch_angle))) * thickness);
  volinfo.Y_add_left = ceil(fabsf(tan(D2R(pitch_angle))) * thickness);
  //volinfo.steplength = min(64, gridYMax);
  volinfo.steplength = min(volinfo.steplength, static_cast<float>(length));

  if (y < volinfo.Y_add_left)
  {
    volinfo.Y_add_left = y - start;
  }
  else if (y + volinfo.steplength + volinfo.Y_add_right > projs.Y() && y + volinfo.steplength < projs.Y())
  {
    volinfo.Y_add_right = vol.height - (y - start) - volinfo.steplength;
  }
  else if (y + volinfo.steplength >= projs.Y())
  {
    volinfo.Y_add_right = 0;
    volinfo.steplength = vol.height - (y - start);
  }

  if (y + volinfo.steplength >= length + start)
  {
    volinfo.steplength = vol.height - (y - start);
  }
  volinfo.slice_steplength = volinfo.Y_add_left + volinfo.steplength + volinfo.Y_add_right;

  volinfo.volsize = (size_t)projs.X() * thickness * volinfo.slice_steplength;
  volinfo.projsize = (size_t)projs.X() * volinfo.slice_steplength * projs.Z();
}
void InitYMAX(int *gridYMax, int maxThreadsSize, MrcStackM &projs, int thickness)
{
  int memlimit =
      deviceProps.totalGlobalMem /
      GetCuTaskDataSizePerPartition(projs.Z(), projs.X(), thickness) * 0.5;
  int gridXMax = min((projs.X() * thickness + maxThreadsSize - 1) / maxThreadsSize,
                     deviceProps.maxGridSize[0]);
  int gridZMax = min(projs.Z(), deviceProps.maxGridSize[2]);
  *gridYMax = min(min(projs.Y(), min(memlimit, deviceProps.maxGridSize[1])),
                  deviceProps.multiProcessorCount * 128 *
                      deviceProps.warpSize / (gridZMax * gridXMax));
}
void InitYMAX(int *gridYMax, int maxThreadsSize, MrcStackM &projs, int thickness, int length)
{
  int memlimit =
      deviceProps.totalGlobalMem /
      GetCuTaskDataSizePerPartition(projs.Z(), projs.X(), thickness) * 0.5;
  int gridXMax = min((projs.X() * thickness + maxThreadsSize - 1) / maxThreadsSize,
                     deviceProps.maxGridSize[0]);
  int gridZMax = min(projs.Z(), deviceProps.maxGridSize[2]);
  *gridYMax = min(min(length, min(memlimit, deviceProps.maxGridSize[1])),
                  deviceProps.multiProcessorCount * 128 *
                      deviceProps.warpSize / (gridZMax * gridXMax));
}

void InitBodyInfo(Bodyinfo &volinfo, float pitch_angle, MrcStackM &projs, int thickness, int length)
{
  volinfo.Y_add_right = ceil(fabsf(tan(D2R(pitch_angle))) * thickness);
  volinfo.Y_add_left = ceil(fabsf(tan(D2R(pitch_angle))) * thickness);
  volinfo.steplength = 64;
  volinfo.steplength = min(volinfo.steplength, static_cast<float>(length));
  volinfo.slice_steplength = volinfo.Y_add_left + volinfo.steplength + volinfo.Y_add_right;
  volinfo.volsize = (size_t)projs.X() * thickness * volinfo.slice_steplength;
  volinfo.projsize = (size_t)projs.X() * projs.Z() * volinfo.slice_steplength;
  volinfo.volslicesize = (size_t)projs.X() * thickness * volinfo.steplength;
}
void InitBodyInfo(Bodyinfo &volinfo, float pitch_angle, MrcStackM &projs, int thickness, int gridYMax, int length)
{
  volinfo.Y_add_right = ceil(fabsf(tan(D2R(pitch_angle))) * thickness);
  volinfo.Y_add_left = ceil(fabsf(tan(D2R(pitch_angle))) * thickness);
  volinfo.steplength = 64;
  volinfo.steplength = max(64, gridYMax);
  volinfo.steplength = min(volinfo.steplength, static_cast<float>(length));
  volinfo.slice_steplength = volinfo.Y_add_left + volinfo.steplength + volinfo.Y_add_right;
  volinfo.volsize = (size_t)projs.X() * thickness * volinfo.slice_steplength;
  volinfo.projsize = (size_t)projs.X() * projs.Z() * volinfo.slice_steplength;
  volinfo.volslicesize = (size_t)projs.X() * thickness * volinfo.steplength;
}
void CuBackProject(Point3DF &origin, MrcStackM &projs,
                   std::vector<SimCoeff> &params, int thickness,
                   MrcStackM &mrcvol, Slice &proj,
                   Volume &vol, float pitch_angle, int start, int length, int add_left)
{
  int gridYMax;
  int maxThreadsSize = 128;
  InitYMAX(&gridYMax, maxThreadsSize, projs, thickness, length);
  Bodyinfo volinfo;
  InitBodyInfo(volinfo, pitch_angle, projs, thickness, length);

  CHECK_CUDA(cudaHostAlloc((void **)(&vol.data), sizeof(float) * volinfo.volslicesize,
                           cudaHostAllocDefault))
  CuTaskDataY cudev;
  CuMallocBPTTaskData(cudev, projs.Z(), projs.X(), thickness, volinfo.slice_steplength);
  float oy = origin.y;
  CHECK_CUDA(cudaMemcpyToSymbol(c_coeff, &(params[0]), sizeof(SimCoeff) * projs.Z(),
                                0, cudaMemcpyHostToDevice))

  for (int y = start; y < start + vol.height; y += volinfo.steplength)
  {
    ReadSliceBlock(vol, volinfo, projs, y, thickness, gridYMax, pitch_angle, start, length);
    std::cout << "BPT reconstructs " << y << "~" << y + volinfo.steplength << std::endl;
    CHECK_CUDA(cudaMemset(cudev.slice, 0, sizeof(float) * volinfo.volsize))

    cudev.y = volinfo.slice_steplength;
    origin.y = oy - y + volinfo.Y_add_left;
    cudaMemcpyToSymbol(c_origin, &origin, sizeof(Point3DF), 0, cudaMemcpyHostToDevice);

    for (int z = 0; z < projs.Z(); z++)
    {
      size_t sliceOffset = z * projs.X() * length + (y - start + add_left - volinfo.Y_add_left) * projs.X();
      float *hostPtr = proj.data + sliceOffset;
      float *devicePtr = cudev.projs + (int)(z * volinfo.slice_steplength * projs.X());
      CHECK_CUDA(cudaMemcpy(devicePtr, hostPtr, sizeof(float) * volinfo.slice_steplength * projs.X(), cudaMemcpyHostToDevice))
    }

    dim3 dimBlock = maxThreadsSize;
    dim3 dim3Grid((projs.X() * thickness + maxThreadsSize - 1) / maxThreadsSize,
                  volinfo.slice_steplength, projs.Z());
    CuBackProjKernel<<<dim3Grid, dimBlock>>>(cudev.slice, cudev.projs, cudev.x,
                                             cudev.z, cudev.y);
    size_t offset = 0;
    if (pitch_angle)
    {
      offset = projs.X() * thickness * volinfo.Y_add_left;
    }
    CHECK_CUDA(cudaMemcpy(vol.data, cudev.slice + offset, sizeof(float) * volinfo.volslicesize, cudaMemcpyDeviceToHost))

    mrcvol.WriteBlock(y, y + volinfo.steplength, 'z', vol.data);
  }
  CuFreeTaskData(cudev);
  cudaFreeHost(vol.data);
}

void CuFBP(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
           int thickness, MrcStackM &mrcvol, Slice &proj, Volume &vol,
           int filterMode, float pitch_angle, int start, int length, int add_left)
{
  int gridYMax;
  int maxThreadsSize = 128;
  InitYMAX(&gridYMax, maxThreadsSize, projs, thickness, length);
  Bodyinfo volinfo;
  InitBodyInfo(volinfo, pitch_angle, projs, thickness, length);

  CuTaskDataY cudev;

  CHECK_CUDA(cudaHostAlloc((void **)(&vol.data), sizeof(float) * volinfo.volslicesize, cudaHostAllocDefault))
  CHECK_CUDA(cudaMemcpyToSymbol(c_coeff, &(params[0]), sizeof(SimCoeff) * projs.Z(), 0, cudaMemcpyHostToDevice))

  CuMallocBPTTaskData(cudev, projs.Z(), projs.X(), thickness, volinfo.slice_steplength);
  float oy = origin.y;
 // ApplyFilterInplace(projs, proj.data, length, filterMode);
  for (int y = start; y < start + vol.height; y += volinfo.steplength)
  {
    ReadSliceBlock(vol, volinfo, projs, y, thickness, gridYMax, pitch_angle, start, length);
    if (filterMode == 0)
      std::cout << "FBP reconstructs " << y << "~" << y + volinfo.steplength << std::endl;
    else
      std::cout << "WBP reconstructs " << y << "~" << y + volinfo.steplength << std::endl;
    cudev.y = volinfo.slice_steplength;
    origin.y = oy - y + volinfo.Y_add_left;

    CHECK_CUDA(cudaMemcpyToSymbol(c_origin, &origin, sizeof(Point3DF), 0, cudaMemcpyHostToDevice))

    for (int z = 0; z < projs.Z(); z++)
    {
      size_t sliceOffset = z * projs.X() * length + (y - start + add_left - volinfo.Y_add_left) * projs.X();
      float *hostPtr = proj.data + sliceOffset;
      float *devicePtr = cudev.projs + (int)(z * volinfo.slice_steplength * projs.X());
      CHECK_CUDA(cudaMemcpy(devicePtr, hostPtr, sizeof(float) * volinfo.slice_steplength * projs.X(), cudaMemcpyHostToDevice))
    }

    dim3 dimBlock = maxThreadsSize;
    dim3 dim3Grid((projs.X() * thickness + maxThreadsSize - 1) / maxThreadsSize,
                  volinfo.slice_steplength, projs.Z());
    cudaMemset(cudev.slice, 0, sizeof(float) * volinfo.volsize);

    CuBackProjKernel<<<dim3Grid, dimBlock>>>(cudev.slice, cudev.projs, cudev.x, cudev.z, cudev.y);
    size_t offset = 0;
    if (pitch_angle)
    {
      offset = projs.X() * thickness * volinfo.Y_add_left;
    }

    CHECK_CUDA(cudaMemcpy(vol.data, cudev.slice + offset, sizeof(float) * volinfo.volslicesize, cudaMemcpyDeviceToHost))
    mrcvol.WriteBlock(y, y + volinfo.steplength, 'z', vol.data);
  }
  CuFreeTaskData(cudev);
  CHECK_CUDA(cudaFreeHost(vol.data))
}

void CuSIRT(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
            int thickness, MrcStackM &mrcvol, Slice &proj, Volume &vol,
            int iteration,
            float gamma,
            float pitch_angle, int start, int length, int add_left)
{
  int gridYMax;
  int maxThreadsSize = 128;
  InitYMAX(&gridYMax, maxThreadsSize, projs, thickness, length);
  Bodyinfo volinfo;
  InitBodyInfo(volinfo, pitch_angle, projs, thickness, gridYMax, length);

  CHECK_CUDA(cudaHostAlloc((void **)(&vol.data), sizeof(float) * volinfo.volslicesize,
                           cudaHostAllocDefault))

  CuTaskDataY cudev;
  CuMallocSIRTTaskData(cudev, projs.Z(), projs.X(), thickness, volinfo.slice_steplength);

  float oy = origin.y;
  CHECK_CUDA(cudaMemcpyToSymbol(c_coeff, &(params[0]), sizeof(SimCoeff) * projs.Z(),
                                0, cudaMemcpyHostToDevice))

  float *valvol;
  CHECK_CUDA(cudaMalloc((void **)&valvol, sizeof(float) * volinfo.volsize))

  float *rewt, *bpwt;
  CHECK_CUDA(cudaMalloc((void **)&rewt, sizeof(float) * volinfo.projsize))
  CHECK_CUDA(cudaMalloc((void **)&bpwt, sizeof(float) * volinfo.volsize))

  cudaArray *projarray;
  cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
  for (int y = start; y < start + vol.height; y += volinfo.steplength)
  {

    ReadSliceBlock(vol, volinfo, projs, y, thickness, gridYMax, pitch_angle, start, length);
    std::cout << "SIRT reconstructs " << y << "~" << y + volinfo.steplength << std::endl;

    cudev.y = volinfo.slice_steplength;
    origin.y = oy - y + volinfo.Y_add_left;

    CHECK_CUDA(cudaMemcpyToSymbol(c_origin, &origin, sizeof(Point3DF), 0, cudaMemcpyHostToDevice))

    dim3 dimBlock = maxThreadsSize;
    dim3 dim3Grid((projs.X() + maxThreadsSize - 1) / maxThreadsSize, thickness, volinfo.slice_steplength);
    dim3 dim2Grid_xy((projs.X() * volinfo.slice_steplength + maxThreadsSize - 1) / maxThreadsSize, projs.Z());
    dim3 dim2Grid_xz((projs.X() * thickness + maxThreadsSize - 1) / maxThreadsSize, projs.Z());

    CHECK_CUDA(cudaMemset(cudev.slice, 0, sizeof(float) * volinfo.volsize))

    for (int z = 0; z < projs.Z(); z++)
    {
      size_t sliceOffset = z * projs.X() * length + (y - start + add_left - volinfo.Y_add_left) * projs.X();
      float *hostPtr = proj.data + sliceOffset;
      float *devicePtr = cudev.projs + (int)(z * volinfo.slice_steplength * projs.X());
      CHECK_CUDA(cudaMemcpy(devicePtr, hostPtr, sizeof(float) * volinfo.slice_steplength * projs.X(), cudaMemcpyHostToDevice))
    }

    CHECK_CUDA(cudaMemset(bpwt, 0, sizeof(float) * volinfo.volsize))
    CHECK_CUDA(cudaMemset(rewt, 0, sizeof(float) * volinfo.projsize))
    CuVolWeightKernel<<<dim3Grid, dimBlock>>>(bpwt, cudev.x, cudev.y, cudev.z, projs.Z());
    CuReWeightKernel<<<dim2Grid_xz, dimBlock>>>(rewt, cudev.x, cudev.y, cudev.z);
    cudaExtent extent;
    extent.width = cudev.x;
    extent.height = cudev.y;
    extent.depth = projs.Z();
    CHECK_CUDA(cudaMalloc3DArray(&projarray, &desc, extent))

    for (int i = 0; i < iteration; i++)
    {
      CHECK_CUDA(cudaMemset(valvol, 0, sizeof(float) * volinfo.volsize))
      CHECK_CUDA(cudaMemset(cudev.s, 0, sizeof(float) * volinfo.projsize))

      CuReprojectKernel<<<dim2Grid_xz, dimBlock>>>(cudev.slice, cudev.s,
                                                   cudev.x, cudev.z, cudev.y);
      CuCalcProjectionDiffKernel<<<dim2Grid_xy, dimBlock>>>(cudev.projs, cudev.s, rewt, cudev.x, cudev.y);

      struct cudaMemcpy3DParms parms = {0};
      parms.dstArray = projarray;
      parms.extent = make_cudaExtent(extent.width, extent.height, extent.depth);
      parms.kind = cudaMemcpyDeviceToDevice;
      parms.srcPos = make_cudaPos(0, 0, 0);
      parms.srcPtr = make_cudaPitchedPtr((void *)cudev.s, extent.width * sizeof(float),
                                         extent.width, extent.height);
      CHECK_CUDA(cudaMemcpy3D(&parms))

      struct cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypeArray;
      resDesc.res.array.array = projarray;
      struct cudaTextureDesc texdesc;
      memset(&texdesc, 0, sizeof(texdesc));
      texdesc.addressMode[0] = cudaAddressModeBorder;
      texdesc.addressMode[1] = cudaAddressModeBorder;
      texdesc.addressMode[2] = cudaAddressModeBorder;
      texdesc.filterMode = cudaFilterModeLinear;
      texdesc.normalizedCoords = false;
      cudaTextureObject_t projtexobj;
      CHECK_CUDA(cudaCreateTextureObject(&projtexobj, &resDesc, &texdesc, NULL))
      CuBackProjWeightAndValueKernel<<<dim3Grid, dimBlock>>>(valvol, projtexobj,
                                                             cudev.x, cudev.y, projs.Z(), cudev.z);

      CuUpdateVolumeByWeightsKernel<<<(volinfo.volsize + maxThreadsSize - 1) / maxThreadsSize, dimBlock>>>(cudev.slice, valvol, bpwt, gamma, volinfo.volsize);
    }
    size_t offset = 0;
    if (pitch_angle)
    {
      offset = projs.X() * thickness * volinfo.Y_add_left;
    }
    CHECK_CUDA(cudaMemcpy(vol.data, cudev.slice + offset, sizeof(float) * volinfo.volslicesize, cudaMemcpyDeviceToHost))

    printf("Writing data to file\n");
    mrcvol.WriteBlock(y, y + volinfo.steplength, 'z', vol.data);

    CHECK_CUDA(cudaFreeArray(projarray))
  }
  CuFreeTaskData(cudev);
  CHECK_CUDA(cudaFree(valvol))
  CHECK_CUDA(cudaFree(bpwt))
  CHECK_CUDA(cudaFreeHost(vol.data))
  CHECK_CUDA(cudaFree(rewt))
}

void CuSART(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
            int thickness, MrcStackM &mrcvol, Slice &proj, Volume &vol,
            int iteration,
            float gamma,
            float pitch_angle, int start, int length, int add_left)
{
  int gridYMax;
  int maxThreadsSize = 128;
  InitYMAX(&gridYMax, maxThreadsSize, projs, thickness, length);
  Bodyinfo volinfo;
  InitBodyInfo(volinfo, pitch_angle, projs, thickness, length);

  float *curProjData;
  CHECK_CUDA(cudaHostAlloc((void **)(&vol.data), sizeof(float) * volinfo.volslicesize,
                           cudaHostAllocDefault))
  CuTaskDataY cudev;
  CuMallocSARTTaskData(cudev, projs.Z(), projs.X(), thickness, volinfo.slice_steplength);
  CHECK_CUDA(cudaMemcpyToSymbol(c_coeff, &(params[0]), sizeof(SimCoeff) * projs.Z(), 0, cudaMemcpyHostToDevice))

  float oy = origin.y;
  float *rewt;
  CHECK_CUDA(cudaMalloc((void **)&rewt, sizeof(float) * volinfo.projsize))

  for (int y = start; y < start + vol.height; y += volinfo.steplength)
  {
    ReadSliceBlock(vol, volinfo, projs, y, thickness, gridYMax, pitch_angle, start, length);

    std::cout << "SART reconstructs " << y << "~" << y + volinfo.steplength << std::endl;

    cudev.y = volinfo.slice_steplength;
    size_t oneprojsize = (size_t)projs.X() * volinfo.slice_steplength;
    origin.y = oy - y + volinfo.Y_add_left;

    CHECK_CUDA(cudaMemcpyToSymbol(c_origin, &origin, sizeof(Point3DF), 0, cudaMemcpyHostToDevice))

    CHECK_CUDA(cudaMemset(cudev.slice, 0, sizeof(float) * volinfo.volsize))
    for (int z = 0; z < projs.Z(); z++)
    {
      size_t sliceOffset = z * projs.X() * length + (y - start + add_left - volinfo.Y_add_left) * projs.X();
      float *hostPtr = proj.data + sliceOffset;
      float *devicePtr = cudev.projs + (int)(z * volinfo.slice_steplength * projs.X());
      CHECK_CUDA(cudaMemcpy(devicePtr, hostPtr, sizeof(float) * volinfo.slice_steplength * projs.X(), cudaMemcpyHostToDevice))
    }
    dim3 dimBlock = maxThreadsSize;
    dim3 dim3Grid((projs.X() + maxThreadsSize - 1) / maxThreadsSize, thickness,
                  volinfo.slice_steplength);
    dim3 dim2Grid((projs.X() * thickness + maxThreadsSize - 1) / maxThreadsSize,
                  projs.Z());
    CHECK_CUDA(cudaMemset(rewt, 0, sizeof(float) * volinfo.projsize))
    CuReWeightKernel<<<dim2Grid, dimBlock>>>(rewt, cudev.x, cudev.y, cudev.z);

    for (int i = 0; i < iteration; i++)
    {
      for (int projIdxStart = 0; projIdxStart < projs.Z(); projIdxStart++)
      {
        curProjData = cudev.projs + (int)(projIdxStart * projs.X() * volinfo.slice_steplength);
        CHECK_CUDA(cudaMemset(cudev.s, 0, sizeof(float) * oneprojsize))
        CuReprojectKernel_SART<<<(projs.X() * thickness + maxThreadsSize - 1) / maxThreadsSize, dimBlock>>>(cudev.slice,
                                                                                                            cudev.s, cudev.x, cudev.z, cudev.y, projIdxStart);
        CuCalcProjectionDiffKernel_SART<<<(projs.X() * volinfo.slice_steplength + maxThreadsSize - 1) / maxThreadsSize, dimBlock>>>(curProjData,
                                                                                                                                    cudev.s, rewt, cudev.x, cudev.y, projIdxStart);
        CuUpdateVolumeByProjDiffKernel_SART<<<dim3Grid, dimBlock>>>(cudev.s, cudev.slice, cudev.x, cudev.y, cudev.z, projIdxStart, gamma);
      }
    }
    size_t offset = 0;
    if (pitch_angle)
    {
      offset = projs.X() * thickness * volinfo.Y_add_left;
    }
    CHECK_CUDA(cudaMemcpy(vol.data, cudev.slice + offset, sizeof(float) * volinfo.volslicesize, cudaMemcpyDeviceToHost))
    printf("Write data to file\n");
    mrcvol.WriteBlock(y, y + volinfo.steplength, 'z', vol.data);
  }
  CuFreeTaskData(cudev);
  CHECK_CUDA(cudaFreeHost(vol.data))
  CHECK_CUDA(cudaFree(rewt))
}