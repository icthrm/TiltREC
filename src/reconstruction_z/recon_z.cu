#include "recon_z.cuh"
extern cudaDeviceProp deviceProps;

void CuBackProjectZ(Point3DF &origin, MrcStackM &projs,
                    std::vector<SimCoeff> &params, int thickness,
                    MrcStackM &mrcvol, Slice &proj,
                    Volume &vol)
{

    size_t maxThreadsSize = deviceProps.maxThreadsPerBlock;

    int steplength = thickness; 
    int projsnum = projs.Z();
    size_t volsize = (size_t)projs.X() * projs.Y() * steplength;
    size_t projsize = (size_t)projs.X() * projs.Y() * projsnum; 
    cudaMallocManaged((void **)(&vol.data), sizeof(float) * volsize);
    CUERR
    CuTaskDataZ cudev;
    CuMallocBPTTaskDataZ(cudev, projsnum, projs.X(), projs.Y(), steplength);
    float *originalProjsData;
    cudaMallocManaged((void **)(&originalProjsData), sizeof(float) * projsize); // the data of proj is extended by steplength
    CUERR
    cudaMemcpy(cudev.coeffs, &(params[0]), sizeof(SimCoeff) * params.size(),
               cudaMemcpyHostToDevice);
    CUERR
    { 
        int j = 0;
        for (j = 0; j + 5 < projs.Z(); j += 5)
            projs.ReadBlock(j, j + 5, 'z', (originalProjsData + (size_t)projs.X() * projs.Y() * j));
        projs.ReadBlock(j, projs.Z(), 'z', (originalProjsData + (size_t)projs.X() * projs.Y() * j));
    }
    CUERR
    cudaMemcpy(cudev.origin, &origin, sizeof(Point3DF), cudaMemcpyHostToDevice);
    CUERR

    for (int z = vol.z; z < vol.z + vol.height; z += steplength)
    {
        std::cout << "BPT reconstructs " << z  << std::endl;
        if (z + steplength >= vol.z + vol.height)
        { // compenstate for margin
            steplength = vol.z + vol.height - z;
            volsize = (size_t)projs.X() * projs.Y() * steplength;
            projsize = (size_t)projs.X() * projs.Y() * projsnum;
            cudev.z = steplength;
        }

        time_t now = time(NULL);
        char *curr_time = ctime(&now);
        std::cout << curr_time << std::endl;

        cudaDeviceSynchronize();
        CUERR
        dim3 dimBlock = maxThreadsSize;
        dim3 dim3Grid((projs.X() * projs.Y() + maxThreadsSize - 1) / maxThreadsSize,
                      steplength, projsnum);
        CuBackProjKernelZ<<<dim3Grid, dimBlock>>>(cudev.origin, cudev.coeffs,
                                                  vol.data, originalProjsData, cudev.x,
                                                  cudev.y, z);
        CUERR
        cudaDeviceSynchronize();
        CUERR
        printf("Write data to file\n");
        {
            int j = 0;
            for (j = 0; j + 20 < thickness; j += 20)
                mrcvol.WriteBlock(j, j + 20, 'z', (vol.data + (size_t)projs.X() * projs.Y() * j));
            mrcvol.WriteBlock(j, thickness, 'z', (vol.data + (size_t)projs.X() * projs.Y() * j));
        }
    }
    time_t now = time(NULL);
    char *curr_time = ctime(&now);
    std::cout << curr_time << std::endl;
    cudaFree(vol.data);
    CuFreeTaskDataZ(cudev);
    cudaFree(originalProjsData);
}

void CuSIRTZ(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
             int thickness, MrcStackM &mrcvol, Slice &proj, Volume &vol,
             int iteration,
             float gamma)
{

    size_t maxThreadsSize = deviceProps.maxThreadsPerBlock;

    int batchsize = 1; 
    int projsnum = projs.Z();
    size_t volsize = (size_t)projs.X() * projs.Y() * thickness;
    size_t oneProjsize = (size_t)projs.X() * projs.Y() * batchsize; 
    cudaMallocManaged((void **)(&vol.data), sizeof(float) * volsize);
    CUERR
    CuTaskDataZ cudev;
    CuMallocSIRTTaskDataZ(cudev, projs.Z(), projs.X(), projs.Y(), thickness, batchsize);
    CUERR
    cudaMemcpy(cudev.origin, &origin, sizeof(Point3DF), cudaMemcpyHostToDevice);
    CUERR
    cudaMemcpy(cudev.coeffs, &(params[0]), sizeof(SimCoeff) * params.size(),
               cudaMemcpyHostToDevice);
    CUERR

    float *valvol, *wtvol;
    dim3 dimBlock = maxThreadsSize;
    dim3 dim3Grid((projs.X() * projs.Y() + maxThreadsSize - 1) / maxThreadsSize,
                  thickness, batchsize);
    dim3 dim2Grid((projs.X() * projs.Y() + maxThreadsSize - 1) / maxThreadsSize,
                  batchsize);
    cudaMallocManaged((void **)&valvol, sizeof(float) * volsize);
    cudaMallocManaged((void **)&wtvol, sizeof(float) * volsize);
    CUERR
    cudaMallocManaged((void **)&proj.data, sizeof(float) * (size_t)projs.X() * projs.Y() * projs.Z());
    { 
        int j = 0;
        for (j = 0; j + 5 < projs.Z(); j += 5)
            projs.ReadBlock(j, j + 5, 'z', (proj.data + (size_t)projs.X() * projs.Y() * j));
        projs.ReadBlock(j, projs.Z(), 'z', (proj.data + (size_t)projs.X() * projs.Y() * j));
    }

    for (int iter = 0; iter < iteration; ++iter)
    {
        cudaMemset(valvol, 0, sizeof(float) * volsize);
        cudaMemset(wtvol, 0, sizeof(float) * volsize);
        for (int projIdxStart = 0; projIdxStart < projsnum; projIdxStart += batchsize) 
        {
            float *curProjData = proj.data + projIdxStart * projs.X() * projs.Y();
            cudaDeviceSynchronize();
            printf("SIRT Iter %d on projs [%d,%d)\n", iter, projIdxStart, projIdxStart + batchsize);
            cudaMemset(cudev.c, 0, sizeof(float) * oneProjsize);
            cudaMemset(cudev.s, 0, sizeof(float) * oneProjsize);
            CuReprojectKernelZ<<<dim3Grid, dimBlock>>>(cudev.origin, cudev.coeffs,
                                                       vol.data, cudev.s, cudev.c,
                                                       cudev.x, cudev.y, 0, projIdxStart); 
            CUERR

            CuCalcProjectionDiffKernelZ<<<dim2Grid, dimBlock>>>(
                curProjData, cudev.s, cudev.c, cudev.x, cudev.y);
            CUERR
            CuBackProjWeightAndValueKernelZ<<<dim3Grid, dimBlock>>>(
                cudev.origin, cudev.coeffs, valvol, wtvol, cudev.s, cudev.x, cudev.y,
                0, projIdxStart);

            cudaDeviceSynchronize();

            CUERR
        }

        CuUpdateVolumeByWeightsKernelZ<<<
            (volsize + maxThreadsSize - 1) / maxThreadsSize, dimBlock>>>(
            vol.data, valvol, wtvol, gamma, volsize);
        CUERR
        cudaDeviceSynchronize();
        CUERR
    }
    cudaDeviceSynchronize();
    CUERR

    printf("Writing data to file\n");
    {
        int j = 0;
        for (j = 0; j + 5 < thickness; j += 5)
            mrcvol.WriteBlock(j, j + 5, 'z', (vol.data + (size_t)projs.X() * projs.Y() * j));
        mrcvol.WriteBlock(j, thickness, 'z', (vol.data + (size_t)projs.X() * projs.Y() * j));
    }
    cudaFree(vol.data);
    CuFreeTaskDataZ(cudev);
    cudaFree(valvol);
    cudaFree(wtvol);
    cudaFree(proj.data);
}

void CuSARTZ(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
             int thickness, MrcStackM &mrcvol, Slice &proj, Volume &vol,
             int iteration,
             float gamma)
{
    size_t maxThreadsSize = deviceProps.maxThreadsPerBlock;

    int batchsize = 1; 
    int projsnum = projs.Z();
    size_t volsize = (size_t)projs.X() * projs.Y() * thickness;
    size_t projsize = (size_t)projs.X() * projs.Y() * batchsize;
    cudaMallocManaged((void **)(&vol.data), sizeof(float) * volsize);
    CUERR
    CuTaskDataZ cudev;
    CuMallocSIRTTaskDataZ(cudev, projs.Z(), projs.X(), projs.Y(), thickness, batchsize);
    CUERR
    cudaMemcpy(cudev.origin, &origin, sizeof(Point3DF), cudaMemcpyHostToDevice);
    CUERR
    cudaMemcpy(cudev.coeffs, &(params[0]), sizeof(SimCoeff) * params.size(),
               cudaMemcpyHostToDevice);
    CUERR

    dim3 dimBlock = maxThreadsSize;
    dim3 dim3Grid((projs.X() * projs.Y() + maxThreadsSize - 1) / maxThreadsSize,
                  thickness, batchsize);
    dim3 dim2Grid((projs.X() * projs.Y() + maxThreadsSize - 1) / maxThreadsSize,
                  batchsize);

    CUERR
    cudaMallocManaged((void **)&proj.data, sizeof(float) * (size_t)projs.X() * projs.Y() * projs.Z());
    { 
        int j = 0;
        for (j = 0; j + 5 < projs.Z(); j += 5)
            projs.ReadBlock(j, j + 5, 'z', (proj.data + (size_t)projs.X() * projs.Y() * j));
        projs.ReadBlock(j, projs.Z(), 'z', (proj.data + (size_t)projs.X() * projs.Y() * j));
    }
    for (int iter = 0; iter < iteration; ++iter)
    {
        for (int projIdxStart = 0; projIdxStart < projsnum; projIdxStart += batchsize) 
        {
            float *curProjData = proj.data + projIdxStart * projs.X() * projs.Y();
            cudaDeviceSynchronize();
            printf("SART Iter %d on projs [%d,%d)\n", iter, projIdxStart, projIdxStart + batchsize);
            cudaMemset(cudev.c, 0, sizeof(float) * projsize);
            cudaMemset(cudev.s, 0, sizeof(float) * projsize);
            CuReprojectKernelZ<<<dim3Grid, dimBlock>>>(cudev.origin, cudev.coeffs,
                                                       vol.data, cudev.s, cudev.c,
                                                       cudev.x, cudev.y, 0, projIdxStart); 
            CUERR
            cudaDeviceSynchronize();
            CuCalcProjectionDiffKernelZ<<<dim2Grid, dimBlock>>>(
                curProjData, cudev.s, cudev.c, cudev.x, cudev.y);
            CUERR
            cudaDeviceSynchronize();
            CuUpdateVolumeByProjDiffKernelZ<<<dim3Grid, dimBlock>>>(
                cudev.origin, cudev.coeffs, vol.data, cudev.s, gamma, cudev.x, cudev.y,
                0, projIdxStart);
            cudaDeviceSynchronize();
            CUERR
        }
    }
    {
        printf("Write data to file\n");
        int j = 0;
        for (j = 0; j + 5 < thickness; j += 5)
            mrcvol.WriteBlock(j, j + 5, 'z', (vol.data + (size_t)projs.X() * projs.Y() * j));
        mrcvol.WriteBlock(j, thickness, 'z', (vol.data + (size_t)projs.X() * projs.Y() * j));
    }

    cudaFree(vol.data);
    CuFreeTaskDataZ(cudev);
    cudaFreeHost(proj.data);
}

void CuFBPZ(Point3DF &origin, MrcStackM &projs,
           std::vector<SimCoeff> &params, int thickness,
           MrcStackM &mrcvol, Slice &proj,
           Volume &vol, int filterMode)
{
    size_t maxThreadsSize = deviceProps.maxThreadsPerBlock;

    int steplength = thickness; 
    int projsnum = projs.Z();
    size_t volsize = (size_t)projs.X() * projs.Y() * steplength;
    size_t projsize = (size_t)projs.X() * projs.Y() * projsnum; 
    cudaMallocManaged((void **)(&vol.data), sizeof(float) * volsize);
    CUERR
    CuTaskDataZ cudev;
    CuMallocBPTTaskDataZ(cudev, projsnum, projs.X(), projs.Y(), steplength);
    float *originalProjsData;
    cudaMallocManaged((void **)(&originalProjsData), sizeof(float) * projsize); // the data of proj is extended by steplength
    CUERR
    cudaMemcpy(cudev.coeffs, &(params[0]), sizeof(SimCoeff) * params.size(),
               cudaMemcpyHostToDevice);
    CUERR
    { 
        int j = 0;
        for (j = 0; j + 5 < projs.Z(); j += 5)
            projs.ReadBlock(j, j + 5, 'z', (originalProjsData + (size_t)projs.X() * projs.Y() * j));
        projs.ReadBlock(j, projs.Z(), 'z', (originalProjsData + (size_t)projs.X() * projs.Y() * j));
    }
    size_t ny = projs.header.ny;

    ApplyFilterInplace(projs, originalProjsData, ny, filterMode); 

    cudaMemcpy(cudev.origin, &origin, sizeof(Point3DF), cudaMemcpyHostToDevice);
    CUERR
    for (int z = vol.z; z < vol.z + vol.height; z += steplength)
    {
        if (z + steplength >= vol.z + vol.height)
        { // compenstate for margin
            steplength = vol.z + vol.height - z;
            volsize = (size_t)projs.X() * projs.Y() * steplength;
            projsize = (size_t)projs.X() * projs.Y() * projsnum;
            cudev.z = steplength;
        }

        time_t now = time(NULL);
        char *curr_time = ctime(&now);
        std::cout << curr_time << std::endl;

        cudaDeviceSynchronize();
        CUERR
        dim3 dimBlock = maxThreadsSize;
        dim3 dim3Grid((projs.X() * projs.Y() + maxThreadsSize - 1) / maxThreadsSize,
                      steplength, projsnum);

        CuBackProjKernelZ<<<dim3Grid, dimBlock>>>(cudev.origin, cudev.coeffs,
                                                  vol.data, originalProjsData, cudev.x,
                                                  cudev.y, z);
        CUERR
        cudaDeviceSynchronize();
        CUERR
        printf("Write data to file\n");
        {
            int j = 0;
            for (j = 0; j + 20 < thickness; j += 20)
                mrcvol.WriteBlock(j, j + 20, 'z', (vol.data + (size_t)projs.X() * projs.Y() * j));
            mrcvol.WriteBlock(j, thickness, 'z', (vol.data + (size_t)projs.X() * projs.Y() * j));
        }
    }
    time_t now = time(NULL);
    char *curr_time = ctime(&now);
    std::cout << curr_time << std::endl;
    cudaFree(vol.data);
    CuFreeTaskDataZ(cudev);
    cudaFree(originalProjsData);
}

void CuADMMZ(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
             int thickness, MrcStackM &mrcvol, Slice &proj, Volume &vol,
             int iteration, int cgiter, float gamma, float soft)
{
    size_t maxThreadsSize = deviceProps.maxThreadsPerBlock;
    int batchsize = 1; 
    int projsnum = projs.Z();
    size_t volsize = (size_t)projs.X() * projs.Y() * thickness; 
    cudaMallocManaged((void **)(&vol.data), sizeof(float) * volsize);
    CUERR

    CuTaskDataZ cudev;
    CuMallocADMMTaskDataZ(cudev, projs.Z(), projs.X(), projs.Y(), thickness, batchsize);
    CUERR
    cudaMemcpy(cudev.origin, &origin, sizeof(Point3DF), cudaMemcpyHostToDevice); 
    CUERR
    cudaMemcpy(cudev.coeffs, &(params[0]), sizeof(SimCoeff) * params.size(), cudaMemcpyHostToDevice);
    CUERR

    dim3 dimBlock = maxThreadsSize;
    dim3 dim1Grid((volsize + maxThreadsSize - 1) / maxThreadsSize);  
    dim3 dim2Grid_xy((projs.X() * projs.Y() + maxThreadsSize - 1) / maxThreadsSize, thickness);
    dim3 dim3Grid((projs.X() * projs.Y() + maxThreadsSize - 1) / maxThreadsSize,
                  thickness, batchsize);
    dim3 dim3Grid_xyz((projs.X() * projs.Y() + maxThreadsSize - 1) / maxThreadsSize,
                  thickness, projs.Z());
    CUERR
    
    cudaMallocManaged((void **)&proj.data, sizeof(float) * (size_t)projs.X() * projs.Y() * projs.Z());
    { 
        int j = 0;
        for (j = 0; j + 5 < projs.Z(); j += 5)
            projs.ReadBlock(j, j + 5, 'z', (proj.data + (size_t)projs.X() * projs.Y() * j));
        projs.ReadBlock(j, projs.Z(), 'z', (proj.data + (size_t)projs.X() * projs.Y() * j));
    }

    cudaEvent_t begin, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&stop);
    cudaEventRecord(begin);
    std::cout << "ADMM reconstruction started." << std::endl;

    float *htb, *x0, *uk, *dk;
    cudaMallocManaged((void **)&htb, sizeof(float) * volsize);
    CUERR
    cudaMallocManaged((void **)&x0, sizeof(float) * volsize);
    CUERR
    cudaMallocManaged((void **)&uk, sizeof(float) * volsize);
    CUERR
    cudaMallocManaged((void **)&dk, sizeof(float) * volsize);
    CUERR
    
    cudaMemset(uk, 0, sizeof(float) * volsize);
    cudaMemset(dk, 0, sizeof(float) * volsize);

    for (int iter = 0; iter < iteration; ++iter)
    {
        cudaMemset(htb, 0, sizeof(float) * volsize);
        cudaMemset(x0, 0, sizeof(float) * volsize);

        for (int projIdxStart = 0; projIdxStart < projsnum; projIdxStart += batchsize) 
        {
            float *curProjData = proj.data + projIdxStart * projs.X() * projs.Y();
            cudaDeviceSynchronize();
            printf("ADMM Iter %d on projs [%d,%d)\n", iter, projIdxStart, projIdxStart + batchsize);

            CuAtb_ADMM_Z<<<dim3Grid, dimBlock>>>(cudev.origin, cudev.coeffs, htb, cudev.x, cudev.y, 
                                                 curProjData, 0, projIdxStart);
            cudaDeviceSynchronize();                                     
            CUERR
            CuATbGammaIt_ADMM_Z<<<dim1Grid, dimBlock>>>(htb, uk, dk, volsize, gamma);
            cudaDeviceSynchronize();
            CUERR
            CuATaGammaI_ADMM_Z(cudev.origin, cudev.coeffs, cudev.s, cudev.c, x0, cudev.x, cudev.y, 
                               volsize, vol.data, gamma, dim1Grid, dim3Grid, dimBlock, 0, projIdxStart); 
            CUERR
            cudaDeviceSynchronize();
        }
        CuApplycg_ADMM_Z(cudev, vol.data, x0, htb, cgiter, gamma, volsize, dim1Grid, dim3Grid_xyz, dimBlock);
        CUERR
        CuSoft_ADMM_Z<<<dim1Grid, dimBlock>>>(uk, dk, soft, cudev.x, volsize);
        CUERR
        cudaDeviceSynchronize();
        CUERR
    }
    cudaDeviceSynchronize();
    CUERR

    printf("Writing data to file\n");
    {
        int j = 0;
        for (j = 0; j + 5 < thickness; j += 5)
            mrcvol.WriteBlock(j, j + 5, 'z', (vol.data + (size_t)projs.X() * projs.Y() * j));
        mrcvol.WriteBlock(j, thickness, 'z', (vol.data + (size_t)projs.X() * projs.Y() * j));
    }
    CHECK_CUDA(cudaEventRecord(stop))
    CHECK_CUDA(cudaEventSynchronize(stop))
    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, begin, stop))
    std::cout << "ADMM reconstruction completed." << std::endl;
    std::cout << "Reconstruction time: " << milliseconds / 1000.0f << "s." << std::endl;
    CHECK_CUDA(cudaEventDestroy(begin))
    CHECK_CUDA(cudaEventDestroy(stop))
    
    cudaFree(vol.data);
    CuFreeTaskDataZ(cudev);
    cudaFree(htb);
    cudaFree(x0);
    cudaFree(uk);
    cudaFree(dk);
    cudaFree(proj.data);
}

