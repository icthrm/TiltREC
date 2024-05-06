#ifndef RECON_Y
#define RECON_Y

// #include <cuda.h>
// #include <cuda_runtime.h>
#include "basic.cuh"
#include "mrcmx/mrcstack.h"

#include <fstream>
#include <iostream>
#include <vector>
#include "../filter/filter_prj.h"
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cusparse.h>
void InitYMAX(int *gridYMax, int maxThreadsSize, MrcStackM &projs, int thickness);
void InitYMAX(int *gridYMax, int maxThreadsSize, MrcStackM &projs, int thickness, int length);

void InitBodyInfo(Bodyinfo &volinfo, float pitch_angle, MrcStackM &projs, int thickness, int length);
void InitBodyInfo(Bodyinfo &volinfo, float pitch_angle, MrcStackM &projs, int thickness, int gridYMax, int length);
//void ReadSliceBlock(Volume &vol, int *Y_add_left, int *Y_add_right, size_t *slice_steplength, size_t *steplength, MrcStackM &projs, float *tmp, int y);
void ReadSliceBlock(Volume &vol, Bodyinfo &volinfo, MrcStackM &projs, float *tmp, int y, int thickness, int gridYMax, float pitch_angle);
void ReadSliceBlock(Volume &vol, Bodyinfo &volinfo, MrcStackM &projs, int y, int thickness, int gridYMax, float pitch_angle, int start, int length);

void CuBackProject(Point3DF &origin, MrcStackM &projs,
                   std::vector<SimCoeff> &params, int thickness,
                   MrcStackM &mrcvol, Slice &proj,
                   Volume &vol, float pitch_angle, int start, int length, int add_left);

void CuFBP(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
           int thickness, MrcStackM &mrcvol, Slice &proj, Volume &vol,
           int filterMode, float pitch_angle, int start, int length, int add_left);

void CuSIRT(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
            int thickness, MrcStackM &mrcvol, Slice &proj, Volume &vol,
            int iteration,
            float gamma,
            float pitch_angle, int start, int length, int add_left);

void CuSART(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
            int thickness, MrcStackM &mrcvol, Slice &proj, Volume &vol,
            int iteration,
            float gamma,
            float pitch_angle, int start, int length, int add_left);

#endif