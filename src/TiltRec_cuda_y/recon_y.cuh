#ifndef RECON_Y
#define RECON_Y

// #include <cuda.h>
// #include <cuda_runtime.h>
#include "basic.cuh"
#include "mrcmx/mrcstack.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <fstream>
#include <iostream>
#include <vector>
#include "../filter/filter_prj.h"
#include "../opts/opts.h"
void InitYMAX(int *gridYMax, int maxThreadsSize, MrcStackM &projs, int thickness);
void InitYMAX(int *gridYMax, int maxThreadsSize, MrcStackM &projs, int thickness, int length);

void InitBodyInfo(Bodyinfo &volinfo, float pitch_angle, MrcStackM &projs, int thickness, int length);
void InitBodyInfo(Bodyinfo &volinfo, float pitch_angle, MrcStackM &projs, int thickness, int gridYMax, int length);
//void ReadSliceBlock(Volume &vol, int *Y_add_left, int *Y_add_right, size_t *slice_steplength, size_t *steplength, MrcStackM &projs, float *tmp, int y);
void ReadSliceBlock(Volume &vol, Bodyinfo &volinfo, MrcStackM &projs, float *tmp, int y, int thickness, int gridYMax, float pitch_angle);
void ReadSliceBlock(Volume &vol, Bodyinfo &volinfo, MrcStackM &projs, int y, int thickness, int gridYMax, float pitch_angle, int start, int length);

void CuBackProject(Point3DF &origin, MrcStackM &projs,
                   std::vector<SimCoeff> &params,
                   MrcStackM &mrcvol, Slice &proj,
                   Volume &vol, int start, int length, int add_left, const options &opt);

void CuFBP(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
            MrcStackM &mrcvol, Slice &proj, Volume &vol,
           int filterMode, int start, int length, int add_left, const options &opt);

void CuSIRT(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
            MrcStackM &mrcvol, Slice &proj, Volume &vol,
            int start, int length, int add_left, const options &opt);

void CuSART(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
            MrcStackM &mrcvol, Slice &proj, Volume &vol,
            int start, int length, int add_left, const options &opt);

#endif