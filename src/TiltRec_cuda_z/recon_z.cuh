#ifndef RECON_Z
#define RECON_Z

#include <cuda.h>
#include <cuda_runtime.h>
#include "basicz.cuh"
#include "../filter/filter_prj.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <ctime>
#include "../reconstruction_y/basic.cuh"


void CuBackProjectZ(Point3DF &origin, MrcStackM &projs,
                    std::vector<SimCoeff> &params, int thickness,
                    MrcStackM &mrcvol, Slice &proj,
                    Volume &vol);

void CuSIRTZ(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
             int thickness, MrcStackM &mrcvol, Slice &proj, Volume &vol,
             int iteration,
             float gamma);

void CuSARTZ(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
             int thickness, MrcStackM &mrcvol, Slice &proj, Volume &vol,
             int iteration,
             float gamma);

void CuFBPZ(Point3DF &origin, MrcStackM &projs,
           std::vector<SimCoeff> &params, int thickness,
           MrcStackM &mrcvol, Slice &proj,
           Volume &vol, int filterMode);
           
void CuADMMZ(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
             int thickness, MrcStackM &mrcvol, Slice &proj, Volume &vol,
             int iteration, int cgiter,
             float gamma, float soft);

#endif