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
#include "../TiltRec_cuda_y/basic.cuh"
#include "../opts/opts.h"
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
void CuBackProjectZ(Point3DF &origin, MrcStackM &projs,
                    std::vector<SimCoeff> &params,
                    MrcStackM &mrcvol, Slice &proj,
                    Volume &vol, const options &opt);

void CuSIRTZ(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
             MrcStackM &mrcvol, Slice &proj, Volume &vol,
             const options &opt);

void CuSARTZ(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
             int thickness, MrcStackM &mrcvol, Slice &proj, Volume &vol,
             int iteration,
             float gamma,const options &opt);

void CuFBPZ(Point3DF &origin, MrcStackM &projs,
           std::vector<SimCoeff> &params, int thickness,
           MrcStackM &mrcvol, Slice &proj,
           Volume &vol, int filterMode,const options &opt);
           
void CuADMMZ(Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
             int thickness, MrcStackM &mrcvol, Slice &proj, Volume &vol,
             int iteration, int cgiter,
             float gamma, float soft,const options &opt);

#endif