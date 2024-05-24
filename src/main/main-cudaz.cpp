#include <fstream>
#include <iostream>
#include <vector>
#include "..//TiltRec_cuda_y/recon_y.cuh"
#include "../TiltRec_cuda_z/recon_z.cuh"
#include "../opts/opts.h"
#include <sstream>
#include <cassert>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <ctime>
#include <cuda.h>
#include <cuda_runtime.h>
// #define BLOCKDIM 512
// #define CHECK_CUDA(func)                                         \
//   {                                                              \
//     cudaError_t status = (func);                                 \
//     if (status != cudaSuccess)                                   \
//     {                                                            \
//       printf("CUDA API failed at line %d with error: %s (%d)\n", \
//              __LINE__, cudaGetErrorString(status), status);      \
//     }                                                            \
//   }
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__, __LINE__, #value, value)
cudaDeviceProp deviceProps;

bool ReadAngles(std::vector<float> &angles, const char *name)
{
  std::ifstream in(name);
  if (!in.good())
  {
    return false;
  }

  while (in.good())
  {
    float val;
    in >> val;
    if (in.fail())
    {
      break;
    }

    angles.push_back(val);
  }
  in.close();
  return true;
}

static void CheckCudaErrorAux(const char *file, unsigned line, const char *statement, cudaError_t err)
{
  if (err == cudaSuccess)
    return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
            << err << ") at " << file << ":" << line << std::endl;
  exit(1);
}

int InitializeCuda(int mpi_rank_id)
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int gpu_id = mpi_rank_id % deviceCount;
  cudaSetDevice(gpu_id);

  CUDA_CHECK_RETURN(cudaGetDeviceProperties(&deviceProps, gpu_id));
  printf("\nCUDA device [%s]", deviceProps.name);
  printf("SM %d.%d\n", deviceProps.major, deviceProps.minor);

  // 	printf("TotalGlobalMem: %ld\n", deviceProps.totalGlobalMem);
  // 	printf("SharedMemPerBlock: %ld\n", deviceProps.sharedMemPerBlock);
  // 	printf("MaxThreadsPerBlock: %d\n", deviceProps.maxThreadsPerBlock);
  // 	printf("MultiProcessorCount: %d\n", deviceProps.multiProcessorCount);
  // 	printf("MaxGridSize: %d %d %d\n", deviceProps.maxGridSize[0], deviceProps.maxGridSize[1], deviceProps.maxGridSize[2]);

  if (deviceProps.major < 2)
  {
    printf("requires SM 2.0 or higher.\n");

    exit(-1);
  }

  cudaDeviceReset();
  return 0;
}

void TranslateAngleToCoefficients(const std::vector<float> &angles,
                                  std::vector<SimCoeff> &coeffs)
{
  coeffs.resize(angles.size());
  for (int i = 0; i < angles.size(); i++)
  {
    float rval = D2R(angles[i]);
    coeffs[i].a[0] = 0;
    coeffs[i].a[1] = cos(rval);
    coeffs[i].a[2] = 0;
    coeffs[i].a[3] = -sin(rval);
    coeffs[i].b[0] = 0;
    coeffs[i].b[1] = 0;
    coeffs[i].b[2] = 1;
    coeffs[i].b[3] = 0;
  }
}

void DecorateSimCoefficients(std::vector<SimCoeff> &coeffs,
                             const Geometry &geo)
{
  float alpha = -D2R(geo.pitch_angle), beta = D2R(geo.offset), t = -geo.zshift;
  float ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);

  for (int i = 0; i < coeffs.size(); i++)
  {
    float a[4], b[4];
    memcpy(a, coeffs[i].a, sizeof(float) * 4);
    memcpy(b, coeffs[i].b, sizeof(float) * 4);
    coeffs[i].a[0] = a[0];
    coeffs[i].a[1] = (a[2] * sa * sb + a[3] * ca * sb + a[1] * cb);
    coeffs[i].a[2] = (a[2] * ca - a[3] * sa);
    coeffs[i].a[3] = (a[3] * ca * cb + a[2] * cb * sa - a[1] * sb);

    coeffs[i].b[0] = b[0];
    coeffs[i].b[1] = (b[2] * sa * sb + b[3] * ca * sb + b[1] * cb);
    coeffs[i].b[2] = (b[2] * ca - b[3] * sa);
    coeffs[i].b[3] = (b[3] * ca * cb + b[2] * cb * sa - b[1] * sb);
  }

  // considering z_shift
  for (int i = 0; i < coeffs.size(); i++)
  {
    float a[4], b[4];
    memcpy(a, coeffs[i].a, sizeof(float) * 4);
    memcpy(b, coeffs[i].b, sizeof(float) * 4);

    coeffs[i].a[0] = a[0] + a[3] * t;
    coeffs[i].a[1] = a[1];
    coeffs[i].a[2] = a[2];
    coeffs[i].a[3] = a[3];

    coeffs[i].b[0] = b[0] + b[3] * t;
    coeffs[i].b[1] = b[1];
    coeffs[i].b[2] = b[2];
    coeffs[i].b[3] = b[3];
  }
}

int ATOM_GPUZ(options &opt, int myid, int procs)
{
  MrcStackM projs, mrcvol;
  if (!projs.ReadFile(opt.input))
  {
    printf("File %s cannot access.\n", opt.input);

    return -1;
  }

  if (myid == 0)
  {
    projs.ReadHeader();
  }
  MPI_Bcast(&(projs.header), sizeof(MRCheader), MPI_CHAR, 0, MPI_COMM_WORLD);

  std::vector<float> angles;
  ReadAngles(angles, opt.angle);

  std::vector<SimCoeff> params;
  TranslateAngleToCoefficients(angles, params);

  Geometry geo;
  geo.offset = opt.offset;
  geo.pitch_angle = 0;
  geo.zshift = opt.zshift;
  printf("geo= %.2f %.2f %.2f\n", geo.offset, geo.pitch_angle, geo.zshift);
  DecorateSimCoefficients(params, geo);

  mrcvol.InitializeHeader();
  mrcvol.SetSize(projs.X(), projs.Y(), opt.thickness);
  mrcvol.WriteToFile(opt.output);

  /******* along Z *******/
  int height;
  int zrem = mrcvol.Z() % procs;
  int volz; // the start slice of reproject per process
  if (myid < zrem)
  {
    height = mrcvol.Z() / procs + 1;
    volz = height * myid;
  }
  else
  {
    height = mrcvol.Z() / procs;
    volz = height * myid + zrem;
  }

  Point3DF origin;
  origin.x = projs.X() * .5;
  origin.y = projs.Y() * .5;
  origin.z = opt.thickness * .5;

  Volume vol(0, 0, volz, mrcvol.Y(), mrcvol.X(), height, NULL);
  Slice proj(projs.X(), projs.Y(), NULL);

  std::cout << "CPU th_" << myid << ": (" << vol.x << "," << vol.y << "," << vol.z << ")"
            << "&(" << vol.width << "," << vol.length << "," << vol.height
            << ")" << std::endl;

  if (myid == 0)
  {
    mrcvol.WriteHeader();
  }
  /** start cuda logic **/
  InitializeCuda(myid);


  if (opt.method == "BPT")
  {
    printf("Start using BPT for reconstruction.\n ");
    CuBackProjectZ(origin, projs, params, opt.thickness, mrcvol, proj, vol);
  }
  else if (opt.method == "FBP")
  {
    printf("Start using FBP for reconstruction. \n ");
    CuFBPZ(origin, projs, params, opt.thickness, mrcvol, proj, vol, 0);
  }
  else if (opt.method == "WBP")
  {
    printf("Start using WBP for reconstruction.\n  ");
    CuFBPZ(origin, projs, params, opt.thickness, mrcvol, proj, vol, 2);
  }
  else if (opt.method == "SIRT")
  {
    printf("Start using SIRT for reconstruction. \n ");
    CuSIRTZ(origin, projs, params, opt.thickness, mrcvol, proj, vol,
            opt.iteration, opt.gamma);
  }
  else if (opt.method == "SART")
  {
    printf("Start using SART for reconstruction.\n  ");
    CuSARTZ(origin, projs, params, opt.thickness, mrcvol, proj, vol,
            opt.iteration, opt.gamma);
  }
  else if (opt.method == "ADMM")
  {
    printf("Start using ADMM for reconstruction.\n  ");
    CuADMMZ(origin, projs, params, opt.thickness, mrcvol, proj, vol,
            opt.iteration, opt.cgiter, opt.gamma, opt.soft);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (myid == 0)
  {
    mrcvol.UpdateHeader();
  }

  projs.Close();
  mrcvol.Close();

  return 0;
}
int main(int argc, char *argv[])
{
  SysInfo info;

  MPI_Init(&argc, &argv); // parallel init
  MPI_Comm_rank(MPI_COMM_WORLD, &(info.id));
  MPI_Comm_size(MPI_COMM_WORLD, &(info.procs));
  MPI_Get_processor_name(info.processor_name, &(info.namelen));

  options opts;
  InitOpts(&opts);

  int result = GetOpts(argc, argv, &opts);
  if (result < 0)
  {
    EX_TRACE("***WRONG INPUT.\n");
    return -1;
  }
  if (result == 0)
  {
    return -1;
  }

  if (info.id == 0)
  {
    PrintOpts(opts);
  }

  ATOM_GPUZ(opts, info.id, info.procs);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize(); // parallel finish

  return 0;
}
