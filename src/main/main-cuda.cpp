#include <fstream>
#include <iostream>
#include <vector>
#include "../reconstruction_y/recon_y.cuh"
#include "../reconstruction_z/recon_z.cuh"
#include "../opts/opts.h"
#include <sstream>
#include <cassert>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>

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

void ExeRecY(options &opt, Point3DF &origin, MrcStackM &projs, std::vector<SimCoeff> &params,
             MrcStackM &mrcvol, Slice &proj, Volume &vol, int start, int length, int add_left)
{
  if (opt.method == "BPT")
  {
    printf("Start using BPT for reconstruction.\n  ");
    CuBackProject(origin, projs, params, opt.thickness, mrcvol, proj, vol, opt.pitch_angle, start, length, add_left);
  }
  else if (opt.method == "SIRT")
  {
    printf("Start using SIRT for reconstruction.\n  ");
    CuSIRT(origin, projs, params, opt.thickness, mrcvol, proj, vol, opt.iteration, opt.gamma, opt.pitch_angle, start, length, add_left);
  }
  else if (opt.method == "SART")
  {
    printf("Start using SART for reconstruction. \n ");
    CuSART(origin, projs, params, opt.thickness, mrcvol, proj, vol, opt.iteration, opt.gamma, opt.pitch_angle, start, length, add_left);
  }
  else if (opt.method == "FBP")
  {
    printf("Start using FBP for reconstruction.\n  ");
    ApplyFilterInplace(projs, proj.data, length, 0);
    CuFBP(origin, projs, params, opt.thickness, mrcvol, proj, vol, 0, opt.pitch_angle, start, length, add_left);
  }
  else if (opt.method == "WBP")
  {
    printf("Start using WBP for reconstruction.\n  ");
    ApplyFilterInplace(projs, proj.data, length, 2);
    CuFBP(origin, projs, params, opt.thickness, mrcvol, proj, vol, 2, opt.pitch_angle, start, length, add_left);
  }
}
int ATOM_GPU(options &opt, int myid, int procs)
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
  geo.pitch_angle = opt.pitch_angle;
  geo.zshift = opt.zshift;

  DecorateSimCoefficients(params, geo);

  /******* for single axis reconstruction, the partial axis is along Y *******/
  int length;
  int yrem = projs.Y() % procs;
  int start, end; // the start ane end slice of reproject per process
  int add_left = ceil(fabsf(tan(D2R(opt.pitch_angle))) * opt.thickness);
  int add_right = ceil(fabsf(tan(D2R(opt.pitch_angle))) * opt.thickness); // left和right值相同

  if (myid < yrem)
  {
    length = projs.Y() / procs + 1;
    start = length * myid;
  }
  else
  {
    length = projs.Y() / procs;
    start = length * myid + yrem;
  }
  end = start + length;

  if (start < add_left)
  {
    add_left = start;
  }
  else if (start + length < projs.Y())
  {
    add_right = projs.Y() - start - length;
  }
  else if (start + length >= projs.Y())
  {
    add_right = 0;
  }
  // std::cout << "projs.Y():" << projs.Y() << " length:" << length << std::endl;
  // std::cout << "add_left:" << add_left << "add_right:" << add_right << std::endl;
  Point3DF origin;
  origin.x = projs.X() * .5;
  origin.y = projs.Y() * .5;
  origin.z = opt.thickness * .5;

  mrcvol.InitializeHeader();
  mrcvol.SetSize(projs.X(), opt.thickness,
                 projs.Y());
  mrcvol.WriteToFile(opt.output);

  Volume vol(0, 0, start, projs.X(), opt.thickness, length,
             NULL);
  Slice proj(projs.X(), projs.Y(), NULL);

  std::cout << myid << ": (" << vol.x << "," << vol.y << "," << vol.z << ")"
            << "&(" << vol.length << "," << length << "," << opt.thickness
            << ")" << std::endl;

  if (myid == 0)
  {
    mrcvol.WriteHeader();
  }
  /** start cuda logic **/
  InitializeCuda(myid);

  length = length + add_left + add_right;

  CHECK_CUDA(cudaHostAlloc((void **)(&proj.data), sizeof(float) * projs.X() * length * projs.Z(), cudaHostAllocDefault))

  float *tmp = nullptr;
  try
  {
    tmp = new float[projs.Z() * projs.X() * length];
  }
  catch (const std::bad_alloc &e)
  {
    std::cerr << "Memory allocation failed: " << e.what() << '\n';
  }
  projs.ReadBlock(start - add_left, end + add_right, 'y', tmp);
  MrcStackM::RotateX(tmp, projs.X(), length, projs.Z(), proj.data); // transfrom to z-order
  delete[] tmp;

  if (myid == 0)
  {

    ExeRecY(opt, origin, projs, params, mrcvol, proj, vol, start, length, add_left);
  }

  float *nextData = nullptr;
  for (int block = 1; block < procs; ++block)
  {

    if (myid == 0)
    {
      int nextLength, nextStart, nextAddLeft, nextheight;
      MPI_Recv(&nextLength, 1, MPI_INT, block, block, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&nextStart, 1, MPI_INT, block, block, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&nextAddLeft, 1, MPI_INT, block, block, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&nextheight, 1, MPI_INT, block, block, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (nextData)
      {
        cudaFreeHost(nextData);
        nextData = nullptr;
      }
      CHECK_CUDA(cudaHostAlloc((void **)(&nextData), sizeof(float) * projs.X() * nextLength * projs.Z(), cudaHostAllocDefault))
      MPI_Recv(nextData, projs.X() * nextLength * projs.Z(), MPI_FLOAT, block, block, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      proj.data = nextData;
      length = nextLength;
      start = nextStart;
      vol.height = nextheight;
      add_left = nextAddLeft;

      ExeRecY(opt, origin, projs, params, mrcvol, proj, vol, start, length, add_left);
    }
    else if (myid == block) // 下一个进程
    {
      // MrcStackM::RotateX(tmp, projs.X(), length, projs.Z(), proj.data); // 其他进程旋转
      //  发送数据到零号进程

      MPI_Send(&length, 1, MPI_INT, 0, myid, MPI_COMM_WORLD);

      MPI_Send(&start, 1, MPI_INT, 0, myid, MPI_COMM_WORLD);

      MPI_Send(&add_left, 1, MPI_INT, 0, myid, MPI_COMM_WORLD);
      MPI_Send(&(vol.height), 1, MPI_INT, 0, myid, MPI_COMM_WORLD);
      // MPI_Send(tmp, projs.X() * length * projs.Z(), MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
      MPI_Send(proj.data, projs.X() * length * projs.Z(), MPI_FLOAT, 0, myid, MPI_COMM_WORLD);
    }
  }

  if (procs > 1 && myid == 0 && nextData)
  {
    CHECK_CUDA(cudaFreeHost(nextData));
  }
  else if (myid != 0)
  {
    CHECK_CUDA(cudaFreeHost(proj.data));
  }
  else if (procs == 1)
  {

    CHECK_CUDA(cudaFreeHost(proj.data));
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (myid == 0)
  {
    mrcvol.UpdateHeader();
  }
  // cudaFreeHost(proj.data);

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

  ATOM_GPU(opts, info.id, info.procs);

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize(); // parallel finish

  return 0;
}
