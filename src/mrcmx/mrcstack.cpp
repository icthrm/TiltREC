#include "mrcstack.h"
#include <cfloat>
#include <limits>
#include <ostream>
#include <iostream>

bool MrcStackM::ReadFile(const char *filename)
{
  int rc = MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY,
                         MPI_INFO_NULL, &mpifile);
  return !rc;
}

bool MrcStackM::WriteToFile(const char *filename)
{
  int rc =
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_RDWR,
                    MPI_INFO_NULL, &output);

  if (rc != MPI_SUCCESS)
  {
    char error_string[BUFSIZ];
    int length_of_error_string;

    MPI_Error_string(rc, error_string, &length_of_error_string);

    fprintf(stderr, "%s\n", error_string);

    return false; 
  }

  size_t psize;
  switch (header.mode)
  {
  case MODE_BYTE:
    psize = sizeof(char);
    break;
  case MODE_SHORT:
    psize = sizeof(short);
    break;
  case MODE_FLOAT:
    psize = sizeof(float);
    break;

  default:
    printf("File type unknown!\n");
    return false;
  }

  MPI_File_set_size(output, sizeof(MrcHeader) + header.next +
                                psize * header.nx * header.ny * header.nz);
  MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL,
                &mpifile);
  return !rc;
}

void MrcStackM::InitializeHeader(int type)
{
  memset(&header, 0, sizeof(MRCheader));

  if (type == 1)
    header.mode = MODE_FLOAT;
  else
    header.mode = MODE_BYTE;
  header.mx = 1;
  header.my = 1;
  header.mz = 1;

  header.xlen = 1;
  header.ylen = 1;
  header.zlen = 1;

  header.alpha = 90;
  header.beta = 90;
  header.gamma = 90;

  header.mapc = 1;
  header.mapr = 2;
  header.maps = 3;

  header.amin = 0;
  header.amax = 255;
  header.amean = 128;

  header.creatid = 1000;
}

void MrcStackM::ReadHeader()
{
  MPI_File_seek(mpifile, 0, MPI_SEEK_SET);
  MPI_File_read(mpifile, &header, sizeof(MRCheader), MPI_CHAR,
                MPI_STATUS_IGNORE);
}

MRCheader &MrcStackM::Header() { return header; }

void MrcStackM::WriteHeader()
{
  MPI_File_seek(output, 0, MPI_SEEK_SET);
  MPI_File_write(output, &header, sizeof(MRCheader), MPI_CHAR,
                 MPI_STATUS_IGNORE);
}

void MrcStackM::UpdateHeader()
{
  size_t pxsize = header.nx * header.ny;
  float *slcdata = new float[pxsize];

  printf("Update head!\n");

  float amin = FLT_MAX;
  float amax = FLT_MIN;

  long double amean = 0;

  // read the first slice
  ReadSliceZ(header.nz * .5, slcdata);

  for (size_t i = pxsize; i--;)
  {
    if (slcdata[i] > amax)
      amax = slcdata[i];
    if (slcdata[i] < amin)
      amin = slcdata[i];
    amean += slcdata[i];
  }
  amean /= pxsize;

  printf("amin is %f, amax is %f, amean is %Lf\n", amin, amax, amean);

  header.amin = amin;
  header.amax = amax;
  header.amean = amean;
  header.nlabl = 0;

  MPI_File_seek(output, 0, MPI_SEEK_SET);
  MPI_File_write(output, &header, sizeof(MRCheader), MPI_CHAR,
                 MPI_STATUS_IGNORE);

  delete[] slcdata;
  printf("Updating finished!\n");
}

void MrcStackM::ReadSlice(int slcN, char axis, float *slcdata)
{
  MPI_Offset glboffset = sizeof(MRCheader) + header.next;
  int psize;

  switch (header.mode)
  {
  case MODE_BYTE:
    psize = sizeof(char);

    break;

  case MODE_SHORT:
    psize = sizeof(short);

    break;

  case MODE_FLOAT:
    psize = sizeof(float);

    break;

  default:
    printf("File type unknown!\n");
    return;
  }

  char *buf;
  size_t bufsize;
  char *cur;

  size_t slcsize = header.nx * header.ny * psize;

  switch (axis)
  {
  case 'z':
  case 'Z':
    bufsize = header.nx * header.ny;
    buf = new char[bufsize * psize];

    MPI_File_seek(mpifile, glboffset + slcN * slcsize, MPI_SEEK_SET);
    MPI_File_read(mpifile, buf, slcsize, MPI_CHAR, MPI_STATUS_IGNORE);
    break;

  case 'x':
  case 'X':
    bufsize = header.nz * header.ny;
    buf = new char[bufsize * psize];
    cur = buf;

    for (int k = 0; k < header.nz; k++)
    {
      MPI_Offset zoffset = glboffset + k * slcsize;
      for (int j = 0; j < header.ny; j++)
      {
        MPI_File_seek(mpifile, zoffset + j * header.nx * psize + slcN,
                      MPI_SEEK_SET);
        MPI_File_read(mpifile, cur, psize, MPI_CHAR, MPI_STATUS_IGNORE);
        cur += psize;
      }
    }
    break;

  case 'y':
  case 'Y':
    bufsize = header.nz * header.nx;
    buf = new char[bufsize * psize];

    for (int k = 0; k < header.nz; k++)
    {
      MPI_File_seek(mpifile, glboffset + slcN * header.nx * psize + k * slcsize,
                    MPI_SEEK_SET);
      MPI_File_read(mpifile, buf + k * header.nx * psize, header.nx * psize,
                    MPI_CHAR, MPI_STATUS_IGNORE);
    }
    break;

  default:
    break;
  }

  switch (header.mode)
  {
  case MODE_BYTE:
    for (int i = bufsize; i--;)
    {
      slcdata[i] = ((unsigned char *)buf)[i];
    }

    break;

  case MODE_SHORT:
    for (int i = bufsize; i--;)
    {
      slcdata[i] = ((short *)buf)[i];
    }

    break;

  case MODE_FLOAT:
    memcpy(slcdata, buf, sizeof(float) * bufsize);

    break;

  default:
    printf("File type unknown!\n");
  }

  delete[] buf;
}

void MrcStackM::ReadSliceX(int slcN, float *slcdata)
{
  ReadSlice(slcN, 'x', slcdata);
}

void MrcStackM::ReadSliceY(int slcN, float *slcdata)
{
  ReadSlice(slcN, 'y', slcdata);
}

void MrcStackM::ReadSliceZ(int slcN, float *slcdata)
{
  ReadSlice(slcN, 'z', slcdata);
}

void MrcStackM::WriteSlice(int slcN, char axis, float *slcdata)
{
  int psize = sizeof(float);
  MPI_Offset offset = sizeof(MrcHeader) + header.next;
  MPI_File_seek(output, offset, MPI_SEEK_SET);

  switch (axis)
  {
  case 'x':
  case 'X':
    offset = slcN * psize;
    MPI_File_seek(output, offset, MPI_SEEK_CUR);
    for (int k = 0; k < header.nz; k++)
    {
      for (int j = 0; j < header.ny; j++)
      {
        MPI_File_write(output, slcdata + k * header.ny + j, 1, MPI_FLOAT,
                       MPI_STATUS_IGNORE);
        offset = (header.nx - 1) * psize;
        MPI_File_seek(output, offset, MPI_SEEK_CUR);
      }
    }
    break;

  case 'y':
  case 'Y':
    offset = header.nx * slcN * psize;
    MPI_File_seek(output, offset, MPI_SEEK_CUR);
    for (int k = 0; k < header.nz; k++)
    {
      MPI_File_write(output, slcdata + k * header.nx, header.nx, MPI_FLOAT,
                     MPI_STATUS_IGNORE);
      offset = header.nx * (header.ny - 1) * psize;
      MPI_File_seek(output, offset, MPI_SEEK_CUR);
    }
    break;

  case 'z':
  case 'Z':
    offset = header.nx * header.ny * psize;
    for (int k = 0; k < slcN; k++)
    {
      MPI_File_seek(output, offset, MPI_SEEK_CUR);
    }
    MPI_File_write(output, slcdata, header.nx * header.ny, MPI_FLOAT,
                   MPI_STATUS_IGNORE);
    break;
  }
}

void MrcStackM::ReadBlock(int start, int end, char axis, float *blockdata)
{
  MPI_Offset glboffset = sizeof(MRCheader) + header.next;
  int psize;

  switch (header.mode)
  {
  case MODE_BYTE:
    psize = sizeof(char);

    break;

  case MODE_SHORT:
    psize = sizeof(short);

    break;

  case MODE_FLOAT:
    psize = sizeof(float);

    break;

  default:
    printf("File type unknown!\n");
    return;
  }

  char *buf;
  size_t bufsize;
  int slcN = start;
  int thickness = end - start; // slice_step

  size_t slcsize = header.nx * header.ny * psize;

  switch (axis)
  {
  case 'z':
  case 'Z':
    bufsize = header.nx * header.ny * thickness;
    buf = new char[bufsize * psize];

    MPI_File_seek(mpifile, glboffset + slcN * slcsize, MPI_SEEK_SET);
    MPI_File_read(mpifile, buf, slcsize * thickness, MPI_CHAR,
                  MPI_STATUS_IGNORE);
    break;

  case 'x':
  case 'X':
    bufsize = header.nz * header.ny * thickness;
    buf = new char[bufsize * psize];
    for (slcN; slcN < end; slcN++)
    {
      char *cur = buf + (slcN - start) * header.nz * header.ny * psize;
      ;

      for (int k = 0; k < header.nz; k++)
      {
        MPI_Offset zoffset = glboffset + k * slcsize;
        for (int j = 0; j < header.ny; j++)
        {
          MPI_File_seek(mpifile, zoffset + j * header.nx * psize + slcN,
                        MPI_SEEK_SET);
          MPI_File_read(mpifile, cur, psize, MPI_CHAR, MPI_STATUS_IGNORE);
          cur += psize;
        }
      }
    }
    break;

  case 'y':
  case 'Y':
    bufsize = header.nz * header.nx * thickness;
    buf = new char[bufsize * psize];
    for (slcN; slcN < end; slcN++)
    {
      char *cur = buf + (slcN - start) * header.nz * header.nx * psize;

      for (int k = 0; k < header.nz; k++)
      {
        MPI_File_seek(mpifile,
                      glboffset + slcN * header.nx * psize + k * slcsize,
                      MPI_SEEK_SET);
        MPI_File_read(mpifile, cur + k * header.nx * psize, header.nx * psize,
                      MPI_CHAR, MPI_STATUS_IGNORE);
      }
    }
    break;

  default:
    break;
  }

  switch (header.mode)
  {
  case MODE_BYTE:
    for (int i = bufsize; i--;)
    {
      blockdata[i] = ((unsigned char *)buf)[i];
    }

    break;

  case MODE_SHORT:
    for (int i = bufsize; i--;)
    {
      blockdata[i] = ((short *)buf)[i];
    }

    break;

  case MODE_FLOAT:
    memcpy(blockdata, buf, sizeof(float) * bufsize);

    break;

  default:
    printf("File type unknown!\n");
  }

  delete[] buf;
}

template <>
MPI_Datatype MrcStackM::get_mpi_type<unsigned char>()
{
  return MPI_UNSIGNED_CHAR;
}

template <>
MPI_Datatype MrcStackM::get_mpi_type<float>()
{
  return MPI_FLOAT;
}
template <typename T>
void MrcStackM::WriteBlock(int start, int end, char axis, float *blockdata)
{
    int psize = sizeof(T);
  size_t del = end - start;

  MPI_Offset offset = sizeof(MrcHeader) + header.next;
  MPI_File_seek(output, offset, MPI_SEEK_SET);
  MPI_Datatype mpi_type = get_mpi_type<T>();

  switch (axis)
  {
  case 'x':
  case 'X':
    offset = start * psize;
    MPI_File_seek(output, offset, MPI_SEEK_CUR);
    for (int k = 0; k < header.nz; k++)
    {
      for (int j = 0; j < header.ny; j++)
      {
        MPI_File_write(output, blockdata + j * del + k * del * header.ny, del,
                       mpi_type, MPI_STATUS_IGNORE);
        offset = (header.nx - del) * psize;
        MPI_File_seek(output, offset, MPI_SEEK_CUR);
      }
    }
    break;

  case 'y':
  case 'Y':
    offset = header.nx * start * psize;
    MPI_File_seek(output, offset, MPI_SEEK_CUR);
    for (int k = 0; k < header.nz; k++)
    {
      MPI_File_write(output, blockdata + k * del * header.nx, del * header.nx,
                     mpi_type, MPI_STATUS_IGNORE);
      offset = header.nx * (header.ny - del) * psize;
      MPI_File_seek(output, offset, MPI_SEEK_CUR);
    }
    break;

  case 'z':
  case 'Z':
    offset = header.nx * header.ny * psize;
    MPI_File_seek(output, start * offset, MPI_SEEK_CUR);
    MPI_File_write(output, blockdata, header.nx * header.ny * del, mpi_type,
                   MPI_STATUS_IGNORE);
    break;
  }
}
// 显式实例化模板函数
template void MrcStackM::WriteBlock<float>(int, int, char, float *);
template void MrcStackM::WriteBlock<unsigned char>(int, int, char, float *);

void MrcStackM::WriteBlockRotX(int start, int end, char axis,
                               float *blockdata)
{
  int psize = sizeof(float);
  size_t del = end - start;

  MPI_Offset offset = sizeof(MrcHeader) + header.next;
  MPI_File_seek(output, offset, MPI_SEEK_SET);

  std::cout << "write block (" << start << "," << end << ")" << std::endl;

  switch (axis)
  {
  case 'x':
  case 'X':
    break;

  case 'y':
  case 'Y':
  {
    size_t slcsize = header.nx * header.nz;
    for (int slcN = start; slcN < end; slcN++)
    {
      for (int k = 0; k < header.ny; k++)
      {
        float *cur = blockdata + (slcN - start) * header.nx + k * slcsize;
        MPI_File_write(output, cur, header.nx, MPI_FLOAT, MPI_STATUS_IGNORE);
      }
    }
  }

  break;

  case 'z':
  case 'Z':
    break;
  }
}

/**not inplace; x, y, z-use the dest order**/
void MrcStackM::RotateX(const float *blockdata, int x, int y, int z,
                        float *rotxblock)
{
  size_t slcsize = x * z;
  for (int i = 0; i < z; i++)
  {
    for (int k = 0; k < y; k++)
    {
      const float *cur = blockdata + i * x + k * slcsize;
      memcpy(rotxblock, cur, sizeof(float) * x);
      rotxblock += x;
    }
  }
}

void MrcStackM::ReadAll(float *mrcdata)
{
  size_t xysize = header.nx * header.ny;
  for (int i = 0; i < header.nz; i++)
  {
    ReadSliceZ(i, mrcdata + xysize);
  }
}

void MrcStackM::Close()
{
  if (mpifile != MPI_FILE_NULL)
  {
    MPI_File_close(&mpifile);
    mpifile = MPI_FILE_NULL;
  }
  if (output != MPI_FILE_NULL)
  {
    MPI_File_close(&output);
    output = MPI_FILE_NULL;
  }
}
