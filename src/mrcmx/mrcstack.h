#ifndef MRCSTACK_H__
#define MRCSTACK_H__

//#include "/usr/local/include/mpi.h"
#include "/usr/local/mpi/include/mpi.h"
#include "mrcheader.h"
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>


/** @brief MrcStack as matrix for MPI */
class MrcStackM {
public:
  MRCheader header;
  MPI_File mpifile;
  MPI_File output;

public:
  enum Mode { MODE_BYTE = 0, MODE_SHORT = 1, MODE_FLOAT = 2 };

public:
  MrcStackM() : mpifile(MPI_FILE_NULL), output(MPI_FILE_NULL) {
    header.mode = MODE_FLOAT;
  }

  ~MrcStackM() {}

  bool ReadFile(const char *filename);

  bool WriteToFile(const char *filename);

  void InitializeHeader();

  void ReadHeader();

  void WriteHeader();

  void UpdateHeader(); // only use the information from middle slice

  MRCheader &Header();

  void ReadSlice(int slcN, char axis, float *slcdata);

  void ReadSliceZ(int slcN, float *slcdata);

  void ReadSliceX(int slcN, float *slcdata);

  void ReadSliceY(int slcN, float *slcdata);

  void WriteSlice(int slcN, char axis, float *slcdata);

  void ReadBlock(int start, int end, char axis,
                 float *blockdata); // not include end

  void WriteBlock(int start, int end, char axis,
                  float *blockdata); // not include end

  void WriteBlockRotX(int start, int end, char axis,
                      float *blockdata); // not include end

  void ReadAll(float *mrcdata);

  void SetSize(int nx, int ny, int nz) {
    header.nx = nx;
    header.ny = ny;
    header.nz = nz;
  }

  int Z() const { return header.nz; }

  void SetZ(int nz) { header.nz = nz; }

  int X() const { return header.nx; }

  void SetX(int nx) { header.nx = nx; }

  int Y() const { return header.ny; }

  void SetY(int ny) { header.ny = ny; }

  void Close();

  static void RotateX(const float *blockdata, int x, int y, int z,
                      float *rotxblock);
};

struct Point3D {
  int x, y, z;
};

inline void AssignValue(Point3D &pt3, int _x, int _y, int _z) {
  pt3.x = _x;
  pt3.y = _y;
  pt3.z = _z;
}

struct Point2D {
  int x, y;
};

inline void AssignValue(Point2D &pt2, int _x, int _y) {
  pt2.x = _x;
  pt2.y = _y;
}

struct Point3DF {
  float x, y, z;
};

inline void AssignValue(Point3DF &pt3f, int _x, int _y, int _z) {
  pt3f.x = _x;
  pt3f.y = _y;
  pt3f.z = _z;
}

struct Point2DF {
  float x, y;
};

inline void AssignValue(Point2DF &pt2f, int _x, int _y) {
  pt2f.x = _x;
  pt2f.y = _y;
}

/** the order of parameters in reconstruction function should be noted
    int _x, int _y, int _z, int _length, int _width, int _height
 **/
struct Volume {
  // volume will not manage the data if _data is provided
public:
  union { // origin
    Point3D coord;
    struct {
      int x, y, z;
    };
  };

  int length; 
  int width;  
  int height;

  float *data;
  bool external;

public:
  Volume(int _length, int _width, int _height, float *_data)
      : length(_length), width(_width), height(_height), data(_data),
        external(true) {
    coord.x = 0;
    coord.y = 0, coord.z = 0;
  }

  Volume(int _length, int _width, int _height)
      : length(_length), width(_width), height(_height), external(false) {
    coord.x = 0;
    coord.y = 0, coord.z = 0;
    data = new float[length * width * height];
  }

  Volume(int _x, int _y, int _z, int _length, int _width, int _height,
         float *_data)
      : length(_length), width(_width), height(_height), data(_data),
        external(true) {
    coord.x = _x;
    coord.y = _y, coord.z = _z;
  }

  Volume(int _x, int _y, int _z, int _length, int _width, int _height)
      : length(_length), width(_width), height(_height), external(false) {
    coord.x = _x;
    coord.y = _y, coord.z = _z;
    data = new float[length * width * height];
  }

  void SetCoord(int _x, int _y, int _z) {
    x = _x;
    y = _y;
    z = _z;
  }

  ~Volume() {
    if (!external) {
      delete[] data;
    }
  }
};

struct Slice { // slice will not manage the data if _data is provided
public:
  union { // origin
    Point2D coord;
    struct {
      int x, y;
    };
  };

  int width;
  int height;

  float *data;
  bool external;

public:
  Slice(int _width, int _height, float *_data)
      : width(_width), height(_height), data(_data), external(true) {
    coord.x = 0;
    coord.y = 0;
  }

  Slice(int _width, int _height)
      : width(_width), height(_height), external(false) {
    coord.x = 0;
    coord.y = 0;
    data = new float[width * height];
  }

  Slice(int _x, int _y, int _width, int _height, float *_data)
      : width(_width), height(_height), data(_data), external(true) {
    coord.x = _x;
    coord.y = _y;
  }

  Slice(int _x, int _y, int _width, int _height)
      : width(_width), height(_height), external(false) {
    coord.x = _x;
    coord.y = _y;
    data = new float[width * height];
  }

  void SetCoord(int _x, int _y) {
    x = _x;
    y = _y;
  }

  ~Slice() {
    if (!external) {
      delete[] data;
    }
  }
};

/*computing proj by the coordinate of a 3D pixel*/
struct Weight {
  int x_min; // x coordinate of the proj
  int y_min; // y coordinate of the proj

  float x_min_del;
  float y_min_del; // weight of the proj
};

struct Geometry {
  float zshift;
  float pitch_angle;
  float offset;
};

#endif
