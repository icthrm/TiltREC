#ifndef FILTER_PRJ_H
#define FILTER_PRJ_H

// #include <opencv/cv.h>
// #include <opencv/cxcore.h>
// #include <opencv/highgui.h>

// #include <opencv/cv.h>
// #include <opencv/cxcore.h>
// #include <opencv/highgui.h>
// #include <opencv.hpp>

// #include "mrcfile_atom.h"
#include "mrcmx/mrcstack.h"

#ifndef TEXT_LINE_MAX
#define TEXT_LINE_MAX 500
#endif
// #include "mrcfiles.h"
// #include "mrcslice.h"

int FileterProjectionSymmetrize(float *prj_real, const char *filter, int filtlength, Slice &projection, int row_pad, int col_pad, int symmetrize_2D_flag, int angle = 0, int my_MPI_thread_id = 0);

int ApplyFilterInplace(MrcStackM &inputProjection, float *data,size_t ny, int mode = 0);
// int Fileter_Projection_Symmetrize_AlongX(float *prj_real, const char *filter, int filtlength, Projection prj, int row_pad, int col_pad, int symmetrize_2D_flag, int angle, int id);
int ApplyRWeightedFilterInplace(MrcStackM &inputProjection, float *data, int mode);

#endif
