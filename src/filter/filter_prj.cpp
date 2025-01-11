#include <opencv4/opencv2/imgproc/imgproc_c.h>
#include <opencv4/opencv2/imgproc/types_c.h>
#include <opencv4/opencv2/highgui/highgui_c.h>
#include <opencv4/opencv2/core/core_c.h>
// #include "/usr/local/cuda/include/cufftw.h"
#include <cuda.h>
#include <cufftw.h>
#include "filter_prj.h"

///////////////////////////////// OpenCV utilities: CvMat.
CvMat *create_CvMat_32F_from_float_data(int rows, int cols, float *data)
{
    CvMat *matrix = cvCreateMatHeader(rows, cols, CV_32FC1);

    int n_bytes_row = cols * sizeof(float);
    cvSetData(matrix, data, n_bytes_row);

    return matrix;
}

/////////////////////////////
// void write_MRC_image_from_CvMat(CvMat *Mat, char *filepath, int angleIdx)
// {
//     int i, j;

//     MPI_File fout;
//     MPI_File_open(MPI_COMM_WORLD, const_cast<char *>(filepath), MPI_MODE_CREATE | MPI_MODE_RDWR,
//                   MPI_INFO_NULL, &fout);

//     MrcHeader *outhead;

//     outhead = (MrcHeader *)malloc(sizeof(MrcHeader));
//     if (angleIdx == 0)
//         printf("Mat->cols is %d, Mat->rows is %d\n", Mat->cols, Mat->rows);

//     mrc_init_head(outhead);
//     outhead->nx = Mat->cols;
//     outhead->ny = Mat->rows;
//     outhead->nz = 61;

//     if (angleIdx == 0)
//         mrc_write_head(fout, outhead);

//     size_t projectionSliceSize = Mat->cols * Mat->rows;

//     float *prj_slice;
//     if ((prj_slice = (float *)malloc(sizeof(float) * projectionSliceSize)) == NULL)
//     {
//         printf(
//             "Error with Function 'filter_prj()'!Can't malloc memery for 'prj_slice'!");
//         exit(1);
//     }
//     memset(prj_slice, 0, sizeof(float) * projectionSliceSize);

//     for (j = 0; j < Mat->rows; j++)

//         for (i = 0; i < Mat->cols; i++)
//             prj_slice[i + j * Mat->cols] = cvGetReal2D(Mat, j, i);
//     // prj_slice[i+j*Mat->cols]=0;

//     mrc_write_slice(fout, outhead, angleIdx, 'Z', prj_slice);

//     free(outhead);
//     outhead = NULL;

//     MPI_File_close(&fout);
//     free(prj_slice);
// }

/////////////////////////////
// void write_MRC_image_from_IplImage(IplImage *image, char *filepath, int angleIdx)
// {
//     int i, j;

//     MPI_File fout;
//     MPI_File_open(MPI_COMM_WORLD, const_cast<char *>(filepath), MPI_MODE_CREATE | MPI_MODE_RDWR,
//                   MPI_INFO_NULL, &fout);

//     MrcHeader *outhead;

//     outhead = (MrcHeader *)malloc(sizeof(MrcHeader));
//     mrc_init_head(outhead);
//     outhead->nx = image->width;
//     outhead->ny = image->height;
//     outhead->nz = 61;
//     outhead->xorg = 0.0;
//     outhead->yorg = 0.0;
//     outhead->zorg = 0.0;

//     if (angleIdx == 0)
//     {
//         mrc_write_head(fout, outhead);
//         PRINT_DEBUG_MSG("outhead->nx is %d, outhead->ny is %d, outhead->nz is %d\n",
//                         outhead->nx, outhead->ny, outhead->nz);
//     }

//     unsigned size_t projectionSliceSize = image->width * image->height;

//     float *prj_slice;
//     if ((prj_slice = (float *)malloc(sizeof(float) * projectionSliceSize)) == NULL)
//     {
//         printf(
//             "Error with Function 'filter_prj()'!Can't malloc memery for 'prj_slice'!");
//         exit(1);
//     }
//     memset(prj_slice, 0, sizeof(float) * projectionSliceSize);

//     /*  float *p_image;

//      for(j=0;j<image->height;j++)
//      {
//      p_image = (float *) (image->imageData+j*image->widthStep);
//      for(i=0;i<image->width;i++)
//      {
//      prj_slice[i+j*image->width]=*p_image++;
//      }
//      }*/

//     for (j = 0; j < image->height; j++)
//     {

//         for (i = 0; i < image->width; i++)
//             prj_slice[i + j * image->width] = cvGetReal2D(image, j, i);
//     }

//     mrc_write_slice(fout, outhead, angleIdx, 'Z', prj_slice);

//     free(outhead);
//     outhead = NULL;

//     MPI_File_close(&fout);

//     free(prj_slice);
// }

/////////////////////////////RamLak
int RamLak(int width, float *ramlak)
{
    int n;
    n = -(width - 1) / 2;

    int i;
    for (i = 0; i < width; i++)
    {
        if (n == 0)
            ramlak[i] = M_PI / 4;
        else if (abs(n % 2) == 1)
            ramlak[i] = -1.0 / (M_PI * n * n);
        else
            ramlak[i] = 0;
        n++;
    }
    return 1;
}

//////////////////////////////SheppLogan
int SheppLogan(int length, float *shepp)
{
    if (length % 2 != 1)
    {
        printf("length %d is not Odd\n", length);
        exit(1);
    }

    float n;
    n = -((float)length - 1.0) / 2.0;

    int i;
    for (i = 0; i < length; i++)
    {
        shepp[i] = (float)(-2.0 / (M_PI * (4.0 * n * n - 1.0)));
        n = n + 1.0;

        //  printf("shepp[%d] is %f\n",i,shepp[i]);
    }

    return 1;
}

//////////////////////////////////
int max(int i, int j)
{
    if (i >= j)
        return i;
    else
        return j;
}

float lg2(float n)
{
    return log(n) / log(2);
}

/*****************************************************************************************************/
// void Show_Mat2D(CvMat *mat, int row, int col)
// {
//     int i, j;
//     for (i = 0; i < row; i++)
//     {
//         for (j = 0; j < col; j++)
//         {
//             PRINT_DEBUG_MSG("%lf  ", cvGet2D(mat, i, j).val[0]);
//         }
//         PRINT_DEBUG_MSG("\n");
//     }
// }

/*****************************************************************************************************/
void symmetrize_IplImage_values_CvRect(IplImage *image, CvRect rect)
{
    if ((image->depth != IPL_DEPTH_32F) && (image->depth != IPL_DEPTH_64F))
    {
        printf(
            "Expected image->depth = %i (IPL_DEPTH_32F) or %i (IPL_DEPTH_64F), but image->depth = %i.\n",
            IPL_DEPTH_32F, IPL_DEPTH_64F, image->depth);
        exit(1);
    }

    // write_MRC_image_from_IplImage(image, "/home/akulo/filter_1D_images/image.mrc");

    int n_x = image->width;
    int n_y = image->height;

    CvMat matrix_src;
    CvMat matrix_dst;

    // BEGIN: Top center.
    // NOTE: A flip about x.
    cvGetSubRect(image, &matrix_src,
                 cvRect(rect.x,
                        (rect.y + rect.height) - (n_y - (rect.y + rect.height)),
                        rect.width, (n_y - (rect.y + rect.height))));
    cvGetSubRect(image, &matrix_dst,
                 cvRect(rect.x, rect.y + rect.height, rect.width,
                        (n_y - (rect.y + rect.height))));
    cvFlip(&matrix_src, &matrix_dst, 0);
    // END: Top center.

    // BEGIN: Bottom center.
    // NOTE: A flip about x.
    cvGetSubRect(image, &matrix_src,
                 cvRect(rect.x, rect.y, rect.width, rect.y));
    cvGetSubRect(image, &matrix_dst, cvRect(rect.x, 0, rect.width, rect.y));
    cvFlip(&matrix_src, &matrix_dst, 0);
    // END: Bottom center.

    // BEGIN: Left top, center, and bottom.
    // NOTE: A flip about y.
    cvGetSubRect(image, &matrix_src, cvRect(rect.x, 0, rect.x, n_y));
    cvGetSubRect(image, &matrix_dst, cvRect(0, 0, rect.x, n_y));
    cvFlip(&matrix_src, &matrix_dst, 1);
    // END: Left top, center, and bottom.

    // BEGIN: Right top, center, and bottom.
    // NOTE: A flip about y.
    cvGetSubRect(image, &matrix_src,
                 cvRect((rect.x + rect.width) - (n_x - (rect.x + rect.width)), 0,
                        n_x - (rect.x + rect.width), n_y));
    cvGetSubRect(image, &matrix_dst,
                 cvRect(rect.x + rect.width, 0, n_x - (rect.x + rect.width), n_y));
    cvFlip(&matrix_src, &matrix_dst, 1);
    // END: Right top, center, and bottom.
}

/*****************************************************************************************************/
int FileterProjectionSymmetrize(float *prj_real, const char *filter, int filterLength,
                                Slice &projection, int row_pad, int col_pad, int symmetrize_2D_flag,
                                int angleIdx, int my_MPI_thread_id)
{
    int i, j;

    int n_x_pad = projection.x + row_pad;
    int n_y_pad = projection.y + col_pad;

    int n_x_pad_conv = filterLength + n_x_pad - 1;
    int n_x_pad_conv_opt = cvGetOptimalDFTSize(n_x_pad_conv);

    int n_y_pad_opt = cvGetOptimalDFTSize(n_y_pad);

    // if (my_MPI_thread_id == 0 && angleIdx == 0)
    //     printf("n_x_pad is %d, n_y_pad is %d, n_x_pad_conv is %d, n_x_pad_conv_opt is %d, n_y_pad_opt is %d\n", n_x_pad, n_y_pad, n_x_pad_conv, n_x_pad_conv_opt, n_y_pad_opt);

    size_t projectionSliceSize = projection.x * projection.y;
    float *prj_slice;
    if ((prj_slice = (float *)malloc(sizeof(float) * projectionSliceSize)) == NULL)
    {
        printf(
            "Error with Function 'filter_prj()'!Can't malloc memery for 'prj_slice'!");
        return 0;
    }
    memset(prj_slice, 0, sizeof(float) * projectionSliceSize);

    for (i = 0; i < projectionSliceSize; i++)
        prj_slice[i] = prj_real[i];

    // original projection (projection.y*projection.x)
    IplImage *projection_org = cvCreateImageHeader(cvSize(projection.x, projection.y),
                                                   IPL_DEPTH_32F, 1);

    int n_bytes_row = projection.x * sizeof(float);
    cvSetData(projection_org, prj_slice, n_bytes_row);


    // pad projection (n_y_pad*n_x_pad)
    IplImage *projection_pad = cvCreateImage(cvSize(n_x_pad, n_y_pad),
                                             IPL_DEPTH_32F, 1);
    cvSetZero(projection_pad);

    CvRect projection_pad_center_ROI = cvRect(row_pad / 2, col_pad / 2, projection.x,
                                              projection.y);
    // printf("projection.x=%d projection.y=%d \n", projection.x, projection.y);
    cvSetImageROI(projection_org, cvRect(0, 0, projection.x, projection.y));
    cvSetImageROI(projection_pad, projection_pad_center_ROI); // NOTE: The extents of this ROI are the same as image0's ROI.
    cvCopy(projection_org, projection_pad, NULL);
    cvResetImageROI(projection_org);
    cvResetImageROI(projection_pad);

    // padding projection_pad using symmetrize
    if (symmetrize_2D_flag)
        symmetrize_IplImage_values_CvRect(projection_pad,
                                          projection_pad_center_ROI);

    // filter
    float *filter_rs = (float *)malloc(filterLength * sizeof(float)); // filter_rs (1*251)
    if (filter_rs == NULL)
    {
        printf("Memory request for filter_rs failed .\n");
        exit(1);
    }

    /* if (!strcmp(filter,"RamLak"))
     RamLak(filterLength, filter_rs);
     else if (!strcmp(filter,"SheppLogan"))
     SheppLogan(filterLength,filter_rs);
     else
     printf("Invalid filter selected.\n");*/
    if (!strcmp(filter, "RamLak"))
        RamLak(filterLength, filter_rs);
    else if (!strcmp(filter, "SheppLogan"))
        SheppLogan(filterLength, filter_rs);
    else
        std::cout << "Invalid filter type.\n"
                  << std::endl;

    // filter_rs_cv (1*251)
    CvMat *filter_rs_cv = create_CvMat_32F_from_float_data(1, filterLength,
                                                           filter_rs);

    // Debug:
    /* Show_Mat2D(filter_rs_cv,1,filterLength);

     exit(0);*/

    // filter_fs_cv (1*n_x_pad_cov_opt)
    CvMat *filter_fs_cv = cvCreateMat(1, n_x_pad_conv_opt, CV_32FC1);
    cvZero(filter_fs_cv);

    CvMat tmp;
    cvGetSubRect(filter_fs_cv, &tmp, cvRect(0, 0, filterLength, 1));
    cvCopy(filter_rs_cv, &tmp, NULL);

    cvDFT(filter_fs_cv, filter_fs_cv, CV_DXT_FORWARD | CV_DXT_ROWS, 1);

    // filter_fs_cv (n_y_pad_opt*n_x_pad_cov_opt)
    CvMat *filter_2D_fs_cv = cvCreateMat(n_y_pad_opt, n_x_pad_conv_opt,
                                         CV_32FC1);
    cvZero(filter_2D_fs_cv);

    cvRepeat(filter_fs_cv, filter_2D_fs_cv);

    // convolution projection_pad and filter_2D_fs_cv into projection_conv
    CvMat *projection_conv = cvCreateMat(n_y_pad_opt, n_x_pad_conv_opt,
                                         CV_32FC1);
    cvZero(projection_conv);

    CvMat matrix_src;
    CvMat matrix_dst;

    cvGetSubRect(projection_conv, &matrix_dst, cvRect(0, 0, n_x_pad, n_y_pad));
    cvCopy(projection_pad, &matrix_dst, NULL);

    if (symmetrize_2D_flag)
    {

        int i, j;
        if (n_x_pad >= n_x_pad_conv_opt - n_x_pad)
        {
            for (j = 0; j < n_y_pad_opt; j++)
                for (i = n_x_pad; i < n_x_pad_conv_opt; i++)
                    cvmSet(projection_conv, j, i,
                           cvmGet(projection_conv, j, 2 * n_x_pad - 1 - i));
        }
        else
        {

            for (j = 0; j < n_y_pad_opt; j++)
                for (i = n_x_pad; i < 2 * n_x_pad; i++)
                    cvmSet(projection_conv, j, i,
                           cvmGet(projection_conv, j, 2 * n_x_pad - 1 - i));
        }
    }

    cvDFT(projection_conv, projection_conv, CV_DXT_FORWARD | CV_DXT_ROWS,
          n_y_pad);

    cvMulSpectrums(projection_conv, filter_2D_fs_cv, projection_conv,
                   CV_DXT_ROWS);

    cvDFT(projection_conv, projection_conv, CV_DXT_INV_SCALE | CV_DXT_ROWS,
          n_y_pad);

    cvGetSubRect(projection_conv, &matrix_src,
                 cvRect(filterLength / 2, 0, projection_pad->width,
                        projection_pad->height));
    cvCopy(&matrix_src, projection_pad, NULL);

    float *p_image;

    for (j = 0; j < projection_pad_center_ROI.height; j++)
    {
        p_image =
            (float *)(projection_pad->imageData + projection_pad_center_ROI.x * sizeof(float) + (projection_pad_center_ROI.y + j) * projection_pad->widthStep);

        for (i = 0; i < projection_pad_center_ROI.width; i++)
        {
            prj_slice[i + j * projection.x] = *p_image++;
        }
    }

    for (i = 0; i < projectionSliceSize; i++)
        prj_real[i] = prj_slice[i];

    free(prj_slice);

    cvReleaseImageHeader(&projection_org);
    cvReleaseImage(&projection_pad);

    free(filter_rs);

    cvReleaseMat(&filter_rs_cv);
    cvReleaseMat(&filter_fs_cv);
    cvReleaseMat(&filter_2D_fs_cv);
    cvReleaseMat(&projection_conv);

    return 1;
}

/*****************************************************************************************************/
int FileterProjectionSymmetrize_AlongX(float *prj_real, char *filter, int filterLength,
                                       Slice &projection, int row_pad, int col_pad, int symmetrize_2D_flag,
                                       int angleIdx, int my_MPI_thread_id)
{
    int i, j;

    int n_x_pad = projection.x + row_pad;
    int n_y_pad = projection.y + col_pad;

    int n_x_pad_conv = filterLength + n_x_pad - 1;
    int n_x_pad_conv_opt = cvGetOptimalDFTSize(n_x_pad_conv);

    int n_y_pad_opt = cvGetOptimalDFTSize(n_y_pad);

    // if (my_MPI_thread_id == 0 && angleIdx == 0)
    //     printf("n_x_pad is %d, n_y_pad is %d, n_x_pad_conv is %d, n_x_pad_conv_opt is %d, n_y_pad_opt is %d\n", n_x_pad, n_y_pad, n_x_pad_conv, n_x_pad_conv_opt, n_y_pad_opt);

    size_t projectionSliceSize = projection.x * projection.y;
    float *prj_slice;
    if ((prj_slice = (float *)malloc(sizeof(float) * projectionSliceSize)) == NULL)
    {
        printf(
            "Error with Function 'filter_prj()'!Can't malloc memery for 'prj_slice'!");
        return 0;
    }
    memset(prj_slice, 0, sizeof(float) * projectionSliceSize);

    /* for(i=0;i<projectionSliceSize;i++)
     prj_slice[i]=prj_real[i+angleIdx*projectionSliceSize];*/

    for (i = 0; i < projection.x; i++)
        for (j = 0; j < projection.y; j++)
            prj_slice[j + i * projection.y] = prj_real[i + j * projection.x];

    // original projection (projection.y*projection.x)
    IplImage *projection_org = cvCreateImageHeader(cvSize(projection.x, projection.y),
                                                   IPL_DEPTH_32F, 1);

    int n_bytes_row = projection.x * sizeof(float);
    cvSetData(projection_org, prj_slice, n_bytes_row);


    // pad projection (n_y_pad*n_x_pad)
    IplImage *projection_pad = cvCreateImage(cvSize(n_x_pad, n_y_pad),
                                             IPL_DEPTH_32F, 1);
    cvSetZero(projection_pad);

    CvRect projection_pad_center_ROI = cvRect(row_pad / 2, col_pad / 2, projection.x,
                                              projection.y);

    cvSetImageROI(projection_org, cvRect(0, 0, projection.x, projection.y));
    cvSetImageROI(projection_pad, projection_pad_center_ROI); // NOTE: The extents of this ROI are the same as image0's ROI.
    cvCopy(projection_org, projection_pad, NULL);
    cvResetImageROI(projection_org);
    cvResetImageROI(projection_pad);


    // padding projection_pad using symmetrize
    if (symmetrize_2D_flag)
        symmetrize_IplImage_values_CvRect(projection_pad,
                                          projection_pad_center_ROI);


    // filter
    float *filter_rs = (float *)malloc(filterLength * sizeof(float)); // filter_rs (1*251)
    if (filter_rs == NULL)
    {
        printf("Memory request for filter_rs failed .\n");
        exit(1);
    }

    if (!strcmp(filter, "RamLak"))
        RamLak(filterLength, filter_rs);
    else if (!strcmp(filter, "SheppLogan"))
        SheppLogan(filterLength, filter_rs);
    else
        printf("Invalid filter selected.\n");

    // filter_rs_cv (1*251)
    CvMat *filter_rs_cv = create_CvMat_32F_from_float_data(1, filterLength,
                                                           filter_rs);

    // Debug:
    /* Show_Mat2D(filter_rs_cv,1,filterLength);

     exit(0);*/

    // filter_fs_cv (1*n_x_pad_cov_opt)
    CvMat *filter_fs_cv = cvCreateMat(1, n_x_pad_conv_opt, CV_32FC1);
    cvZero(filter_fs_cv);

    CvMat tmp;
    cvGetSubRect(filter_fs_cv, &tmp, cvRect(0, 0, filterLength, 1));
    cvCopy(filter_rs_cv, &tmp, NULL);

    cvDFT(filter_fs_cv, filter_fs_cv, CV_DXT_FORWARD | CV_DXT_ROWS, 1);

    // filter_fs_cv (n_y_pad_opt*n_x_pad_cov_opt)
    CvMat *filter_2D_fs_cv = cvCreateMat(n_y_pad_opt, n_x_pad_conv_opt,
                                         CV_32FC1);
    cvZero(filter_2D_fs_cv);

    cvRepeat(filter_fs_cv, filter_2D_fs_cv);

    // convolution projection_pad and filter_2D_fs_cv into projection_conv
    CvMat *projection_conv = cvCreateMat(n_y_pad_opt, n_x_pad_conv_opt,
                                         CV_32FC1);
    cvZero(projection_conv);

    CvMat matrix_src;
    CvMat matrix_dst;

    cvGetSubRect(projection_conv, &matrix_dst, cvRect(0, 0, n_x_pad, n_y_pad));
    cvCopy(projection_pad, &matrix_dst, NULL);

    if (symmetrize_2D_flag)
    {
        /*  cvGetSubRect(projection_pad, &matrix_src, cvRect(n_x_pad - (n_x_pad_conv_opt - n_x_pad), 0, n_x_pad_conv_opt - n_x_pad, n_y_pad));
         cvFlip(&matrix_src, &matrix_dst, 1);// END: Symmetrize right.  Only a single mirrored copy is necessary.*/

        int i, j;
        if (n_x_pad >= n_x_pad_conv_opt - n_x_pad)
        {
            for (j = 0; j < n_y_pad_opt; j++)
                for (i = n_x_pad; i < n_x_pad_conv_opt; i++)
                    cvmSet(projection_conv, j, i,
                           cvmGet(projection_conv, j, 2 * n_x_pad - 1 - i));
        }
        else
        {

            for (j = 0; j < n_y_pad_opt; j++)
                for (i = n_x_pad; i < 2 * n_x_pad; i++)
                    cvmSet(projection_conv, j, i,
                           cvmGet(projection_conv, j, 2 * n_x_pad - 1 - i));
        }
    }

    cvDFT(projection_conv, projection_conv, CV_DXT_FORWARD | CV_DXT_ROWS,
          n_y_pad);

    cvMulSpectrums(projection_conv, filter_2D_fs_cv, projection_conv,
                   CV_DXT_ROWS);

    cvDFT(projection_conv, projection_conv, CV_DXT_INV_SCALE | CV_DXT_ROWS,
          n_y_pad);

    cvGetSubRect(projection_conv, &matrix_src,
                 cvRect(filterLength / 2, 0, projection_pad->width,
                        projection_pad->height));
    cvCopy(&matrix_src, projection_pad, NULL);

    float *p_image;

    for (j = 0; j < projection_pad_center_ROI.height; j++)
    {
        p_image =
            (float *)(projection_pad->imageData + projection_pad_center_ROI.x * sizeof(float) + (projection_pad_center_ROI.y + j) * projection_pad->widthStep);

        for (i = 0; i < projection_pad_center_ROI.width; i++)
        {
            prj_slice[i + j * projection.x] = *p_image++;
        }
    }

    /* for(i=0;i<projectionSliceSize;i++)
     prj_real[i+angleIdx*projectionSliceSize]=prj_slice[i];*/

    for (i = 0; i < projection.x; i++)
        for (j = 0; j < projection.y; j++)
            prj_real[i + j * projection.x] = prj_slice[j + i * projection.y];

    free(prj_slice);

    cvReleaseImageHeader(&projection_org);
    cvReleaseImage(&projection_pad);

    free(filter_rs);

    cvReleaseMat(&filter_rs_cv);
    cvReleaseMat(&filter_fs_cv);
    cvReleaseMat(&filter_2D_fs_cv);
    cvReleaseMat(&projection_conv);

    return 1;
}

static void fft2buf_padding_1D(float *buf, float *fft, int nx_orig, int nx_final, int ny_orig, int ny_final)
{
    int nxb = nx_final + 2 - nx_final % 2;
    int i;
#pragma omp parallel for num_threads(threadNumber)
    for (i = 0; i < nx_orig * ny_orig; i++)
    {
        buf[i] = 0.0;
    }
#pragma omp parallel for num_threads(threadNumber)
    for (i = 0; i < ny_orig; i++)
    {
        memcpy(buf + i * nx_orig, fft + i * nxb, sizeof(float) * nx_orig);
    }
}

static void buf2fft_padding_1D(float *buf, float *fft, int nx_orig, int nx_final, int ny_orig, int ny_final)
{
    int nxb = nx_final + 2 - nx_final % 2;
    int nxp = nx_final - nx_orig + 1;
    int i, j;
#pragma omp parallel for num_threads(threadNumber)
    for (i = 0; i < (nx_final + 2 - nx_final % 2) * ny_final; i++)
    {
        fft[i] = 0.0;
    }
#pragma omp parallel for num_threads(threadNumber)
    for (i = 0; i < ny_orig; i++)
    {
        memcpy(fft + i * nxb, buf + i * nx_orig, sizeof(float) * nx_orig);
        for (j = nx_orig; j < nx_final; j++) // padding for continuity 
        {
            fft[i * nxb + j] = (float(j - nx_orig + 1) / float(nxp)) * buf[i * nx_orig] +
                               (float(nx_final - j) / float(nxp)) * buf[(i + 1) * nx_orig - 1];
        }
    }
}

static void filter_Rweighting_1D_many(float *data, int Nx, int Ny, float radial, float sigma)
{
    int Nx_padding = int(Nx / 10);
    int Nx_final = Nx + Nx_padding;
    fftwf_plan plan_fft, plan_ifft;
    long long tmp=static_cast<long long>(Nx_final + 2 - Nx_final % 2) * static_cast<long long>(Ny);
    float *bufc = new float[tmp]; 


    plan_fft = fftwf_plan_many_dft_r2c(1, &Nx_final, Ny, (float *)bufc, NULL, 1, (Nx_final + 2 - Nx_final % 2),
                                       reinterpret_cast<fftwf_complex *>(bufc), NULL, 1,
                                       (Nx_final + 2 - Nx_final % 2) / 2, FFTW_ESTIMATE);
    plan_ifft = fftwf_plan_many_dft_c2r(1, &Nx_final, Ny, reinterpret_cast<fftwf_complex *>(bufc), NULL, 1,
                                        (Nx_final + 2 - Nx_final % 2) / 2, (float *)bufc, NULL, 1,
                                        (Nx_final + 2 - Nx_final % 2), FFTW_ESTIMATE);

    buf2fft_padding_1D(data, bufc, Nx, Nx_final, Ny, Ny);

    fftwf_execute(plan_fft);

    int radial_Nx = int(floor((Nx_final)*radial));
    float sigma_Nx = float(Nx_final) * sigma;
    // loop: Ny (all Fourier components for y-axis)
    // #pragma omp parallel for num_threads(threadNumber)
    for (int j = 0; j < Ny; j++)
    {
        // loop: Nx_final+2-Nx_final%2 (all Fourier components for x-axis)
        for (int i = 0; i < Nx_final + 2 - Nx_final % 2; i += 2)
        {
            if (i == 0) // DC
            {
                bufc[i + j * (Nx_final + 2 - Nx_final % 2)] *= 0.2;
                bufc[i + 1 + j * (Nx_final + 2 - Nx_final % 2)] *= 0.2;
            }
            else if (i / 2 <= radial_Nx) // radial
            {
                bufc[i + j * (Nx_final + 2 - Nx_final % 2)] *= (i / 2);
                bufc[i + 1 + j * (Nx_final + 2 - Nx_final % 2)] *= (i / 2);
            }
            else // Gaussian falloff
            {
                bufc[i + j * (Nx_final + 2 - Nx_final % 2)] =
                    bufc[i + j * (Nx_final + 2 - Nx_final % 2)] * float(radial_Nx) *
                    exp(-float((i / 2 - radial_Nx) * (i / 2 - radial_Nx)) / (sigma_Nx * sigma_Nx));
                bufc[i + 1 + j * (Nx_final + 2 - Nx_final % 2)] =
                    bufc[i + 1 + j * (Nx_final + 2 - Nx_final % 2)] * float(radial_Nx) *
                    exp(-float((i / 2 - radial_Nx) * (i / 2 - radial_Nx)) / (sigma_Nx * sigma_Nx));
            }
        }
    }

    fftwf_execute(plan_ifft);
    fft2buf_padding_1D(data, bufc, Nx, Nx_final, Ny, Ny);
    // #pragma omp parallel for num_threads(threadNumber)
    for (int i = 0; i < Nx * Ny; i++) 
    {
        data[i] /= Nx;
    }

    fftwf_destroy_plan(plan_fft);
    fftwf_destroy_plan(plan_ifft);
    delete[] bufc; 
}

int ApplyFilterInplace(MrcStackM &inputProjection, float *data,size_t ny ,int mode)
{
    if (data == NULL)
    {
        printf("Apply filter error, data pointer is NULL !\n");
        return -1;
    }
    // printf("input.nx=%d input.ny=%d\n", inputProjection.header.nx, inputProjection.header.ny);
    Slice projectionParameter(0, 0, NULL);
    projectionParameter.x = inputProjection.header.nx;
  //  projectionParameter.y = inputProjection.header.ny; 
    projectionParameter.y =ny;
   // size_t oneSliceSize = inputProjection.header.nx * inputProjection.header.ny;
    size_t oneSliceSize = inputProjection.header.nx * ny;
    float weighting_radial = 0.05, weighting_sigma = 0.5;
    switch (mode)
    {
    case 0: //
    {
        int filtlength = 251;
        // char *filter="RamLak";
        char *filter = "SheppLogan";
        int row_pad = 2; // x-padded
        int col_pad = 2; // y-padded
        int symmetrize_2D_flag = 1;

        for (int angleIdx = 0; angleIdx < inputProjection.header.nz; angleIdx++)
        {
            float *currentSlice = data + oneSliceSize * angleIdx;
            FileterProjectionSymmetrize(currentSlice, filter, filtlength, projectionParameter, row_pad, col_pad, symmetrize_2D_flag, angleIdx);
        }
    };
    break;
    case 1:
    {
        int filtlength = 251;
        char *filter = "RamLak";
        // char *filter = "SheppLogan";
        int row_pad = 2; // x-padded
        int col_pad = 2; // y-padded
        int symmetrize_2D_flag = 1;

        for (int angleIdx = 0; angleIdx < inputProjection.header.nz; angleIdx++)
        {
            float *currentSlice = data + oneSliceSize * angleIdx;
            FileterProjectionSymmetrize(currentSlice, filter, filtlength, projectionParameter, row_pad, col_pad, symmetrize_2D_flag, angleIdx);
        }
    };
    break;
    case 2:
    {
        for (int angleIdx = 0; angleIdx < inputProjection.header.nz; angleIdx++)
        {
            float *currentSlice = data + oneSliceSize * angleIdx;
            filter_Rweighting_1D_many(currentSlice, inputProjection.header.nx, ny, weighting_radial, weighting_sigma);
        }
    };
    break;
    default:
        printf("Apply filter error, mode wrong !\n");
        return -2;
        break;
    }
    return 0;
}