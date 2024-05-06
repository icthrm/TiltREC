#ifndef MRCHEADER_H__
#define MRCHEADER_H__

#define MRC_MODE_BYTE            0
#define MRC_MODE_SHORT           1
#define MRC_MODE_FLOAT           2
#define MRC_MODE_COMPLEX_SHORT   3
#define MRC_MODE_COMPLEX_FLOAT   4
#define MRC_MODE_USHORT          6
#define MRC_MODE_RGB             16

#define MRC_CFLOAT_REAL_IMAG     20            /* COMPLEX FLOAT mode */
#define MRC_CFLOAT_AMP_RAD       21           /* COMPLEX FLOAT mode, but in amplitude and phase(  ) form */
#define MRC_CFLOAT_AMP_DEG       22           /* COMPLEX FLOAT mode, but in amplitude and phase( degree ) form */


#define MRC_LABEL_SIZE         80
#define MRC_NEXTRA             16
#define MRC_NLABELS            10
#define MRC_HEADER_SIZE        1024   /* Length of Header is 1024 Bytes. */
#define MRC_MAXCSIZE           3

/* The header structure for MRC files */
#pragma pack(1)
typedef struct MRCheader
{
    int   nx;         /*  # of Columns                  */
    int   ny;         /*  # of Rows                     */
    int   nz;         /*  # of Sections.                */
    int   mode;       /*  given by #define MRC_MODE...  */

    int   nxstart;    /*  Starting point of sub image.  */
    int   nystart;
    int   nzstart;

    int   mx;         /* Grid size in x, y, and z       */
    int   my;
    int   mz;

    float   xlen;       /* length of x element in um.     */
    float   ylen;       /* get scale = xlen/nx ...        */
    float   zlen;

    float   alpha;      /* cell angles, ignore */
    float   beta;
    float   gamma;

    int   mapc;       /* map coloumn 1=x,2=y,3=z.       */
    int   mapr;       /* map row     1=x,2=y,3=z.       */
    int   maps;       /* map section 1=x,2=y,3=z.       */

    float   amin;
    float   amax;
    float   amean;

    short   ispg;       /* image type */
    short   nsymbt;     /* space group number */


    /* 64 bytes */

    int   next;
    short   creatid;  /* Creator id, hvem = 1000, DeltaVision = -16224 */


    char    blank[30];

    short   nint;
    short   nreal;
    short   sub;
    short   zfac;

    float   min2;
    float   max2;
    float   min3;
    float   max3;
    float   min4;
    float   max4;


    short   idtype;
    short   lens;
    short   nd1;     /* Devide by 100 to get float value. */
    short   nd2;
    short   vd1;
    short   vd2;
    float   tiltangles[6];  /* 0,1,2 = original:  3,4,5 = current */


    float   xorg;
    float   yorg;
    float   zorg;
    char    cmap[4];
    char    stamp[4];
    float   rms;

    int nlabl;
    char  labels[10][80];

} MrcHeader;
#pragma pack()
/* END OF HEAD CODE */


#endif