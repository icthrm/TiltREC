#include "mrcmx/mrcstack.h"
#include "../opts/opts-mpi.h"
#include <fstream>
#include <iostream>
#include <vector>
#include "../filter/filter_prj.h"
#define MILLION 1000000

#define PI_180 0.01745329252f

#ifndef PI
#define PI 3.14159265358979323846
#endif

#define D2R(__ANGLE__) ((__ANGLE__) * PI_180)

void TestMrcWriteToFile(MrcStackM &testMrc, const char *filename)
{
	testMrc.SetZ(1);
	testMrc.WriteToFile(filename);
}

// 只写第150层的mrc
void TestMrcWriteBlock(MrcStackM &testMrc, int start, int end, char axis, float *blockdata)
{
	if (not('z' == axis || 'Z' == axis))
	{
		printf("Wrong test write block axis!\n");
		exit(-10);
	}
	const int TestWriteSlice = 151;
	if (start <= TestWriteSlice && TestWriteSlice < end)
	{
		int num = end - start;
		int sliceSize = testMrc.header.nx * testMrc.header.ny;
		testMrc.WriteBlock(0, 1, axis, blockdata + (TestWriteSlice - start) * sliceSize); // WriteBlock函数不包括end
	}
}

struct Coeff
{ // 20个双精度数，前10个是a后10个是b
	union
	{
		double p[20];
		struct
		{
			double a[10];
			double b[10];
		};
	};
};

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

void TranslateAngleToCoefficients(const std::vector<float> &angles,

								  std::vector<Coeff> &coeffs)
{
	coeffs.resize(angles.size());

	for (int i = 0; i < angles.size(); i++)
	{
		memset(coeffs[i].p, 0, sizeof(double) * 20);
		float beta = D2R(angles[i]);
		coeffs[i].a[0] = 0;			 //
		coeffs[i].a[1] = cos(beta);	 // x
		coeffs[i].a[2] = 0;			 // y
		coeffs[i].a[3] = -sin(beta); // z
		coeffs[i].b[0] = 0;			 //
		coeffs[i].b[1] = 0;			 // x
		coeffs[i].b[2] = 1;			 // y
		coeffs[i].b[3] = 0;			 // z
	}
	// printf("addr %p %p %p %p\n", &coeffs[0].a[0], &coeffs[0].a[1], &coeffs[0].b[0], &coeffs[0].b[1]);
	// exit(-1);
}

/*solve inverse transfroms defined by Geometry; substitute the inversion into
 * coefficients*/
void DecorateCoefficients(std::vector<Coeff> &coeffs, const Geometry &geo)
{
	double alpha = -D2R(geo.pitch_angle), beta = D2R(geo.offset), t = -geo.zshift;
	double ca = cos(alpha), sa = sin(alpha), cb = cos(beta), sb = sin(beta);
	double ca2 = ca * ca, sa2 = sa * sa, cb2 = cb * cb, sb2 = sb * sb;

	for (int i = 0; i < coeffs.size(); i++)
	{
		double a[10], b[10];
		memcpy(a, coeffs[i].a, sizeof(double) * 10);
		memcpy(b, coeffs[i].b, sizeof(double) * 10);
		coeffs[i].a[0] = a[0];
		coeffs[i].a[1] = (a[2] * sa * sb + a[3] * ca * sb + a[1] * cb); //*x
		coeffs[i].a[2] = (a[2] * ca - a[3] * sa);						//*y
		coeffs[i].a[3] = (a[3] * ca * cb + a[2] * cb * sa - a[1] * sb); //*z
		coeffs[i].a[4] = (a[4] * ca * cb - a[5] * cb * sa - a[6] * sa2 * sb -
						  2 * a[9] * ca * sa * sb + a[6] * ca2 * sb +
						  2 * a[8] * ca * sa * sb); //*x*y
		coeffs[i].a[5] =
			(a[4] * cb2 * sa - a[5] * ca * sb2 + a[5] * ca * cb2 -
			 2 * a[7] * cb * sb - a[4] * sa * sb2 + 2 * a[9] * ca2 * cb * sb +
			 2 * a[8] * cb * sa2 * sb + 2 * a[6] * ca * cb * sa * sb); //*x*z
		coeffs[i].a[6] =
			(2 * a[8] * ca * cb * sa - a[4] * ca * sb + a[5] * sa * sb +
			 a[6] * ca2 * cb - a[6] * cb * sa2 - 2 * a[9] * ca * cb * sa); //*y*z
		coeffs[i].a[7] =
			(a[7] * cb2 + a[5] * ca * cb * sb + a[9] * ca2 * sb2 +
			 a[8] * sa2 * sb2 + a[4] * cb * sa * sb + a[6] * ca * sa * sb2); //*x^2
		coeffs[i].a[8] = (a[8] * ca2 + a[9] * sa2 - a[6] * ca * sa);		 //*y^2
		coeffs[i].a[9] = (a[7] * sb2 + a[9] * ca2 * cb2 + a[8] * cb2 * sa2 -
						  a[4] * cb * sa * sb + a[6] * ca * cb2 * sa -
						  a[5] * ca * cb * sb); //*z^2

		coeffs[i].b[0] = b[0];
		coeffs[i].b[1] = (b[2] * sa * sb + b[3] * ca * sb + b[1] * cb); //*x
		coeffs[i].b[2] = (b[2] * ca - b[3] * sa);						//*y
		coeffs[i].b[3] = (b[3] * ca * cb + b[2] * cb * sa - b[1] * sb); //*z
		coeffs[i].b[4] = (b[4] * ca * cb - b[5] * cb * sa - b[6] * sa2 * sb -
						  2 * b[9] * ca * sa * sb + b[6] * ca2 * sb +
						  2 * b[8] * ca * sa * sb); //*x*y
		coeffs[i].b[5] =
			(b[4] * cb2 * sa - b[5] * ca * sb2 + b[5] * ca * cb2 -
			 2 * b[7] * cb * sb - b[4] * sa * sb2 + 2 * b[9] * ca2 * cb * sb +
			 2 * b[8] * cb * sa2 * sb + 2 * b[6] * ca * cb * sa * sb); //*x*z
		coeffs[i].b[6] =
			(2 * b[8] * ca * cb * sa - b[4] * ca * sb + b[5] * sa * sb +
			 b[6] * ca2 * cb - b[6] * cb * sa2 - 2 * b[9] * ca * cb * sa); //*y*z
		coeffs[i].b[7] =
			(b[7] * cb2 + b[5] * ca * cb * sb + b[9] * ca2 * sb2 +
			 b[8] * sa2 * sb2 + b[4] * cb * sa * sb + b[6] * ca * sa * sb2); //*x^2
		coeffs[i].b[8] = (b[8] * ca2 + b[9] * sa2 - b[6] * ca * sa);		 //*y^2
		coeffs[i].b[9] = (b[7] * sb2 + b[9] * ca2 * cb2 + b[8] * cb2 * sa2 -
						  b[4] * cb * sa * sb + b[6] * ca * cb2 * sa -
						  b[5] * ca * cb * sb); //*z^2
	}

	// considering z_shift
	for (int i = 0; i < coeffs.size(); i++)
	{
		double a[10], b[10];
		memcpy(a, coeffs[i].a, sizeof(double) * 10);
		memcpy(b, coeffs[i].b, sizeof(double) * 10);

		coeffs[i].a[0] = a[0] + a[3] * t + a[9] * t * t;
		coeffs[i].a[1] = a[1] + a[5] * t;	  //*x
		coeffs[i].a[2] = a[2] + a[6] * t;	  //*y
		coeffs[i].a[3] = a[3] + 2 * a[9] * t; //*z

		coeffs[i].b[0] = b[0] + b[3] * t + b[9] * t * t;
		coeffs[i].b[1] = b[1] + b[5] * t;	  //*x
		coeffs[i].b[2] = b[2] + b[6] * t;	  //*y
		coeffs[i].b[3] = b[3] + 2 * b[9] * t; //*z
	}
}

void (*funL)(const Coeff &, double, double, double, double *);

void WarpPosition(const Coeff &coeff, double X, double Y, double Z, double *n)
{
	n[0] = coeff.a[0] + coeff.a[1] * X + coeff.a[2] * Y + coeff.a[3] * Z +
		   coeff.a[4] * X * Y + coeff.a[5] * X * Z + coeff.a[6] * Y * Z +
		   coeff.a[7] * X * X + coeff.a[8] * Y * Y + coeff.a[9] * Z * Z;
	n[1] = coeff.b[0] + coeff.b[1] * X + coeff.b[2] * Y + coeff.b[3] * Z +
		   coeff.b[4] * X * Y + coeff.b[5] * X * Z + coeff.b[6] * Y * Z +
		   coeff.b[7] * X * X + coeff.b[8] * Y * Y + coeff.b[9] * Z * Z;
}

void LinearPosition(const Coeff &coeff, double X, double Y, double Z,
					double *n)
{
	n[0] = coeff.a[0] + coeff.a[1] * X + coeff.a[2] * Y + coeff.a[3] * Z;
	n[1] = coeff.b[0] + coeff.b[1] * X + coeff.b[2] * Y + coeff.b[3] * Z;
}

void ValCoef(const Point3DF &origin, const Point3D &coord, const Coeff &coeff,
			 Weight *wt)
{
	double x, y;

	double X, Y, Z, n[2];
	X = coord.x - origin.x;
	Y = coord.y - origin.y;
	Z = coord.z - origin.z;

	// 	funL(coeff, X, Y, Z, n);
	n[0] = coeff.a[0] + coeff.a[1] * X + coeff.a[2] * Y + coeff.a[3] * Z;
	n[1] = coeff.b[0] + coeff.b[1] * X + coeff.b[2] * Y + coeff.b[3] * Z;

	x = n[0] + origin.x;
	y = n[1] + origin.y;

	wt->x_min = floor(x);
	wt->y_min = floor(y);

	wt->x_min_del = x - wt->x_min;
	wt->y_min_del = y - wt->y_min;
}

void Reproject(const Point3DF &origin, const Volume &vol, const Coeff &coeff,
			   Slice &reproj_val, Slice &reproj_wt, int thickness, int length, int width)
{

	Point3D coord;
	int n;

	for (int z = 0; z < thickness; z++)
	{
		float *vdrefz = vol.data + z * (size_t)width * length;
		coord.z = z;

		for (int y = 0; y < length; y++)
		{
			float *vdrefy = vdrefz + y * width;
			coord.y = y;

			for (int x = 0; x < width; x++)
			{
				float *vdrefx = vdrefy + x;
				coord.x = x;
				Weight wt;
				ValCoef(origin, coord, coeff, &wt);

				if (wt.x_min >= 0 && wt.x_min < width && wt.y_min >= 0 &&
					wt.y_min < length)
				{									 //(x_min, y_min)
					n = wt.x_min + wt.y_min * width; // index in reproj
					reproj_val.data[n] +=
						(1 - wt.x_min_del) * (1 - wt.y_min_del) * (*vdrefx);
					reproj_wt.data[n] += (1 - wt.x_min_del) * (1 - wt.y_min_del);
				}
				if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < width &&
					wt.y_min >= 0 && wt.y_min < length)
				{										 //(x_min+1, y_min)
					n = wt.x_min + 1 + wt.y_min * width; // index in reproj
					reproj_val.data[n] += wt.x_min_del * (1 - wt.y_min_del) * (*vdrefx);
					reproj_wt.data[n] += wt.x_min_del * (1 - wt.y_min_del);
				}
				if (wt.x_min >= 0 && wt.x_min < width && (wt.y_min + 1) >= 0 &&
					(wt.y_min + 1) < length)
				{										   //(x_min, y_min+1)
					n = wt.x_min + (wt.y_min + 1) * width; // index in reproj
					reproj_val.data[n] += (1 - wt.x_min_del) * wt.y_min_del * (*vdrefx);
					reproj_wt.data[n] += (1 - wt.x_min_del) * wt.y_min_del;
				}
				if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < width &&
					(wt.y_min + 1) >= 0 &&
					(wt.y_min + 1) < length)
				{												 //(x_min+1, y_min+1)
					n = (wt.x_min + 1) + (wt.y_min + 1) * width; // index in reproj
					reproj_val.data[n] += wt.x_min_del * wt.y_min_del * (*vdrefx);
					reproj_wt.data[n] += wt.x_min_del * wt.y_min_del;
				}
			}
		}
	}
}
void Reproject(const Point3DF &origin, const Volume &vol, const Coeff &coeff,
			   float *reproj_val, float *reproj_wt, int thickness, int length, int width)
{

	Point3D coord;
	int n;

	// for (int z = 0; z < thickness; z++)
	for (int y = 0; y < length; y++)
	{
		// float *vdrefz = vol.data + z * (size_t)width * length;
		// coord.z = z;
		float *vdrefy = vol.data + y * (size_t)width * thickness;
		coord.y = y;

		// for (int y = 0; y < length; y++)
		for (int z = 0; z < thickness; z++)
		{
			// float *vdrefy = vdrefz + y * width;
			// coord.y = y;
			float *vdrefz = vdrefy + z * (size_t)width;
			coord.z = z;
			for (int x = 0; x < width; x++)
			{
				float *vdrefx = vdrefz + x;
				coord.x = x;
				Weight wt;
				ValCoef(origin, coord, coeff, &wt);

				if (wt.x_min >= 0 && wt.x_min < width && wt.y_min >= 0 &&
					wt.y_min < length)
				{									 //(x_min, y_min)
					n = wt.x_min + wt.y_min * width; // index in reproj
					reproj_val[n] +=
						(1 - wt.x_min_del) * (1 - wt.y_min_del) * (*vdrefx);
					reproj_wt[n] += (1 - wt.x_min_del) * (1 - wt.y_min_del);
				}
				if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < width &&
					wt.y_min >= 0 && wt.y_min < length)
				{										 //(x_min+1, y_min)
					n = wt.x_min + 1 + wt.y_min * width; // index in reproj
					reproj_val[n] += wt.x_min_del * (1 - wt.y_min_del) * (*vdrefx);
					reproj_wt[n] += wt.x_min_del * (1 - wt.y_min_del);
				}
				if (wt.x_min >= 0 && wt.x_min < width && (wt.y_min + 1) >= 0 &&
					(wt.y_min + 1) < length)
				{										   //(x_min, y_min+1)
					n = wt.x_min + (wt.y_min + 1) * width; // index in reproj
					reproj_val[n] += (1 - wt.x_min_del) * wt.y_min_del * (*vdrefx);
					reproj_wt[n] += (1 - wt.x_min_del) * wt.y_min_del;
				}
				if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < width &&
					(wt.y_min + 1) >= 0 &&
					(wt.y_min + 1) < length)
				{												 //(x_min+1, y_min+1)
					n = (wt.x_min + 1) + (wt.y_min + 1) * width; // index in reproj
					reproj_val[n] += wt.x_min_del * wt.y_min_del * (*vdrefx);
					reproj_wt[n] += wt.x_min_del * wt.y_min_del;
				}
			}
		}
	}
}
void BilinearValue(const Slice &slc, const Weight &wt, float *val,
				   float *vwt)
{
	int n;
	if (wt.x_min >= 0 && wt.x_min < slc.width && wt.y_min >= 0 &&
		wt.y_min < slc.height)
	{ //(x_min, y_min)
		n = wt.x_min + wt.y_min * slc.width;
		*val += (1 - wt.x_min_del) * (1 - wt.y_min_del) * slc.data[n];
		*vwt += (1 - wt.x_min_del) * (1 - wt.y_min_del);
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < slc.width && wt.y_min >= 0 &&
		wt.y_min < slc.height)
	{ //(x_min+1, y_min)
		n = wt.x_min + 1 + wt.y_min * slc.width;
		*val += wt.x_min_del * (1 - wt.y_min_del) * slc.data[n];
		*vwt += wt.x_min_del * (1 - wt.y_min_del);
	}
	if (wt.x_min >= 0 && wt.x_min < slc.width && (wt.y_min + 1) >= 0 &&
		(wt.y_min + 1) < slc.height)
	{ //(x_min, y_min+1)
		n = wt.x_min + (wt.y_min + 1) * slc.width;
		*val += (1 - wt.x_min_del) * wt.y_min_del * slc.data[n];
		*vwt += (1 - wt.x_min_del) * wt.y_min_del;
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < slc.width &&
		(wt.y_min + 1) >= 0 && (wt.y_min + 1) < slc.height)
	{ //(x_min+1, y_min+1)
		n = wt.x_min + 1 + (wt.y_min + 1) * slc.width;
		*val += wt.x_min_del * wt.y_min_del * slc.data[n];
		*vwt += wt.x_min_del * wt.y_min_del;
	}
}
void BilinearValue(int width, int length, float *proj, const Weight &wt, float *val,
				   float *vwt)
{
	int n;
	if (wt.x_min >= 0 && wt.x_min < width && wt.y_min >= 0 &&
		wt.y_min < length)
	{ //(x_min, y_min)
		n = wt.x_min + wt.y_min * width;
		*val += (1 - wt.x_min_del) * (1 - wt.y_min_del) * proj[n];
		*vwt += (1 - wt.x_min_del) * (1 - wt.y_min_del);
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < width && wt.y_min >= 0 &&
		wt.y_min < length)
	{ //(x_min+1, y_min)
		n = wt.x_min + 1 + wt.y_min * width;
		*val += wt.x_min_del * (1 - wt.y_min_del) * proj[n];
		*vwt += wt.x_min_del * (1 - wt.y_min_del);
	}
	if (wt.x_min >= 0 && wt.x_min < width && (wt.y_min + 1) >= 0 &&
		(wt.y_min + 1) < length)
	{ //(x_min, y_min+1)
		n = wt.x_min + (wt.y_min + 1) * width;
		*val += (1 - wt.x_min_del) * wt.y_min_del * proj[n];
		*vwt += (1 - wt.x_min_del) * wt.y_min_del;
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < width &&
		(wt.y_min + 1) >= 0 && (wt.y_min + 1) < length)
	{ //(x_min+1, y_min+1)
		n = wt.x_min + 1 + (wt.y_min + 1) * width;
		*val += wt.x_min_del * wt.y_min_del * proj[n];
		*vwt += wt.x_min_del * wt.y_min_del;
	}
}
void BackProject(const Point3DF &origin, MrcStackM &projs, Volume &vol,
				 Coeff coeffv[], int start, int length, Slice &proj, int thickness)
{
	// Slice proj(projs.X(), projs.Y());
	// Slice proj(projs.X(), length);
	Point3D coord;

	// memset(vol.data, 0, sizeof(float) * length * vol.width * vol.height);

	for (int idx = 0; idx < projs.Z(); idx++)
	{
		printf("Begin to read %d projection\n", idx);
		float *oneproj = proj.data + projs.X() * length * idx;

		for (int y = 0; y < length; y++)
		{

			float *vdrefy = vol.data + y * (size_t)projs.X() * thickness;

			coord.y = y;

			for (int z = 0; z < thickness; z++)
			{
				float *vdrefz = vdrefy + z * (size_t)projs.X();
				coord.z = z;

				for (int x = 0; x < projs.X(); x++)
				{
					coord.x = x;
					Weight wt;
					float s = 0, c = 0;

					ValCoef(origin, coord, coeffv[idx], &wt);
					BilinearValue(projs.X(), length, oneproj, wt, &s, &c);

					if (c)
					{
						*(vdrefz + x) += (float)(s / c);
					}
				}
			}
		}
	}
}

void FBP(const Point3DF &origin, MrcStackM &projs, Volume &vol,
		 Coeff coeffv[], int start, int length, Slice &proj, int thickness, int filterMode)
{
	Point3D coord;

	ApplyFilterInplace(projs, proj.data, length, filterMode);
	for (int idx = 0; idx < projs.Z(); idx++)
	{
		printf("BPT begin to read %d projection\n", idx);
		float *oneproj = proj.data + projs.X() * length * idx;

		for (int y = 0; y < length; y++)
		{
			float *vdrefy = vol.data + y * (size_t)projs.X() * thickness;

			coord.y = y;

			for (int z = 0; z < thickness; z++)
			{
				float *vdrefz = vdrefy + z * (size_t)projs.X();
				coord.z = z;

				for (int x = 0; x < projs.X(); x++)
				{
					coord.x = x;
					Weight wt;
					float s = 0, c = 0;

					ValCoef(origin, coord, coeffv[idx], &wt);
					BilinearValue(projs.X(), length, oneproj, wt, &s, &c);

					if (c)
					{
						*(vdrefz + x) += (float)(s / c);
					}
				}
			}
		}
	}
}
void UpdateVolumeByProjDiff(const Point3DF &origin, float *diff,
							Volume &vol, float gamma, const Coeff &coeff, int thickness, int length, int width)
{

	Point3D coord;

	// for (int z = 0; z < thickness; z++)
	for (int y = 0; y < length; y++)
	{
		// float *vdrefz = vol.data + z * (size_t)width * length;
		// coord.z = z;

		float *vdrefy = vol.data + y * (size_t)width * thickness;
		coord.y = y;

		// for (int y = 0; y < length; y++)
		for (int z = 0; z < thickness; z++)
		{
			// float *vdrefy = vdrefz + y * width;
			// coord.y = y;
			float *vdrefz = vdrefy + z * (size_t)width;
			coord.z = z;
			for (int x = 0; x < width; x++)
			{
				coord.x = x;
				Weight wt;
				float s = 0, c = 0;

				ValCoef(origin, coord, coeff, &wt);
				BilinearValue(width, length, diff, wt, &s, &c);

				if (c)
				{
					*(vdrefz + x) += (float)(s / c) * gamma;
				}
			}
		}
	}
}

void UpdateWeightsByProjDiff(const Point3DF &origin, float *diff,
							 float *values, float *weights,
							 const Coeff &coeff, int thickness, int length, int width)
{

	Point3D coord;

	// for (int z = 0; z < values.height; z++)
	for (int y = 0; y < length; y++)
	{
		// float *vdrefz = values.data + z * (size_t)values.width * values.length;
		// float *wdrefz = weights.data + z * (size_t)weights.width * weights.length;
		// coord.z = z + values.z;
		float *vdrefy = values + y * static_cast<size_t>(width * thickness);
		float *wdrefy = weights + y * static_cast<size_t>(width * thickness);
		coord.y = y;
		//	for (int y = 0; y < values.length; y++)
		for (int z = 0; z < thickness; z++)
		{
			// float *vdrefy = vdrefz + y * values.width;
			// float *wdrefy = wdrefz + y * weights.width;
			// coord.y = y + values.y;
			float *vdrefz = vdrefy + z * width;
			float *wdrefz = wdrefy + z * width;
			coord.z = z;

			for (int x = 0; x < width; x++)
			{
				float *vdrefx = vdrefz + x;
				float *wdrefx = wdrefz + x;
				coord.x = x;
				Weight wt;
				float s = 0, c = 0;

				ValCoef(origin, coord, coeff, &wt);
				// BilinearValue(diff, wt, &s, &c);
				BilinearValue(width, length, diff, wt, &s, &c);
				*vdrefx += s;
				*wdrefx += c;
			}
		}
	}
}

void UpdateVolumeByWeights(Volume &vol, float *values, float *weights,
						   float gamma, int thickness, int length, int width)
{
	// size_t pxsize = vol.height * vol.width * vol.length;
	size_t size = static_cast<size_t>(length) * static_cast<size_t>(width) * static_cast<size_t>(thickness);
	for (size_t i = size; i--;)
	{
		if (weights[i])
		{
			vol.data[i] += values[i] / weights[i] * gamma;
		}
	}
}

void SART(const Point3DF &origin, MrcStackM &projs, Volume &vol, Coeff coeffv[],
		  int iteration // 迭代次数
		  ,
		  float gamma // 松弛变量
		  ,
		  int start, int length, Slice &proj, int thickness)
{
	// int pxsize = projs.X() * projs.Y();
	// Slice reproj_val(projs.X(), projs.Y()); // reprojection value
	// Slice reproj_wt(projs.X(), projs.Y());	// reprojection weight
	// Slice projection(projs.X(), projs.Y());
	// int pxsize = projs.X() * length;
	// size_t pxsize = static_cast<size_t>(length) * static_cast<size_t>(projs.X()) * static_cast<size_t>(projs.Z());
	// Slice reproj_val(projs.X(), length, NULL); // reprojection value
	// Slice reproj_wt(projs.X(), length, NULL);  // reprojection weight
	//  Slice projection(projs.X(), length);
	size_t pxsize = static_cast<size_t>(length) * static_cast<size_t>(projs.X());
	float *reproj_val = (float *)malloc(sizeof(float) * pxsize);
	float *reproj_wt = (float *)malloc(sizeof(float) * pxsize);

	for (int i = 0; i < iteration; i++)
	{
		printf("SART iteraion %d\n", i);
		for (int idx = 0; idx < projs.Z(); idx++)
		{
			memset(reproj_val, 0, sizeof(float) * pxsize);
			memset(reproj_wt, 0, sizeof(float) * pxsize);

			Reproject(origin, vol, coeffv[idx], reproj_val, reproj_wt, thickness, length, projs.X());

			// MPI_Allreduce(MPI_IN_PLACE, reproj_val, pxsize, MPI_FLOAT, MPI_SUM,
			// 			  MPI_COMM_WORLD);
			// MPI_Allreduce(MPI_IN_PLACE, reproj_wt, pxsize, MPI_FLOAT, MPI_SUM,
			// 			  MPI_COMM_WORLD);

			printf("SART begin to read %d projection (iteraion "
				   "%d)\n",
				   idx, i);
			// projs.ReadSliceZ(idx, projection.data);
			float *projectiondata = proj.data + projs.X() * length * idx;
			for (int n = 0; n < pxsize; n++)
			{
				if (reproj_wt[n] != 0)
				{
					reproj_val[n] /= reproj_wt[n];
				}
				reproj_val[n] = projectiondata[n] - reproj_val[n];
			}

			UpdateVolumeByProjDiff(origin, reproj_val, vol, gamma, coeffv[idx], thickness, length, projs.X());
		}
	}
	free(reproj_val);
	free(reproj_wt);
}

void SIRT(const Point3DF &origin, MrcStackM &projs, Volume &vol, Coeff coeffv[],
		  int iteration, float gamma, int start, int length, Slice &proj, int thickness)
{

	size_t pxsize = static_cast<size_t>(length) * static_cast<size_t>(projs.X());
	// Slice reproj_val(projs.X(), projs.Y()); // reprojection value
	// Slice reproj_wt(projs.X(), projs.Y());	// reprojection weight
	// Slice projection(projs.X(), projs.Y());
	float *reproj_val = (float *)malloc(sizeof(float) * pxsize);
	float *reproj_wt = (float *)malloc(sizeof(float) * pxsize);

	// Volume valvol(vol.x, vol.y, vol.z, vol.length, vol.width, vol.height);
	// Volume wtvol(vol.x, vol.y, vol.z, vol.length, vol.width, vol.height);
	size_t size = static_cast<size_t>(length) * static_cast<size_t>(projs.X()) * static_cast<size_t>(thickness);
	float *valvol = (float *)malloc(sizeof(float) * size);
	float *wtvol = (float *)malloc(sizeof(float) * size);
	// if (wtvol == NULL)
	// {
	// 	// 如果malloc返回NULL，打印错误信息并退出程序
	// 	std::cerr << "内存分配失败。" << std::endl;
	// 	exit(1);
	// }

	for (int i = 0; i < iteration; i++)
	{
		memset(valvol, 0, sizeof(float) * size);
		memset(wtvol, 0, sizeof(float) * size);
		printf("SIRT iteraion %d\n", i);
		for (int idx = 0; idx < projs.Z(); idx++)
		{
			memset(reproj_val, 0, sizeof(float) * pxsize);
			memset(reproj_wt, 0, sizeof(float) * pxsize);

			// Reproject(origin, vol, coeffv[idx], reproj_val,
			// 		  reproj_wt); // vol is not changed during iteration

			Reproject(origin, vol, coeffv[idx], reproj_val, reproj_wt, thickness, length, projs.X());

			// MPI_Allreduce(MPI_IN_PLACE, reproj_val, pxsize, MPI_FLOAT, MPI_SUM,
			// 			  MPI_COMM_WORLD);
			// MPI_Allreduce(MPI_IN_PLACE, reproj_wt, pxsize, MPI_FLOAT, MPI_SUM,
			// 			  MPI_COMM_WORLD);

			// printf("SIRT begin to read %d projection for %d z-coordinate (iteraion "
			// 	   "%d)\n",
			// 	   idx, vol.z, i);
			// projs.ReadSliceZ(idx, projection.data);
			float *projectiondata = proj.data + projs.X() * length * idx;
			for (int n = 0; n < pxsize; n++)
			{
				if (reproj_wt[n])
				{
					reproj_val[n] /= reproj_wt[n];
				}
				reproj_val[n] = projectiondata[n] - reproj_val[n];
			}

			UpdateWeightsByProjDiff(origin, reproj_val, valvol, wtvol, coeffv[idx], thickness, length, projs.X());
		}

		// UpdateVolumeByWeights(vol, valvol, wtvol, gamma, thickness, length, projs.X());
		for (size_t i = size; i--;)
		{
			if (wtvol[i])
			{
				vol.data[i] += valvol[i] / wtvol[i] * gamma;
			}
		}
	}
	free(reproj_val);
	free(reproj_wt);
	free(valvol);
	free(wtvol);
}

struct SysInfo
{
	int id;
	int procs;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int namelen;
};

int ATOM(options &opt, int myid, int procs)
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
	// printf("success\n");
	mrcvol.InitializeHeader();
	// mrcvol.SetSize(projs.X(), projs.Y(), opt.thickness); // test---------------------
	mrcvol.SetSize(projs.X(), opt.thickness,
				   projs.Y());
	std::vector<float> angles;
	ReadAngles(angles, opt.angle);

	std::vector<float> xangles;

	// if (opt.xangle[0] != '\0')
	// {
	// 	ReadAngles(xangles, opt.xangle);
	// }
	// else
	// {
	xangles.resize(angles.size(), 0.0);
	// }

	std::vector<Coeff> params;
	TranslateAngleToCoefficients(angles, params);

	Geometry geo;
	geo.offset = opt.offset;
	geo.pitch_angle = opt.pitch_angle;
	geo.zshift = opt.zshift;

	DecorateCoefficients(params, geo);

	// int height;
	// int zrem = mrcvol.Z() % procs;
	// int volz; // the start slice of reproject per process

	// if (myid < zrem)
	// {
	// 	height = mrcvol.Z() / procs + 1;
	// 	volz = height * myid;
	// }
	// else
	// {
	// 	height = mrcvol.Z() / procs;
	// 	volz = height * myid + zrem;
	// }
	int length;
	int yrem = projs.Y() % procs;
	int start, end; // the start ane end slice of reproject per process
	int add_left = ceil(fabsf(tan(D2R(opt.pitch_angle))) * opt.thickness);
	int add_right = ceil(fabsf(tan(D2R(opt.pitch_angle))) * opt.thickness);

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
	length = length + add_left + add_right;
	Point3DF origin;

	origin.x = mrcvol.X() * .5;
	origin.y = mrcvol.Y() * .5;
	origin.z = opt.thickness * .5;

	// Volume vol(0, 0, volz, mrcvol.Y(), mrcvol.X(), height);
	Volume vol(0, 0, 0, projs.X(), opt.thickness, length,
			   NULL);

	std::cout << myid << ": (" << vol.x << "," << vol.y << "," << start << ")"
			  << "&(" << vol.length << "," << length << "," << opt.thickness
			  << ")" << std::endl;

	if (myid == 0)
	{
		printf("origin.x is %f, origin.y is %f, origin.z is %f\n", origin.x,
			   origin.y, origin.z);
	}
	// TestMrcWriteToFile(mrcvol, opt.output);
	mrcvol.WriteToFile(opt.output); // test------------------------------

	if (myid == 0)
	{
		mrcvol.WriteHeader();
	}
	size_t size = static_cast<size_t>(length) * static_cast<size_t>(projs.X()) * static_cast<size_t>(opt.thickness);

	try
	{
		vol.data = new float[size];
	}
	catch (std::bad_alloc &ba)
	{
		std::cerr << "Failed to allocate memory: " << ba.what() << '\n';
	}

	/**********************reconstruction along Z-axis*******************/
	const char *reference = opt.initial;
	if (reference[0] != '\0')
	{

		MrcStackM init;
		init.ReadFile(reference);
		init.ReadHeader();
		init.ReadBlock(start, start + length, 'z', vol.data);
		init.Close();
	}
	else
	{
		memset(vol.data, 0, sizeof(float) * size);
	}

	size_t projsize = static_cast<size_t>(length) * static_cast<size_t>(projs.X()) * static_cast<size_t>(projs.Z());

	float *tmp = nullptr;
	try
	{
		tmp = new float[projsize];
	}
	catch (std::bad_alloc &ba)
	{
		std::cerr << "Failed to allocate memory: " << ba.what() << '\n';
	}

	Slice proj(projs.X(), projs.Y(), NULL);
	try
	{
		proj.data = new float[projsize];
	}
	catch (std::bad_alloc &ba)
	{
		std::cerr << "Failed to allocate memory: " << ba.what() << '\n';
	}

	projs.ReadBlock(start - add_left, end + add_right, 'y', tmp);
	MrcStackM::RotateX(tmp, projs.X(), length, projs.Z(), proj.data);

	origin.y = origin.y - start + add_left;

	if (opt.method == "BPT")
	{
		BackProject(origin, projs, vol, &params[0], start, length, proj, opt.thickness);
	}
	else if (opt.method == "SART")
	{
		SART(origin, projs, vol, &params[0], opt.iteration, opt.gamma, start, length, proj, opt.thickness);
	}
	else if (opt.method == "SIRT")
	{
		SIRT(origin, projs, vol, &params[0], opt.iteration, opt.gamma, start, length, proj, opt.thickness);
	}
	else if (opt.method == "FBP")
	{
		ApplyFilterInplace(projs, proj.data, length, 0);
		BackProject(origin, projs, vol, &params[0], start, length, proj, opt.thickness);
	}
	else if (opt.method == "WBP")
	{
		ApplyFilterInplace(projs, proj.data, length, 2);
		BackProject(origin, projs, vol, &params[0], start, length, proj, opt.thickness);
	}
	// mrcvol.WriteBlock(vol.z, vol.z + height, 'z', vol.data);
	// if (myid == 0)
	mrcvol.WriteBlock(start, start + length, 'z', vol.data); // test-------------------------
	// TestMrcWriteBlock(mrcvol, vol.z, vol.z + height, 'z', vol.data); // test-------------------------

	MPI_Barrier(MPI_COMM_WORLD);

	if (myid == 0)
	{
		mrcvol.UpdateHeader();
	}

	projs.Close();
	mrcvol.Close();
	// free(vol.data);
	delete[] tmp;
	delete[] proj.data;
	delete[] vol.data;
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

	if (GetOpts(argc, argv, &opts) <= 0)
	{
		EX_TRACE("***WRONG INPUT.\n");
		return -1;
	}

	if (info.id == 0)
	{
		PrintOpts(opts);
	}

	ATOM(opts, info.id, info.procs);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize(); // parallel finish

	return 0;
}
