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
struct Coeff
{
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
		// float alpha = D2R(xangles[i]);

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
			   Slice &reproj_val, Slice &reproj_wt)
{

	Point3D coord;
	int n;

	for (int z = 0; z < vol.height; z++)
	{
		float *vdrefz = vol.data + z * (size_t)vol.width * vol.length;
		coord.z = z + vol.z;

		for (int y = 0; y < vol.length; y++)
		{
			float *vdrefy = vdrefz + y * vol.width;
			coord.y = y + vol.y;

			for (int x = 0; x < vol.width; x++)
			{
				float *vdrefx = vdrefy + x;
				coord.x = x + vol.x;
				Weight wt;
				ValCoef(origin, coord, coeff, &wt);

				if (wt.x_min >= 0 && wt.x_min < vol.width && wt.y_min >= 0 &&
					wt.y_min < vol.length)
				{										 //(x_min, y_min)
					n = wt.x_min + wt.y_min * vol.width; // index in reproj
					reproj_val.data[n] +=
						(1 - wt.x_min_del) * (1 - wt.y_min_del) * (*vdrefx);
					reproj_wt.data[n] += (1 - wt.x_min_del) * (1 - wt.y_min_del);
				}
				if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < vol.width &&
					wt.y_min >= 0 && wt.y_min < vol.length)
				{											 //(x_min+1, y_min)
					n = wt.x_min + 1 + wt.y_min * vol.width; // index in reproj
					reproj_val.data[n] += wt.x_min_del * (1 - wt.y_min_del) * (*vdrefx);
					reproj_wt.data[n] += wt.x_min_del * (1 - wt.y_min_del);
				}
				if (wt.x_min >= 0 && wt.x_min < vol.width && (wt.y_min + 1) >= 0 &&
					(wt.y_min + 1) < vol.length)
				{											   //(x_min, y_min+1)
					n = wt.x_min + (wt.y_min + 1) * vol.width; // index in reproj
					reproj_val.data[n] += (1 - wt.x_min_del) * wt.y_min_del * (*vdrefx);
					reproj_wt.data[n] += (1 - wt.x_min_del) * wt.y_min_del;
				}
				if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < vol.width &&
					(wt.y_min + 1) >= 0 &&
					(wt.y_min + 1) < vol.length)
				{													 //(x_min+1, y_min+1)
					n = (wt.x_min + 1) + (wt.y_min + 1) * vol.width; // index in reproj
					reproj_val.data[n] += wt.x_min_del * wt.y_min_del * (*vdrefx);
					reproj_wt.data[n] += wt.x_min_del * wt.y_min_del;
				}
			}
		}
	}
}

inline void BilinearValue(const Slice &slc, const Weight &wt, float *val,
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
inline void BilinearValue(const Slice &slc, float *slcdata, const Weight &wt, float *val,
						  float *vwt)
{
	int n;
	if (wt.x_min >= 0 && wt.x_min < slc.width && wt.y_min >= 0 &&
		wt.y_min < slc.height)
	{ //(x_min, y_min)
		n = wt.x_min + wt.y_min * slc.width;
		*val += (1 - wt.x_min_del) * (1 - wt.y_min_del) * slcdata[n];
		*vwt += (1 - wt.x_min_del) * (1 - wt.y_min_del);
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < slc.width && wt.y_min >= 0 &&
		wt.y_min < slc.height)
	{ //(x_min+1, y_min)
		n = wt.x_min + 1 + wt.y_min * slc.width;
		*val += wt.x_min_del * (1 - wt.y_min_del) * slcdata[n];
		*vwt += wt.x_min_del * (1 - wt.y_min_del);
	}
	if (wt.x_min >= 0 && wt.x_min < slc.width && (wt.y_min + 1) >= 0 &&
		(wt.y_min + 1) < slc.height)
	{ //(x_min, y_min+1)
		n = wt.x_min + (wt.y_min + 1) * slc.width;
		*val += (1 - wt.x_min_del) * wt.y_min_del * slcdata[n];
		*vwt += (1 - wt.x_min_del) * wt.y_min_del;
	}
	if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < slc.width &&
		(wt.y_min + 1) >= 0 && (wt.y_min + 1) < slc.height)
	{ //(x_min+1, y_min+1)
		n = wt.x_min + 1 + (wt.y_min + 1) * slc.width;
		*val += wt.x_min_del * wt.y_min_del * slcdata[n];
		*vwt += wt.x_min_del * wt.y_min_del;
	}
}
void BackProject(const Point3DF &origin, MrcStackM &projs, Volume &vol,
				 Coeff coeffv[])
{
	Slice proj(projs.X(), projs.Y());
	Point3D coord;

	memset(vol.data, 0, sizeof(float) * vol.length * vol.width * vol.height);

	for (int idx = 0; idx < projs.Z(); idx++)
	{

		projs.ReadSliceZ(idx, proj.data);

		for (int z = 0; z < vol.height; z++)
		{
			float *vdrefz = vol.data + z * (size_t)vol.width * vol.length;
			coord.z = z + vol.z;

			for (int y = 0; y < vol.length; y++)
			{
				float *vdrefy = vdrefz + y * (size_t)vol.width;
				coord.y = y + vol.y;

				for (int x = 0; x < vol.width; x++)
				{
					coord.x = x + vol.x;
					Weight wt;
					float s = 0, c = 0;

					ValCoef(origin, coord, coeffv[idx], &wt);
					BilinearValue(proj, wt, &s, &c);

					if (c)
					{
						*(vdrefy + x) += (float)(s / c);
					}
				}
			}
		}
	}
	delete[] proj.data;
}

void FBP(const Point3DF &origin, MrcStackM &projs, Volume &vol,
		 Coeff coeffv[], int filterMode)
{
	size_t projsize = static_cast<size_t>(projs.Y()) * static_cast<size_t>(projs.X()) * static_cast<size_t>(projs.Z());

	Slice proj(projs.X(), projs.Y(), NULL);
	try
	{
		proj.data = new float[projsize];
	}
	catch (std::bad_alloc &ba)
	{
		std::cerr << "Failed to allocate memory: " << ba.what() << '\n';
	}

	Point3D coord;
	memset(vol.data, 0, sizeof(float) * vol.length * vol.width * vol.height);

	projs.ReadBlock(0, projs.Z(), 'z', proj.data);
	ApplyFilterInplace(projs, proj.data, projs.header.ny, filterMode);

	for (int idx = 0; idx < projs.Z(); idx++)
	{

		for (int z = 0; z < vol.height; z++)
		{
			float *vdrefz = vol.data + z * (size_t)vol.width * vol.length;
			coord.z = z + vol.z;

			for (int y = 0; y < vol.length; y++)
			{
				float *vdrefy = vdrefz + y * (size_t)vol.width;
				coord.y = y + vol.y;

				for (int x = 0; x < vol.width; x++)
				{
					coord.x = x + vol.x;
					Weight wt;
					float s = 0, c = 0;
					float *oneproj = proj.data + projs.X() * projs.Y() * idx;
					ValCoef(origin, coord, coeffv[idx], &wt);
					BilinearValue(proj, oneproj, wt, &s, &c);

					if (c)
					{
						*(vdrefy + x) += (float)(s / c);
					}
				}
			}
		}
	}
	delete[] proj.data;
}

void UpdateVolumeByProjDiff(const Point3DF &origin, const Slice &diff,
							Volume &vol, float gamma, const Coeff &coeff)
{

	Point3D coord;

	for (int z = 0; z < vol.height; z++)
	{
		float *vdrefz = vol.data + z * (size_t)vol.width * vol.length;
		coord.z = z + vol.z;

		for (int y = 0; y < vol.length; y++)
		{
			float *vdrefy = vdrefz + y * vol.width;
			coord.y = y + vol.y;

			for (int x = 0; x < vol.width; x++)
			{
				coord.x = x + vol.x;
				Weight wt;
				float s = 0, c = 0;

				ValCoef(origin, coord, coeff, &wt);
				BilinearValue(diff, wt, &s, &c);

				if (c)
				{
					*(vdrefy + x) += (float)(s / c) * gamma;
				}
			}
		}
	}
}

void UpdateWeightsByProjDiff(const Point3DF &origin, const Slice &diff,
							 Volume &values, Volume &weights,
							 const Coeff &coeff)
{

	Point3D coord;

	for (int z = 0; z < values.height; z++)
	{
		float *vdrefz = values.data + z * (size_t)values.width * values.length;
		float *wdrefz = weights.data + z * (size_t)weights.width * weights.length;
		coord.z = z + values.z;

		for (int y = 0; y < values.length; y++)
		{
			float *vdrefy = vdrefz + y * values.width;
			float *wdrefy = wdrefz + y * weights.width;
			coord.y = y + values.y;

			for (int x = 0; x < values.width; x++)
			{
				coord.x = x + values.x;
				Weight wt;
				float s = 0, c = 0;

				ValCoef(origin, coord, coeff, &wt);
				BilinearValue(diff, wt, &s, &c);

				*(vdrefy + x) += s;
				*(wdrefy + x) += c;
			}
		}
	}
}

void UpdateVolumeByWeights(Volume &vol, Volume &values, Volume &weights,
						   float gamma)
{
	size_t pxsize = vol.height * vol.width * vol.length;

	for (size_t i = pxsize; i--;)
	{
		if (weights.data[i])
		{
			vol.data[i] += values.data[i] / weights.data[i] * gamma;
		}
	}
}

void SART(const Point3DF &origin, MrcStackM &projs, Volume &vol, Coeff coeffv[],
		  int iteration // 迭代次数
		  ,
		  float gamma // 松弛变量
)
{
	int pxsize = projs.X() * projs.Y();
	Slice reproj_val(projs.X(), projs.Y()); // reprojection value
	Slice reproj_wt(projs.X(), projs.Y());	// reprojection weight
	Slice projection(projs.X(), projs.Y());

	for (int i = 0; i < iteration; i++)
	{

		for (int idx = 0; idx < projs.Z(); idx++)
		{
			printf("Begin to read %d projection (iteraion %d)\n", idx, i);
			memset(reproj_val.data, 0, sizeof(float) * pxsize);
			memset(reproj_wt.data, 0, sizeof(float) * pxsize);

			Reproject(origin, vol, coeffv[idx], reproj_val, reproj_wt);

			MPI_Allreduce(MPI_IN_PLACE, reproj_val.data, pxsize, MPI_FLOAT, MPI_SUM,
						  MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, reproj_wt.data, pxsize, MPI_FLOAT, MPI_SUM,
						  MPI_COMM_WORLD);

			// printf("SART begin to read %d projection for %d z-coordinate (iteraion "
			// 	   "%d)\n",
			// 	   idx, vol.z, i);
			projs.ReadSliceZ(idx, projection.data);

			for (int n = 0; n < pxsize; n++)
			{
				if (reproj_wt.data[n] != 0)
				{
					reproj_val.data[n] /= reproj_wt.data[n];
				}
				reproj_val.data[n] = projection.data[n] - reproj_val.data[n];
			}

			UpdateVolumeByProjDiff(origin, reproj_val, vol, gamma, coeffv[idx]);
		}
	}
}

void SIRT(const Point3DF &origin, MrcStackM &projs, Volume &vol, Coeff coeffv[],
		  int iteration, float gamma)
{
	int pxsize = projs.X() * projs.Y();
	Slice reproj_val(projs.X(), projs.Y()); // reprojection value
	Slice reproj_wt(projs.X(), projs.Y());	// reprojection weight
	Slice projection(projs.X(), projs.Y());

	Volume valvol(vol.x, vol.y, vol.z, vol.length, vol.width, vol.height);
	Volume wtvol(vol.x, vol.y, vol.z, vol.length, vol.width, vol.height);

	for (int i = 0; i < iteration; i++)
	{
		memset(valvol.data, 0,
			   sizeof(float) * valvol.length * valvol.width * valvol.height);
		memset(wtvol.data, 0,
			   sizeof(float) * wtvol.length * wtvol.width * wtvol.height);

		for (int idx = 0; idx < projs.Z(); idx++)
		{
			printf("Begin to read %d projection (iteraion %d)\n", idx, i);
			memset(reproj_val.data, 0, sizeof(float) * pxsize);
			memset(reproj_wt.data, 0, sizeof(float) * pxsize);

			Reproject(origin, vol, coeffv[idx], reproj_val,
					  reproj_wt); // vol is not changed during iteration

			MPI_Allreduce(MPI_IN_PLACE, reproj_val.data, pxsize, MPI_FLOAT, MPI_SUM,
						  MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, reproj_wt.data, pxsize, MPI_FLOAT, MPI_SUM,
						  MPI_COMM_WORLD);

			// printf("SIRT begin to read %d projection for %d z-coordinate (iteraion "
			// 	   "%d)\n",
			// 	   idx, vol.z, i);
			projs.ReadSliceZ(idx, projection.data);

			for (int n = 0; n < pxsize; n++)
			{
				if (reproj_wt.data[n])
				{
					reproj_val.data[n] /= reproj_wt.data[n];
				}
				reproj_val.data[n] = projection.data[n] - reproj_val.data[n];
			}

			UpdateWeightsByProjDiff(origin, reproj_val, valvol, wtvol, coeffv[idx]);
		}

		UpdateVolumeByWeights(vol, valvol, wtvol, gamma);
	}
	delete[] reproj_val.data;
	delete[] reproj_wt.data;
	delete[] projection.data;
}
void Reproject_admm_htb(const Point3DF &origin, const Volume &vol, const Coeff &coeff, float *htb, const Slice &slc)
{
	Point3D coord;
	int n;
	size_t volsize = vol.width * vol.length * vol.height;
	for (int z = 0; z < vol.height; z++)
	{
		float *vdrefz = vol.data + z * (size_t)vol.width * vol.length;
		float *htb_z = htb + z * (size_t)vol.width * vol.length; 
		coord.z = z + vol.z;

		for (int y = 0; y < vol.length; y++)
		{
			float *vdrefy = vdrefz + y * vol.width;
			float *htb_y = htb_z + y * vol.width;
			coord.y = y + vol.y;

			for (int x = 0; x < vol.width; x++)
			{
				float *vdrefx = vdrefy + x;
				float *htb_x = htb_y + x;
				coord.x = x + vol.x;
				Weight wt;
				ValCoef(origin, coord, coeff, &wt);
				
				if (wt.x_min >= 0 && wt.x_min < slc.width && wt.y_min >= 0 &&
					wt.y_min < slc.height)
				{	//(x_min, y_min)
					n = wt.x_min + wt.y_min * vol.width;							
					*htb_x += (1 - wt.x_min_del) * (1 - wt.y_min_del) * slc.data[n]; 
				}
				if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < slc.width &&
					wt.y_min >= 0 && wt.y_min < slc.height)
				{											 //(x_min+1, y_min)
					n = wt.x_min + 1 + wt.y_min * vol.width;
					*htb_x += wt.x_min_del * (1 - wt.y_min_del) * slc.data[n];
				}
				if (wt.x_min >= 0 && wt.x_min < slc.width && (wt.y_min + 1) >= 0 &&
					(wt.y_min + 1) < slc.height)
				{											   //(x_min, y_min+1)
					n = wt.x_min + (wt.y_min + 1) * vol.width;
					*htb_x += (1 - wt.x_min_del) * wt.y_min_del * slc.data[n];
				}
				if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < slc.width &&
					(wt.y_min + 1) >= 0 &&
					(wt.y_min + 1) < slc.height)
				{													 //(x_min+1, y_min+1)
					n = (wt.x_min + 1) + (wt.y_min + 1) * vol.width;
					*htb_x += wt.x_min_del * wt.y_min_del * slc.data[n];
				}
			}
		}
	}
}
void Reproject_admm_atax(const Point3DF &origin, const Volume &vol, float *x0, Coeff coeffv[],
						 float *atax, int proj)
{
	Point3D coord;
	int n;
	size_t volsize = vol.width * vol.length * vol.height;
	float *ax = (float *)malloc(sizeof(float) * vol.width * vol.length);
	float *w = (float *)malloc(sizeof(float) * vol.width * vol.length);
	int pxsize = vol.width * vol.length;
	float a = 0;

	for (int idx = 0; idx < proj; idx++)
	{
		memset(ax, 0, sizeof(float) * vol.width * vol.length); 
		memset(w, 0, sizeof(float) * vol.width * vol.length); 
		for (int z = 0; z < vol.height; z++)
		{

			float *vdrefz = x0 + z * (size_t)vol.width * vol.length;
			coord.z = z + vol.z;

			for (int y = 0; y < vol.length; y++)
			{
				float *vdrefy = vdrefz + y * vol.width;
				coord.y = y + vol.y;

				for (int x = 0; x < vol.width; x++)
				{
					float *vdrefx = vdrefy + x;
					coord.x = x + vol.x;
					Weight wt;
					ValCoef(origin, coord, coeffv[idx], &wt);
					if (wt.x_min >= 0 && wt.x_min < vol.width && wt.y_min >= 0 &&
						wt.y_min < vol.length)
					{										 //(x_min, y_min)
						n = wt.x_min + wt.y_min * vol.width; 

						ax[n] += (1 - wt.x_min_del) * (1 - wt.y_min_del) * (*vdrefx); 
						w[n] += (1 - wt.x_min_del) * (1 - wt.y_min_del);
					}
					if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < vol.width &&
						wt.y_min >= 0 && wt.y_min < vol.length)
					{											 //(x_min+1, y_min)
						n = wt.x_min + 1 + wt.y_min * vol.width; 
						ax[n] += wt.x_min_del * (1 - wt.y_min_del) * (*vdrefx);
						w[n] += wt.x_min_del * (1 - wt.y_min_del);
					}
					if (wt.x_min >= 0 && wt.x_min < vol.width && (wt.y_min + 1) >= 0 &&
						(wt.y_min + 1) < vol.length)
					{											   //(x_min, y_min+1)
						n = wt.x_min + (wt.y_min + 1) * vol.width;

						ax[n] += (1 - wt.x_min_del) * wt.y_min_del * (*vdrefx);
						w[n] += (1 - wt.x_min_del) * wt.y_min_del;	
					}
					if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < vol.width &&
						(wt.y_min + 1) >= 0 &&
						(wt.y_min + 1) < vol.length)
					{													 //(x_min+1, y_min+1)
						n = (wt.x_min + 1) + (wt.y_min + 1) * vol.width; 
						ax[n] += wt.x_min_del * wt.y_min_del * (*vdrefx);
						w[n] += wt.x_min_del * wt.y_min_del;
					}
				}
			}
		}
		MPI_Allreduce(MPI_IN_PLACE, ax, pxsize, MPI_FLOAT, MPI_SUM,
					  MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, w, pxsize, MPI_FLOAT, MPI_SUM,
					  MPI_COMM_WORLD);


		for (int z = 0; z < vol.height; z++)
		{
			float *atax_z = atax + z * (size_t)vol.width * vol.length;
			coord.z = z + vol.z;

			for (int y = 0; y < vol.length; y++)
			{
				float *atax_y = atax_z + y * vol.width;

				coord.y = y + vol.y;

				for (int x = 0; x < vol.width; x++)
				{
					float *atax_x = atax_y + x; 

					coord.x = x + vol.x;
					Weight wt;
					ValCoef(origin, coord, coeffv[idx], &wt);

					if (wt.x_min >= 0 && wt.x_min < vol.width && wt.y_min >= 0 &&
						wt.y_min < vol.length)
					{										 //(x_min, y_min)
						n = wt.x_min + wt.y_min * vol.width;
						if (fabs(w[n]) > 10e-6)
						{ 
							*atax_x += (1 - wt.x_min_del) * (1 - wt.y_min_del) * ax[n] / w[n];
						}
					} 
					if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < vol.width &&
						wt.y_min >= 0 && wt.y_min < vol.length)
					{											 //(x_min+1, y_min)
						n = wt.x_min + 1 + wt.y_min * vol.width; 
						if (fabs(w[n]) > 10e-6)
						{
							*atax_x += wt.x_min_del * (1 - wt.y_min_del) * ax[n] / w[n];
						}
					}
					if (wt.x_min >= 0 && wt.x_min < vol.width && (wt.y_min + 1) >= 0 &&
						(wt.y_min + 1) < vol.length)
					{											   //(x_min, y_min+1)
						n = wt.x_min + (wt.y_min + 1) * vol.width;
						if (fabs(w[n]) > 10e-6)
						{ 
							*atax_x += (1 - wt.x_min_del) * wt.y_min_del * ax[n] / w[n];
						}
					}
					if ((wt.x_min + 1) >= 0 && (wt.x_min + 1) < vol.width &&
						(wt.y_min + 1) >= 0 &&
						(wt.y_min + 1) < vol.length)
					{													 //(x_min+1, y_min+1)
						n = (wt.x_min + 1) + (wt.y_min + 1) * vol.width; 
						if (fabs(w[n]) > 10e-6)
						{
							*atax_x += wt.x_min_del * wt.y_min_del * ax[n] / w[n];
						}
					}
				}
			}
		}
	}
	free(ax);
	free(w);
}

void ATbmuLT(float *atb, float *uk, float *dk, int width, int length, int height, float mu)
{
	size_t volsize = width * length * height;
	for (int z = 0; z < height; z++)
	{
		float *drez_atb = atb + z * (size_t)width * length;
		float *drez_u = uk + z * (size_t)width * length;
		float *drez_d = dk + z * (size_t)width * length;

		for (int y = 0; y < length; y++)
		{
			float *drey_atb = drez_atb + y * width;
			float *drey_u = drez_u + y * width;
			float *drey_d = drez_d + y * width;

			for (int x = 0; x < width; x++)
			{
				float *drex_atb = drey_atb + x;
				float *drex_u = drey_u + x;
				float *drex_d = drey_d + x;
				
				*drex_atb += ((*drex_u) - (*drex_d)) * mu;
			}
		}
	}
}
void ATAmuLTL(float *vol, float *ata, float mu,
			  int width, int length, int height)
{
	size_t n;

	for (int z = 0; z < height; z++)
	{
		float *drez = vol + z * (size_t)width * length; 
		float *drez_ATA = ata + z * (size_t)width * length;

		for (int y = 0; y < length; y++)
		{
			float *drey = drez + y * width;
			float *drey_ATA = drez_ATA + y * width;

			for (int x = 0; x < width; x++)
			{
				float *drex = drey + x;
				float *dreATA = drey_ATA + x;						
				*dreATA += mu * (*drex);
			}
		}
	}
}
void applycg(Volume &vol, float *atax0, float *ATb, int numberIteration, float mu, 
             const Point3DF &origin, Coeff coeffv[], int proj, MrcStackM &projs)
{
	size_t volsize = vol.length * vol.width * vol.height;
	float *r0, *p0;

	if ((r0 = (float *)malloc(sizeof(float) * volsize)) == NULL)
	{
		printf("false2");
	}
	if ((p0 = (float *)malloc(sizeof(float) * volsize)) == NULL)
	{
		printf("false3");
	}
	float *ax = (float *)malloc(sizeof(float) * volsize);

	double d1;
	double d0;
	double beta = 0;

	for (int k = 0; k < volsize; k++)
	{
		r0[k] = ATb[k] - atax0[k];
		p0[k] = r0[k];
	}

	float *h0;
	if ((h0 = (float *)malloc(sizeof(float) * volsize)) == NULL)
	{
		printf("false1");
	}

	for (int i = 0; i < numberIteration; i++)
	{
		memset(h0, 0, sizeof(float) * volsize);
		{
			Reproject_admm_atax(origin, vol, p0, coeffv, h0, proj);	  
			ATAmuLTL(p0, h0, mu, vol.width, vol.length, vol.height);
		}

		double alpha = 0;
		double d2 = 0;
		d1 = 0;
		d0 = 0;
		for (int k = 0; k < volsize; k++)
		{
			d0 += r0[k] * r0[k];
			d2 += h0[k] * p0[k]; 
		}
		MPI_Allreduce(MPI_IN_PLACE, &d0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &d2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		if (d2 > 10e-6 || -d2 > 10e-6) 
		{
			alpha = d0 / d2;
		}
		else
		{
			alpha = 0;
		}

		for (int k = 0; k < volsize; k++)
		{
			vol.data[k] = vol.data[k] + alpha * p0[k]; 
		}

		{
			Reproject_admm_atax(origin, vol, vol.data, coeffv, ax, proj);
			ATAmuLTL(vol.data, ax, mu, vol.width, vol.length, vol.height);
		}

		for (int k = 0; k < volsize; k++)
		{
			r0[k] = ATb[k] - ax[k];  
			d1 += r0[k] * h0[k];
		}
		MPI_Allreduce(MPI_IN_PLACE, &d1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		if (d2 > 10e-6 || -d2 > 10e-6)
		{
			beta = d1 / d2;
		}
		else
		{
			beta = 0;
		}

		for (int k = 0; k < volsize; k++)
		{	
			p0[k] = r0[k] + beta * p0[k]; 	
		}
	}
	free(h0);
	free(r0);
	free(p0);
	free(ax);
}

void soft_admm(float *u_k, float *d_k, float *voldata, float soft, size_t volsize)
{
	for (size_t k = 0; k < volsize; k++)
	{
		if (voldata[k] + d_k[k] < -soft)
		{
			u_k[k] = voldata[k] + d_k[k] + soft;
		}
		else if (voldata[k] + d_k[k] > soft)
		{
			u_k[k] = voldata[k] + d_k[k] - soft;
		}
		else
		{
			u_k[k] = 0;
		}
	}
}
void ADMM(const Point3DF &origin, MrcStackM &projs, Volume &vol, Coeff coeffv[], 
		  int maxOutIter, int numberIteration, float mu, float soft)
{
	int pxsize = projs.X() * projs.Y(); 
	size_t volsize = vol.length * vol.width * vol.height; 

	Slice proj(projs.X(), projs.Y()); 
	Point3D coord; 


	float *htb = (float *)malloc(sizeof(float) * volsize); 

	memset(htb, 0, sizeof(float) * volsize); 

	for (int idx = 0; idx < projs.Z(); idx++) 
	{
		projs.ReadSliceZ(idx, proj.data); 
		Reproject_admm_htb(origin, vol, coeffv[idx], htb, proj);
	}

	float *u_k = (float *)malloc(sizeof(float) * volsize);
	float *d_k = (float *)malloc(sizeof(float) * volsize);
	memset(u_k, 0, sizeof(float) * volsize);
	memset(d_k, 0, sizeof(float) * volsize);

	for (int i = 0; i < maxOutIter; i++)
	{
		printf("**  ADMM iter:%d  **\n", i);
		float *x0 = (float *)malloc(sizeof(float) * volsize);
		memset(x0, 0, sizeof(float) * volsize);
	
		ATbmuLT(htb, u_k, d_k, vol.width, vol.length, vol.height, mu);

		{
			Reproject_admm_atax(origin, vol, vol.data, coeffv, x0, projs.Z()); 
			ATAmuLTL(vol.data, x0, mu, vol.width, vol.length, vol.height); 
		}

		applycg(vol, x0, htb, numberIteration, mu, origin, coeffv, projs.Z(), projs); 
		free(x0);

		//update uk
		soft_admm(u_k, d_k, vol.data, soft, volsize);
		
		//update dk
		for (size_t k = 0; k < volsize; k++)
		{
			d_k[k] += vol.data[k] - u_k[k];
		}
		
	}

	free(u_k);
	free(d_k);
	free(htb);
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
	mrcvol.SetSize(projs.X(), projs.Y(), opt.thickness); // test---------------------

	std::vector<float> angles;
	ReadAngles(angles, opt.angle);

	std::vector<Coeff> params;
	TranslateAngleToCoefficients(angles, params);

	Geometry geo;
	geo.offset = opt.offset;
	geo.pitch_angle = opt.pitch_angle;
	geo.zshift = opt.zshift;

	DecorateCoefficients(params, geo);

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

	Volume vol(0, 0, volz, mrcvol.Y(), mrcvol.X(), height);

	std::cout << myid << ": (" << vol.x << "," << vol.y << "," << vol.z << ")"
			  << "&(" << vol.width << "," << vol.length << "," << vol.height
			  << ")" << std::endl;

	Point3DF origin;

	origin.x = mrcvol.X() * .5;
	origin.y = mrcvol.Y() * .5;
	origin.z = mrcvol.Z() * .5;

	if (myid == 0)
	{
		printf("origin.x is %f, origin.y is %f, origin.z is %f\n", origin.x,
			   origin.y, origin.z);
	}

	size_t size = static_cast<size_t>(vol.length) * static_cast<size_t>(vol.width) * static_cast<size_t>(vol.height);


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
	if (reference[0] == '\0')
	{
		size_t batchSize = vol.width * vol.length * 2;
		size_t totalSize = static_cast<size_t>(vol.width) * static_cast<size_t>(vol.length) * static_cast<size_t>(vol.height);
		float *ptr = vol.data;
		for (size_t i = 0; i < totalSize; i += batchSize)
		{
			size_t currentBatchSize = std::min(batchSize, totalSize - i);
			memset(ptr, 0, currentBatchSize * sizeof(float));
			ptr += currentBatchSize;
		}
	}
	else
	{
		MrcStackM init;
		init.ReadFile(reference);
		init.ReadHeader();
		init.ReadBlock(vol.z, vol.z + vol.height, 'z', vol.data);
		init.Close();
	}
	if (opt.method == "BPT")
	{
		printf("Start using BPT for reconstruction. \n ");
		BackProject(origin, projs, vol, &params[0]);
	}
	else if (opt.method == "SART")
	{
		printf("Start using SART for reconstruction. \n ");
		SART(origin, projs, vol, &params[0], opt.iteration, opt.gamma);
	}
	else if (opt.method == "SIRT")
	{
		printf("Start using SIRT for reconstruction.\n  ");
		SIRT(origin, projs, vol, &params[0], opt.iteration, opt.gamma);
	}
	else if (opt.method == "FBP")
	{
		printf("Start using FBP for reconstruction. \n ");
		FBP(origin, projs, vol, &params[0], 0);
	}
	else if (opt.method == "WBP")
	{
		printf("Start using WBP for reconstruction.\n  ");
		FBP(origin, projs, vol, &params[0], 2);
	}
	else if (opt.method == "ADMM")
	{
		ADMM(origin, projs, vol, &params[0], opt.iteration, opt.cgiter, opt.gamma, opt.soft);
	}

	// TestMrcWriteToFile(mrcvol, opt.output);
	mrcvol.WriteToFile(opt.output); // test------------------------------

	if (myid == 0)
	{
		mrcvol.WriteHeader();
	}

	mrcvol.WriteBlock<float>(vol.z, vol.z + height, 'z', vol.data); // test-------------------------
	// TestMrcWriteBlock(mrcvol, vol.z, vol.z + height, 'z', vol.data); // test-------------------------

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
