#pragma once
#include "ImagePyramid.h"
//#include <Eigen/Dense>
//#include <Eigen/Sparse>



class OpticalFlow_Huber_L1
{

public:
	enum DESCRIPTOR { POINT_INTENSITY, PATCH_INTENSITY, D1, CENSUS};

    struct Data
	{
		float** I1;
		float** I2;
		float** alpha; //weight image on prior term
		float** uprior;
		float** vprior;
		float** u;
		float** v;
		int nCols;
		int nRows;
	};

	struct Parms
	{
		int warps;
		int outerIters;
		int innerIters;
		float scalefactor;
		int levels;
		int minSize;
		float Lambda; //weight parameter for data term
		float theta; //weight parameter for (u-1)^2
		float epsilon; // huber epsilon
		DESCRIPTOR descriptor;
	};


private:
	float** m_I1;
	float** m_I2;

	float** m_alpha;
	float** m_u1prior;
	float** m_u2prior;

	//descriptors for image1 and image2
	float*** m_D1;
	float*** m_D2;

    //dual variables for v1, v2
	float** m_p1x;
	float** m_p1y;

	float** m_p2x;
	float** m_p2y;

	//primal variables for v1, v2
	float** m_v1;
	float** m_v2;
	float** m_v1_;
	float** m_v2_;


	//primal variables for u1 and u2
	float** m_u1;
	float** m_u10;
	float** m_u1_;

	float** m_u2;
	float** m_u20;
	float** m_u2_;


	//size of the current image
	int m_nCols;
	int m_nRows;

	//full res image
	int m_nCols0;
	int m_nRows0;

	ImagePyramid m_imagePyramid1;
	ImagePyramid m_imagePyramid2;
	ImagePyramid m_alphaPyramid;
	ImagePyramid m_uPriorPyramid;
	ImagePyramid m_vPriorPyramid;
	Parms m_parms;

	int m_nDescriptors;

	float** m_I2w;
	float*** m_Dx;
	float*** m_Dy;

	//Dual variables for descriptors needed for L1 term
	float*** m_qD;


	void (OpticalFlow_Huber_L1::*compute_feature_map)(float***, float**);

	void warp();
	void compute_feature_map_derivatives();
	void compute_feature_map_D1(float*** D, float** img);
	void compute_feature_map_PatchIntensity(float*** D, float** img);
	void compute_feature_map_PointIntensity(float*** D, float** img);
	void compute_feature_map_Census(float*** D, float** img);
	void updateUV();
	void updateV();
	void updateU();
	void updateV_Dual(int i, int j);
	void updateV_Primal(int i, int j);
	void updateU_Dual(int i, int j);
	void updateU_Primal(int i, int j);
	inline float Tex2D(float** t, int w, int h, float x, float y);
	float atPt(float** image, int y, int x);

public:
	OpticalFlow_Huber_L1();
	~OpticalFlow_Huber_L1();

	void setup(Data inputData, Parms intParms);

	void run();
};