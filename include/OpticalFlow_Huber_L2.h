#pragma once
#include "ImagePyramid.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>



class OpticalFlow_Huber_L2
{

public:
	enum DESCRIPTOR { POINT_INTENSITY, PATCH_INTENSITY, D1, CENSUS };

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
		int iterations;
		float scalefactor;
		int levels;
		int minSize;
		float Lambda; //weight on data term
		float epsilon; // huber epsilon
		DESCRIPTOR descriptor;
        bool bInitializeWithPrior;
	};

    float** m_I2w;


private:
	float** m_I1;
	float** m_I2;

	float** m_alpha;
	float** m_uprior;
	float** m_vprior;

	//primal variables
	float** m_u;
	float** m_u0;
	float** m_u_;

	float** m_v;
	float** m_v0;
	float** m_v_;

	//dual variables for u,v
	float** m_pux;
	float** m_puy;

	float** m_pvx;
	float** m_pvy;

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

	//arrays needed for L2 data term
	float** m_Dx2;
	float** m_Dy2;
	float** m_DxDy;
	float** m_DxR;
	float** m_DyR;

	//descriptors for image1 and image2
	float*** m_D1;
	float*** m_D2;

	void (OpticalFlow_Huber_L2::*compute_feature_map)(float***, float**);

	void warp();
	void compute_feature_map_derivatives();
	void compute_feature_map_D1(float*** D, float** img);
	void compute_feature_map_PatchIntensity(float*** D, float** img);
	void compute_feature_map_PointIntensity(float*** D, float** img);
	void compute_feature_map_Census(float*** D, float** img);
	void solvePrimalDual();
	void updateDual();
	void updatePrimal();
	inline float Tex2D(float** t, int w, int h, float x, float y);
	float atPt(float** image, int y, int x);

public:
	OpticalFlow_Huber_L2();
	~OpticalFlow_Huber_L2();

	void setup(Data inputData, Parms intParms);

	void run();
};