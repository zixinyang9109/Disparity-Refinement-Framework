#pragma once
#include "ImagePyramid.h"
//#include <Eigen/Dense>
//#include <Eigen/Sparse>



class Stereo_Huber_L2
{

public:
	enum DESCRIPTOR { POINT_INTENSITY, PATCH_INTENSITY, D1, CENSUS };

	 struct Data
	{
		float** I1;
		float** I2;
		float** alpha; //weight image on prior term
		float** uprior;
		float** u;
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
		float lambda; //weight on data term
		float epsilon; // huber epsilon
		DESCRIPTOR descriptor;
        bool withPrior;
	};

    float diff_last;


private:

    float diff;

	float** m_I1;
	float** m_I2;

	float** m_alpha;
	float** m_uprior;

	//primal variables
	float** m_u;
	float** m_u0;
	float** m_u_;


	//dual variables for u,v
	float** m_pux;
	float** m_puy;

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
	Parms m_parms;

	int m_nDescriptors;

	float** m_I2w;

	//arrays needed for L2 data term
	float** m_Dx2;
	float** m_DxR;

	//descriptors for image1 and image2
	float*** m_D1;
	float*** m_D2;

	void (Stereo_Huber_L2::*compute_feature_map)(float***, float**);

	void warp();
	void compute_feature_map_derivatives();
	void compute_feature_map_D1(float*** D, float** img);
	void compute_feature_map_PatchIntensity(float*** D, float** img);
	void compute_feature_map_PointIntensity(float*** D, float** img);
	void compute_feature_map_Census(float*** D, float** img);
	void solvePrimalDual();
	void updateDual();
	void updatePrimal();
	void updateDual(int i, int j);
	void updatePrimal(int i, int j);
	inline float Tex2D(float** t, int w, int h, float x, float y);
	float atPt(float** image, int y, int x);

public:
	Stereo_Huber_L2();
	~Stereo_Huber_L2();

	void setup(Data inputData, Parms intParms);

	void run();
};