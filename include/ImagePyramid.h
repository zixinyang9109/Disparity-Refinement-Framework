#pragma once


class ImagePyramid
{

protected:
	int m_nLevels;

	int *m_nCols;
	int *m_nRows;

	float m_scaleFactor;
	float ***m_Pyramid;

	bool m_bImg;
	bool m_bAllocated;


public:

	ImagePyramid();
	~ImagePyramid();

	void allocatePyramid(int nCols, int nRows, float scaleFactor, int nLevels);
	void createPyramid(float** image, float scale = 1.0f);

	void createPyramid(float **image, int nCols, int nRows, float scaleFactor,  int nLevels, float scale = 1.0f);


	float **getImage(int iLevel);
	int getRows(int iLevel);
	int getCols(int iLevel);
	void dealloc();

	static void resizeImage(float** src, int width, int height,
		int newWidth, int newHeight, float** out, float scale = 1.0f);

	static inline float Tex2D(float **t, int w, int h, float x, float y);

protected:
};