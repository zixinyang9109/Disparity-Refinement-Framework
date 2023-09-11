#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "ImageIO.h"
#include "gaussianFilter.h"
#include "ImagePyramid.h"

///////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////
ImagePyramid::ImagePyramid()
{
	m_nLevels = 1;

	m_nCols = NULL;
	m_nRows = NULL;

	m_Pyramid = NULL;

	m_bImg = false;
	m_bAllocated = false;

}

//////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////
ImagePyramid::~ImagePyramid()
{
	dealloc();

}

//////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////
void ImagePyramid::dealloc()
{
	FreeArray(m_nCols);
	FreeArray(m_nRows);

	m_nCols = NULL;
	m_nRows = NULL;

	if (m_Pyramid)
	{
		//do not free zero image
		for (int i = 1; i < m_nLevels; i++)
			FreeImage(m_Pyramid[i]);

		delete[] m_Pyramid;
		m_Pyramid = NULL;
	}

	m_bImg = false;
}
//////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////
float ** ImagePyramid::getImage(int iLevel)
{
	if (!m_bImg)
		return NULL;

	if (iLevel < 0 || iLevel > m_nLevels - 1)
		return NULL;

	return m_Pyramid[iLevel];

}


//////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////
int ImagePyramid::getRows(int iLevel)
{
	if (!m_bImg)
		return -1;

	if (iLevel < 0 || iLevel > m_nLevels - 1)
		return -1;

	return m_nRows[iLevel];
}

//////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////
int ImagePyramid::getCols(int iLevel)
{

	if (!m_bImg)
		return -1;

	if (iLevel < 0 || iLevel > m_nLevels - 1)
		return -1;

	return m_nCols[iLevel];
}

//////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////
void ImagePyramid::allocatePyramid(int nCols, int nRows, float scaleFactor, int nLevels)
{
	m_nLevels = nLevels;

	m_nCols = AllocateArray<int>(nLevels);
	m_nRows = AllocateArray<int>(nLevels);
	m_Pyramid = new float** [nLevels];


	//m_gaussianPyramid[0] = image;
	m_nCols[0] = nCols;
	m_nRows[0] = nRows;

	for (int level = 1; level < m_nLevels; level++)
	{
		//printf("current level %d\n", currentLevel);

		int nw = m_nCols[level - 1];
		int nh = m_nRows[level - 1];

		//int snw = round((float)nw * scaleFactor);
		//int snh = round((float)nh * scaleFactor);

		int snw = ((float)nw * scaleFactor);
		int snh = ((float)nh * scaleFactor);

		m_Pyramid[level] = AllocateImage<float>(snw, snh);

		m_nCols[level] = snw;
		m_nRows[level] = snh;
	}

	m_bAllocated = true;
}

//////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////
void ImagePyramid::createPyramid(float** image, float scale)
{


	m_Pyramid[0] = image;

	for (int level = 1; level < m_nLevels; level++)
	{
		//printf("current level %d\n", currentLevel);

		int nw = m_nCols[level-1];
		int nh = m_nRows[level-1];

		int snw = m_nCols[level];
		int snh = m_nRows[level];

			resizeImage(m_Pyramid[level-1], nw, nh,
				snw, snh, (float**)m_Pyramid[level], scale);

	}

	m_bImg = true;
}

///////////////////////////////////////////////////////////////////////////////
//
//////////////////////////////////////////////////////////////////////////////
void ImagePyramid::createPyramid(float** image, int nCols, int nRows, float scaleFactor, int nLevels, float scale)
{
	m_nLevels = nLevels;

	m_nCols = AllocateArray<int>(nLevels);
	m_nRows = AllocateArray<int>(nLevels);
	m_Pyramid = new float**[nLevels];

	m_Pyramid[0] = image;
	m_nCols[0] = nCols;
	m_nRows[0] = nRows;


	for (int level = 1; level < m_nLevels; level++)
	{
		//printf("current level %d\n", currentLevel);

		int nw = m_nCols[level-1];
		int nh = m_nRows[level-1];

		int snw = round((float)nw * scaleFactor);
		int snh = round((float)nh * scaleFactor);

		//int snw = ((float)nw * scaleFactor);
		//int snh = ((float)nh * scaleFactor);

		m_Pyramid[level] = AllocateImage<float>(snw, snh);

		resizeImage(m_Pyramid[level-1], nw, nh,
			snw, snh, (float **)m_Pyramid[level], scale);

		m_nCols[level] = snw;
		m_nRows[level] = snh;
	}

	m_bImg = true;

}

///////////////////////////////////////////////////////////////////////////////
/// \brief host texture fetch
///
/// read from arbitrary position within image using bilinear interpolation
/// out of range coords are mirrored
/// \param[in]  t   texture raw data
/// \param[in]  w   texture width
/// \param[in]  h   texture height
/// \param[in]  s   texture stride
/// \param[in]  x   x coord of the point to fetch value at
/// \param[in]  y   y coord of the point to fetch value at
/// \return fetched value
///////////////////////////////////////////////////////////////////////////////
inline float ImagePyramid::Tex2D(float **t, int w, int h, float x, float y)
{
	// integer parts in floating point format
	float intPartX, intPartY;

	// get fractional parts of coordinates
	float dx = fabsf(modff(x, &intPartX));
	float dy = fabsf(modff(y, &intPartY));

	// assume pixels are squares
	// one of the corners
	int ix0 = (int)intPartX;
	int iy0 = (int)intPartY;

	// mirror out-of-range position
	if (ix0 < 0) ix0 = abs(ix0 + 1);

	if (iy0 < 0) iy0 = abs(iy0 + 1);

	if (ix0 >= w) ix0 = w * 2 - ix0 - 1;

	if (iy0 >= h) iy0 = h * 2 - iy0 - 1;

	// corner which is opposite to (ix0, iy0)
	int ix1 = ix0 + 1;
	int iy1 = iy0 + 1;

	if (ix1 >= w) ix1 = w * 2 - ix1 - 1;

	if (iy1 >= h) iy1 = h * 2 - iy1 - 1;

	float res = t[iy0][ix0] * (1.0f - dx) * (1.0f - dy);
	res += t[iy0][ix1] * dx * (1.0f - dy);
	res += t[iy1][ix0] * (1.0f - dx) * dy;
	res += t[iy1][ix1] * dx * dy;

	return res;
}

///////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////
void ImagePyramid::resizeImage(float** src, int width, int height,
	int newWidth, int newHeight, float** out, float scale)
{
	float factor = (float)width / (float)newWidth;

	float off = 0.5f * (1.0f / factor) - 1.0f + 0.5f;

	if (factor > 1.0f)
	{
		float sigma = factor / 3.0f;

		GaussianFilter filter(sigma, sigma);

		filter.filter2D(src, height, width);
	}

	for (int i = 0; i < newHeight; ++i)
	{
		for (int j = 0; j < newWidth; ++j)
		{
			float x = (float)j * factor + off;
			float y = (float)i * factor + off;
			out[i][j] = Tex2D(src, width, height, x, y) * scale;
		}
	}
}

