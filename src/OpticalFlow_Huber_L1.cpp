#include <math.h>
#include "ImageIO.h"
//#include "matrix3D.h"
//#include "Common.h"
#include "ImagePyramid.h"
//#include <Eigen/Dense>
//#include <Eigen/Sparse>
//#include "fit.h"
#include "OpticalFlow_Huber_L1.h"
#include <omp.h>



////////////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////////
OpticalFlow_Huber_L1::OpticalFlow_Huber_L1()
{
	//dual variables
	m_p1x = NULL;
	m_p1y = NULL;

	m_p2x = NULL;
	m_p2y = NULL;

	m_qD = NULL;

	//primal
	m_u10 = NULL;
	m_u1_ = NULL;

	m_u20 = NULL;
	m_u2_ = NULL;

	m_v1_ = NULL;
	m_v2_ = NULL;
	m_v1 = NULL;
	m_v2 = NULL;

	m_nDescriptors = 8;

	m_I2w = NULL;

	m_Dx = NULL;
	m_Dy = NULL;
	m_D1 = NULL;
	m_D2 = NULL;

}

////////////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////////
OpticalFlow_Huber_L1::~OpticalFlow_Huber_L1()
{

	FreeImage(m_p1x);
	FreeImage(m_p1y);
	FreeImage(m_p2x);
	FreeImage(m_p2y);

	FreeImage(m_I2w);

	FreeImage(m_u10);
	FreeImage(m_u20);
	FreeImage(m_u1_);
	FreeImage(m_u2_);

	FreeImage(m_v1);
	FreeImage(m_v2);
	FreeImage(m_v1_);
	FreeImage(m_v2_);

	FreeVolume(m_D1, m_nDescriptors);
	FreeVolume(m_D2, m_nDescriptors);
	FreeVolume(m_qD, m_nDescriptors);
	FreeVolume(m_Dx, m_nDescriptors);
	FreeVolume(m_Dy, m_nDescriptors);

}

////////////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////////
void OpticalFlow_Huber_L1::setup(Data inputData, Parms parms)
{
	m_I1 = inputData.I1;
	m_I2 = inputData.I2;

	m_u1 = inputData.u;
	m_u2 = inputData.v;

	m_alpha = inputData.alpha;
	m_u1prior = inputData.uprior;
	m_u2prior = inputData.vprior;

	m_nCols0 = inputData.nCols;
	m_nRows0 = inputData.nRows;

	m_parms = parms;

	if (parms.descriptor == PATCH_INTENSITY)
	{
		m_nDescriptors = 9;
		compute_feature_map = &OpticalFlow_Huber_L1::compute_feature_map_PatchIntensity;
	}
	else if (parms.descriptor == POINT_INTENSITY)
	{
		m_nDescriptors = 1;
		compute_feature_map = &OpticalFlow_Huber_L1::compute_feature_map_PointIntensity;
	}
	else if (parms.descriptor == D1)
	{
		m_nDescriptors = 8;
		compute_feature_map = &OpticalFlow_Huber_L1::compute_feature_map_D1;
	}
	else
	{
		m_nDescriptors = 8;
		compute_feature_map = &OpticalFlow_Huber_L1::compute_feature_map_Census;
	}

	//create image pyramids
	m_imagePyramid1.createPyramid(m_I1, m_nCols0, m_nRows0, parms.scalefactor, parms.levels);
	m_imagePyramid2.createPyramid(m_I2, m_nCols0, m_nRows0, parms.scalefactor, parms.levels);

	if (m_alpha)
	{
		m_alphaPyramid.createPyramid(m_alpha, m_nCols0, m_nRows0, parms.scalefactor, parms.levels);
		m_uPriorPyramid.createPyramid(m_u1prior, m_nCols0, m_nRows0, parms.scalefactor, parms.levels, parms.scalefactor);
		m_vPriorPyramid.createPyramid(m_u2prior, m_nCols0, m_nRows0, parms.scalefactor, parms.levels, parms.scalefactor);
	}

	m_I2w = AllocateImage<float>(m_nCols0, m_nRows0);

	//dual variables for v
	m_p1x = AllocateImage<float>(m_nCols0, m_nRows0);
	m_p1y = AllocateImage<float>(m_nCols0, m_nRows0);
	m_p2x = AllocateImage<float>(m_nCols0, m_nRows0);
	m_p2y = AllocateImage<float>(m_nCols0, m_nRows0);

	//primal variables for u
	m_u10 = AllocateImage<float>(m_nCols0, m_nRows0);
	m_u20 = AllocateImage<float>(m_nCols0, m_nRows0);
	m_u1_ = AllocateImage<float>(m_nCols0, m_nRows0);
	m_u2_ = AllocateImage<float>(m_nCols0, m_nRows0);

	//primal variables for v
	m_v1 = AllocateImage<float>(m_nCols0, m_nRows0);
	m_v2 = AllocateImage<float>(m_nCols0, m_nRows0);
	m_v1_ = AllocateImage<float>(m_nCols0, m_nRows0);
	m_v2_ = AllocateImage<float>(m_nCols0, m_nRows0);

	//allocate memory for features
	m_D1 = AllocateVolume<float>(m_nCols0, m_nRows0, m_nDescriptors);
	m_D2 = AllocateVolume<float>(m_nCols0, m_nRows0, m_nDescriptors);

	//allocate memory for Dual variables for descriptors
	m_qD = AllocateVolume<float>(m_nCols0, m_nRows0, m_nDescriptors);

	//allocate memory for descriptor derivatives
	m_Dx = AllocateVolume<float>(m_nCols0, m_nRows0, m_nDescriptors);
	m_Dy = AllocateVolume<float>(m_nCols0, m_nRows0, m_nDescriptors);

	memset(m_p1x[0], 0, sizeof(float) * m_nCols0 * m_nRows0);
	memset(m_p2x[0], 0, sizeof(float) * m_nCols0 * m_nRows0);
	memset(m_p1y[0], 0, sizeof(float) * m_nCols0 * m_nRows0);
	memset(m_p2y[0], 0, sizeof(float) * m_nCols0 * m_nRows0);
	memset(m_u1[0], 0, sizeof(float) * m_nCols0 * m_nRows0);
	memset(m_u2[0], 0, sizeof(float) * m_nCols0 * m_nRows0);
	memset(m_v1[0], 0, sizeof(float) * m_nCols0 * m_nRows0);
	memset(m_v2[0], 0, sizeof(float) * m_nCols0 * m_nRows0);

	for(int d = 0;  d < m_nDescriptors; d++)
		memset(m_qD[d][0], 0, sizeof(float) * m_nCols0 * m_nRows0);

}

////////////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////////
void OpticalFlow_Huber_L1::run()
{
	float** temp = AllocateImage<float>(m_nCols0, m_nRows0);

	for (int level = m_parms.levels - 1; level >= 0; level--)
	{
		printf("level : %d\n", level);
		m_I1 = m_imagePyramid1.getImage(level);
		m_I2 = m_imagePyramid2.getImage(level);

		m_nCols = m_imagePyramid1.getCols(level);
		m_nRows = m_imagePyramid1.getRows(level);

		//compute feature map of I1
		(this->*compute_feature_map)(m_D1, m_I1);

		if (m_alpha)
		{
			m_alpha = m_alphaPyramid.getImage(level);
			m_u1prior = m_uPriorPyramid.getImage(level);
			m_u2prior = m_vPriorPyramid.getImage(level);
		}

		for (int k = 0; k < m_parms.warps; k++)
		{
			memcpy(m_u10[0], m_u1[0], m_nCols0 * m_nRows0 * sizeof(float));
			memcpy(m_u20[0], m_u2[0], m_nCols0 * m_nRows0 * sizeof(float));
			memcpy(m_u1_[0], m_u1[0], m_nCols0 * m_nRows0 * sizeof(float));
			memcpy(m_u2_[0], m_u2[0], m_nCols0 * m_nRows0 * sizeof(float));

			memcpy(m_v1_[0], m_v1[0], m_nCols0 * m_nRows0 * sizeof(float));
			memcpy(m_v2_[0], m_v2[0], m_nCols0 * m_nRows0 * sizeof(float));

			printf("warp : %d\n", k);

			//use current (u,v) estimate to warp image I2
			warp();

			//compute features of warped I2
			(this->*compute_feature_map)(m_D2, m_I2w);

			//calculate gradient of I2w features with respect to (u,v)
			compute_feature_map_derivatives();

			//update (u,v) estimate
			updateUV();
		}

		if (level != 0)
		{
			//upscale primal variables(u,v)
			ImagePyramid::resizeImage(m_u1, m_nCols, m_nRows, m_imagePyramid1.getCols(level - 1),
				m_imagePyramid1.getRows(level - 1), temp, 2);
			memcpy(m_u1[0], temp[0], m_nCols0 * m_nRows0 * sizeof(float));

			ImagePyramid::resizeImage(m_u2, m_nCols, m_nRows, m_imagePyramid1.getCols(level - 1),
				m_imagePyramid1.getRows(level - 1), temp, 2);
			memcpy(m_u2[0], temp[0], m_nCols0 * m_nRows0 * sizeof(float));

			ImagePyramid::resizeImage(m_v1, m_nCols, m_nRows, m_imagePyramid1.getCols(level - 1),
				m_imagePyramid1.getRows(level - 1), temp, 2);
			memcpy(m_v1[0], temp[0], m_nCols0 * m_nRows0 * sizeof(float));

			ImagePyramid::resizeImage(m_v2, m_nCols, m_nRows, m_imagePyramid1.getCols(level - 1),
				m_imagePyramid1.getRows(level - 1), temp, 2);
			memcpy(m_v2[0], temp[0], m_nCols0 * m_nRows0 * sizeof(float));

			//////upscale dual variables
			////ImagePyramid::resizeImage(m_pux, m_nCols, m_nRows, m_imagePyramid1.getCols(level - 1),
			////	m_imagePyramid1.getRows(level - 1), temp);
			////memcpy(m_pux[0], temp[0], m_nCols0 * m_nRows0 * sizeof(float));

			////ImagePyramid::resizeImage(m_puy, m_nCols, m_nRows, m_imagePyramid1.getCols(level - 1),
			////	m_imagePyramid1.getRows(level - 1), temp);
			////memcpy(m_puy[0], temp[0], m_nCols0 * m_nRows0 * sizeof(float));

			////ImagePyramid::resizeImage(m_pvx, m_nCols, m_nRows, m_imagePyramid1.getCols(level - 1),
			////	m_imagePyramid1.getRows(level - 1), temp);
			////memcpy(m_pvx[0], temp[0], m_nCols0 * m_nRows0 * sizeof(float));

			////ImagePyramid::resizeImage(m_pvy, m_nCols, m_nRows, m_imagePyramid1.getCols(level - 1),
			////	m_imagePyramid1.getRows(level - 1), temp);
			////memcpy(m_pvy[0], temp[0], m_nCols0 * m_nRows0 * sizeof(float));

			memset(m_p1x[0], 0, m_nCols0 * m_nRows0 * sizeof(float));
			memset(m_p1y[0], 0, m_nCols0 * m_nRows0 * sizeof(float));
			memset(m_p2x[0], 0, m_nCols0 * m_nRows0 * sizeof(float));
			memset(m_p2y[0], 0, m_nCols0 * m_nRows0 * sizeof(float));

			for (int d = 0; d < m_nDescriptors; d++)
				memset(m_qD[d][0], 0, sizeof(float) * m_nCols0 * m_nRows0);

		}
		
	}

	warp();
	memcpy(m_I2[0], m_I2w[0], m_nCols0 * m_nRows0 * sizeof(float));
}

/////////////////////////////////////////////////////////////////////////////////////
//Warp I2 using current estimate of (u,v) 
/////////////////////////////////////////////////////////////////////////////////////
void OpticalFlow_Huber_L1::warp()
{
	float x, y;
	for (int i = 0; i < m_nRows; i++)
	{
		for (int j = 0; j < m_nCols; j++)
		{
			x = (float)j + m_u1[i][j];
			y = (float)i + m_u2[i][j];
			m_I2w[i][j] = Tex2D(m_I2, m_nCols, m_nRows, x, y);
		}
	}

}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void OpticalFlow_Huber_L1::compute_feature_map_D1(float*** D, float** img)
{
	int offX[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
	int offY[] = { -1, -1, -1, 0, 0, 1, 1, 1 };

	//for(int i = 0; i < m_nDescriptors; i++)
	//	memset(D[i][0], 0, m_nCols0 * m_nRows0 * sizeof(float));

	for (int i = 0; i < m_nRows; i++)
	{
		for (int j = 0; j < m_nCols; j++)
		{
			float norm = 0.0f;
			for (int d = 0; d < m_nDescriptors; d++)
			{ 
				float fac = atPt(img, i + offY[d], j + offX[d]);
				D[d][i][j] = fac - img[i][j];
				norm += D[d][i][j] * D[d][i][j];
			}

			norm = sqrt(norm);

			for (int d = 0; d < m_nDescriptors; d++)
			{
				if (norm != 0.0f)
					D[d][i][j] /= norm;
				else
					D[d][i][j] = 0.0f;

			}

		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void OpticalFlow_Huber_L1::compute_feature_map_Census(float*** D, float** img)
{
	int offX[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
	int offY[] = { -1, -1, -1, 0, 0, 1, 1, 1 };

	//for (int i = 0; i < m_nDescriptors; i++)
	//	memset(D[i][0], 0, m_nCols0 * m_nRows0 * sizeof(float));

	for (int i = 0; i < m_nRows; i++)
	{
		for (int j = 0; j < m_nCols; j++)
		{
			float norm = 0.0f;
			for (int d = 0; d < m_nDescriptors; d++)
			{
				float fac = atPt(img, i + offY[d], j + offX[d]);
				if(fac - img[i][j] > 0.0f)
					D[d][i][j] = 1.0f;
				else
					D[d][i][j] = 0.0f;
			}

		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void OpticalFlow_Huber_L1::compute_feature_map_PatchIntensity(float*** D, float** img)
{
	int offX[] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	int offY[] = { -1, -1, 1, 0, 0, 0, 1, 1, 1 };

	//for (int i = 0; i < m_nDescriptors; i++)
	//	memset(D[i][0], 0, m_nCols0 * m_nRows0 * sizeof(float));

	for (int i = 0; i < m_nRows; i++)
	{
		for (int j = 0; j < m_nCols; j++)
		{
			for (int d = 0; d < m_nDescriptors; d++)
			{
				float fac = atPt(img, i + offY[d], j + offX[d]);
				D[d][i][j] = fac;
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void OpticalFlow_Huber_L1::compute_feature_map_PointIntensity(float*** D, float** img)
{

	//for (int i = 0; i < m_nDescriptors; i++)
	//	memset(D[i][0], 0, m_nCols0 * m_nRows0 * sizeof(float));

	for (int i = 0; i < m_nRows; i++)
	{
		for (int j = 0; j < m_nCols; j++)
		{
			D[0][i][j] = img[i][j];
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void OpticalFlow_Huber_L1::compute_feature_map_derivatives()
{
	float dDdx, dDdy;


	for (int i = 0; i < m_nRows; i++)
	{
		for (int j = 0; j < m_nCols; j++)
		{
			for (int d = 0; d < m_nDescriptors; d++)
			{

				//calculate feature gradient
				if (j == 0)
					dDdx = m_D2[d][i][1] - m_D2[d][i][0]; 
				else if (j == m_nCols - 1)
				    dDdx = m_D2[d][i][m_nCols - 1] - m_D2[d][i][m_nCols - 2];
				else 
					dDdx = 0.5f *( m_D2[d][i][j + 1] - m_D2[d][i][j - 1]); 

				if (i == 0)
					dDdy = m_D2[d][1][j] - m_D2[d][0][j];
				else if (i == m_nRows - 1)
					dDdy = m_D2[d][m_nRows - 1][j] - m_D2[d][m_nRows - 2][j];
				else
					dDdy = 0.5f * (m_D2[d][i + 1][j] - m_D2[d][i - 1][j]); 


				m_Dx[d][i][j] = dDdx;
				m_Dy[d][i][j] = dDdy;

			}

		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void OpticalFlow_Huber_L1::updateUV()
{
	////memcpy(m_u0[0], m_u[0], m_nCols0 * m_nRows0 * sizeof(float));
	////memcpy(m_v0[0], m_v[0], m_nCols0 * m_nRows0 * sizeof(float));
	////memcpy(m_u_[0], m_u[0], m_nCols0 * m_nRows0 * sizeof(float));
	////memcpy(m_v_[0], m_v[0], m_nCols0 * m_nRows0 * sizeof(float));

	for (int k = 0; k < m_parms.outerIters; k++)
	{
		printf("iterations : %d\n", k);

		for(int l =0; l < m_parms.innerIters; l++)
			updateV();

		for (int l = 0; l < m_parms.innerIters; l++)
			updateU();

	}

}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void OpticalFlow_Huber_L1::updateV()
{
	//float norm;
	//float v1x, v1y, v2x, v2y;
	//float sigma = 0.5f;
	//float fac = sigma * m_parms.epsilon;
	//float tau = 0.25f;
	//float divpv1, divpv2;
	//float v1bar, v2bar, v1Old, v2Old;
	//float con1 = m_parms.theta  / (m_parms.theta + tau);
	//float con2 = tau / m_parms.theta;

	//update dual variable
#pragma omp parallel for
	for (int i = 0; i < m_nRows; i++)
	{
		for (int j = 0; j < m_nCols; j++)
		{

			updateV_Dual(i, j);
			//if (j < m_nCols - 1)
			//{
			//	v1x = m_v1_[i][j + 1] - m_v1_[i][j];
			//	v2x = m_v2_[i][j + 1] - m_v2_[i][j];
			//}
			//else
			//	v1x = v2x = 0.0f;

			//if (i < m_nRows - 1)
			//{
			//	v1y = m_v1_[i + 1][j] - m_v1_[i][j];
			//	v2y = m_v2_[i + 1][j] - m_v2_[i][j];
			//}
			//else
			//	v1y = v2y = 0.0f;

			//m_p1x[i][j] = (m_p1x[i][j] + sigma * v1x) / (1.0f + fac);
			//m_p2x[i][j] = (m_p2x[i][j] + sigma * v2x) / (1.0f + fac);

			//m_p1y[i][j] = (m_p1y[i][j] + sigma * v1y) / (1.0f + fac);
			//m_p2y[i][j] = (m_p2y[i][j] + sigma * v2y) / (1.0f + fac);

			////project onto a unit ball
			//norm = sqrt(m_p1x[i][j] * m_p1x[i][j] + m_p1y[i][j] * m_p1y[i][j]);
			//norm = max(1.0f, norm);
			//m_p1x[i][j] /= norm;
			//m_p1y[i][j] /= norm;

			//norm = sqrt(m_p2x[i][j] * m_p2x[i][j] + m_p2y[i][j] * m_p2y[i][j]);
			//norm = max(1.0f, norm);
			//m_p2x[i][j] /= norm;
			//m_p2y[i][j] /= norm;
		}
	}

	//update primal variable
#pragma omp parallel for
	for (int i = 0; i < m_nRows; i++)
	{
		for (int j = 0; j < m_nCols; j++)
		{
			updateV_Primal(i, j);
			//divpv1 = divpv2 = 0.0f;

			//if (i > 0 && i < m_nRows - 1)
			//{
			//	divpv1 += m_p1y[i - 1][j] - m_p1y[i][j];
			//	divpv2 += m_p2y[i - 1][j] - m_p2y[i][j];
			//}
			//else if (i == 0)
			//{
			//	divpv1 += -m_p1y[i][j];
			//	divpv2 += -m_p2y[i][j];
			//}
			//else
			//{
			//	divpv1 += m_p1y[i - 1][j];
			//	divpv2 += m_p2y[i - 1][j];
			//}


			//if (j > 0 && j < m_nCols - 1)
			//{
			//	divpv1 += m_p1x[i][j - 1] - m_p1x[i][j];
			//	divpv2 += m_p2x[i][j - 1] - m_p2x[i][j];
			//}
			//else if (j == 0)
			//{
			//	divpv1 += -m_p1x[i][j];
			//	divpv2 += -m_p2x[i][j];
			//}
			//else
			//{
			//	divpv1 += m_p1x[i][j - 1];
			//	divpv2 += m_p2x[i][j - 1];
			//}

			//v1bar = m_v1[i][j] - tau * divpv1;
			//v2bar = m_v2[i][j] - tau * divpv2;

			//v1Old = m_v1[i][j];
			//v2Old = m_v2[i][j];

			//m_v1[i][j] = con1 * (con2 * m_u1[i][j] + v1bar);
			//m_v2[i][j] = con1 * (con2 * m_u2[i][j] + v2bar);

			//m_v1_[i][j] = 2.0f * m_v1[i][j] - v1Old;
			//m_v2_[i][j] = 2.0f * m_v2[i][j] - v2Old;

			//if (isnan(m_v1_[i][j]))
			//	int mm = 0;

			//if (isnan(m_v2_[i][j]))
			//	int mm = 0;
		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void OpticalFlow_Huber_L1::updateV_Dual(int i, int j)
{
	float norm;
	float v1x, v1y, v2x, v2y;
	float sigma = 0.5f;
	float fac = sigma * m_parms.epsilon;
	float tau = 0.25f;
	float divpv1, divpv2;
	float v1bar, v2bar, v1Old, v2Old;
	float con1 = m_parms.theta / (m_parms.theta + tau);
	float con2 = tau / m_parms.theta;

	//update dual variable
	//for (int i = 0; i < m_nRows; i++)
	//{
	//	for (int j = 0; j < m_nCols; j++)
	//	{
	if (j < m_nCols - 1)
	{
		v1x = m_v1_[i][j + 1] - m_v1_[i][j];
		v2x = m_v2_[i][j + 1] - m_v2_[i][j];
	}
	else
		v1x = v2x = 0.0f;

	if (i < m_nRows - 1)
	{
		v1y = m_v1_[i + 1][j] - m_v1_[i][j];
		v2y = m_v2_[i + 1][j] - m_v2_[i][j];
	}
	else
		v1y = v2y = 0.0f;

	m_p1x[i][j] = (m_p1x[i][j] + sigma * v1x) / (1.0f + fac);
	m_p2x[i][j] = (m_p2x[i][j] + sigma * v2x) / (1.0f + fac);

	m_p1y[i][j] = (m_p1y[i][j] + sigma * v1y) / (1.0f + fac);
	m_p2y[i][j] = (m_p2y[i][j] + sigma * v2y) / (1.0f + fac);

	//project onto a unit ball
	norm = sqrt(m_p1x[i][j] * m_p1x[i][j] + m_p1y[i][j] * m_p1y[i][j]);
	norm = max(1.0f, norm);
	m_p1x[i][j] /= norm;
	m_p1y[i][j] /= norm;

	norm = sqrt(m_p2x[i][j] * m_p2x[i][j] + m_p2y[i][j] * m_p2y[i][j]);
	norm = max(1.0f, norm);
	m_p2x[i][j] /= norm;
	m_p2y[i][j] /= norm;
	//	}
	//}
}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void OpticalFlow_Huber_L1::updateV_Primal(int i, int j)
{
	float norm;
	float v1x, v1y, v2x, v2y;
	float sigma = 0.5f;
	float fac = sigma * m_parms.epsilon;
	float tau = 0.25f;
	float divpv1, divpv2;
	float v1bar, v2bar, v1Old, v2Old;
	float con1 = m_parms.theta / (m_parms.theta + tau);
	float con2 = tau / m_parms.theta;
	//update primal variable
	//for (int i = 0; i < m_nRows; i++)
	//{
	//	for (int j = 0; j < m_nCols; j++)
	//	{
			divpv1 = divpv2 = 0.0f;

			if (i > 0 && i < m_nRows - 1)
			{
				divpv1 += m_p1y[i - 1][j] - m_p1y[i][j];
				divpv2 += m_p2y[i - 1][j] - m_p2y[i][j];
			}
			else if (i == 0)
			{
				divpv1 += -m_p1y[i][j];
				divpv2 += -m_p2y[i][j];
			}
			else
			{
				divpv1 += m_p1y[i - 1][j];
				divpv2 += m_p2y[i - 1][j];
			}


			if (j > 0 && j < m_nCols - 1)
			{
				divpv1 += m_p1x[i][j - 1] - m_p1x[i][j];
				divpv2 += m_p2x[i][j - 1] - m_p2x[i][j];
			}
			else if (j == 0)
			{
				divpv1 += -m_p1x[i][j];
				divpv2 += -m_p2x[i][j];
			}
			else
			{
				divpv1 += m_p1x[i][j - 1];
				divpv2 += m_p2x[i][j - 1];
			}

			v1bar = m_v1[i][j] - tau * divpv1;
			v2bar = m_v2[i][j] - tau * divpv2;

			v1Old = m_v1[i][j];
			v2Old = m_v2[i][j];

			m_v1[i][j] = con1 * (con2 * m_u1[i][j] + v1bar);
			m_v2[i][j] = con1 * (con2 * m_u2[i][j] + v2bar);

			m_v1_[i][j] = 2.0f * m_v1[i][j] - v1Old;
			m_v2_[i][j] = 2.0f * m_v2[i][j] - v2Old;


	//	}
	//}
}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void OpticalFlow_Huber_L1::updateU()
{
	float sum;
	float sigma;
	float tau1, tau2; 
	float sum1, sum2;
	float norm;
	float u1bar, u2bar;
	float u1Old, u2Old;

	//update dual variable
#pragma omp parallel for
	for (int i = 0; i < m_nRows; i++)
	{
		for (int j = 0; j < m_nCols; j++)
		{
			updateU_Dual(i, j);
			//norm = 0.0f;
			//for (int d = 0; d < m_nDescriptors; d++)
			//{

			//	sum = m_Dx[d][i][j] * (m_u1_[i][j] - m_u10[i][j]) + m_Dy[d][i][j] * (m_u2_[i][j] - m_u20[i][j]) + m_D2[d][i][j] - m_D1[d][i][j];
			//	sigma = abs(m_Dx[d][i][j]) + abs(m_Dy[d][i][j]);

			//	if (sigma == 0.0f)
			//		sigma = 1.0f;

			//	if(sum != 0.0f )
			//		m_qD[d][i][j] += sum / sigma;
			//		//m_qD[d][i][j] += m_parms.Lambda * sum / sigma;s

			//	norm += m_qD[d][i][j] * m_qD[d][i][j];
			//}
			//
			// norm = max(sqrt(norm), 1.0f);

			//for (int d = 0; d < m_nDescriptors; d++)
			//{
			//	m_qD[d][i][j] /= norm;

			//}

		}
	}

	//update primal variable
#pragma omp parallel for
	for (int i = 0; i < m_nRows; i++)
	{
		for (int j = 0; j < m_nCols; j++)
		{
			updateU_Primal(i, j);
			//sum1 = sum2 = 0.0f;
			//tau1 = tau2 = 0.0f;
			//for (int d = 0; d < m_nDescriptors; d++)
			//{
			//	sum1 += m_Dx[d][i][j] * m_qD[d][i][j];
			//	tau1 += abs(m_Dx[d][i][j]);

			//	sum2 += m_Dy[d][i][j] * m_qD[d][i][j];
			//	tau2 += abs(m_Dy[d][i][j]);
			//}

			//if (tau1 == 0.0f)
			//	tau1 = 1.0f;

			//if (tau2 == 0.0f)
			//	tau2 = 1.0f;

			//tau1 = 1.0f / tau1;
			//tau2 = 1.0f / tau2;

			//u1bar = m_u1[i][j] - tau1 * sum1;
			//u2bar = m_u2[i][j] - tau2 * sum2;

			//u1Old = m_u1[i][j];
			//u2Old = m_u2[i][j];

			//float theta = m_parms.theta * m_parms.Lambda;

			//if (m_alpha)
			//{
			//	m_u1[i][j] = m_parms.theta * (tau1 * m_v1[i][j] / m_parms.theta + m_alpha[i][j] * tau1 * m_u1prior[i][j] + u1bar) /
			//	            (m_parms.theta + tau1 + m_alpha[i][j] * tau1);

			//	m_u2[i][j] = m_parms.theta * (tau2 * m_v2[i][j] / m_parms.theta + m_alpha[i][j] * tau2 * m_u2prior[i][j] + u2bar) /
			//		(m_parms.theta + tau2 + m_alpha[i][j] * tau2);
			//}
			//else
			//{
			//	m_u1[i][j] = theta * (tau1 * m_v1[i][j] /theta +  u1bar) /
			//		(theta + tau1);

			//	m_u2[i][j] = m_parms.theta * (tau2 * m_v2[i][j] / m_parms.theta +  u2bar) /
			//		(m_parms.theta + tau2);
			//}

			//m_u1_[i][j] = 2.0f * m_u1[i][j] - u1Old;
			//m_u2_[i][j] = 2.0f * m_u2[i][j] - u2Old;

		}
	}
}


/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void OpticalFlow_Huber_L1::updateU_Dual(int i, int j)
{
	float sum;
	float sigma;
	float tau1, tau2;
	float sum1, sum2;
	float norm;
	float u1bar, u2bar;
	float u1Old, u2Old;

	//update dual variable
	//for (int i = 0; i < m_nRows; i++)
	//{
	//	for (int j = 0; j < m_nCols; j++)
	//	{
	norm = 0.0f;
	for (int d = 0; d < m_nDescriptors; d++)
	{

		sum = m_Dx[d][i][j] * (m_u1_[i][j] - m_u10[i][j]) + m_Dy[d][i][j] * (m_u2_[i][j] - m_u20[i][j]) + m_D2[d][i][j] - m_D1[d][i][j];
		sigma = abs(m_Dx[d][i][j]) + abs(m_Dy[d][i][j]);

		if (sigma == 0.0f)
			sigma = 1.0f;

		if (sum != 0.0f)
			m_qD[d][i][j] += sum / sigma;
		//m_qD[d][i][j] += m_parms.Lambda * sum / sigma;s

		norm += m_qD[d][i][j] * m_qD[d][i][j];
	}

	norm = max(sqrt(norm), 1.0f);

	for (int d = 0; d < m_nDescriptors; d++)
	{
		m_qD[d][i][j] /= norm;

	}

	//	}
	//}
}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void OpticalFlow_Huber_L1::updateU_Primal(int i, int j)
{
	float sum;
	float sigma;
	float tau1, tau2;
	float sum1, sum2;
	float norm;
	float u1bar, u2bar;
	float u1Old, u2Old;

	//update primal variable
	//for (int i = 0; i < m_nRows; i++)
	//{
	//	for (int j = 0; j < m_nCols; j++)
	//	{

			sum1 = sum2 = 0.0f;
			tau1 = tau2 = 0.0f;
			for (int d = 0; d < m_nDescriptors; d++)
			{
				sum1 += m_Dx[d][i][j] * m_qD[d][i][j];
				tau1 += abs(m_Dx[d][i][j]);

				sum2 += m_Dy[d][i][j] * m_qD[d][i][j];
				tau2 += abs(m_Dy[d][i][j]);
			}

			if (tau1 == 0.0f)
				tau1 = 1.0f;

			if (tau2 == 0.0f)
				tau2 = 1.0f;

			tau1 = 1.0f / tau1;
			tau2 = 1.0f / tau2;

			u1bar = m_u1[i][j] - tau1 * sum1;
			u2bar = m_u2[i][j] - tau2 * sum2;

			u1Old = m_u1[i][j];
			u2Old = m_u2[i][j];

			float theta = m_parms.theta * m_parms.Lambda;

			if (m_alpha)
			{
				m_u1[i][j] = theta * (tau1 * m_v1[i][j] / theta + m_alpha[i][j] * tau1 * m_u1prior[i][j] + u1bar) /
					(theta + tau1 + m_alpha[i][j] * tau1);

				m_u2[i][j] = theta * (tau2 * m_v2[i][j] / theta + m_alpha[i][j] * tau2 * m_u2prior[i][j] + u2bar) /
					(theta + tau2 + m_alpha[i][j] * tau2);
			}
			else
			{
				m_u1[i][j] = theta * (tau1 * m_v1[i][j] / theta + u1bar) /
					(theta + tau1);

				m_u2[i][j] = m_parms.theta * (tau2 * m_v2[i][j] / m_parms.theta + u2bar) /
					(m_parms.theta + tau2);
			}

			m_u1_[i][j] = 2.0f * m_u1[i][j] - u1Old;
			m_u2_[i][j] = 2.0f * m_u2[i][j] - u2Old;

	//	}
	//}
}
///////////////////////////////////////////////////////////////////////////////
/// \brief host texture fetch
///
/// read from arbitrary position within image using bilinear interpolation
/// out of range coords are clamped to boundary (mirrored)
/// \param[in]  t   texture raw data
/// \param[in]  w   texture width
/// \param[in]  h   texture height
/// \param[in]  s   texture stride
/// \param[in]  x   x coord of the point to fetch value at
/// \param[in]  y   y coord of the point to fetch value at
/// \return fetched value
///////////////////////////////////////////////////////////////////////////////
inline float OpticalFlow_Huber_L1::Tex2D(float** t, int w, int h, float x, float y)
{
	// integer parts in floating point format
	float intPartX, intPartY;

	// get fractional parts of coordinates
	float dx = fabsf(modff(x, &intPartX));
	float dy = fabsf(modff(y, &intPartY));

	// assume pixels are squaresx
	// one of the corners
	int ix0 = (int)intPartX;
	int iy0 = (int)intPartY;

	// mirror out-of-range position
	if (ix0 < 0) ix0 = 0;// abs(ix0 + 1);

	if (iy0 < 0) iy0 = 0;// abs(iy0 + 1);

	if (ix0 >= w) ix0 = w - 1;// w * 2 - ix0 - 1;

	if (iy0 >= h) iy0 = h - 1;// h * 2 - iy0 - 1;

	// corner which is opposite to (ix0, iy0)
	int ix1 = ix0 + 1;
	int iy1 = iy0 + 1;

	if (ix1 >= w) ix1 = w - 1;// w * 2 - ix1 - 1;

	if (iy1 >= h) iy1 = h - 1; // h * 2 - iy1 - 1;

	float res = t[iy0][ix0] * (1.0f - dx) * (1.0f - dy);
	res += t[iy0][ix1] * dx * (1.0f - dy);
	res += t[iy1][ix0] * (1.0f - dx) * dy;
	res += t[iy1][ix1] * dx * dy;

	return res;
}

/////////////////////////////////////////////////////////////////////////////////////
//Clamp pixel to the boundary (//mirror boundary pixels)
/////////////////////////////////////////////////////////////////////////////////////
float OpticalFlow_Huber_L1::atPt(float**image, int y, int x)
{
	if (x < 0)
		x = 0; //abs(x + 1);
	if (x > m_nCols - 1)
		x = m_nCols - 1; // m_nCols * 2 - x - 1;

	if (y < 0)
		y = 0; //abs(y + 1)
	if (y > m_nRows - 1)
		y = m_nRows - 1; //m_nRows *2 - y - 1;

	return image[y][x];

}
