#include <math.h>
#include "ImageIO.h"
#include "ImagePyramid.h"
#include "Stereo_Huber_L2.h"
#include <omp.h>



////////////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////////
Stereo_Huber_L2::Stereo_Huber_L2()
{
	//dual variables
	m_pux = NULL;
	m_puy = NULL;

	//primal
	m_u0 = NULL;
	m_u_ = NULL;

	m_nDescriptors = 8;

	m_I2w = NULL;
	m_Dx2 = NULL;
	m_DxR = NULL;

	m_D1 = NULL;
	m_D2 = NULL;

}

////////////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////////
Stereo_Huber_L2::~Stereo_Huber_L2()
{

	FreeImage(m_pux);
	FreeImage(m_puy);


	FreeImage(m_I2w);
	FreeImage(m_Dx2);
	FreeImage(m_DxR);

	FreeImage(m_u_);

	FreeVolume(m_D1, m_nDescriptors);
	FreeVolume(m_D2, m_nDescriptors);

}

////////////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////////
void Stereo_Huber_L2::setup(Data inputData, Parms parms)
{
	m_I1 = inputData.I1;
	m_I2 = inputData.I2;

	m_u = inputData.u;

	m_alpha = inputData.alpha;
	m_uprior = inputData.uprior;

	m_nCols0 = inputData.nCols;
	m_nRows0 = inputData.nRows;

	m_parms = parms;

	if (parms.descriptor == PATCH_INTENSITY)
	{
		m_nDescriptors = 9;
		compute_feature_map = &Stereo_Huber_L2::compute_feature_map_PatchIntensity;
	}
	else if (parms.descriptor == POINT_INTENSITY)
	{
		m_nDescriptors = 1;
		compute_feature_map = &Stereo_Huber_L2::compute_feature_map_PointIntensity;
	}
	else if (parms.descriptor == D1)
	{
		m_nDescriptors = 8;
		compute_feature_map = &Stereo_Huber_L2::compute_feature_map_D1;
	}
	else
	{
		m_nDescriptors = 8;
		compute_feature_map = &Stereo_Huber_L2::compute_feature_map_Census;
	}

	//create image pyramids
	m_imagePyramid1.createPyramid(m_I1, m_nCols0, m_nRows0, parms.scalefactor, parms.levels);
	m_imagePyramid2.createPyramid(m_I2, m_nCols0, m_nRows0, parms.scalefactor, parms.levels);

	if (m_alpha)
	{
		m_alphaPyramid.createPyramid(m_alpha, m_nCols0, m_nRows0, parms.scalefactor, parms.levels);
	}

    if(parms.withPrior) {
        m_uPriorPyramid.createPyramid(m_uprior, m_nCols0, m_nRows0, parms.scalefactor, parms.levels, parms.scalefactor);
    }

	m_I2w = AllocateImage<float>(m_nCols0, m_nRows0);
	//dual variables
	m_pux = AllocateImage<float>(m_nCols0, m_nRows0);
	m_puy = AllocateImage<float>(m_nCols0, m_nRows0);

	//arrays for L2 term
	m_Dx2 = AllocateImage<float>(m_nCols0, m_nRows0);
	m_DxR = AllocateImage<float>(m_nCols0, m_nRows0);

	//primal variables
	m_u0 = AllocateImage<float>(m_nCols0, m_nRows0);
	m_u_ = AllocateImage<float>(m_nCols0, m_nRows0);

	//allocate memory for features
	m_D1 = AllocateVolume<float>(m_nCols0, m_nRows0, m_nDescriptors);
	m_D2 = AllocateVolume<float>(m_nCols0, m_nRows0, m_nDescriptors);

	memset(m_pux[0], 0, sizeof(float) * m_nCols0 * m_nRows0);
	memset(m_puy[0], 0, sizeof(float) * m_nCols0 * m_nRows0);
	memset(m_u[0], 0, sizeof(float) * m_nCols0 * m_nRows0);


}

////////////////////////////////////////////////////////////////////////////////////
//
///////////////////////////////////////////////////////////////////////////////////
void Stereo_Huber_L2::run()
{
	float** temp = AllocateImage<float>(m_nCols0, m_nRows0);

	for (int level = m_parms.levels - 1; level >= 0; level--)
	{
//		printf("level : %d\n", level);
		m_I1 = m_imagePyramid1.getImage(level);
		m_I2 = m_imagePyramid2.getImage(level);

		m_nCols = m_imagePyramid1.getCols(level);
		m_nRows = m_imagePyramid1.getRows(level);

		//compute feature map of I1
		(this->*compute_feature_map)(m_D1, m_I1);

		if (m_alpha)
		{
			m_alpha = m_alphaPyramid.getImage(level);
            //m_uprior = m_uPriorPyramid.getImage(level);
		}

        if (m_parms.withPrior){
            m_uprior = m_uPriorPyramid.getImage(level);

            for (int i = 0; i < m_nRows; i++)
            {
                for (int j = 0; j < m_nCols; j++)
                {
//                    cout<<m_u[i][j]<<endl;
//                    cout<< m_uprior[i][j]<<endl;
                    m_u[i][j]  = m_uprior[i][j];
                }
            }
            m_parms.withPrior = false;

        }


		for (int k = 0; k < m_parms.warps; k++)
		{
			memcpy(m_u0[0], m_u[0], m_nCols0 * m_nRows0 * sizeof(float));
			memcpy(m_u_[0], m_u[0], m_nCols0 * m_nRows0 * sizeof(float));

//			printf("warp : %d\n", k);

			//use current (u,v) estimate to warp image I2
			warp();

			//compute features of warped I2
			(this->*compute_feature_map)(m_D2, m_I2w);

			//calculate gradient of I2w features with respect to (u,v)
			compute_feature_map_derivatives();

			//update (u,v) estimate
			solvePrimalDual();
		}

		if (level != 0)
		{
			//upscale primal variables(u,v)
			ImagePyramid::resizeImage(m_u, m_nCols, m_nRows, m_imagePyramid1.getCols(level - 1),
				m_imagePyramid1.getRows(level - 1), temp, 2);
			memcpy(m_u[0], temp[0], m_nCols0 * m_nRows0 * sizeof(float));

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

			memset(m_pux[0], 0, m_nCols0 * m_nRows0 * sizeof(float));
			memset(m_puy[0], 0, m_nCols0 * m_nRows0 * sizeof(float));
		}
		
	}

	warp();
//    cout<<"last mean diff: "<< diff_last<<endl;

	memcpy(m_I2[0], m_I2w[0], m_nCols0 * m_nRows0 * sizeof(float));
}

/////////////////////////////////////////////////////////////////////////////////////
//Warp I2 using current estimate of (u,v) 
/////////////////////////////////////////////////////////////////////////////////////
void Stereo_Huber_L2::warp()
{
	float x, y;
//#pragma omp parallel for
	for (int i = 0; i < m_nRows; i++)
	{
		for (int j = 0; j < m_nCols; j++)
		{
			x = (float)j + m_u[i][j];
			y = (float)i;
			m_I2w[i][j] = Tex2D(m_I2, m_nCols, m_nRows, x, y);
		}
	}

}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void Stereo_Huber_L2::compute_feature_map_D1(float*** D, float** img)
{
	int offX[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
	int offY[] = { -1, -1, -1, 0, 0, 1, 1, 1 };

	//for(int i = 0; i < m_nDescriptors; i++)
	//	memset(D[i][0], 0, m_nCols0 * m_nRows0 * sizeof(float));
//#pragma omp parallel for
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
void Stereo_Huber_L2::compute_feature_map_Census(float*** D, float** img)
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
				if (fac - img[i][j] > 0.0f)
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
void Stereo_Huber_L2::compute_feature_map_PatchIntensity(float*** D, float** img)
{
	int offX[] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
	int offY[] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };

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
void Stereo_Huber_L2::compute_feature_map_PointIntensity(float*** D, float** img)
{


	for (int i = 0; i < m_nDescriptors; i++)
		memset(D[i][0], 0, m_nCols0 * m_nRows0 * sizeof(float));

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
void Stereo_Huber_L2::compute_feature_map_derivatives()
{
	float dDdx, R, Dt;

	memset(m_Dx2[0], 0, m_nCols0 * m_nRows0 * sizeof(float));
	memset(m_DxR[0], 0, m_nCols0 * m_nRows0 * sizeof(float));


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

				Dt = m_D2[d][i][j] - m_D1[d][i][j];
				R = Dt - dDdx * m_u0[i][j];

				//L2 term
				m_Dx2[i][j] += dDdx * dDdx;
				m_DxR[i][j] += dDdx * R;

			}

			m_Dx2[i][j] /= (float)m_nDescriptors;
			m_DxR[i][j] /= (float)m_nDescriptors;

		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void Stereo_Huber_L2::solvePrimalDual()
{


	for (int k = 0; k < m_parms.iterations; k++)
	{
//		printf("iterations : %d\n", k);


		updateDual();


        diff = 0;
		updatePrimal();
        diff = diff/(float)(m_nRows*m_nCols);
//        cout<<"diff: "<<diff<<endl;
        diff_last =diff;


	}

}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void Stereo_Huber_L2::updateDual()
{
	//float norm;
	//float ux, uy, vx, vy;
	//float sigma = 0.5f; // 1.0f / (2.0f + m_parms.epsilon);
	//float fac = sigma * m_parms.epsilon;

#pragma omp parallel for
	for (int i = 0; i < m_nRows; i++)
	{
		for (int j = 0; j < m_nCols; j++)
		{

			updateDual(i, j);

		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void Stereo_Huber_L2::updateDual(int i, int j)
{
	float norm;
	float ux, uy, vx, vy;
	float sigma = 0.5f; // 1.0f / (2.0f + m_parms.epsilon);
	float fac = sigma * m_parms.epsilon;

	////#pragma omp parallel for
	//for (int i = 0; i < m_nRows; i++)
	//{
	//	for (int j = 0; j < m_nCols; j++)
	//	{

			if (j < m_nCols - 1)
			{
				ux = m_u_[i][j + 1] - m_u_[i][j];
			}
			else
				ux = vx = 0.0f;

			if (i < m_nRows - 1)
			{
				uy = m_u_[i + 1][j] - m_u_[i][j];
			}
			else
				uy = vy = 0.0f;

			m_pux[i][j] = (m_pux[i][j] + sigma * ux) / (1.0f + fac);

			m_puy[i][j] = (m_puy[i][j] + sigma * uy) / (1.0f + fac);

			//project onto a unit ball
			norm = sqrt(m_pux[i][j] * m_pux[i][j] + m_puy[i][j] * m_puy[i][j]);
			norm = max(1.0f, norm);
			m_pux[i][j] /= norm;
			m_puy[i][j] /= norm;

	//	}
	//}
}
/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void Stereo_Huber_L2::updatePrimal()
{
	//float tau = 0.25f;
	//float divpu, divpv;
	//float ubar, vbar;
	//float uOld, vOld;
	//float m11, m12, m21, m22, mI11, mI12, mI21, mI22;
	//float b1, b2, det;

#pragma omp parallel for
	for (int i = 0; i < m_nRows; i++)
	{
		for (int j = 0; j < m_nCols; j++)
		{

			updatePrimal(i, j); // update diff

		}
	}
}

/////////////////////////////////////////////////////////////////////////////////////
//
/////////////////////////////////////////////////////////////////////////////////////
void Stereo_Huber_L2::updatePrimal(int i, int j)
{
	float tau = 0.25f;
	float divpu, divpv;
	float ubar, vbar;
	float uOld, vOld;
	float m11, m12, m21, m22, mI11, mI12, mI21, mI22;
	float b1, b2, det;

	//#pragma omp parallel for
	//for (int i = 0; i < m_nRows; i++)
	//{
	//	for (int j = 0; j < m_nCols; j++)
	//	{
			divpu = divpv = 0.0f;

			if (i > 0 && i < m_nRows - 1)
			{
				divpu += m_puy[i - 1][j] - m_puy[i][j];
			}
			else if (i == 0)
			{
				divpu += -m_puy[i][j];
			}
			else
			{
				divpu += m_puy[i - 1][j];
			}


			if (j > 0 && j < m_nCols - 1)
			{
				divpu += m_pux[i][j - 1] - m_pux[i][j];
			}
			else if (j == 0)
			{
				divpu += -m_pux[i][j];
			}
			else
			{
				divpu += m_pux[i][j - 1];
			}

			ubar = m_u[i][j] - tau * divpu;

			//save previous values of u and v
			uOld = m_u[i][j];

			if (m_alpha)
			{
				m_u[i][j] = (ubar + tau * m_alpha[i][j] * m_uprior[i][j] - m_parms.lambda * tau * m_DxR[i][j]) /
					(1.0f + m_alpha[i][j] * tau + m_parms.lambda * tau * m_Dx2[i][j]);
			}
			else
			{
				m_u[i][j] = (ubar - m_parms.lambda * tau * m_DxR[i][j]) / (1.0f + m_parms.lambda * tau * m_Dx2[i][j]);
			}


			m_u_[i][j] = 2.0f * m_u[i][j] - uOld;

            diff = diff + 100*abs(m_u[i][j] - uOld)/abs(uOld);

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
inline float Stereo_Huber_L2::Tex2D(float** t, int w, int h, float x, float y)
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

	// clamp out-of-range position
	if (ix0 < 0) ix0 = 0;// abs(ix0 + 1);

	if (iy0 < 0) iy0 = 0;// abs(iy0 + 1);

	if (ix0 >= w) ix0 = w - 1;// w * 2 - ix0 - 1;

	if (iy0 >= h) iy0 = h - 1;// h * 2 - iy0 - 1;

	// corner which is opposite to (ix0, iy0)
	int ix1 = ix0 + 1;
	int iy1 = iy0 + 1;

	if (ix1 >= w) ix1 = w - 1;// w * 2 - ix1 - 1;

	if (iy1 >= h) iy1 = h - 1; // h * 2 - iy1 - 1;

	//// mirror out-of-range position
	//if (ix0 < 0) ix0 = abs(ix0 + 1);

	//if (iy0 < 0) iy0 = abs(iy0 + 1);

	//if (ix0 >= w) ix0 = w * 2 - ix0 - 1;

	//if (iy0 >= h) iy0 = h * 2 - iy0 - 1;

	//// corner which is opposite to (ix0, iy0)
	//int ix1 = ix0 + 1;
	//int iy1 = iy0 + 1;

	//if (ix1 >= w) ix1 =  w * 2 - ix1 - 1;

	//if (iy1 >= h) iy1 = h * 2 - iy1 - 1;

	float res = t[iy0][ix0] * (1.0f - dx) * (1.0f - dy);
	res += t[iy0][ix1] * dx * (1.0f - dy);
	res += t[iy1][ix0] * (1.0f - dx) * dy;
	res += t[iy1][ix1] * dx * dy;

	return res;
}

/////////////////////////////////////////////////////////////////////////////////////
//Clamp pixel to the boundary (//mirror boundary pixels)
/////////////////////////////////////////////////////////////////////////////////////
float Stereo_Huber_L2::atPt(float**image, int y, int x)
{
	if (x < 0)
		x = 0; //abs(x + 1);
	if (x > m_nCols - 1)
		x = m_nCols - 1; // m_nCols * 2 - x - 1;

	if (y < 0)
		y = 0; //abs(y + 1)
	if (y > m_nRows - 1)
		y = m_nRows - 1; //m_nRows *2 - y - 1;

	//if (x < 0)
	//	x = abs(x + 1);
	//if (x > m_nCols - 1)
	//	x = m_nCols * 2 - x - 1;

	//if (y < 0)
	//	y = abs(y + 1);
	//if (y > m_nRows - 1)
	//	y = m_nRows *2 - y - 1;

	return image[y][x];

}
