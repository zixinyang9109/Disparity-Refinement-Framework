#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>


void FlowToColor(float **u, float **v, unsigned char **R, unsigned char **G, unsigned char **B, int nRows, int nCols, float flowscale)
{

	float colorwheelR[55] = { 255, 255,	255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
	255, 213, 170, 128, 85, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 19, 39, 58, 78, 98, 117, 137, 156,
	176, 196, 215, 235, 255, 255, 255, 255, 255, 255 };
	float colorwheelG[55] = { 0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 232, 209, 186, 163,
		140, 116, 93, 70, 47, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	float colorwheelB[55] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 63, 127, 191, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
		255, 255, 255, 255, 255, 213, 170, 128, 85, 43 };

	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			float du = u[i][j] / flowscale;
			float dv = v[i][j] / flowscale;

			int ncols = 55;
			float rad = sqrtf(du * du + dv * dv);
			float a = atan2(-dv, -du) / 3.14159f;
			float fk = (a + 1) / 2 * ((float)ncols - 1) + 1;
			int k0 = floorf(fk); //colorwheel index lower bound
			int k1 = k0 + 1; //colorwheel index upper bound
			if (k1 == ncols + 1) 
			{
				k1 = 1;
			}
			float f = fk - (float)k0;

			float colR = (1 - f) * (colorwheelR[k0] / 255.0f) + f * (colorwheelR[k1] / 255.0f);
			float colG = (1 - f) * (colorwheelG[k0] / 255.0f) + f * (colorwheelG[k1] / 255.0f);
			float colB = (1 - f) * (colorwheelB[k0] / 255.0f) + f * (colorwheelB[k1] / 255.0f);

			if (rad <= 1) {
				colR = 1 - rad * (1 - colR);
				colG = 1 - rad * (1 - colG);
				colB = 1 - rad * (1 - colB);
			}
			else {
				colR = colR * 0.75;
				colG = colG * 0.75;
				colB = colB * 0.75;
			}

			R[i][j] = (unsigned char)(colR*255);
			G[i][j] = (unsigned char)(colG*255);
			B[i][j] = (unsigned char)(colB*255);
		}
	}

}