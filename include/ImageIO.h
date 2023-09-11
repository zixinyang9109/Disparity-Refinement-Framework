#pragma once
#include <string>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <bitset>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void ToMat(float** image, Mat &output);
void ToMat_scale(float** image, Mat &output, float scale);
void ToMat_scale_u16(float** image, Mat &output, float scale);
void Mat2Raw(float** im,Mat img);
void Mat2Raw_scale(float** im,Mat img,float scale);
void Mat2Raw_scale_8uc1(float** im,Mat img,float scale);



template <typename T> static T*** AllocateVolume(int numCol, int numRow, int numFrame);
template <typename T> static T** AllocateImage(int ncols, int nrows);

template <typename T> static void FreeImage(T** image);

template <typename T> static void FreeVolume(T*** Image, int numFrame);

template <typename T> static void WriteImage(std::string dirIn, std::string fileIn, T** proj, int numCol, int numRow);
template <typename T> static void ReadImage(std::string dirIn, std::string fileIn, T** proj, int numCol, int numRow);

template <typename T1, typename T2> static void ReadImage(std::string dirIn, std::string fileIn, T1** proj, int numCol, int numRow);

template <class T> static T* AllocateArray(int numElements);
template <class T> static void FreeArray(T* array);

template <typename T> T* AllocateArray(int numElements)
{
	try
	{
		T* Array = new T[numElements];
		return Array;
	}
	catch (std::bad_alloc)
	{
		std::cerr << "Not enough memory could be allocated by the system." << std::endl;    //TODO: other recovery strategies??
		assert(false);
		return NULL;
	}

}

template <typename T> void FreeArray(T* Array)
{
	if (Array == NULL)
		return;

	if (Array != NULL)
	{
		delete[] Array;
		Array = NULL;
	}
}

template <typename T> T*** AllocateVolume(int numCol, int numRow, int numFrame)
{
    try{
        T*** Image = new T**[numFrame];
        for (int i = 0; i< numFrame; i++)
        {
            Image[i] = new T*[numRow];
			Image[i][0] = new T[numRow*numCol];
			for(int j = 1; j < numRow; j++)
				Image[i][j] = Image[i][j-1] + numCol;
        }
       return Image;
    }
    catch(std::bad_alloc)
    {
        std::cerr<<"Not enough memory could be allocated by the system."<<std::endl;    //TODO: other recovery strategies??
        assert(false);
        return NULL;
    }
}
template <typename T> T** AllocateImage(int ncols, int nrows)
{
	try
	{
		T** ptr = new T*[nrows];
		T* pool = new T[nrows*ncols];
		for (unsigned i = 0; i < nrows; ++i, pool += ncols )
			ptr[i] = pool;
		return ptr;
	}
	catch(std::bad_alloc)
	{
		std::cerr<<"Not enough memory could be allocated by the system."<<std::endl;    //TODO: other recovery strategies??
		assert(false);
		return NULL;
	}
}
template <typename T> void FreeVolume(T*** Image, int numFrame)
{
    if(Image == NULL)
        return;
    for (int i = 0; i < numFrame; i++) 
        {
            if(Image[i] != NULL)
            {
                delete [] Image[i][0];
				delete [] Image[i];
                Image[i]= NULL;
            }
    }
    delete[] Image;
    Image= NULL;
}
template <typename T> void FreeImage(T** image)
{
    if(image == NULL)
        return;

    if(image != NULL)
    {
        delete [] image[0];
		delete [] image;
        image= NULL;
    }
    delete[] image;
    image= NULL;
}

template <typename T> void WriteImage(std::string dirIn, std::string fileIn, T** proj, int numCol, int numRow)
{
	string file = dirIn + fileIn;
	FILE *fp;
	fp = fopen(file.c_str(), "wb"); // ab

	fwrite(&proj[0][0], numRow*numCol*sizeof(T), 1, fp);

	fclose(fp);
}

template <typename T> void ReadImage(std::string dirIn, std::string fileIn, T** proj, int numCol, int numRow)
{
	string file = dirIn + fileIn;
	FILE *fp;
	fp = fopen(file.c_str(), "rb");

	int num = fread(proj[0], sizeof(T), numRow*numCol, fp);

	fclose(fp);
}

template <typename T1, typename T2> void ReadImage(std::string dirIn, std::string fileIn, T1** proj, int numCol, int numRow)
{
	string file = dirIn + fileIn;
	FILE *fp;
	fp = fopen(file.c_str(), "rb");

	T2 *temp = new T2[numRow*numCol];

	int num = fread(temp, sizeof(T2), numRow*numCol, fp);
	fclose(fp);

	int ind = 0;
	T1 *ptr = proj[0];
	for (int i = 0; i < numRow*numCol; i++)
	{
		ptr[ind] = (T1)temp[ind];
		ind++;
	}
	delete[] temp;
}


