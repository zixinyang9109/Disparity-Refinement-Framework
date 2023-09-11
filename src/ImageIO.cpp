#include <string>
#include <assert.h>
#include "ImageIO.h"
using namespace std;

void ToMat(float** image, Mat &output){

    int ImageCols = output.cols;
    int ImageRows = output.rows;

    for (int i = 0; i < ImageRows; i++) {
        for (int j = 0; j < ImageCols; j++) {
            float value = abs(image[i][j]);
            //cout<<"value"<<value<<endl;
            output.data[i * ImageCols + j] = value;
        }
    }

//    double min, max;
//    cv::minMaxLoc(output, &min, &max);
//    cout<<"disparity gt the min "<< min<<endl;
//    cout<<"disparity gt the max "<< max<<endl;
}

void ToMat_scale_u16(float** image, Mat &output, float scale){

    int ImageCols = output.cols;
    int ImageRows = output.rows;

    for (int i = 0; i < ImageRows; i++) {
        for (int j = 0; j < ImageCols; j++) {
            float value = abs(image[i][j]);
            //cout<<"value"<<value<<endl;
            //output.data[i * ImageCols + j] = value * scale;
            output.at<unsigned short >(i,j) = value*256.0;
        }
    }

//    double min, max;
//    cv::minMaxLoc(output, &min, &max);
//    cout<<"disparity gt the min "<< min<<endl;
//    cout<<"disparity gt the max "<< max<<endl;
}

void ToMat_scale(float** image, Mat &output, float scale){

    int ImageCols = output.cols;
    int ImageRows = output.rows;

    for (int i = 0; i < ImageRows; i++) {
        for (int j = 0; j < ImageCols; j++) {
            float value = abs(image[i][j]);
            //cout<<"value"<<value<<endl;
            output.data[i * ImageCols + j] = value * scale;
        }
    }

//    double min, max;
//    cv::minMaxLoc(output, &min, &max);
//    cout<<"disparity gt the min "<< min<<endl;
//    cout<<"disparity gt the max "<< max<<endl;
}

void Mat2Raw(float** im,Mat img){

    int ImageCols = img.cols;
    int ImageRows = img.rows;

    for (int i = 0; i < ImageRows; i++) {
        for (int j = 0; j < ImageCols; j++) {
            im[i][j] = img.data[i*ImageCols+j];
            //cout<<" value in image structure "<< im[i][j]<<endl;
            //cout<<" value in mat structure "<< (int) img.data[i*ImageCols+j]<<endl;
        }
    }

}

void Mat2Raw_scale(float** im,Mat img,float scale){

    int ImageCols = img.cols;
    int ImageRows = img.rows;

    for (int i = 0; i < ImageRows; i++) {
        for (int j = 0; j < ImageCols; j++) {
            //im[i][j] = img.data[i*ImageCols+j]/scale;
            im[i][j] = img.at<unsigned short >(i,j)/scale;
            //cout<<" value in image structure "<< im[i][j]/scale<<endl;
            //cout<<" value in mat structure "<<  (float) img.data[i*ImageCols+j]<<endl;
            //cout<< img.at<unsigned short >(i,j)/scale<<endl;
        }
    }

}

void Mat2Raw_scale_8uc1(float** im,Mat img,float scale){

    int ImageCols = img.cols;
    int ImageRows = img.rows;

    for (int i = 0; i < ImageRows; i++) {
        for (int j = 0; j < ImageCols; j++) {
            //im[i][j] = img.data[i*ImageCols+j]/scale;
            im[i][j] = img.at<uchar>(i,j)/scale;
            //cout<<" value in image structure "<< im[i][j]/scale<<endl;
            //cout<<" value in mat structure "<<  (float) img.data[i*ImageCols+j]<<endl;
            //cout<< img.at<unsigned short >(i,j)/scale<<endl;
        }
    }

}