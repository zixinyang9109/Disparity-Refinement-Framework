#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "ImageIO.h"
#include "OpticalFlow_Huber_L2.h"
#include "Stereo_Huber_L2.h"
#include <boost/format.hpp>
#include <regex>
#include <pangolin/pangolin.h>
#include <vector>

#include <string>
#include <unistd.h>

int main(){

    string path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/AAnet/final_001.png";
    Mat im = imread(path,IMREAD_ANYDEPTH);

    Mat im_u8 = Mat::zeros(im.rows,im.cols,CV_8UC1);;
    Mat im_f = Mat::zeros(im.rows,im.cols,CV_8UC1);;

    im.convertTo(im_u8,CV_8UC1,1.0/256);//
    cv::medianBlur	(im_u8,im_f,21);
    imshow("org",im_u8);
    imshow("filter",im_f);
    waitKey();


}