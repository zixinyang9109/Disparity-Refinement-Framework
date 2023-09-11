// AlgorithmDriver.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <iostream>
#include "ImageIO.h"
#include "OpticalFlow_Huber_L2.h"
#include <boost/format.hpp>
#include <regex>


using namespace std;
using namespace cv;
using namespace Eigen;

void FlowToColor(float** u, float** v, unsigned char** R, unsigned char** G, unsigned char** B, int nRows, int nCols, float flowscale);


/*** save final optical flow and save warped image***/
void run(const string left_file, const string right_file, const string out_file, const string warp_file){

    cv::Mat left = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_file,cv::IMREAD_GRAYSCALE);

//    cv::imshow("left",left);
//    cv::imshow("right", right);
//    cv::waitKey();

    int ImageCols = left.cols;
    int ImageRows = left.rows;

    float** u = AllocateImage<float>(ImageCols, ImageRows);
    float** v = AllocateImage<float>(ImageCols, ImageRows);

    float** image1 = AllocateImage<float>(ImageCols, ImageRows);
    float** image2 = AllocateImage<float>(ImageCols, ImageRows);
    float** image1_warp;// = AllocateImage<float>(ImageCols, ImageRows);

    Mat2Raw(image1,left);
    Mat2Raw(image2,right);

//    string dirIn = "/home/yzx/CLionProjects/Stereo_Matching/Optical_Huber_L2/";
//    string fileIn = "re_left_640x512.raw";// "I1_640x480.raw";//"re_left_1280x1024.raw";
//    ReadImage<float>(dirIn, fileIn, image1, ImageCols, ImageRows);
//
//    fileIn = "re_right_640x512.raw";//"I2_640x480.raw";
//    ReadImage<float>(dirIn, fileIn, image2, ImageCols, ImageRows);

//    OpticalFlow_Huber_L2::Parms inputParms{};
//    inputParms.epsilon = 0.001f;
//    inputParms.iterations = 50;
//    inputParms.Lambda = 15.0f;//100.0f;
//    //patch = 120  D1 = 15.0f;
//    inputParms.levels = 6;
//    inputParms.minSize = 48;
//    inputParms.scalefactor = 0.5f;
//    inputParms.warps = 5;
//    inputParms.descriptor = OpticalFlow_Huber_L2::D1;//PATCH_INTENSITY;

    OpticalFlow_Huber_L2::Parms inputParms{};
    inputParms.epsilon = 0.001f;
    inputParms.iterations = 50;
    inputParms.Lambda = 15.0f;//100.0f;
    //patch = 120  D1 = 15.0f;
    inputParms.levels = 8;
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 5;
    inputParms.descriptor = OpticalFlow_Huber_L2::D1;//PATCH_INTENSITY;

    OpticalFlow_Huber_L2::Data data{};
    data.alpha = NULL;
    data.I1 = image1;
    data.I2 = image2;
    data.u = u;
    data.v = v;
    data.uprior = NULL;
    data.vprior = NULL;
    data.nCols = ImageCols;
    data.nRows = ImageRows;

    OpticalFlow_Huber_L2 of;
    of.setup(data, inputParms);
    of.run();

    Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    ToMat(u, flow_u);

    image1_warp = of.m_I2w;
    Mat image1_warp_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    ToMat(image1_warp, image1_warp_mat);
    cv::imwrite(warp_file,image1_warp_mat);

    string iml_target_str = regex_replace(warp_file, regex("warp"), "target");
    cv::imwrite(iml_target_str,left);

    string iml_source_str = regex_replace(warp_file, regex("warp"), "source");
    cv::imwrite(iml_source_str,right);

    Mat save_disp;
    flow_u.convertTo(save_disp,CV_16UC1,256);
    cv::imwrite(out_file,save_disp);


    //cv::imwrite(out_file,flow_u);


//    double min, max;
//    cv::minMaxLoc(flow_u, &min, &max);
//    cout<<"disparity gt the min "<< min<<endl;
//    cout<<"disparity gt the max "<< max<<endl;

//    cv::imshow("flow u",flow_u);
//    cv::imshow("Image1 warped from Image2",image1_warp_mat);
//    cv::waitKey();

    //cout<<*image1[0]<<endl;
//    fileIn = "I1_warped" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
//    WriteImage(dirIn, fileIn, image1_warp, ImageCols, ImageRows);

//    fileIn = "u_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
//    WriteImage(dirIn, fileIn, u, ImageCols, ImageRows);

//    fileIn = "v_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
//    WriteImage(dirIn, fileIn, v, ImageCols, ImageRows);

    FreeImage(image1);
    FreeImage(image2);
    FreeImage(u);
    FreeImage(v);

}


void OpticalFlowTest()
{
    int ImageCols = 640;
    int ImageRows = 512;

    float** u = AllocateImage<float>(ImageCols, ImageRows);
    float** v = AllocateImage<float>(ImageCols, ImageRows);

    float** image1 = AllocateImage<float>(ImageCols, ImageRows);
    float** image2 = AllocateImage<float>(ImageCols, ImageRows);
    float** image1_warp;// = AllocateImage<float>(ImageCols, ImageRows);

    string dirIn = "/home/yzx/CLionProjects/Stereo_Matching/Optical_Huber_L2/Data/";
    string fileIn = "re_left_640x512.raw";// "I1_640x480.raw";//"re_left_1280x1024.raw";
    ReadImage<float>(dirIn, fileIn, image1, ImageCols, ImageRows);

    fileIn = "re_right_640x512.raw";//"I2_640x480.raw";
    ReadImage<float>(dirIn, fileIn, image2, ImageCols, ImageRows);

    OpticalFlow_Huber_L2::Parms inputParms{};
    inputParms.epsilon = 0.1f;//0.001f
    inputParms.iterations = 50;
    inputParms.Lambda = 12.0f;//100.0f;
    //patch = 120  D1 = 15.0f;
    inputParms.levels = 6;
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 5;
    inputParms.descriptor = OpticalFlow_Huber_L2::D1;//PATCH_INTENSITY;

    //cout<<*image1[0]<<endl;

    OpticalFlow_Huber_L2::Data data{};
    data.alpha = NULL;
    data.I1 = image1;
    data.I2 = image2;
    data.u = u;
    data.v = v;
    data.uprior = NULL;
    data.vprior = NULL;
    data.nCols = ImageCols;
    data.nRows = ImageRows;

    OpticalFlow_Huber_L2 of;
    of.setup(data, inputParms);

    double Time;
    time_t start, end;

    Time = 0.0;
    time(&start);
    of.run();

    time(&end);

    Time += difftime(end, start);

    printf("OpticalFlow time:  %6.3f seconds\n\n", Time);

    image1_warp = of.m_I2w;
    Mat image1_warp_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    ToMat_scale(image1_warp, image1_warp_mat, 255.0);
    fileIn = "I1_warped" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    WriteImage(dirIn, fileIn, image1_warp, ImageCols, ImageRows);

    fileIn = "u_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    WriteImage(dirIn, fileIn, u, ImageCols, ImageRows);
    Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    ToMat(u, flow_u);

    double min, max;
    cv::minMaxLoc(flow_u, &min, &max);
    cout<<"disparity gt the min "<< min<<endl;
    cout<<"disparity gt the max "<< max<<endl;

    cv::imshow("flow u",flow_u);
    cv::imshow("Image1 warped from Image2",image1_warp_mat);
    cv::waitKey();

    fileIn = "v_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    WriteImage(dirIn, fileIn, v, ImageCols, ImageRows);

    FreeImage(image1);
    FreeImage(image2);
    FreeImage(u);
    FreeImage(v);

}

void test_image_structure(){
    // test image data structure
    // target 1. read images. 2. save in image structure. 3. save in raw images
    // read images
    string disp_Dir = "/media/yzx/Elements/Dataset/Colon/Simulator and PublicData/new_case/Result/AAnet/001.png";
    string im_Dir_l = "/media/yzx/Elements/Dataset/Colon/Simulator and PublicData/new_case/left/MYS_L00000.png";
    //string im_Dir_l = "/media/yzx/Elements/Dataset/SCARED_Keyframes/Dataset1/keyframe_1/re_left.jpg";
    string im_Dir_r = "/media/yzx/Elements/Dataset/Colon/Simulator and PublicData/new_case/right/MYS_R00000.png";

    Mat img1 = imread(im_Dir_l,cv::IMREAD_GRAYSCALE);

    int ImageCols = img1.cols;
    int ImageRows = img1.rows;

    float** im = AllocateImage<float>(ImageCols,ImageRows);

    for (int i = 0; i < ImageRows; i++) {
        for (int j = 0; j < ImageCols; j++) {
            im[i][j] = img1.data[i*ImageCols+j];
            cout<<" value in image structure "<< im[i][j]<<endl;
            cout<<" value in mat structure "<< (int) img1.data[i*ImageCols+j]<<endl;
        }
    }

    string fileIn = "testImg" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    string dirIn = "/home/yzx/CLionProjects/Stereo_Matching/Optical_Huber_L2/";
    WriteImage(dirIn, fileIn, im, ImageCols, ImageRows);

}

void test_SERV(){

    /***SERV***/
    // 内参
    double fx = 9.9640068207290187e+02, fy = 9.9640068207290187e+02, cx = 3.7502582168579102e+02, cy =  2.4026374816894531e+02;
    // 基线
    double fb = 5.4301732344712009e+03;

    // 文件路径
    int index = 13;
    boost::format fmt("%03d.png");
    string name = (fmt % index).str();
    cout<<name<<endl;

    string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/"+ name;
    string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/"+ name;
    string disparity_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/BDIS/"+ name;
    string gt_dis_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/"+ name;
    string gt_depth_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/DepthL/"+ name;

    // read images
    cv::Mat left = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_file,cv::IMREAD_GRAYSCALE);

    cv::Mat disparity_gt = cv::imread(gt_dis_file,cv::IMREAD_ANYDEPTH);
    cv::Mat disparity = cv::imread(disparity_file,cv::IMREAD_ANYDEPTH); //8cu1

    int ImageCols = left.cols;
    int ImageRows = left.rows;

    // save to raw
    float** im_l_raw = AllocateImage<float>(ImageCols,ImageRows);
    float** im_r_raw = AllocateImage<float>(ImageCols,ImageRows);
    float** disp_raw = AllocateImage<float>(ImageCols,ImageRows);
    float scale = 256;

    Mat2Raw(im_l_raw, left);
    Mat2Raw(im_r_raw,right);
    Mat2Raw_scale(disp_raw,disparity,scale);

    // set parameters
    OpticalFlow_Huber_L2::Parms inputParms{};
    inputParms.epsilon = 0.001f;
    inputParms.iterations = 50;//50;
    inputParms.Lambda = 15.0f;//100.0f;
    //patch = 120  D1 = 15.0f;
    inputParms.levels = 6;
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 5;
    inputParms.descriptor = OpticalFlow_Huber_L2::D1;//PATCH_INTENSITY;

    // set input
    float** u = AllocateImage<float>(ImageCols, ImageRows);
    float** v = AllocateImage<float>(ImageCols, ImageRows);
    OpticalFlow_Huber_L2::Data data{};
    data.alpha = NULL;
    data.I1 = im_l_raw;
    data.I2 = im_r_raw;
    data.u = u;
    data.v = v;
    data.uprior = NULL;
    data.vprior = NULL;
    data.nCols = ImageCols;
    data.nRows = ImageRows;

    // run
    OpticalFlow_Huber_L2 of;
    of.setup(data, inputParms);
    of.run();

    // visualize result
    Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);

    for (int i = 0; i < ImageRows; i++) {
        for (int j = 0; j < ImageCols; j++) {
            float value = abs(u[i][j]);
            cout<<"value "<<value<<endl;
            flow_u.data[i * ImageCols + j] = value;
        }
    }

    double min, max;
    cv::minMaxLoc(flow_u, &min, &max);
    cout<<"disparity the min "<< min<<endl;
    cout<<"disparity the max "<< max<<endl;

    cv::imshow("img l",left);
    cv::imshow("flow u",flow_u);
    cv::waitKey();

    string fileIn = "u_serv" + to_string(index)+ to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    string dirIn = "/home/yzx/CLionProjects/Stereo_Matching/Optical_Huber_L2/";
    WriteImage(dirIn, fileIn, u, ImageCols, ImageRows);


}

void test_colon(){

    /***colon***/
    // 文件路径
    int index = 1;

    boost::format fmt1("MYS_L%05d.png");
    boost::format fmt2("MYS_R%05d.png");
//    boost::format fmt3("%03d.png");

    string name_im_l = (fmt1 % index).str();
    string name_im_r = (fmt2 % index).str();

//    string name_depth = (fmt3 % index).str();
//    string name_pred_depth = (fmt3 % (index)).str();


    string left_file = "/media/yzx/Elements/Dataset/Colon/Simulator and PublicData/new_case/left/" + name_im_l;
    string right_file = "/media/yzx/Elements/Dataset/Colon/Simulator and PublicData/new_case/right/" + name_im_r;
    //string gt_depth_file = "/media/yzx/Elements/Dataset/Colon/Simulator and PublicData/new_case/Depth/"+name_depth;
    //string pred_depth_file = "/home/yzx/CLionProjects/Stereo_Matching/SDR-master/test/FP_depth" + name_pred_depth;

    // read images
    cv::Mat left = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_file,cv::IMREAD_GRAYSCALE);
//    cv::imshow("img l",left);
//    cv::imshow("img r",right);
//    cv::waitKey();

    //cv::Mat disparity_gt = cv::imread(gt_dis_file,cv::IMREAD_ANYDEPTH);
    //cv::Mat disparity = cv::imread(disparity_file,cv::IMREAD_ANYDEPTH); //8cu1

    int ImageCols = left.cols;
    int ImageRows = left.rows;

    // save to raw
    float** im_l_raw = AllocateImage<float>(ImageCols,ImageRows);
    float** im_r_raw = AllocateImage<float>(ImageCols,ImageRows);
    //float** disp_raw = AllocateImage<float>(ImageCols,ImageRows);
    float scale = 256;

    Mat2Raw(im_l_raw, left);
    Mat2Raw(im_r_raw,right);
    //Mat2Raw(disp_raw,disparity,scale);

    // set parameters
    OpticalFlow_Huber_L2::Parms inputParms{};
    inputParms.epsilon = 0.001f;
    inputParms.iterations = 50;//50;
    inputParms.Lambda = 15.0f;//100.0f;
    //patch = 120  D1 = 15.0f;
    inputParms.levels = 6;
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 5;
    inputParms.descriptor = OpticalFlow_Huber_L2::D1;//PATCH_INTENSITY;

    // set input
    float** u = AllocateImage<float>(ImageCols, ImageRows);
    float** v = AllocateImage<float>(ImageCols, ImageRows);
    OpticalFlow_Huber_L2::Data data{};
    data.alpha = NULL;
    data.I1 = im_l_raw;
    data.I2 = im_r_raw;
    data.u = u;
    data.v = v;
    data.uprior = NULL;
    data.vprior = NULL;
    data.nCols = ImageCols;
    data.nRows = ImageRows;

    // run
    OpticalFlow_Huber_L2 of;
    of.setup(data, inputParms);
    of.run();

    // visualize result
    Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);

    for (int i = 0; i < ImageRows; i++) {
        for (int j = 0; j < ImageCols; j++) {
            float value = abs(u[i][j]);
            cout<<"value "<<value<<endl;
            flow_u.data[i * ImageCols + j] = value;
        }
    }

    double min, max;
    cv::minMaxLoc(flow_u, &min, &max);
    cout<<"disparity the min "<< min<<endl;
    cout<<"disparity the max "<< max<<endl;

    cv::imshow("img l",left);
    cv::imshow("img r",right);
    cv::imshow("flow u",flow_u);
    cv::waitKey();

    string fileIn = to_string(index)+"u_colon_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    string dirIn = "/home/yzx/CLionProjects/Stereo_Matching/Optical_Huber_L2/";
    WriteImage(dirIn, fileIn, u, ImageCols, ImageRows);
}

void run_SERV(){

    string txt_name = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/list.txt";
    string root_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images";

    std::ifstream file(txt_name);
    string left_str;
//    string right_str;
//    string save_str;


    while (std::getline(file, left_str))
    {
        // Process str
        left_str =  root_path + left_str;
        string right_str =  regex_replace(left_str, regex("\\left"), "right");
        string out_str = regex_replace(left_str, regex("\\.png"), "_disp.png");
        out_str = regex_replace(out_str,regex("\\images/left"), "Results/Optical_flow");
        string warp_str = regex_replace(out_str,regex("disp"), "warp") ;
        cout<< left_str<<endl;
        cout<< right_str<<endl;
        cout<<out_str<<endl;

        run(left_str, right_str, out_str, warp_str);

    }
}

void run_SCARED(){
    string txt_name = "/media/yzx/Elements/Dataset/SCARED_Keyframes/left_list.txt";
    string root_dir = "/media/yzx/Elements/Dataset/SCARED_Keyframes";
    string out_dir = "/media/yzx/Elements/Dataset/SCARED_Keyframes/Result/Optical_flow/";
    string warp_dir = "/media/yzx/Elements/Dataset/SCARED_Keyframes/Result/Optical_flow/warp/";

    std::ifstream file(txt_name);
    string left_str;

    int i =0;

    while (std::getline(file, left_str))
    {
        // Process str
        i = i + 1;
        left_str = root_dir + left_str;
        string right_str = regex_replace(left_str, regex("\\left"), "right");

        boost::format fmt("%s/%03d.png");

        string out_str = (fmt % out_dir  % i).str();

        //string warp_dir = out_dir
        string warp_str = (fmt % warp_dir % i).str();

//        cout<< left_str<<endl;
//        cout<< right_str<<endl;
//        cout<<out_str<<endl;

        run(left_str,right_str,out_str,warp_str);

    }

}

void run_Colon(){
    string txt_name = "/media/yzx/Elements/Dataset/Colon/Simulator and PublicData/new_case/left.txt";

    string root_dir = "/media/yzx/Elements/Dataset/Colon/Simulator and PublicData/new_case";
    string out_dir = "/media/yzx/Elements/Dataset/Colon/Simulator and PublicData/new_case/Result/Optical_flow/";

    std::ifstream file(txt_name);
    string left_str;

    int i =0;


    while (std::getline(file, left_str))
    {
        // Process str
        i = i+1;
        left_str = root_dir + left_str;
        string right_str = regex_replace(left_str, regex("\\left"), "right");
        right_str = regex_replace(right_str, regex("\\L"), "R");

        boost::format fmt("%s/%03d.png");

        string out_str = (fmt % out_dir  % i).str();
        string warp_str = regex_replace(out_str, regex("\\.png"), "warp.png");

//        cout<< left_str<<endl;
//        cout<< right_str<<endl;
//        cout<<out_str<<endl;

        run(left_str, right_str, out_str, warp_str);

    }

}

void run_SERV_with_prior(){

    string txt_name = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/list.txt";
    string root_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images";

    std::ifstream file(txt_name);
    string left_str;
//    string right_str;
//    string save_str;


    while (std::getline(file, left_str))
    {
        // Process str
        left_str =  root_path + left_str;
        string right_str =  regex_replace(left_str, regex("\\left"), "right");
        string out_str = regex_replace(left_str, regex("\\.png"), "_disp.png");
        out_str = regex_replace(out_str,regex("\\images/left"), "Results/Optical_flow");
        string warp_str = regex_replace(out_str,regex("disp"), "warp") ;
        cout<< left_str<<endl;
        cout<< right_str<<endl;
        cout<<out_str<<endl;

        run(left_str, right_str, out_str, warp_str);

    }
}

int main()
{
//    run_SCARED();
//    run_SERV();
//    run_Colon();


    //testDepthMap();

    OpticalFlowTest();


    //test_SERV();




//    string txt_name = "/media/yzx/Elements/Dataset/SCARED_Keyframes/left_list.txt";
//    string root_dir = "/media/yzx/Elements/Dataset/SCARED_Keyframes";
//
//    std::ifstream file(txt_name);
//    string left_str;
//    vector<string> im_list;
//    while (std::getline(file, left_str))
//    {
//        left_str = root_dir + left_str;
//        im_list.push_back(left_str);
//    }
//
//    cout<<im_list[0]<<endl;
//
//    int index_arr[13] = {3,4,5,7,8,10,11,12,13,20,21,22,23};
//
//    //int index = 4;
//    for (int index:index_arr) {
//
//        string left_file = im_list[index];
//        string right_file =  regex_replace(left_file, regex("left"), "right");
//
//        boost::format fmt("%03d_disp.png");
//        string name = (fmt % (index+1)).str();
//
//        boost::format fmt1("%03d_warp.png");
//        string name_warp = (fmt1 % (index+1)).str();
//
//        string out_file = "/media/yzx/Elements/Dataset/SCARED_Keyframes/Result/test_Result/" + name;
//        string warp_file = "/media/yzx/Elements/Dataset/SCARED_Keyframes/Result/test_Result/" + name_warp;
//
//        run(left_file, right_file, out_file, warp_file);
//
//    }



//    string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/006.png";
//    string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/006.png";
//    string out_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/test_Result/006_disp.png";
//    string warp_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/test_Result/006_warp.png";
//    run(left_file, right_file, out_file, warp_file);


    // 文件路径
//    int index = 1;
//    boost::format fmt("%03d.png");
//    string name = (fmt % index).str();
//    //cout<<name<<endl;
//

    //test_colon();
    //test_SERV();
    //OpticalFlowTest();


}

