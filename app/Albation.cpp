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
#include <numeric>


using namespace std;
using namespace cv;
using namespace Eigen;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
constexpr auto Invalid_Float = std::numeric_limits<float>::infinity();

void visualize(float **img, int row, int col, const string& title) {
    Mat mat = Mat::zeros(row,col,CV_8UC1);
    ToMat(img, mat);

    imshow(title,mat);

    waitKey();
}

double calRMSE_mask(Mat disparity_gt, Mat pred_disp, Mat valid_mask){
    int num = 0;
    double Sum = 0;
    double RMSE;
    for (int v = 0; v < disparity_gt.rows; v++) {
        for (int u = 0; u < disparity_gt.cols; u++) {
            auto dis_gt = (double) disparity_gt.at<unsigned short >(v, u);
            auto dis_pred = (double) pred_disp.at<unsigned short >(v, u);
            int mask_value = valid_mask.at<uchar>(v,u);
            if (dis_gt > 0.0 && dis_pred > 0.0 && mask_value==0) {
                double error_value = abs(dis_pred-dis_gt)/256.0;
                Sum = Sum + error_value * error_value;
                num = num + 1;}
        }
    }

    RMSE = sqrt(Sum/num);

    return RMSE;
}

double calRMSE(Mat disparity_gt, Mat pred_disp){
    int num = 0;
    double Sum = 0;
    double RMSE;
    for (int v = 0; v < disparity_gt.rows; v++) {
        for (int u = 0; u < disparity_gt.cols; u++) {
            auto dis_gt = (double) disparity_gt.at<unsigned short >(v, u);
            auto dis_pred = (double) pred_disp.at<unsigned short >(v, u);
            if (dis_gt > 0.0 && dis_pred > 0.0 ) {
                double error_value = abs(dis_pred-dis_gt)/256.0;
                Sum = Sum + error_value * error_value;
                num = num + 1;}
        }
    }

    RMSE = sqrt(Sum/num);

    return RMSE;
}

double caldepthRMSE_mask(Mat disparity_gt, Mat pred_disp, Mat valid_mask, double fb){
    int num = 0;
    double Sum = 0;
    double RMSE;
    for (int v = 0; v < disparity_gt.rows; v++) {
        for (int u = 0; u < disparity_gt.cols; u++) {
            auto dis_gt = (double) disparity_gt.at<unsigned short >(v, u)/256.0;
            auto dis_pred = (double) pred_disp.at<unsigned short >(v, u)/256.0;
            int mask_value = valid_mask.at<uchar>(v,u);
            if (dis_gt > 0.0 && dis_pred > 0.0 && mask_value==0) {
                double depth_gt = fb / dis_gt;
                double depth_pred = fb/dis_pred;
                double error_value = abs(depth_pred-depth_gt);
                Sum = Sum + error_value * error_value;
                num = num + 1;}
        }
    }

    RMSE = sqrt(Sum/num);

    return RMSE;
}

double caldepthRMSE(Mat disparity_gt, Mat pred_disp, double fb){
    int num = 0;
    double Sum = 0;
    double RMSE;
    for (int v = 0; v < disparity_gt.rows; v++) {
        for (int u = 0; u < disparity_gt.cols; u++) {
            auto dis_gt = (double) disparity_gt.at<unsigned short >(v, u)/256.0;
            auto dis_pred = (double) pred_disp.at<unsigned short >(v, u)/256.0;
            if (dis_gt > 0.0 && dis_pred > 0.0 ) {
                double depth_gt =dis_gt;//  fb /
                double depth_pred = fb/dis_pred;
                double error_value = abs(depth_pred-depth_gt);
                Sum = Sum + error_value * error_value;
                num = num + 1;}
        }
    }

    RMSE = sqrt(Sum/num);

    return RMSE;
}

double cal_std(const vector<double>& v){
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / v.size() - mean * mean);

    return stdev;
}


class Refinement{

public:
    int scale_im;
    int orgCols;
    int orgRows;
    int ImageCols;
    int ImageRows;
    Stereo_Huber_L2::Parms inputParms;
    Stereo_Huber_L2::Data data{};
    Stereo_Huber_L2 of;
    Mat gt;
    Mat disp;
    Mat valid_mask;
    Mat ref_result;

    void setup_Parms(Stereo_Huber_L2::Parms inputParms_){
        inputParms = inputParms_;
    }

    void setup_Data(const string& left_file, const string& right_file, const string& disp_file, int scale_im_) {

        scale_im = scale_im_;
        Mat left_org = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
        Mat right_org = cv::imread(right_file,cv::IMREAD_GRAYSCALE);
        Mat disp_mat_org = cv::imread(disp_file,IMREAD_ANYDEPTH);
        disp = disp_mat_org;

        orgCols = left_org.cols;
        orgRows = left_org.rows;

//        cout<<"col "<<orgCols<<endl;
//        cout<<"row "<<orgRows<<endl;

        ImageCols = orgCols/scale_im;
        ImageRows = orgRows/scale_im;

//        cout<<"col "<<ImageCols<<endl;
//        cout<<"row "<<ImageRows<<endl;

        // resize imgs and disp
        Mat left, right, disp_mat;// flow;
        cv::resize(left_org, left, Size(ImageCols, ImageRows));
        cv::resize(right_org, right, Size(ImageCols, ImageRows));
        cv::resize(disp_mat_org, disp_mat, Size(ImageCols, ImageRows));
        disp_mat = disp_mat/scale_im;

        float** image1 = AllocateImage<float>(ImageCols, ImageRows);
        float** image2 = AllocateImage<float>(ImageCols, ImageRows);
        float** u = AllocateImage<float>(ImageCols, ImageRows);
        float** disp_raw = AllocateImage<float>(ImageCols, ImageRows);

        Mat2Raw(image1,left);
        Mat2Raw(image2,right);
        float disp_scale = -256.0;
        //  Mat2Raw_scale(flow_raw,flow, disp_scale);
        Mat2Raw_scale(disp_raw,disp_mat, disp_scale);


        data.I1 = image1;
        data.I2 = image2;
        data.u = u;
        data.nCols = ImageCols;
        data.nRows = ImageRows;
        data.alpha = nullptr;

        if(inputParms.withPrior){
            data.uprior = disp_raw;
        }
        else{
            data.uprior = nullptr;
        }//deep_raw;//NULL;deep_raw;

    }

    void run(bool vis= false){
        of.setup(data, inputParms);
        of.run();

//        double scale = 256;
//        Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
//        ToMat(data.u, flow_u);
//        flow_u.convertTo(flow_u_16,CV_16UC1, scale);

        Mat flow_u_16= Mat::zeros(ImageRows,ImageCols,CV_16UC1);
        ToMat_scale_u16(data.u,flow_u_16,256.0);
        resize(flow_u_16,flow_u_16,Size(ImageCols * scale_im,ImageRows * scale_im));
        flow_u_16 = flow_u_16 * scale_im;
        ref_result = flow_u_16;

        if (vis){
            visualize(data.u,ImageRows,ImageCols,"flow");
            imshow("ref", ref_result);
            waitKey();
        }

    }


    void save(string save_path){

//        cv::medianBlur	(ref_result,ref_result,5);
//        cv::medianBlur	(ref_result,ref_result,5);
        imwrite(save_path,ref_result);
    }

};



void testone(){
    Refinement Final;
    // set parameters
    Stereo_Huber_L2::Parms inputParms{};
    inputParms.epsilon = 0.1f;
    inputParms.iterations = 30;//50;//50;
    inputParms.lambda = 6;//15.0f;//100.0f; 6
    //patch = 120  D1 = 15.0f;
    inputParms.levels = 5;//6;//6; //6
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 5;
    inputParms.descriptor = Stereo_Huber_L2::D1;//D1;//
    inputParms.withPrior = false;//true;//true;

    Final.setup_Parms(inputParms);

    //int index = 1;
    double RMSE = 0;
    double RMSE_mask = 0;
    vector<double> RMSE_result;
    vector<double> RMSE_result_mask;

    double RMSE_depth = 0;
    double RMSE_depth_mask = 0;
    vector<double> RMSE_result_depth;
    vector<double> RMSE_result_depth_mask;

    for (int index=1;index<17;index++) {

        boost::format fmt("%03d.png");
        boost::format fmt1("%03dmask.png");
        string name = (fmt % index).str();
//    boost::format fmt1("%03d_disp.png");

        string left_file = "/media/yzx/yzx_store/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
        string right_file = "/media/yzx/yzx_store/Dataset/SERV-CT-ALL/images/right/" + name;//013.png";
        string disp_deep = "/media/yzx/yzx_store/Dataset/SERV-CT-ALL/Results/PSMnet/" +name; //refine_
        //013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
        //string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);
        string valid_mask_dir = "/media/yzx/yzx_store/Dataset/SERV-CT-ALL/Mask/" + (fmt1 % index).str();
        //string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
        string gt_path = "/media/yzx/yzx_store/Dataset/SERV-CT-ALL/images/gt_disp/" + name;
        string gt_depth_path = "/media/yzx/yzx_store/Dataset/SERV-CT-ALL/images/gt/" + name;

        int scale_im = 2;
        Final.setup_Data(left_file, right_file, disp_deep, scale_im);
        Final.run(false);


        Mat gt = cv::imread(gt_path, IMREAD_ANYDEPTH);
        Mat gt_depth = cv::imread(gt_depth_path, IMREAD_ANYDEPTH);
        Mat valid_mask = cv::imread(valid_mask_dir, IMREAD_GRAYSCALE);
//        imshow("mask", valid_mask);
//        waitKey();

        double fb;
        if (index <= 8) {
            fb = 5.4301732344712009e+03;
        } else {
            fb = 5.1446987361551292e+03;
        }

        double rmse_mask, rmse;
        rmse = calRMSE(gt, Final.ref_result);
        rmse_mask = calRMSE_mask(gt, Final.ref_result, valid_mask);

        RMSE_result.push_back(rmse);
        RMSE_result_mask.push_back(rmse_mask);

        RMSE = RMSE +rmse;
        RMSE_mask = RMSE_mask + rmse_mask;

        double rmse_depth_mask, rmse_depth;
        rmse_depth = caldepthRMSE(gt_depth, Final.ref_result,fb);
        rmse_depth_mask = caldepthRMSE_mask(gt, Final.ref_result, valid_mask,fb);

        RMSE_result_depth.push_back(rmse_depth);
        RMSE_result_depth_mask.push_back(rmse_depth_mask);

        RMSE_depth = RMSE_depth + rmse_depth;
        RMSE_depth_mask = RMSE_depth_mask + rmse_depth_mask;

    }

    RMSE = RMSE/16.0;
    RMSE_mask = RMSE_mask/16.0;

    RMSE_depth = RMSE_depth/16.0;
    RMSE_depth_mask = RMSE_depth_mask/16.0;

//    & 2.95 $\pm$ 1.78 & 2.38 $\pm$ 1.00
//                                   &   18.53  $\pm$    4.63  &    3.09  $\pm$    0.94

    cout<< "|  RMSE disp  |  RMSE depth  |  RMSE disp mask  | RMSE depth mask  |"<<endl;
    cout<< " & "<< RMSE<<" $pm$ "<<cal_std(RMSE_result)
    <<" & "<<RMSE_depth<<" $pm$ "<<cal_std(RMSE_result_depth)
    <<" & "<<RMSE_mask<<" $pm$ "<<cal_std(RMSE_result_mask)
    << " & "<<RMSE_depth_mask<<" $pm$ "<<cal_std(RMSE_result_depth_mask)<<endl;
//    & 2.95 $\pm$ 1.78 & 2.38 $\pm$ 1.00
//                                   &   18.53  $\pm$    4.63  &    3.09  $\pm$    0.94

//    cout<<"RMSE "<<RMSE<<"  std "<<cal_std(RMSE_result)<<endl;
//    cout<<"RMSE mask "<<RMSE_mask<<"  std "<<cal_std(RMSE_result_mask)<<endl;
//
//    cout<<"RMSE depth "<<RMSE_depth<<"  std "<<cal_std(RMSE_result_depth)<<endl;
//    cout<<"RMSE depth mask "<<RMSE_depth_mask<<"  std "<<cal_std(RMSE_result_depth_mask)<<endl;


    //valid_mask.setTo(1,valid_mask>0);
//    double MAD = Final.eval(gt_path, valid_mask_dir, fb);
//    cout<<"MAD: "<<MAD<<endl;

}

int main(){
    testone();

}