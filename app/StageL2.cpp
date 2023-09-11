#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "Stereo_Huber_L1.h"
#include "ImageIO.h"
#include "OpticalFlow_Huber_L2.h"
#include "Stereo_Huber_L2.h"
#include <boost/format.hpp>
#include <regex>
#include <pangolin/pangolin.h>
#include <vector>

#include <string>
#include <unistd.h>

using namespace std;
using namespace cv;

void visualize(float **img, int row, int col, const string& title) {
    Mat mat = Mat::zeros(row,col,CV_8UC1);
    ToMat(img, mat);

    imshow(title,mat);

    waitKey();
}

double calMAD(Mat disparity_gt, Mat pred_disp, double fb, double scale){
    int num = 0;
    double MAD = 0;
    for (int v = 0; v < disparity_gt.rows; v++) {
        for (int u = 0; u < disparity_gt.cols; u++) {
            auto dis_gt = (double) disparity_gt.at<unsigned short >(v, u);
            dis_gt = dis_gt/scale;
            auto dis_pred = (double) pred_disp.at<unsigned short >(v, u);
            dis_pred = dis_pred/scale;
            if (dis_gt <= 0.0 ||dis_pred <= 0.0) continue; //  dis_gt >= 196.0 ||
            double depth_gt = fb / dis_gt;
            double depth_pred = fb/dis_pred;
            double error = abs(depth_pred-depth_gt);
            MAD = MAD + error;
            num = num + 1;
        }
    }
    MAD = MAD/num;

    return MAD;

}

double calMAD_mask(Mat disparity_gt, Mat pred_disp, double fb, double scale, Mat valid_mask){
    int num = 0;
    double MAD = 0;
    for (int v = 0; v < disparity_gt.rows; v++) {
        for (int u = 0; u < disparity_gt.cols; u++) {
            auto dis_gt = (double) disparity_gt.at<unsigned short >(v, u);
            dis_gt = dis_gt/scale;
            auto dis_pred = (double) pred_disp.at<unsigned short >(v, u);
            dis_pred = dis_pred/scale;

            int mask_value = valid_mask.at<uchar>(v,u);
//            cout<<dis_gt<<endl;
//            cout<<dis_pred<<endl;
//            cout<<mask_value<<endl;
            if (dis_gt > 0.0 && dis_pred > 0.0 && mask_value>0) {
                //cout<<"valid"<<endl;
                double error_value;
                if(fb>0){
                    double depth_gt = fb / dis_gt;
                    double depth_pred = fb/dis_pred;
                    error_value = abs(depth_pred-depth_gt);}
                else{
                    error_value = abs(dis_pred-dis_gt);}

                //cout<<num<<endl;

                MAD = MAD + error_value;
                num = num + 1;}
            //cout<<"MAD"<<MAD<<endl;
            //cout<<"num"<<num<<endl;
        }
    }
    //cout<<"MAD"<<MAD<<endl;
    //cout<<"num"<<num<<endl;

    MAD = MAD/num;

    return MAD;

}

class Refinement{

public:
    int scale_im;
    int orgCols;
    int orgRows;
    int ImageCols;
    int ImageRows;
    Stereo_Huber_L1::Parms inputParms;
    Stereo_Huber_L1::Data data{};
    Stereo_Huber_L1 of;
    Mat gt;
    Mat disp;
    Mat valid_mask;
    Mat ref_result;

    void setup_Parms(Stereo_Huber_L1::Parms inputParms_){
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
        if (vis){
            visualize(data.u,ImageRows,ImageCols,"flow");}
    }

    double eval(const string& gt_path, const string& valid_mask_dir, double fb, bool vis=false){
        gt = cv::imread(gt_path,IMREAD_ANYDEPTH);
        valid_mask = cv::imread(valid_mask_dir,IMREAD_GRAYSCALE);
        valid_mask.setTo(1,valid_mask>0);
        double scale = 256;

        Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(data.u, flow_u);

        Mat flow_u_16= Mat::zeros(ImageRows,ImageCols,CV_16UC1);
        flow_u.convertTo(flow_u_16,CV_16UC1, scale);
        resize(flow_u_16,flow_u_16,Size(ImageCols*scale_im,ImageRows*scale_im));
        flow_u_16 = flow_u_16*scale_im;
        ref_result = flow_u_16;
        if(vis) {
            imshow("ref", ref_result);
            waitKey();
        }
//        if (save){
//            imwrite(save_path,flow_u_16);
//        }

        return calMAD_mask(gt, flow_u_16, fb, scale, valid_mask);

    }

    void save(string save_path){
        double scale = 256;

        Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(data.u, flow_u);

        Mat flow_u_16= Mat::zeros(ImageRows,ImageCols,CV_16UC1);
        flow_u.convertTo(flow_u_16,CV_16UC1, scale);
        resize(flow_u_16,flow_u_16,Size(ImageCols*scale_im,ImageRows*scale_im));
        flow_u_16 = flow_u_16*scale_im;
        ref_result = flow_u_16;
//        cv::medianBlur	(ref_result,ref_result,5);
//        cv::medianBlur	(ref_result,ref_result,5);
        imwrite(save_path,ref_result);
    }

    double eval_input(double fb){
        return calMAD_mask(gt, disp, fb, 256, valid_mask);

    }


};
void testone(){
    Refinement Final;
    // set parameters
    Stereo_Huber_L1::Parms inputParms{};
    inputParms.epsilon = 0.1f;
    inputParms.outerIters = 30;
    inputParms.innerIters = 5;
    inputParms.theta = 0.3f;
    inputParms.lambda = 1.0f; //patch = 120  D1 = 15.0f;
    inputParms.levels = 6;
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 5;//20
    inputParms.descriptor = Stereo_Huber_L1::D1;
    inputParms.withPrior= false;

    Final.setup_Parms(inputParms);

    int index = 1;

    boost::format fmt("%03d.png");
    string name = (fmt%index).str();
//    boost::format fmt1("%03d_disp.png");

    string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
    string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/"+name;//013.png";
    string disp_deep =  "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/refine_"+name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
    //string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);
    string valid_mask_dir = regex_replace(disp_deep, regex("refine_"), "valid_mask");
    //string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
    string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;

    double fb;
    if (index<=8){
        fb = 5.4301732344712009e+03;
    }
    else{
        fb = 5.1446987361551292e+03;
    }

    int scale_im = 2;
    Final.setup_Data( left_file, right_file,disp_deep, scale_im);
    Final.run(true);
    double MAD = Final.eval(gt_path, valid_mask_dir, fb, true);
    cout<<"MAD: "<<MAD<<endl;

}

void test_SCARED(){
    Refinement Final;
    // set parameters
    Stereo_Huber_L1::Parms inputParms{};
    inputParms.epsilon = 0.3f;
    inputParms.outerIters = 30;
    inputParms.innerIters = 5;
    inputParms.theta = 0.3f;
    inputParms.lambda = 1.0f; //patch = 120  D1 = 15.0f;
    inputParms.levels = 1;
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 1;//20
    inputParms.descriptor = Stereo_Huber_L1::D1;
    inputParms.withPrior= true;

    Final.setup_Parms(inputParms);

    //vector<string> Methods={"STTR","AAnet","LEAStereo","PSMnet"};
    string method = "AAnet";
    boost::format fmt("%03d.png");
    string txt_name = "/media/yzx/Elements/Dataset/SCARED_Keyframes/left_list.txt";
    string root_dir = "/media/yzx/Elements/Dataset/SCARED_Keyframes";
    string result_dir = "/media/yzx/Elements/Dataset/SCARED_Keyframes/Result/";
    string gt_root = "/media/yzx/Elements/Dataset/SCARED_Keyframes/Result_depth/GT_disp/";

    std::ifstream file(txt_name);
    string left_file;
    vector<string> im_list;

    while (std::getline(file, left_file)) {
        // Process str
        im_list.push_back(left_file);
    }

    cout << method << endl;
    int i = 19;
    left_file = root_dir + im_list[i];
    i = i +1;
    string right_file = regex_replace(left_file, regex("\\left"), "right");
    string name = (fmt % i).str();
    string disp_deep = result_dir + method + "/refine_" + name;
    string valid_mask_dir = result_dir + method + "/valid_mask" + name;
//            string gt_path = "/media/yzx/Elements/Dataset/SCARED_Keyframes/Result_depth/GT" + name;
    string save_path = "/media/yzx/Elements/Dataset/SCARED_Keyframes/Result/"+ method+"/final"+ name;

    Mat disp = cv::imread(disp_deep,IMREAD_ANYDEPTH);
    Mat  left =cv::imread(left_file);
    imshow("left",left);
    imshow("disp",disp);
    waitKey();

    int scale_im = 2;
    Final.setup_Data(left_file, right_file, disp_deep, scale_im);
    Final.run(false);
    string gt_path = gt_root + name;
    double MAD = Final.eval(gt_path, valid_mask_dir, 0, true);
    double MAD_input = Final.eval_input(0);
    cout<<"Input "<<MAD_input<<endl;
    cout<<"Ref "<<MAD<<endl;

    // Final.save(save_path);
}


int main(){
    testone();
    //test_SCARED();
}