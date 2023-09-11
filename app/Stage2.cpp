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


using namespace std;
using namespace cv;
using namespace Eigen;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
constexpr auto Invalid_Float = std::numeric_limits<float>::infinity();

void visualize(float **img, int row, int col, const string& title) {
    Mat mat = Mat::zeros(row,col,CV_16UC1);
    ToMat_scale_u16(img, mat, 256.0);

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

    double eval(const string& gt_path, const string& valid_mask_dir, double fb, bool vis=false){
        gt = cv::imread(gt_path,IMREAD_ANYDEPTH);
        valid_mask = cv::imread(valid_mask_dir,IMREAD_GRAYSCALE);
        valid_mask.setTo(1,valid_mask>0);
        double scale = 256;

        if(vis) {
            imshow("ref", ref_result);
            waitKey();
        }
//        if (save){
//            imwrite(save_path,flow_u_16);
//        }

       return calMAD_mask(gt, ref_result, fb, scale, valid_mask);

    }

    void save(string save_path){

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
    Stereo_Huber_L2::Parms inputParms{};
    inputParms.epsilon = 0.1f;
    inputParms.iterations = 50;//50;//50;
    inputParms.lambda = 12;//15.0f;//100.0f;
    //patch = 120  D1 = 15.0f;
    inputParms.levels = 6;//6;//6; //6
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 5;
    inputParms.descriptor = Stereo_Huber_L2::D1;//PATCH_INTENSITY;
    inputParms.withPrior = false;//true;//true;

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
    double MAD = Final.eval(gt_path, valid_mask_dir, fb);
    cout<<"MAD: "<<MAD<<endl;
}

void testone_prior(){
    Refinement Final;
    // set parameters
    Stereo_Huber_L2::Parms inputParms{};

    inputParms.epsilon = 0.1f;
    inputParms.iterations = 30;//50;//50;
    inputParms.lambda = 6;//15.0f;//100.0f;
    //patch = 120  D1 = 15.0f;
    inputParms.levels = 4;//6;//6; //6
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 5;
    inputParms.descriptor = Stereo_Huber_L2::D1;//PATCH_INTENSITY;
    inputParms.withPrior = true;//true;//true;

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
    double MAD = Final.eval(gt_path, valid_mask_dir, fb);
    cout<<"MAD: "<<MAD<<endl;
}

void test_SERV(){
    Refinement Final;
    // set parameters
    Stereo_Huber_L2::Parms inputParms{};
    inputParms.epsilon = 0.1f;
    inputParms.iterations = 50;//50;//50;
    inputParms.lambda = 12;//15.0f;//100.0f;
    //patch = 120  D1 = 15.0f;
    inputParms.levels = 6;//6;//6; //6
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 5;
    inputParms.descriptor = Stereo_Huber_L2::D1;//PATCH_INTENSITY;
    inputParms.withPrior = false;//true;//true;

    Final.setup_Parms(inputParms);

    double ref_error = 0;
    double org_error = 0;


    for (int index=1;index<17;index++) {

        boost::format fmt("%03d.png");
        string name = (fmt % index).str();
//    boost::format fmt1("%03d_disp.png");

        string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
        string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/" + name;//013.png";
        string disp_deep = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/STTR/refine_"+name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
        //string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);refine_
        string valid_mask_dir = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/STTR/valid_mask" +
                                name;//regex_replace(disp_deep, regex("refine_"), "valid_mask");
        //string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
        string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;
        string save_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/STTR/final_"+name;

        double fb;
        if (index <= 8) {
            fb = 5.4301732344712009e+03;
        } else {
            fb = 5.1446987361551292e+03;
        }

        int scale_im = 2;
        Final.setup_Data(left_file, right_file, disp_deep, scale_im);
        Final.run(false);
        double MAD = Final.eval(gt_path, valid_mask_dir, fb);
        Final.save(save_path);
        //cout << "MAD: " << MAD << endl;
        ref_error = ref_error + MAD;
        org_error = org_error + Final.eval_input(fb);

    }
    cout<<"mean op "<< ref_error/16<<endl;
    cout<<"mean org "<< org_error/16<<endl;
}

void test_SERV_allmethods(){
    Refinement Final;
    // set parameters
    Stereo_Huber_L2::Parms inputParms{};
    inputParms.epsilon = 0.1f;
    inputParms.iterations = 30;//50;//50;
    inputParms.lambda = 6;//15.0f;//100.0f;
    //patch = 120  D1 = 15.0f;
    inputParms.levels = 4;//6;//6; //6
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 5;
    inputParms.descriptor = Stereo_Huber_L2::D1;//PATCH_INTENSITY;
    inputParms.withPrior = true;//true;//true;

    Final.setup_Parms(inputParms);

    vector<string> Methods={"STTR","AAnet","LEAStereo","PSMnet"};

    for (auto method:Methods){
        cout<<method<<endl;

    double ref_error = 0;
    double org_error = 0;

    for (int index=1;index<17;index++) {

        boost::format fmt("%03d.png");
        string name = (fmt % index).str();
//    boost::format fmt1("%03d_disp.png");

        string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
        string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/" + name;//013.png";
        string disp_deep = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/"+method+"/refine_"+name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
        //string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);refine_
        string valid_mask_dir = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/"+method+"/valid_mask" +
                                name;//regex_replace(disp_deep, regex("refine_"), "valid_mask");
        //string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
        string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;
        string save_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/"+method+"/final_"+name;

        double fb;
        if (index <= 8) {
            fb = 5.4301732344712009e+03;
        } else {
            fb = 5.1446987361551292e+03;
        }

        int scale_im = 2;
        Final.setup_Data(left_file, right_file, disp_deep, scale_im);
        Final.run(false);
        double MAD = Final.eval(gt_path, valid_mask_dir, fb);
        Final.save(save_path);
        //cout << "MAD: " << MAD << endl;
        ref_error = ref_error + MAD;
        org_error = org_error + Final.eval_input(fb);

    }
    cout<<"mean op "<< ref_error/16<<endl;
    cout<<"mean org "<< org_error/16<<endl;
    cout<<"\n";}
}

void test_SCARED_allmethods(){
    Refinement Final;
    // set parameters
    Stereo_Huber_L2::Parms inputParms{};
    inputParms.epsilon = 0.1f;
    inputParms.iterations = 30;//50;//50;
    inputParms.lambda = 6;//15.0f;//100.0f;
    //patch = 120  D1 = 15.0f;
    inputParms.levels = 4;//6;//6; //6
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 5;
    inputParms.descriptor = Stereo_Huber_L2::D1;//PATCH_INTENSITY;
    inputParms.withPrior = true;//true;//true;

    Final.setup_Parms(inputParms);

    vector<string> Methods={"STTR","AAnet","LEAStereo","PSMnet"};
    boost::format fmt("%03d.png");
    string txt_name = "/media/yzx/Elements/Dataset/SCARED_Keyframes/left_list.txt";
    string root_dir = "/media/yzx/Elements/Dataset/SCARED_Keyframes";
    string result_dir = "/media/yzx/Elements/Dataset/SCARED_Keyframes/Result/";
    int scale_im = 2;


    for (auto method:Methods) {
        cout << method << endl;
        int i = 0;
        std::ifstream file(txt_name);
        string left_file;

        while (std::getline(file, left_file)) {
            // Process str

            left_file = root_dir + left_file;
            string right_file = regex_replace(left_file, regex("\\left"), "right");
            i = i + 1;
            string name = (fmt % i).str();
            string disp_deep = result_dir + method + "/refine_" + name;
            string valid_mask_dir = result_dir + method + "/valid_mask" + name;
//            string gt_path = "/media/yzx/Elements/Dataset/SCARED_Keyframes/Result_depth/GT" + name;
            string save_path = "/media/yzx/Elements/Dataset/SCARED_Keyframes/Result/"+ method+"/final"+ name;


            Final.setup_Data(left_file, right_file, disp_deep, scale_im);
            Final.run(false);
            Final.save(save_path);

        }

    }
}

void test_SCARED(){
    Refinement Final;
    // set parameters
    Stereo_Huber_L2::Parms inputParms{};
    inputParms.epsilon = 0.2f;
    inputParms.iterations = 1;//50;//50;
    inputParms.lambda = 6;//15.0f;//100.0f;
    //patch = 120  D1 = 15.0f;
    inputParms.levels = 2;//6;//6; //6
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 3;
    inputParms.descriptor = Stereo_Huber_L2::D1;//PATCH_INTENSITY;
    inputParms.withPrior = true;//true;//true;

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
    int i = 1;
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

    int scale_im = 4;
    Final.setup_Data(left_file, right_file, disp_deep, scale_im);
    Final.run(false);
    string gt_path = gt_root + name;
    double MAD = Final.eval(gt_path, valid_mask_dir, 0, true);
    double MAD_input = Final.eval_input(0);
    cout<<"Input "<<MAD_input<<endl;
    cout<<"Ref "<<MAD<<endl;

   // Final.save(save_path);
}

void select_level(){
    vector<int> Level={2,3,4,5,6};

    for (auto level:Level) {
        cout<<"Level "<<level<<endl;
        Refinement Final;
        // set parameters
        Stereo_Huber_L2::Parms inputParms{};
        inputParms.epsilon = 0.001f;
        inputParms.iterations = 50;//50;//50;
        inputParms.lambda = 12;//15.0f;//100.0f;
        //patch = 120  D1 = 15.0f;
        inputParms.levels = level;//6;//6; //6
        inputParms.minSize = 48;
        inputParms.scalefactor = 0.5f;
        inputParms.warps = 5;
        inputParms.descriptor = Stereo_Huber_L2::D1;//PATCH_INTENSITY;
        inputParms.withPrior = true;//true;//true;

        Final.setup_Parms(inputParms);

        double ref_error = 0;
        double org_error = 0;


        for (int index = 1; index < 17; index++) {

            boost::format fmt("%03d.png");
            string name = (fmt % index).str();
//    boost::format fmt1("%03d_disp.png");

            string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
            string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/" + name;//013.png";
            string disp_deep = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/refine_" +
                               name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
            //string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);refine_
            string valid_mask_dir = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/valid_mask" +
                                    name;//regex_replace(disp_deep, regex("refine_"), "valid_mask");
            //string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
            string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;

            double fb;
            if (index <= 8) {
                fb = 5.4301732344712009e+03;
            } else {
                fb = 5.1446987361551292e+03;
            }

            int scale_im = 2;
            Final.setup_Data(left_file, right_file, disp_deep, scale_im);
            Final.run(false);
            double MAD = Final.eval(gt_path, valid_mask_dir, fb);
            //cout << "MAD: " << MAD << endl;
            ref_error = ref_error + MAD;
            org_error = org_error + Final.eval_input(fb);

        }
        cout << "mean op " << ref_error / 16 << endl;
        cout << "mean org " << org_error / 16 << endl;

    }
}

void select_eps(){vector<float> EPS={0.02,0.01,0.03,0.04,0.05};

    for (auto eps:EPS) {
        cout<<"EPS "<<eps<<endl;
        Refinement Final;
        // set parameters
        Stereo_Huber_L2::Parms inputParms{};
        inputParms.epsilon = eps;
        inputParms.iterations = 50;//50;//50;
        inputParms.lambda = 12;//15.0f;//100.0f;
        //patch = 120  D1 = 15.0f;
        inputParms.levels = 3;//6;//6; //6
        inputParms.minSize = 48;
        inputParms.scalefactor = 0.5f;
        inputParms.warps = 5;
        inputParms.descriptor = Stereo_Huber_L2::D1;//PATCH_INTENSITY;
        inputParms.withPrior = true;//true;//true;

        Final.setup_Parms(inputParms);

        double ref_error = 0;
        double org_error = 0;


        for (int index = 1; index < 17; index++) {

            boost::format fmt("%03d.png");
            string name = (fmt % index).str();
//    boost::format fmt1("%03d_disp.png");

            string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
            string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/" + name;//013.png";
            string disp_deep = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/refine_" +
                               name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
            //string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);refine_
            string valid_mask_dir = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/valid_mask" +
                                    name;//regex_replace(disp_deep, regex("refine_"), "valid_mask");
            //string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
            string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;

            double fb;
            if (index <= 8) {
                fb = 5.4301732344712009e+03;
            } else {
                fb = 5.1446987361551292e+03;
            }

            int scale_im = 2;
            Final.setup_Data(left_file, right_file, disp_deep, scale_im);
            Final.run(false);
            double MAD = Final.eval(gt_path, valid_mask_dir, fb);
            //cout << "MAD: " << MAD << endl;
            ref_error = ref_error + MAD;
            org_error = org_error + Final.eval_input(fb);

        }
        cout << "mean op " << ref_error / 16 << endl;
        cout << "mean org " << org_error / 16 << endl;

    }
}

void test_mad(){
    int index = 1;
    boost::format fmt("%03d.png");
    string name = (fmt % index).str();
    string disp_deep = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/"+name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
    //string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);refine_
    string valid_mask_dir = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/valid_mask" +
                            name;//regex_replace(disp_deep, regex("refine_"), "valid_mask");
    //string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
    string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;

    Mat gt = cv::imread(gt_path,IMREAD_ANYDEPTH);
    Mat disp = cv::imread(disp_deep,IMREAD_ANYDEPTH);
    Mat valid_mask = cv::imread(valid_mask_dir,IMREAD_GRAYSCALE);
    valid_mask.setTo(1,valid_mask>0);

    double fb;
    if (index <= 8) {
        fb = 5.4301732344712009e+03;
    } else {
        fb = 5.1446987361551292e+03;
    }

    double err = calMAD_mask(gt, disp, fb, 256, valid_mask);
    cout<<"erro "<<err<<endl;
}

void select_lam(){
    vector<float> Var={7,8,9,10,11,12};

    for (auto var:Var) {
        cout<<"Lambda "<<var<<endl;
        Refinement Final;
        // set parameters
        Stereo_Huber_L2::Parms inputParms{};
        inputParms.epsilon = 0.01;
        inputParms.iterations = 50;//50;//50;
        inputParms.lambda = var;// 12;//15.0f;//100.0f;
        //patch = 120  D1 = 15.0f;
        inputParms.levels = 3;//6;//6; //6
        inputParms.minSize = 48;
        inputParms.scalefactor = 0.5f;
        inputParms.warps = 5;
        inputParms.descriptor = Stereo_Huber_L2::D1;//PATCH_INTENSITY;
        inputParms.withPrior = true;//true;//true;

        Final.setup_Parms(inputParms);

        double ref_error = 0;
        double org_error = 0;


        for (int index = 1; index < 17; index++) {

            boost::format fmt("%03d.png");
            string name = (fmt % index).str();
//    boost::format fmt1("%03d_disp.png");

            string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
            string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/" + name;//013.png";
            string disp_deep = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/PSMnet/refine_" +
                               name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
            //string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);refine_
            string valid_mask_dir = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/PSMnet/valid_mask" +
                                    name;//regex_replace(disp_deep, regex("refine_"), "valid_mask");
            //string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
            string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;

            double fb;
            if (index <= 8) {
                fb = 5.4301732344712009e+03;
            } else {
                fb = 5.1446987361551292e+03;
            }

            int scale_im = 2;
            Final.setup_Data(left_file, right_file, disp_deep, scale_im);
            Final.run(false);
            double MAD = Final.eval(gt_path, valid_mask_dir, fb);
            //cout << "MAD: " << MAD << endl;
            ref_error = ref_error + MAD;
            org_error = org_error + Final.eval_input(fb);

        }
        cout << "mean op " << ref_error / 16 << endl;
        cout << "mean org " << org_error / 16 << endl;

    }
}

void testother(){

    Refinement Final;
    // set parameters
    Stereo_Huber_L2::Parms inputParms{};
    inputParms.epsilon = 0.1f;
    inputParms.iterations = 30;//50;//50;
    inputParms.lambda = 6;//15.0f;//100.0f;
    //patch = 120  D1 = 15.0f;
    inputParms.levels = 4;//6;//6; //6
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 5;
    inputParms.descriptor = Stereo_Huber_L2::D1;//PATCH_INTENSITY;
    inputParms.withPrior = true;//true;//true;

    Final.setup_Parms(inputParms);


    boost::format fmt("%04d.png");
    boost::format fmt1("%010d.jpg");
    //int index = 0;// 100 800
    vector<string> folders={"rectified01","rectified04","rectified05","rectified08","rectified09","rectified11"};
    vector<int> indexs={0,100,300,400,500,600,700,800};

    for (const string& folder:folders){
        for (auto index:indexs){
            string name_disp = (fmt % index).str();
            string name_im = (fmt1 % index).str();
            string left_file = "/media/yzx/Elements/Dataset/Hamlyn_all/hamlyn_data/"+folder+"/image01/"+ name_im;
            string right_file = "/media/yzx/Elements/Dataset/Hamlyn_all/hamlyn_data/"+folder+"/image02/" + name_im;
            string disp_deep = "/media/yzx/Elements/Dataset/Hamlyn_all/hamlyn_data/" + folder +"/result/refine_" + name_disp;//0000.png"; //"/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/013.png";// "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/AAnet/001.png";
            string out_dir = "/media/yzx/Elements/Dataset/Hamlyn_all/hamlyn_data/" + folder +"/result/final_"+name_disp;

            int scale_im = 1;
            Final.setup_Data( left_file, right_file,disp_deep, scale_im);
            Final.run(false);
            Final.save(out_dir);


        }
    }




}

int main(){

    //test_SERV_allmethods();
    //test_SERV();
    test_SCARED_allmethods();
    //test_SCARED();
    //testother();
    //testone_prior();

}