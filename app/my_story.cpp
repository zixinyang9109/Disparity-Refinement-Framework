//
// Created by yzx on 2/8/22.
//

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include "ImageIO.h"
#include "OpticalFlow_Huber_L2.h"
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


void showDepthPoints(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {

    const float red[3] = {1.0, 0, 0};
    glPointSize(8);
    glBegin(GL_POINTS);
    for (auto &p: pointcloud) {
        glColor3f(red[0], red[1], red[2]);
        glVertex3d(p[0], p[1], p[2]);
    }
    glEnd();

}

void showDepth(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {

    glPointSize(2);
    glBegin(GL_POINTS);
    for (auto &p: pointcloud) {
        glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
        glVertex3d(p[0], p[1], p[2]);
    }
    glEnd();

}
// 在pangolin中画图，已写好，无需调整
void Visualize(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud,const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud_gt) {

    if (pointcloud.empty()) {
        cerr << "Point cloud is empty!" << endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);


        showDepth(pointcloud);  // color
        showDepthPoints(pointcloud_gt); // red

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}

void getPointCloud(
        cv::Mat left,cv::Mat disparity_gt,double scale,double cx,double cy,double fx,double fy,double fb,vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud){

    for (int v = 0; v < left.rows; v++) {
        for (int u = 0; u < left.cols; u++) {
            double dis_value = (double) disparity_gt.at<unsigned short >(v, u);
            dis_value = dis_value/scale;
            //cout<<"dis value "<<dis_value<<endl;
            if (dis_value <= 0.0 || dis_value >= 196.0) continue;

            Vector6d point; // 前三维为xyz,第四维为颜色

            // 根据双目模型计算 point 的位置
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double depth = fb / dis_value;
            // cout<<"depth value "<<depth<<endl;
            point[0] = x * depth;
            point[1] = y * depth;
            point[2] = depth;

            point[5] = left.data[v * left.step + u * left.channels()];   // blue
            point[4] = left.data[v * left.step + u * left.channels() + 1]; // green
            point[3] = left.data[v * left.step + u * left.channels() + 2]; // red

            pointcloud.push_back(point);
        }
    }

    // 画出点云
//    showPointCloud(pointcloud);

}

double calMAD(Mat disparity_gt, Mat pred_disp, double fb, double scale){
    int num = 0;
    double MAD = 0;
    for (int v = 0; v < disparity_gt.rows; v++) {
        for (int u = 0; u < disparity_gt.cols; u++) {
            double dis_gt = (double) disparity_gt.at<unsigned short >(v, u);
            dis_gt = dis_gt/scale;
            double dis_pred = (double) pred_disp.at<unsigned short >(v, u);
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
            double dis_gt = (double) disparity_gt.at<unsigned short >(v, u);
            dis_gt = dis_gt/scale;
            double dis_pred = (double) pred_disp.at<unsigned short >(v, u);
            dis_pred = dis_pred/scale;

            int mask_value = valid_mask.at<uchar>(v,u);
//            cout<<dis_gt<<endl;
//            cout<<dis_pred<<endl;
//            cout<<mask_value<<endl;
            if (dis_gt <= 0.0  ||dis_pred <= 0.0||mask_value==0) continue;
            //cout<<"valid"<<endl;
            double depth_gt = fb / dis_gt;
            double depth_pred = fb/dis_pred;
            double error = abs(depth_pred-depth_gt);
            //cout<<num<<endl;
            MAD = MAD + error;
            num = num + 1;
            //cout<<"MAD"<<MAD<<endl;
            //cout<<"num"<<num<<endl;
        }
    }
    //cout<<"MAD"<<MAD<<endl;
    //cout<<"num"<<num<<endl;

    MAD = MAD/num;

    return MAD;

}

float Tex2D(float** t, const int w, const int h, const float x, const float y)
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

void warp_u(const int Rows, const int Cols, float** m_u, float** I_source, float** I_warp)
{
    float x, y;
    for (int i = 0; i < Rows; i++)
    {
        for (int j = 0; j < Cols; j++)
        {
            x = (float)j + m_u[i][j];
            y = (float)i;
            I_warp[i][j] = Tex2D(I_source, Cols, Rows, x, y);
        }
    }

}

void valid_range(const int Rows, const int Cols, float** m_u, float** valid_mask)
{
    int x, y;
    for (int i = 0; i < Rows; i++)
    {
        for (int j = 0; j < Cols; j++)
        {
            x = j + m_u[i][j];
            y = (int)i;
            if (x >= 0 && x < Cols && y >= 0 && y < Rows && m_u[i][j]!=0) {
                valid_mask[i][j] = 255;
            }
            else{
                valid_mask[i][j] = 0;
            }

        }
    }

}

void test_prior(){

    string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/013.png";
    string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/013.png";

    string disp_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/013_disp.png";
    string disp_deep =  "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/AAnet/013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//

    cv::Mat left = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_file,cv::IMREAD_GRAYSCALE);
    Mat flow_mat = cv::imread(disp_flow,IMREAD_ANYDEPTH );
    Mat deep_mat = cv::imread(disp_deep,IMREAD_ANYDEPTH);

    double min, max;
    cv::minMaxLoc(deep_mat, &min, &max);
    cout<<"disparity gt the min "<< min/256<<endl;
    cout<<"disparity gt the max "<< max/256<<endl;

    cv::imshow("flow",flow_mat);
    cv::imshow("deep",deep_mat);
    waitKey();

    int ImageCols = left.cols;
    int ImageRows = left.rows;

    float** image1 = AllocateImage<float>(ImageCols, ImageRows);
    float** image2 = AllocateImage<float>(ImageCols, ImageRows);

    float** u = AllocateImage<float>(ImageCols, ImageRows);
    float** v = AllocateImage<float>(ImageCols, ImageRows);
    float** alpha = AllocateImage<float>(ImageCols, ImageRows);

    //float** flow_raw = AllocateImage<float>(ImageCols, ImageRows); //u
    float** deep_raw =  AllocateImage<float>(ImageCols, ImageRows); //u
    float** v_raw =  AllocateImage<float>(ImageCols, ImageRows); //v

    Mat v_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    //Mat alpha_mat = Mat::ones(ImageRows,ImageCols,CV_8UC1);

    memset(alpha[0], 0, ImageCols * ImageRows * sizeof(float));
    memset(v[0], 0, ImageCols * ImageRows * sizeof(float));


    Mat2Raw(image1,left);
    Mat2Raw(image2,right);
    //Mat2Raw(v_raw,v_mat);
    //Mat2Raw(alpha,alpha_mat);

    float disp_scale = -256.0;
    //Mat2Raw_scale(flow_raw,flow_mat, disp_scale);
    Mat2Raw_scale(deep_raw,deep_mat, disp_scale);
    //cout<<deep_raw[10][10]<<endl;

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
    inputParms.bInitializeWithPrior= false;

    OpticalFlow_Huber_L2::Data data{};
    data.I1 = image1;
    data.I2 = image2;
    data.u = u;
    data.v = v;
    data.nCols = ImageCols;
    data.nRows = ImageRows;

    data.alpha = alpha;//alpha;//NULL;
    data.uprior = deep_raw;
    data.vprior = v_raw;


//    OpticalFlow_Huber_L2::Data data{};
//    data.alpha = NULL;
//    data.I1 = image1;
//    data.I2 = image2;
//    data.u = u;
//    data.v = v;
//    data.uprior = NULL;
//    data.vprior = NULL;
//    data.nCols = ImageCols;
//    data.nRows = ImageRows;

    OpticalFlow_Huber_L2 of;
    of.setup(data, inputParms);
    of.run();

    Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    ToMat(u, flow_u);

    cv::minMaxLoc(flow_u, &min, &max);
    cout<<"disparity pred the min "<< min<<endl;
    cout<<"disparity pred the max "<< max<<endl;

    float** image1_warp;
    image1_warp = of.m_I2w;
    Mat image1_warp_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    ToMat(image1_warp, image1_warp_mat);

    cv::imshow(" warped",image1_warp_mat);
    cv::imshow("left",left);
    cv::imshow(" optimized u",flow_u);
    cv::waitKey();

}

// Cols: height, Rows: width
void cal_dist_map(const int Rows, const int Cols, float** input, float** output,
                  int radius, int radius_step, int num_radius_step ){

    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j <Cols; j++) {
            //wnd_data.clear();
            // get window data around i,j
            float disp_win = abs(input[i][j]);//input[i*width+j];
            float disp_sum = 0;
            int num = 0;
            for (int k = 0; k < num_radius_step; ++k) {
                int radius_c = radius_step + radius;
                for (int r = -radius_c; r <= radius_c;) {
                    for (int c = -radius_c; c <= radius_c;) {
                        const int row = i + r;
                        const int col = j + c;
                        if (row >= 0 && row < Rows && col >= 0 && col < Cols ) { //&& r != 0 && c != 0
                            disp_sum = disp_sum + abs(input[row][col]);
                            num = num + 1;
                        }
                        c = c + radius_c/2;
                    }
                    r = r + radius_c/2;
                }
            }
            float mean_disp_sum = disp_sum/num;
            //cout << " value " << abs(mean_disp_sum - disp_win)<<endl;
            output[i][j] = abs(mean_disp_sum - disp_win)/mean_disp_sum;//  abs(mean_disp_sum - disp_win);
        }
    }

}

void confidence_map_warp(const int Rows, const int Cols, float** input, float** output, float** map, float alpha){

    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j <Cols; j++) {
            //wnd_data.clear();
            // get window data around i,j
            float value_in = abs(input[i][j]);//input[i*width+j];
            float value_out = abs(output[i][j]);
            float confidence_value = 1-alpha*(abs(value_in-value_out)/value_in);

            if(confidence_value>0){
                map[i][j] = confidence_value *255;
            }else{
                map[i][j] = 0;
            }
        }
    }
}

void confidence_map_dist(const int Rows, const int Cols, float** dist,float** output, float alpha){

    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j <Cols; j++) {
            //wnd_data.clear();
            // get window data around i,j
            float value = 1- alpha*abs(dist[i][j]);//input[i*width+j];

            if(value>0){
                output[i][j] = value *255 ;
            }else{
                output[i][j] = 0;
            }
        }
    }

}

void mul(const int Rows, const int Cols, float** im1,float** im2, float** output){

    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j <Cols; j++) {
            //wnd_data.clear();
            // get window data around i,j
            float value = (im1[i][j]/255)*(im2[i][j]/255);//1- alpha*abs(dist[i][j]);//input[i*width+j];
            output[i][j] = value *255 ;
        }
    }

}

void mul_all(const int Rows, const int Cols, float** im1, float** im2, float** im3, float** im4, float** output){

    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j <Cols; j++) {
            //wnd_data.clear();
            // get window data around i,j
            float value = (im1[i][j]/255)*(im2[i][j]/255)*(im3[i][j]/255)*(im4[i][j]/255);//1- alpha*abs(dist[i][j]);//input[i*width+j];
            output[i][j] = value *255 ;
        }
    }

}

void test_lost(){

    // test loss
    boost::format fmt("%03d.png");
    int index = 10;
    string name = (fmt % index).str();

    // 0. read left, right images, predicted results
    string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;
    string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/" + name;

    string disp_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/005_disp.png";
    string disp_deep = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/AAnet/" + name; //"/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/013.png";// "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/AAnet/001.png";

    bool use_photo = true;

    cv::Mat left = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_file,cv::IMREAD_GRAYSCALE);
    Mat flow_mat = cv::imread(disp_flow,IMREAD_ANYDEPTH );
    Mat deep_mat = cv::imread(disp_deep,IMREAD_ANYDEPTH);

    int ImageCols = left.cols; // width
    int ImageRows = left.rows; // height

    float** image1 = AllocateImage<float>(ImageCols, ImageRows); // images
    float** image2 = AllocateImage<float>(ImageCols, ImageRows);
    Mat2Raw(image1,left);
    Mat2Raw(image2,right);

    float** flow_raw = AllocateImage<float>(ImageCols, ImageRows); // flows
    float** deep_raw =  AllocateImage<float>(ImageCols, ImageRows);

    float disp_scale = -256.0;
    Mat2Raw_scale(flow_raw,flow_mat, disp_scale);
    Mat2Raw_scale(deep_raw,deep_mat, disp_scale);

    float** valid_mask = AllocateImage<float>(ImageCols, ImageRows);
    valid_range(ImageRows, ImageCols, deep_raw, valid_mask);
    //valid_range(ImageRows, ImageCols, flow_raw, valid_mask);
    Mat valid_mask_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    ToMat(valid_mask,valid_mask_mat);
    imshow("valid mask ",valid_mask_mat);
    imshow("flow_mat",flow_mat);
    waitKey();


    float** target = AllocateImage<float>(ImageCols, ImageRows);
    float** source = AllocateImage<float>(ImageCols, ImageRows);


    // 2. compute feature map
    if(use_photo){
        memcpy(target[0], image1[0], ImageCols * ImageRows * sizeof(float));
        memcpy(source[0], image2[0], ImageCols * ImageRows * sizeof(float));
    }else{
        cout<<"To be done"<<endl;
    }

    // 3. warped
    float** warp_deep = AllocateImage<float>(ImageCols, ImageRows);
    float** warp_flow = AllocateImage<float>(ImageCols, ImageRows);

    warp_u(ImageRows, ImageCols, deep_raw,  source,warp_deep);
    warp_u(ImageRows, ImageCols, flow_raw,  source,warp_flow);

    float** warp_deep_confidence = AllocateImage<float>(ImageCols, ImageRows);
    float** warp_flow_confidence = AllocateImage<float>(ImageCols, ImageRows);

    float alpha = 5;
    confidence_map_warp(ImageRows, ImageCols, target, warp_deep, warp_deep_confidence, alpha);
    confidence_map_warp(ImageRows, ImageCols, target, warp_flow, warp_flow_confidence, alpha);

    Mat warp_deep_confidence_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    ToMat(warp_deep_confidence, warp_deep_confidence_mat);

    Mat warp_flow_confidence_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    ToMat(warp_flow_confidence, warp_flow_confidence_mat);

    double min, max;
    cv::minMaxLoc(warp_deep_confidence_mat, &min, &max);
    cout<<" the min "<< min<<endl;
    cout<<" the max "<< max<<endl;

    cv::imshow("warp deep confidence",warp_deep_confidence_mat);
    cv::imshow("warp flow confidence",warp_flow_confidence_mat);
    cv::imshow("left",left);
    waitKey();

    // 4. compute loss
//    Mat Diff_deep, Diff_flow;
//    Mat warp_deep_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
//    ToMat(warp_deep, warp_deep_mat);
//    Mat warp_flow_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
//    ToMat(warp_flow, warp_flow_mat);
//    cv::absdiff(warp_deep_mat, left, Diff_deep);
//    cv::absdiff(warp_flow_mat, left, Diff_flow);
//
//    cv::imshow("flow",flow_mat);
//    cv::imshow("deep",deep_mat);
//
//    cv::imshow("warp deep",warp_deep_mat);
//    cv::imshow("warp flow",warp_flow_mat);
//    cv::imshow("left",left);
//    //waitKey();
//
//    cv::imshow("diff deep",255*(Diff_deep/left));
//    cv::imshow("diff flow",255*(Diff_flow/left));
//    cv::imshow("diff deep inv",255-255*(Diff_deep/left));
//    cv::imshow("diff flow inv",255-255*(Diff_flow/left));
//    cv::imshow("left",left);
//    waitKey();

    // 5. masks with regards to color and smoothness

    cv::Mat left_color = cv::imread(left_file);
    Mat hsv;
    cv::cvtColor(left_color,hsv,cv::COLOR_BGR2HSV);
    Mat im1_s;
    int channelIdx = 1;
    //double min, max;
    extractChannel(hsv, im1_s, channelIdx);
    cv::minMaxLoc(im1_s, &min, &max);
    cout<<" s channel the min "<< min/256<<endl;
    cout<<" s channel the max "<< max/256<<endl;
    cv::Mat mask_s = cv::Mat(ImageCols, ImageRows, CV_8UC1);
    mask_s = (im1_s>=(0.05*256));
    cv::minMaxLoc(mask_s, &min, &max);
    cout<<" mask s channel the min "<< min<<endl;
    cout<<" mask s channel the max "<< max<<endl;


    float** dist = AllocateImage<float>(ImageCols, ImageRows);
    float** dist_conf = AllocateImage<float>(ImageCols, ImageRows);
    int radius_step = 10;
    int num_radius_step = 4;
    int radius = 10;
    float alpha_dist =10;

    cal_dist_map( ImageRows, ImageCols, deep_raw, dist,
       radius, radius_step, num_radius_step );

    confidence_map_dist(ImageRows, ImageCols, dist, dist_conf, alpha_dist);

    Mat dist_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    ToMat(dist_conf, dist_mat);

    Mat Mask_dist_f_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    float** Mask_dist_f = AllocateImage<float>(ImageCols, ImageRows);
    mul(ImageRows, ImageCols, warp_deep_confidence, dist_conf, Mask_dist_f);
    ToMat(Mask_dist_f, Mask_dist_f_mat);

    imshow("ovel all",Mask_dist_f_mat);

    cv::minMaxLoc(dist_mat, &min, &max);
    cout<<" the max value " << max <<endl;
    cout<<" the min value " << min <<endl;

    imshow("saturation",im1_s);
    imshow("mask s",mask_s);
    cv::imshow("dist",dist_mat);//*255/max

    waitKey();


    // 6. set weight

    // 7. optimize


}

void value_raw(const int Rows, const int Cols, float** input){
    float min = 0;
    float max = 0;

    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j <Cols; j++) {
            //wnd_data.clear();
            // get window data around i,j
            float value_in = abs(input[i][j]);//input[i*width+j];
            //cout<< " current value "<<value_in;
            if (value_in<min){
                min = value_in;
            }
            if (value_in>max){
                max = value_in;
            }
        }
    }
//    cout<< " min value is "<< min<<endl;
//    cout<<" max value is "<< max<<endl;
}

void scale_raw(const int Rows, const int Cols, float** input,float scale){

    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j <Cols; j++) {
            input[i][j] = input[i][j]*scale;


        }
    }

}

void confidence(string left_file, string right_file, string disp_deep, string out_dir, bool visualize, bool save = false) {

    float alpha_warp = 5;
    float th_s = 0.1;
    int radius_step = 10;
    int num_radius_step = 4;
    int radius = 10;
    float alpha_dist = 10;

    // 0. read left, right images, predicted results
    cv::Mat left = cv::imread(left_file, cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_file, cv::IMREAD_GRAYSCALE);
    Mat deep_mat = cv::imread(disp_deep, IMREAD_ANYDEPTH);

    int ImageCols = left.cols; // width
    int ImageRows = left.rows; // height

    float **image1 = AllocateImage<float>(ImageCols, ImageRows); // images
    float **image2 = AllocateImage<float>(ImageCols, ImageRows);
    Mat2Raw(image1, left);
    Mat2Raw(image2, right);

    float **deep_raw = AllocateImage<float>(ImageCols, ImageRows);
    float disp_scale = -256.0;
    Mat2Raw_scale(deep_raw, deep_mat, disp_scale);

    // 1. get the valid mask
    float **valid_mask = AllocateImage<float>(ImageCols, ImageRows);
    valid_range(ImageRows, ImageCols, deep_raw, valid_mask); // 0 ~ 255

    // 2. get dist map
    // 2.1 warp
    float **warp_deep = AllocateImage<float>(ImageCols, ImageRows);
    warp_u(ImageRows, ImageCols, deep_raw, image2, warp_deep);

    // 2.2 cal confidence
    float **warp_deep_confidence = AllocateImage<float>(ImageCols, ImageRows); // 0 ~ 255
    confidence_map_warp(ImageRows, ImageCols, image1, warp_deep, warp_deep_confidence, alpha_warp);

    // 3. saturation mask
    float **mask_s = AllocateImage<float>(ImageCols, ImageRows);
    cv::Mat left_color = cv::imread(left_file);
    Mat hsv;
    cv::cvtColor(left_color, hsv, cv::COLOR_BGR2HSV);
    Mat im1_s;
    int channelIdx = 1;
    //double min, max;
    extractChannel(hsv, im1_s, channelIdx);
    cv::Mat mask_s_mat = cv::Mat(ImageCols, ImageRows, CV_8UC1);
    mask_s_mat = (im1_s >= (th_s * 255));
    Mat2Raw(mask_s, mask_s_mat);


    // 4. distance map
    float **dist = AllocateImage<float>(ImageCols, ImageRows);
    float **dist_conf = AllocateImage<float>(ImageCols, ImageRows);
    cal_dist_map(ImageRows, ImageCols, deep_raw, dist,
                 radius, radius_step, num_radius_step);
    confidence_map_dist(ImageRows, ImageCols, dist, dist_conf, alpha_dist);

    //sum mask together
    float **final_mask = AllocateImage<float>(ImageCols, ImageRows);
    mul_all(ImageRows, ImageCols, valid_mask, mask_s, warp_deep_confidence, dist_conf, final_mask);

    Mat final_mask_mat = Mat::zeros(ImageRows, ImageCols, CV_8UC1);
    ToMat(final_mask, final_mask_mat);

    Mat valid_mask_mat = Mat::zeros(ImageRows, ImageCols, CV_8UC1);
    ToMat(valid_mask, valid_mask_mat);

    //  visualize
    if (visualize) {
        Mat warp_deep_confidence_mat = Mat::zeros(ImageRows, ImageCols, CV_8UC1);
        ToMat(warp_deep_confidence, warp_deep_confidence_mat);

        Mat dist_mat = Mat::zeros(ImageRows, ImageCols, CV_8UC1);
        ToMat(dist_conf, dist_mat);

        imshow("warp confidence", warp_deep_confidence_mat);
        imshow("dist confidence", dist_mat);
        imshow("saturation", mask_s_mat);
        imshow("valid mask", valid_mask_mat);
        imshow("final", final_mask_mat);
        waitKey();
    }


    if (save) {
    cv::imwrite(out_dir, final_mask_mat);
    // string valid_mask_out_dir;
    string valid_mask_out_dir = regex_replace(out_dir, regex("mask"), "valid_mask");
    cv::imwrite(valid_mask_out_dir, valid_mask_mat);
    }
}

void save_confidence_SERV() {
    // test loss
    boost::format fmt("%03d.png");
    //int index = 1;
    //string method_name = "LEAStereo";//"LEAStereo";
    vector<string> methods{"STTR", "LEAStereo", "AAnet"};

    for (string method_name: methods) {


        for (int index = 1; index < 17; index++) {
            string name = (fmt % index).str();
            string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;
            string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/" + name;
            string disp_deep = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/" + method_name + "/" + name; //"/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/013.png";// "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/AAnet/001.png";
            string out_dir = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/" + method_name + "/mask" + name;
            confidence(left_file, right_file, disp_deep, out_dir, false);
        }
    }
}

void compare_deep_flow(){
    double error_deep =0;
    double error_flow =0;
    double error_deep_mask =0;
    double error_flow_mask =0;
    for (int index=1;index<17;index++){
        bool use_alpha = true;
        bool visual = false;
        //boost::format fmt("%03d.png");
        boost::format fmt("%03d.png");
        string name = (fmt%index).str();
        boost::format fmt1("%03d_disp.png");

        string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
        string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/"+name;//013.png";
        string disp_deep =  "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/"+name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
        string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);
        string valid_mask_dir = regex_replace(disp_deep, regex(name), "valid_mask"+name);
        string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
        string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;


        Mat left = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
        Mat right = cv::imread(right_file,cv::IMREAD_GRAYSCALE);
        Mat prior_mat = cv::imread(confidence);
        Mat deep_mat = cv::imread(disp_deep,IMREAD_ANYDEPTH);
        Mat flow = cv::imread(optical_flow,IMREAD_ANYDEPTH);
        Mat gt = cv::imread(gt_path,IMREAD_ANYDEPTH);
        Mat valid_mask = cv::imread(valid_mask_dir,IMREAD_GRAYSCALE);
        valid_mask.setTo(1,valid_mask>0);

        double scale = 256;

        double fx,fy,cx,cy,fb;

        if (index<=8){
            // 内参
            fx = 9.9640068207290187e+02, fy =  9.9640068207290187e+02, cx = 3.7502582168579102e+02, cy =  2.4026374816894531e+02;
            // 基线
            fb = 5.4301732344712009e+03;
        }
        else{
            fx = 9.3469009076912857e+02, fy =  9.3469009076912857e+02, cx = 3.2071961212158203e+02, cy =  2.7268288040161133e+02;
            fb = 5.1446987361551292e+03;
        }

//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
//    getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);
//
//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;
//    getPointCloud(left,flow,scale,cx,cy,fx,fy,fb,pointcloud_pred);
//
//    Visualize(pointcloud_gt,pointcloud_pred);

        double MAD_deep_mask = calMAD_mask(gt, deep_mat, fb, scale, valid_mask);
        double MAD_flow_mask = calMAD_mask(gt, flow, fb, scale, valid_mask);

        double MAD_deep = calMAD(gt, deep_mat, fb, scale);
        double MAD_flow = calMAD(gt, flow, fb, scale);
       // double MAD_flow_optimized = calMAD_mask(gt, flow_u_16, fb, scale, valid_mask);

        cout<<"index: "<<index<<endl;
        cout<<"MAD deep "<< MAD_deep<<endl;
        cout<<"MAD flow "<<MAD_flow<<endl;
        cout<<"MAD deep mask "<< MAD_deep_mask<<endl;
        cout<<"MAD flow mask "<<MAD_flow_mask<<endl;
        cout<<"\n"<<endl;

        error_deep = error_deep + MAD_deep;
        error_flow = error_flow + MAD_flow;

        error_deep_mask = error_deep_mask + MAD_deep_mask;
        error_flow_mask = error_flow_mask + MAD_flow_mask;

    }
    cout<<"mean deep "<< error_deep/16<<endl;
    cout<<"mean flow "<<error_flow/16<<endl;
    cout<<"mean deep mask "<< error_deep_mask/16<<endl;
    cout<<"mean flow mask "<<error_flow_mask/16<<endl;
}

void refinement(){

    int index = 16;

    bool use_alpha = true;
    bool visual = true;
    //boost::format fmt("%03d.png");
    boost::format fmt("%03d.png");
    string name = (fmt%index).str();
    boost::format fmt1("%03d_disp.png");

    string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
    string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/"+name;//013.png";
    string disp_deep =  "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/"+name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
    string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);
    string valid_mask_dir = regex_replace(disp_deep, regex(name), "valid_mask"+name);
    string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
    string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;


    Mat left = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
    Mat right = cv::imread(right_file,cv::IMREAD_GRAYSCALE);
    Mat prior_mat = cv::imread(confidence);
    Mat deep_mat = cv::imread(disp_deep,IMREAD_ANYDEPTH);
    Mat flow = cv::imread(optical_flow,IMREAD_ANYDEPTH);
    Mat gt = cv::imread(gt_path,IMREAD_ANYDEPTH);
    Mat valid_mask = cv::imread(valid_mask_dir,IMREAD_GRAYSCALE);
    valid_mask.setTo(1,valid_mask>0);

    double scale = 256;

    double fx,fy,cx,cy,fb;

    if (index<=8){
        // 内参
        fx = 9.9640068207290187e+02, fy =  9.9640068207290187e+02, cx = 3.7502582168579102e+02, cy =  2.4026374816894531e+02;
        // 基线
        fb = 5.4301732344712009e+03;
    }
    else{
        fx = 9.3469009076912857e+02, fy =  9.3469009076912857e+02, cx = 3.2071961212158203e+02, cy =  2.7268288040161133e+02;
        fb = 5.1446987361551292e+03;
    }

//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
//    getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);
//
//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;
//    getPointCloud(left,flow,scale,cx,cy,fx,fy,fb,pointcloud_pred);
//
//    Visualize(pointcloud_gt,pointcloud_pred);

    int ImageCols = left.cols;
    int ImageRows = left.rows;

    float** image1 = AllocateImage<float>(ImageCols, ImageRows);
    float** image2 = AllocateImage<float>(ImageCols, ImageRows);

    float** u = AllocateImage<float>(ImageCols, ImageRows);
    float** v = AllocateImage<float>(ImageCols, ImageRows);
    float** alpha = AllocateImage<float>(ImageCols, ImageRows);

    if(use_alpha){
        //prior_mat.setTo(0,prior_mat<100);
//        imshow("prior",prior_mat);
//        waitKey();
        float scale_alpha = 0.00005;//0.05 ; //0.1,0.05
        Mat2Raw_scale_8uc1(alpha,prior_mat,255);//2000,1000
        scale_raw(ImageRows, ImageCols, alpha, scale_alpha);
        value_raw(ImageRows, ImageCols, alpha);
    }else{
        cout<<" set weight to zero, only use prior to initialize"<<endl;
        memset(alpha[0], 0, ImageCols * ImageRows * sizeof(float));
    }

    //cout<< &alpha[155][155]<<endl;
    //Mat alpha_mat = Mat::ones(ImageRows,ImageCols,CV_8UC1);

    //float** flow_raw = AllocateImage<float>(ImageCols, ImageRows); //u
    float** deep_raw =  AllocateImage<float>(ImageCols, ImageRows); //u
    float** v_raw =  AllocateImage<float>(ImageCols, ImageRows); //v

    Mat v_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    memset(v[0], 0, ImageCols * ImageRows * sizeof(float));

    Mat2Raw(image1,left);
    Mat2Raw(image2,right);
    //Mat2Raw(v_raw,v_mat);
    //Mat2Raw(alpha,alpha_mat);

    float disp_scale = -256.0;
    //Mat2Raw_scale(flow_raw,flow_mat, disp_scale);
    Mat2Raw_scale(deep_raw,deep_mat, disp_scale);
    //cout<<deep_raw[10][10]<<endl;

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

    OpticalFlow_Huber_L2::Data data{};
    data.I1 = image1;
    data.I2 = image2;
    data.u = u;
    data.v = v;
    data.nCols = ImageCols;
    data.nRows = ImageRows;

    data.alpha = alpha;// NULL; //alpha;//alpha;//alpha;//NULL;
    data.uprior = deep_raw;//NULL;deep_raw;
    data.vprior = v_raw;//NULL; v_raw;

    OpticalFlow_Huber_L2 of;
    of.setup(data, inputParms);
    of.run();

    Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    ToMat(u, flow_u);

    float** image1_warp;
    image1_warp = of.m_I2w;
    Mat image1_warp_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    ToMat(image1_warp, image1_warp_mat);

    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
    //getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);

    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;

    Mat flow_u_16= Mat::zeros(ImageRows,ImageCols,CV_16UC1);
    Mat flow_u_mask;
    flow_u_mask = flow_u.mul(valid_mask);

  //  waitKey();
    flow_u_mask.convertTo(flow_u_16,CV_16UC1, scale);

    double min, max;
    cv::minMaxLoc(deep_mat, &min, &max);
    cout<<"disparity gt the min "<< min/256<<endl;
    cout<<"disparity gt the max "<< max/256<<endl;

    cv::minMaxLoc(prior_mat, &min, &max);
    cout<<"prior the min "<< min<<endl;
    cout<<"prior the max "<< max<<endl;

    cv::minMaxLoc(flow_u, &min, &max);
    cout<<"disparity pred the min "<< min<<endl;
    cout<<"disparity pred the max "<< max<<endl;

    cv::minMaxLoc(flow_u_16, &min, &max);
    cout<<"16UC1 the min "<< min/256<<endl;
    cout<<"16UC1 the max "<< max/256<<endl;

    cv::minMaxLoc(valid_mask, &min, &max);
    cout<<"disparity gt the min "<< min<<endl;
    cout<<"disparity gt the max "<< max<<endl;


    double MAD_deep = calMAD_mask(gt, deep_mat, fb, scale, valid_mask);
    double MAD_flow = calMAD_mask(gt, flow, fb, scale, valid_mask);
    double MAD_flow_optimized = calMAD_mask(gt, flow_u_16, fb, scale, valid_mask);

    imshow("flow_u_16",flow_u_16);
    waitKey();

    cout<<"MAD deep "<< MAD_deep<<endl;
    cout<<"MAD flow "<<MAD_flow<<endl;
    cout<<"MAD flow optimized "<<MAD_flow_optimized<<endl;

    if(visual) {

        cv::imshow("prior", prior_mat);
        cv::imshow("deep", deep_mat);
        cv::imshow(" warped", image1_warp_mat);
        cv::imshow("left", left);
        cv::imshow("flow", flow);
        imshow("mask flow", flow_u_mask);
//    Mat diff = cv::abs(flow-flow_u);
//    cv::imshow("diff",diff);
        cv::imshow(" optimized u", flow_u);
        cv::imshow("optimized u mask", flow_u_mask);
        cv::waitKey();

        getPointCloud(left, deep_mat, scale, cx, cy, fx, fy, fb, pointcloud_gt);
        //getPointCloud(left, flow_u_16, scale, cx, cy, fx, fy, fb, pointcloud_pred);
        getPointCloud(left, flow, scale, cx, cy, fx, fy, fb, pointcloud_pred);
        Visualize(pointcloud_gt, pointcloud_pred);
    }

    cout<<"\n"<<endl;


    //getPointCloud(left,flow_u_16, scale, cx, cy, fx, fy, fb,pointcloud_gt);
//    getPointCloud(left,flow_u_16,1,cx,cy,fx,fy,fb,pointcloud_gt);
//    getPointCloud(left,deep_mat, scale,cx,cy,fx,fy,fb,pointcloud_pred);
//    Visualize(pointcloud_gt,pointcloud_pred);


}

void refinement_all(float scale_alpha){
    double error_deep =0;
    double error_flow =0;
    double error_deep_mask =0;
    double error_flow_mask =0;
    double error_op =0;
    double error_op_mask =0;

    bool use_alpha = true;
    bool visual = true;
    bool show_every = false;
    bool alpha_weight = true;
    float lambda = 15;

    //int index = 10;
    for (int index=1;index<17;index++){

    //boost::format fmt("%03d.png");
    boost::format fmt("%03d.png");
    string name = (fmt%index).str();
    boost::format fmt1("%03d_disp.png");

    string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
    string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/"+name;//013.png";
    string disp_deep =  "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/"+name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
    string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);
    string valid_mask_dir = regex_replace(disp_deep, regex(name), "valid_mask"+name);
    string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
    string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;


    Mat left = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
    Mat right = cv::imread(right_file,cv::IMREAD_GRAYSCALE);
    Mat prior_mat = cv::imread(confidence);
    Mat deep_mat = cv::imread(disp_deep,IMREAD_ANYDEPTH);
    Mat flow = cv::imread(optical_flow,IMREAD_ANYDEPTH);
    Mat gt = cv::imread(gt_path,IMREAD_ANYDEPTH);
    Mat valid_mask = cv::imread(valid_mask_dir,IMREAD_GRAYSCALE);
    valid_mask.setTo(1,valid_mask>0);

    double scale = 256;

    double fx,fy,cx,cy,fb;

    if (index<=8){
        // 内参
        fx = 9.9640068207290187e+02, fy =  9.9640068207290187e+02, cx = 3.7502582168579102e+02, cy =  2.4026374816894531e+02;
        // 基线
        fb = 5.4301732344712009e+03;
    }
    else{
        fx = 9.3469009076912857e+02, fy =  9.3469009076912857e+02, cx = 3.2071961212158203e+02, cy =  2.7268288040161133e+02;
        fb = 5.1446987361551292e+03;
    }

//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
//    getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);
//
//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;
//    getPointCloud(left,flow,scale,cx,cy,fx,fy,fb,pointcloud_pred);
//
//    Visualize(pointcloud_gt,pointcloud_pred);

    int ImageCols = left.cols;
    int ImageRows = left.rows;

    float** image1 = AllocateImage<float>(ImageCols, ImageRows);
    float** image2 = AllocateImage<float>(ImageCols, ImageRows);

    float** u = AllocateImage<float>(ImageCols, ImageRows);
    float** v = AllocateImage<float>(ImageCols, ImageRows);
    float** alpha = AllocateImage<float>(ImageCols, ImageRows);

    if(use_alpha){
        if(alpha_weight){
             Mat2Raw_scale_8uc1(alpha,prior_mat,255);//2000,1000
             scale_raw(ImageRows, ImageCols, alpha, scale_alpha);
        }else{
            Mat alpha_mat = Mat::ones(ImageRows,ImageCols,CV_8UC1);
            alpha_mat.setTo(scale_alpha,alpha_mat>0);
            Mat2Raw(alpha,alpha_mat);
        }
//        prior_mat.setTo(0,prior_mat<100);
//        prior_mat.setTo(255, prior_mat>0);
//        imshow("prior",prior_mat);
//        waitKey();
//   float scale_alpha = 0.00005;//0.05 ; //0.1,0.05
// value_raw(ImageRows, ImageCols, alpha);
    }else{
        cout<<" set weight to zero, only use prior to initialize"<<endl;
        memset(alpha[0], 0, ImageCols * ImageRows * sizeof(float));
    }

    //cout<< &alpha[155][155]<<endl;


    //float** flow_raw = AllocateImage<float>(ImageCols, ImageRows); //u
    float** deep_raw =  AllocateImage<float>(ImageCols, ImageRows); //u
    float** v_raw =  AllocateImage<float>(ImageCols, ImageRows); //v

    Mat v_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    memset(v[0], 0, ImageCols * ImageRows * sizeof(float));

    Mat2Raw(image1,left);
    Mat2Raw(image2,right);
    //Mat2Raw(v_raw,v_mat);


    float disp_scale = -256.0;
    //Mat2Raw_scale(flow_raw,flow_mat, disp_scale);
    Mat2Raw_scale(deep_raw,deep_mat, disp_scale);
    //cout<<deep_raw[10][10]<<endl;

    // set parameters
    OpticalFlow_Huber_L2::Parms inputParms{};
    inputParms.epsilon = 0.001f;
    inputParms.iterations = 50;//50;
    inputParms.Lambda = lambda;//100.0f;
    //patch = 120  D1 = 15.0f;
    inputParms.levels = 6;
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 5;
    inputParms.descriptor = OpticalFlow_Huber_L2::D1;//PATCH_INTENSITY;

    OpticalFlow_Huber_L2::Data data{};
    data.I1 = image1;
    data.I2 = image2;
    data.u = u;
    data.v = v;
    data.nCols = ImageCols;
    data.nRows = ImageRows;

    data.alpha = alpha;// NULL; //alpha;//alpha;//alpha;//NULL;
    data.uprior = deep_raw;//NULL;deep_raw;
    data.vprior = v_raw;//NULL; v_raw;

    OpticalFlow_Huber_L2 of;
    of.setup(data, inputParms);
    of.run();

    Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    ToMat(u, flow_u);

    float** image1_warp;
    image1_warp = of.m_I2w;
    Mat image1_warp_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
    ToMat(image1_warp, image1_warp_mat);

    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
    //getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);

    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;

    Mat flow_u_16= Mat::zeros(ImageRows,ImageCols,CV_16UC1);
    Mat flow_u_mask;
    flow_u_mask = flow_u.mul(valid_mask);

    //  waitKey();
    flow_u_mask.convertTo(flow_u_16,CV_16UC1, scale);

    double MAD_deep_mask = calMAD_mask(gt, deep_mat, fb, scale, valid_mask);
    double MAD_flow_mask = calMAD_mask(gt, flow, fb, scale, valid_mask);
    double MAD_deep = calMAD(gt, deep_mat, fb, scale);
    double MAD_flow = calMAD(gt, flow, fb, scale);
    double MAD_flow_optimized = calMAD(gt, flow_u_16, fb, scale);
    double MAD_flow_optimized_mask = calMAD_mask(gt, flow_u_16, fb, scale, valid_mask);

    if(show_every){
        cout<<"index: "<<index<<endl;
        cout<<"MAD deep "<< MAD_deep<<endl;
        cout<<"MAD flow "<<MAD_flow<<endl;
        cout<<"MAD deep mask "<< MAD_deep_mask<<endl;
        cout<<"MAD flow mask "<<MAD_flow_mask<<endl;
        cout<<"Optimized MAD  "<< MAD_flow_optimized<<endl;
        cout<<"Optimized MAD mask "<<MAD_flow_optimized_mask<<endl;
        cout<<"\n"<<endl;
    }

    error_deep = error_deep + MAD_deep;
    error_flow = error_flow + MAD_flow;
    error_deep_mask = error_deep_mask + MAD_deep_mask;
    error_flow_mask = error_flow_mask + MAD_flow_mask;
    error_op = error_op + MAD_flow_optimized;
    error_op_mask = error_op_mask + MAD_flow_optimized_mask;

    }
    cout<<"mean deep "<< error_deep/16<<endl;
    cout<<"mean flow "<<error_flow/16<<endl;
    cout<<"mean op "<< error_op/16<<endl;
    cout<<"mean deep mask "<< error_deep_mask/16<<endl;
    cout<<"mean flow mask "<<error_flow_mask/16<<endl;
    cout<<"mean op mask "<<error_op_mask/16<<endl;

}

void select_lambda(float lambda){
    double error_deep =0;
    double error_flow =0;
    double error_deep_mask =0;
    double error_flow_mask =0;
    double error_op =0;
    double error_op_mask =0;
    bool use_alpha = false;
    bool visual = true;
    bool show_every = false;

    //int index = 10;
    for (int index=1;index<17;index++){

        //boost::format fmt("%03d.png");
        boost::format fmt("%03d.png");
        string name = (fmt%index).str();
        boost::format fmt1("%03d_disp.png");

        string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
        string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/"+name;//013.png";
        string disp_deep =  "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/"+name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
        string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);
        string valid_mask_dir = regex_replace(disp_deep, regex(name), "valid_mask"+name);
        string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
        string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;


        Mat left = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
        Mat right = cv::imread(right_file,cv::IMREAD_GRAYSCALE);
        Mat prior_mat = cv::imread(confidence);
        Mat deep_mat = cv::imread(disp_deep,IMREAD_ANYDEPTH);
        Mat flow = cv::imread(optical_flow,IMREAD_ANYDEPTH);
        Mat gt = cv::imread(gt_path,IMREAD_ANYDEPTH);
        Mat valid_mask = cv::imread(valid_mask_dir,IMREAD_GRAYSCALE);
        valid_mask.setTo(1,valid_mask>0);

        double scale = 256;

        double fx,fy,cx,cy,fb;

        if (index<=8){
            // 内参
            fx = 9.9640068207290187e+02, fy =  9.9640068207290187e+02, cx = 3.7502582168579102e+02, cy =  2.4026374816894531e+02;
            // 基线
            fb = 5.4301732344712009e+03;
        }
        else{
            fx = 9.3469009076912857e+02, fy =  9.3469009076912857e+02, cx = 3.2071961212158203e+02, cy =  2.7268288040161133e+02;
            fb = 5.1446987361551292e+03;
        }

//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
//    getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);
//
//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;
//    getPointCloud(left,flow,scale,cx,cy,fx,fy,fb,pointcloud_pred);
//
//    Visualize(pointcloud_gt,pointcloud_pred);

        int ImageCols = left.cols;
        int ImageRows = left.rows;

        float** image1 = AllocateImage<float>(ImageCols, ImageRows);
        float** image2 = AllocateImage<float>(ImageCols, ImageRows);

        float** u = AllocateImage<float>(ImageCols, ImageRows);
        float** v = AllocateImage<float>(ImageCols, ImageRows);
        float** alpha = AllocateImage<float>(ImageCols, ImageRows);



        //cout<< &alpha[155][155]<<endl;


        //float** flow_raw = AllocateImage<float>(ImageCols, ImageRows); //u
        float** deep_raw =  AllocateImage<float>(ImageCols, ImageRows); //u
        float** v_raw =  AllocateImage<float>(ImageCols, ImageRows); //v

        Mat v_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        memset(v[0], 0, ImageCols * ImageRows * sizeof(float));

        Mat2Raw(image1,left);
        Mat2Raw(image2,right);
        //Mat2Raw(v_raw,v_mat);


        float disp_scale = -256.0;
        //Mat2Raw_scale(flow_raw,flow_mat, disp_scale);
        Mat2Raw_scale(deep_raw,deep_mat, disp_scale);
        //cout<<deep_raw[10][10]<<endl;

        // set parameters
        OpticalFlow_Huber_L2::Parms inputParms{};
        inputParms.epsilon = 0.001f;
        inputParms.iterations = 50;//50;
        inputParms.Lambda = lambda;//15.0f;//100.0f;
        //patch = 120  D1 = 15.0f;
        inputParms.levels = 6;
        inputParms.minSize = 48;
        inputParms.scalefactor = 0.5f;
        inputParms.warps = 5;
        inputParms.descriptor = OpticalFlow_Huber_L2::D1;//PATCH_INTENSITY;

        OpticalFlow_Huber_L2::Data data{};
        data.I1 = image1;
        data.I2 = image2;
        data.u = u;
        data.v = v;
        data.nCols = ImageCols;
        data.nRows = ImageRows;

        data.alpha = NULL;// NULL; //alpha;//alpha;//alpha;//NULL;
        data.uprior = NULL;//deep_raw;//NULL;deep_raw;
        data.vprior = NULL;//v_raw;//NULL; v_raw;

        OpticalFlow_Huber_L2 of;
        of.setup(data, inputParms);
        of.run();

        Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(u, flow_u);

        float** image1_warp;
        image1_warp = of.m_I2w;
        Mat image1_warp_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(image1_warp, image1_warp_mat);

        vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
        //getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);

        vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;

        Mat flow_u_16= Mat::zeros(ImageRows,ImageCols,CV_16UC1);
//        Mat flow_u_mask;
//        flow_u_mask = flow_u.mul(valid_mask);

        //  waitKey();
        flow_u.convertTo(flow_u_16,CV_16UC1, scale);

        double MAD_deep_mask = calMAD_mask(gt, deep_mat, fb, scale, valid_mask);
        double MAD_flow_mask = calMAD_mask(gt, flow, fb, scale, valid_mask);
        double MAD_deep = calMAD(gt, deep_mat, fb, scale);
        double MAD_flow = calMAD(gt, flow, fb, scale);
        double MAD_flow_optimized = calMAD(gt, flow_u_16, fb, scale);
        double MAD_flow_optimized_mask = calMAD_mask(gt, flow_u_16, fb, scale, valid_mask);

        if(show_every){
            cout<<"index: "<<index<<endl;
            cout<<"MAD deep "<< MAD_deep<<endl;
            cout<<"MAD flow "<<MAD_flow<<endl;
            cout<<"MAD deep mask "<< MAD_deep_mask<<endl;
            cout<<"MAD flow mask "<<MAD_flow_mask<<endl;
            cout<<"Optimized MAD  "<< MAD_flow_optimized<<endl;
            cout<<"Optimized MAD mask "<<MAD_flow_optimized_mask<<endl;
            cout<<"\n"<<endl;
        }

        error_deep = error_deep + MAD_deep;
        error_flow = error_flow + MAD_flow;
        error_deep_mask = error_deep_mask + MAD_deep_mask;
        error_flow_mask = error_flow_mask + MAD_flow_mask;
        error_op = error_op + MAD_flow_optimized;
        error_op_mask = error_op_mask + MAD_flow_optimized_mask;

    }
    cout<<"mean deep "<< error_deep/16<<endl;
    cout<<"mean flow "<<error_flow/16<<endl;
    cout<<"mean op "<< error_op/16<<endl;
    cout<<"mean deep mask "<< error_deep_mask/16<<endl;
    cout<<"mean flow mask "<<error_flow_mask/16<<endl;
    cout<<"mean op mask "<<error_op_mask/16<<endl;

}

void try_initialization(){
    double error_deep =0;
    double error_flow =0;
    double error_deep_mask =0;
    double error_flow_mask =0;
    double error_op =0;
    double error_op_mask =0;
    bool use_alpha = true;
    bool visual = true;
    bool show_every = true;
    float lambda = 12;
    bool bInitializeWithPrior= true;

    //int index = 10;
    for (int index=1;index<17;index++){

        //boost::format fmt("%03d.png");
        boost::format fmt("%03d.png");
        string name = (fmt%index).str();
        boost::format fmt1("%03d_disp.png");

        string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
        string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/"+name;//013.png";
        string disp_deep =  "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/"+name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
        string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);
        string valid_mask_dir = regex_replace(disp_deep, regex(name), "valid_mask"+name);
        string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
        string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;


        Mat left = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
        Mat right = cv::imread(right_file,cv::IMREAD_GRAYSCALE);
        Mat prior_mat = cv::imread(confidence);
        Mat deep_mat = cv::imread(disp_deep,IMREAD_ANYDEPTH);
        Mat flow = cv::imread(optical_flow,IMREAD_ANYDEPTH);
        Mat gt = cv::imread(gt_path,IMREAD_ANYDEPTH);
        Mat valid_mask = cv::imread(valid_mask_dir,IMREAD_GRAYSCALE);
        valid_mask.setTo(1,valid_mask>0);

        double scale = 256;

        double fx,fy,cx,cy,fb;

        if (index<=8){
            // 内参
            fx = 9.9640068207290187e+02, fy =  9.9640068207290187e+02, cx = 3.7502582168579102e+02, cy =  2.4026374816894531e+02;
            // 基线
            fb = 5.4301732344712009e+03;
        }
        else{
            fx = 9.3469009076912857e+02, fy =  9.3469009076912857e+02, cx = 3.2071961212158203e+02, cy =  2.7268288040161133e+02;
            fb = 5.1446987361551292e+03;
        }

//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
//    getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);
//
//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;
//    getPointCloud(left,flow,scale,cx,cy,fx,fy,fb,pointcloud_pred);
//
//    Visualize(pointcloud_gt,pointcloud_pred);

        int ImageCols = left.cols;
        int ImageRows = left.rows;

        float** image1 = AllocateImage<float>(ImageCols, ImageRows);
        float** image2 = AllocateImage<float>(ImageCols, ImageRows);

        float** u = AllocateImage<float>(ImageCols, ImageRows);
        float** v = AllocateImage<float>(ImageCols, ImageRows);
        float** alpha = AllocateImage<float>(ImageCols, ImageRows);

        //cout<< &alpha[155][155]<<endl;
        float** flow_raw = AllocateImage<float>(ImageCols, ImageRows); //u
        float** deep_raw =  AllocateImage<float>(ImageCols, ImageRows); //u
        float** v_raw =  AllocateImage<float>(ImageCols, ImageRows); //v

        Mat v_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        memset(v[0], 0, ImageCols * ImageRows * sizeof(float));

        Mat2Raw(image1,left);
        Mat2Raw(image2,right);
        //Mat2Raw(v_raw,v_mat);

        float disp_scale = -256.0;
        Mat2Raw_scale(flow_raw,flow, disp_scale);
        Mat2Raw_scale(deep_raw,deep_mat, disp_scale);
        //cout<<deep_raw[10][10]<<endl;

        // set parameters
        OpticalFlow_Huber_L2::Parms inputParms{};
        inputParms.epsilon = 0.001f;
        inputParms.iterations = 50;//50;
        inputParms.Lambda = lambda;//15.0f;//100.0f;
        //patch = 120  D1 = 15.0f;
        inputParms.levels = 6;//6; //6
        inputParms.minSize = 48;
        inputParms.scalefactor = 0.5f;
        inputParms.warps = 5;
        inputParms.descriptor = OpticalFlow_Huber_L2::D1;//PATCH_INTENSITY;
        inputParms.bInitializeWithPrior = true;//true;//true;

        OpticalFlow_Huber_L2::Data data{};
        data.I1 = image1;
        data.I2 = image2;
        data.u = u;
        data.v = v;
        data.nCols = ImageCols;
        data.nRows = ImageRows;

        data.alpha = alpha;// NULL; //alpha;//alpha;//alpha;//NULL;
        data.uprior = deep_raw;//deep_raw;//NULL;deep_raw;
        data.vprior = v_raw;//NULL; v_raw;

        OpticalFlow_Huber_L2 of;
        of.setup(data, inputParms);
        of.run();

        Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(u, flow_u);

        float** image1_warp;
        image1_warp = of.m_I2w;
        Mat image1_warp_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(image1_warp, image1_warp_mat);

        vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
        //getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);

        vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;

        Mat flow_u_16= Mat::zeros(ImageRows,ImageCols,CV_16UC1);
//        Mat flow_u_mask;
//        flow_u_mask = flow_u.mul(valid_mask);

        //  waitKey();
        flow_u.convertTo(flow_u_16,CV_16UC1, scale);

        double MAD_deep_mask = calMAD_mask(gt, deep_mat, fb, scale, valid_mask);
        double MAD_flow_mask = calMAD_mask(gt, flow, fb, scale, valid_mask);
        double MAD_deep = calMAD(gt, deep_mat, fb, scale);
        double MAD_flow = calMAD(gt, flow, fb, scale);
        double MAD_flow_optimized = calMAD(gt, flow_u_16, fb, scale);
        double MAD_flow_optimized_mask = calMAD_mask(gt, flow_u_16, fb, scale, valid_mask);

        if(show_every){
            cout<<"index: "<<index<<endl;
            cout<<"MAD deep "<< MAD_deep<<endl;
            cout<<"MAD flow "<<MAD_flow<<endl;
            cout<<"MAD deep mask "<< MAD_deep_mask<<endl;
            cout<<"MAD flow mask "<<MAD_flow_mask<<endl;
            cout<<"Optimized MAD  "<< MAD_flow_optimized<<endl;
            cout<<"Optimized MAD mask "<<MAD_flow_optimized_mask<<endl;
            cout<<"\n"<<endl;
        }

        error_deep = error_deep + MAD_deep;
        error_flow = error_flow + MAD_flow;
        error_deep_mask = error_deep_mask + MAD_deep_mask;
        error_flow_mask = error_flow_mask + MAD_flow_mask;
        error_op = error_op + MAD_flow_optimized;
        error_op_mask = error_op_mask + MAD_flow_optimized_mask;

    }
    cout<<"mean deep "<< error_deep/16<<endl;
    cout<<"mean flow "<<error_flow/16<<endl;
    cout<<"mean op "<< error_op/16<<endl;
    cout<<"mean deep mask "<< error_deep_mask/16<<endl;
    cout<<"mean flow mask "<<error_flow_mask/16<<endl;
    cout<<"mean op mask "<<error_op_mask/16<<endl;

}

void try_initialization_ref(){
    double error_deep =0;
    double error_flow =0;
    double error_deep_mask =0;
    double error_flow_mask =0;
    double error_op =0;
    double error_op_mask =0;
    bool use_alpha = true;
    bool visual = true;
    bool show_every = true;
    float lambda = 12;
    bool bInitializeWithPrior= true;

    //int index = 10;
    for (int index=1;index<17;index++){

        //boost::format fmt("%03d.png");
        boost::format fmt("%03d.png");
        string name = (fmt%index).str();
        boost::format fmt1("%03d_disp.png");

        string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
        string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/"+name;//013.png";
        string disp_deep =  "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/STTR/"+name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
        string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);
        string valid_mask_dir = regex_replace(disp_deep, regex(name), "valid_mask"+name);
        string ref_disp_deep = regex_replace(disp_deep, regex(name), "refine_"+name);
        string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
        string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;


        Mat left = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
        Mat right = cv::imread(right_file,cv::IMREAD_GRAYSCALE);
        Mat prior_mat = cv::imread(confidence);
        Mat deep_mat = cv::imread(ref_disp_deep,IMREAD_ANYDEPTH);
        Mat flow = cv::imread(optical_flow,IMREAD_ANYDEPTH);
        Mat gt = cv::imread(gt_path,IMREAD_ANYDEPTH);
        Mat valid_mask = cv::imread(valid_mask_dir,IMREAD_GRAYSCALE);
        valid_mask.setTo(1,valid_mask>0);

//        double min, max;
//        cv::minMaxLoc(deep_mat, &min, &max);
//        cout<<"disparity deep the min "<< min/256<<endl;
//        cout<<"disparity deep the max "<< max/256<<endl;

        double scale = 256;

        double fx,fy,cx,cy,fb;

        if (index<=8){
            // 内参
            fx = 9.9640068207290187e+02, fy =  9.9640068207290187e+02, cx = 3.7502582168579102e+02, cy =  2.4026374816894531e+02;
            // 基线
            fb = 5.4301732344712009e+03;
        }
        else{
            fx = 9.3469009076912857e+02, fy =  9.3469009076912857e+02, cx = 3.2071961212158203e+02, cy =  2.7268288040161133e+02;
            fb = 5.1446987361551292e+03;
        }

//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
//    getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);
//
//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;
//    getPointCloud(left,flow,scale,cx,cy,fx,fy,fb,pointcloud_pred);
//
//    Visualize(pointcloud_gt,pointcloud_pred);

        int ImageCols = left.cols;
        int ImageRows = left.rows;

        float** image1 = AllocateImage<float>(ImageCols, ImageRows);
        float** image2 = AllocateImage<float>(ImageCols, ImageRows);

        float** u = AllocateImage<float>(ImageCols, ImageRows);
        float** v = AllocateImage<float>(ImageCols, ImageRows);
        float** alpha = AllocateImage<float>(ImageCols, ImageRows);

        //cout<< &alpha[155][155]<<endl;
        float** flow_raw = AllocateImage<float>(ImageCols, ImageRows); //u
        float** deep_raw =  AllocateImage<float>(ImageCols, ImageRows); //u
        float** v_raw =  AllocateImage<float>(ImageCols, ImageRows); //v

        Mat v_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        memset(v[0], 0, ImageCols * ImageRows * sizeof(float));

        Mat2Raw(image1,left);
        Mat2Raw(image2,right);
        //Mat2Raw(v_raw,v_mat);

        float disp_scale = -256.0;
        Mat2Raw_scale(flow_raw,flow, disp_scale);
        Mat2Raw_scale(deep_raw,deep_mat, disp_scale);
        //cout<<deep_raw[10][10]<<endl;

        // set parameters
        OpticalFlow_Huber_L2::Parms inputParms{};
        inputParms.epsilon = 0.001f;
        inputParms.iterations = 50;//50;
        inputParms.Lambda = lambda;//15.0f;//100.0f;
        //patch = 120  D1 = 15.0f;
        inputParms.levels = 6;//6; //6
        inputParms.minSize = 48;
        inputParms.scalefactor = 0.5f;
        inputParms.warps = 5;
        inputParms.descriptor = OpticalFlow_Huber_L2::D1;//PATCH_INTENSITY;
        inputParms.bInitializeWithPrior = true;//true;//true;

        OpticalFlow_Huber_L2::Data data{};
        data.I1 = image1;
        data.I2 = image2;
        data.u = u;
        data.v = v;
        data.nCols = ImageCols;
        data.nRows = ImageRows;

        data.alpha = alpha;// NULL; //alpha;//alpha;//alpha;//NULL;
        data.uprior = deep_raw;//deep_raw;//NULL;deep_raw;
        data.vprior = v_raw;//NULL; v_raw;

        OpticalFlow_Huber_L2 of;
        of.setup(data, inputParms);
        of.run();

        Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(u, flow_u);

        float** image1_warp;
        image1_warp = of.m_I2w;
        Mat image1_warp_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(image1_warp, image1_warp_mat);

        vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
        //getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);

        vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;

        Mat flow_u_16= Mat::zeros(ImageRows,ImageCols,CV_16UC1);
//        Mat flow_u_mask;
//        flow_u_mask = flow_u.mul(valid_mask);

        //  waitKey();
        flow_u.convertTo(flow_u_16,CV_16UC1, scale);

        double MAD_deep_mask = calMAD_mask(gt, deep_mat, fb, scale, valid_mask);
        double MAD_flow_mask = calMAD_mask(gt, flow, fb, scale, valid_mask);
        double MAD_deep = calMAD(gt, deep_mat, fb, scale);
        double MAD_flow = calMAD(gt, flow, fb, scale);
        double MAD_flow_optimized = calMAD(gt, flow_u_16, fb, scale);
        double MAD_flow_optimized_mask = calMAD_mask(gt, flow_u_16, fb, scale, valid_mask);

        if(show_every){
            cout<<"index: "<<index<<endl;
            cout<<"MAD deep "<< MAD_deep<<endl;
            cout<<"MAD flow "<<MAD_flow<<endl;
            cout<<"MAD deep mask "<< MAD_deep_mask<<endl;
            cout<<"MAD flow mask "<<MAD_flow_mask<<endl;
            cout<<"Optimized MAD  "<< MAD_flow_optimized<<endl;
            cout<<"Optimized MAD mask "<<MAD_flow_optimized_mask<<endl;
            cout<<"\n"<<endl;
        }

        error_deep = error_deep + MAD_deep;
        error_flow = error_flow + MAD_flow;
        error_deep_mask = error_deep_mask + MAD_deep_mask;
        error_flow_mask = error_flow_mask + MAD_flow_mask;
        error_op = error_op + MAD_flow_optimized;
        error_op_mask = error_op_mask + MAD_flow_optimized_mask;

    }
    cout<<"mean deep "<< error_deep/16<<endl;
    cout<<"mean flow "<<error_flow/16<<endl;
    cout<<"mean op "<< error_op/16<<endl;
    cout<<"mean deep mask "<< error_deep_mask/16<<endl;
    cout<<"mean flow mask "<<error_flow_mask/16<<endl;
    cout<<"mean op mask "<<error_op_mask/16<<endl;

}

void initialize_level(int level){
    double error_deep =0;
    double error_flow =0;
    double error_deep_mask =0;
    double error_flow_mask =0;
    double error_op =0;
    double error_op_mask =0;
    bool use_alpha = false;
    bool visual = true;
    bool show_every = false;
    float lambda = 12;
    bool bInitializeWithPrior= true;

    //int index = 10;
    for (int index=1;index<17;index++){

        //boost::format fmt("%03d.png");
        boost::format fmt("%03d.png");
        string name = (fmt%index).str();
        boost::format fmt1("%03d_disp.png");

        string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
        string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/"+name;//013.png";
        string disp_deep =  "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/"+name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
        string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);
        string valid_mask_dir = regex_replace(disp_deep, regex(name), "valid_mask"+name);
        string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
        string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;


        Mat left = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
        Mat right = cv::imread(right_file,cv::IMREAD_GRAYSCALE);
        Mat prior_mat = cv::imread(confidence);
        Mat deep_mat = cv::imread(disp_deep,IMREAD_ANYDEPTH);
        Mat flow = cv::imread(optical_flow,IMREAD_ANYDEPTH);
        Mat gt = cv::imread(gt_path,IMREAD_ANYDEPTH);
        Mat valid_mask = cv::imread(valid_mask_dir,IMREAD_GRAYSCALE);
        valid_mask.setTo(1,valid_mask>0);

        double scale = 256;

        double fx,fy,cx,cy,fb;

        if (index<=8){
            // 内参
            fx = 9.9640068207290187e+02, fy =  9.9640068207290187e+02, cx = 3.7502582168579102e+02, cy =  2.4026374816894531e+02;
            // 基线
            fb = 5.4301732344712009e+03;
        }
        else{
            fx = 9.3469009076912857e+02, fy =  9.3469009076912857e+02, cx = 3.2071961212158203e+02, cy =  2.7268288040161133e+02;
            fb = 5.1446987361551292e+03;
        }

//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
//    getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);
//
//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;
//    getPointCloud(left,flow,scale,cx,cy,fx,fy,fb,pointcloud_pred);
//
//    Visualize(pointcloud_gt,pointcloud_pred);

        int ImageCols = left.cols;
        int ImageRows = left.rows;

        float** image1 = AllocateImage<float>(ImageCols, ImageRows);
        float** image2 = AllocateImage<float>(ImageCols, ImageRows);

        float** u = AllocateImage<float>(ImageCols, ImageRows);
        float** v = AllocateImage<float>(ImageCols, ImageRows);
        float** alpha = AllocateImage<float>(ImageCols, ImageRows);

        //cout<< &alpha[155][155]<<endl;
        float** flow_raw = AllocateImage<float>(ImageCols, ImageRows); //u
        float** deep_raw =  AllocateImage<float>(ImageCols, ImageRows); //u
        float** v_raw =  AllocateImage<float>(ImageCols, ImageRows); //v

        Mat v_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        memset(v[0], 0, ImageCols * ImageRows * sizeof(float));

        Mat2Raw(image1,left);
        Mat2Raw(image2,right);
        //Mat2Raw(v_raw,v_mat);

        float disp_scale = -256.0;
        Mat2Raw_scale(flow_raw,flow, disp_scale);
        Mat2Raw_scale(deep_raw,deep_mat, disp_scale);
        //cout<<deep_raw[10][10]<<endl;

        // set parameters
        OpticalFlow_Huber_L2::Parms inputParms{};
        inputParms.epsilon = 0.001f;
        inputParms.iterations = 50;//50;
        inputParms.Lambda = lambda;//15.0f;//100.0f;
        //patch = 120  D1 = 15.0f;
        inputParms.levels = level;//6;//6; //6
        inputParms.minSize = 48;
        inputParms.scalefactor = 0.5f;
        inputParms.warps = 5;
        inputParms.descriptor = OpticalFlow_Huber_L2::D1;//PATCH_INTENSITY;
        inputParms.bInitializeWithPrior = true;//true;//true;

        OpticalFlow_Huber_L2::Data data{};
        data.I1 = image1;
        data.I2 = image2;
        data.u = u;
        data.v = v;
        data.nCols = ImageCols;
        data.nRows = ImageRows;

        data.alpha = alpha;// NULL; //alpha;//alpha;//alpha;//NULL;
        data.uprior = deep_raw;//deep_raw;//NULL;deep_raw;
        data.vprior = v_raw;//NULL; v_raw;

        OpticalFlow_Huber_L2 of;
        of.setup(data, inputParms);
        of.run();

        Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(u, flow_u);

        float** image1_warp;
        image1_warp = of.m_I2w;
        Mat image1_warp_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(image1_warp, image1_warp_mat);

        vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
        //getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);

        vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;

        Mat flow_u_16= Mat::zeros(ImageRows,ImageCols,CV_16UC1);
        Mat flow_u_mask;
        flow_u_mask = flow_u.mul(valid_mask);

        //  waitKey();
        flow_u_mask.convertTo(flow_u_16,CV_16UC1, scale);

        double MAD_deep_mask = calMAD_mask(gt, deep_mat, fb, scale, valid_mask);
        double MAD_flow_mask = calMAD_mask(gt, flow, fb, scale, valid_mask);
        double MAD_deep = calMAD(gt, deep_mat, fb, scale);
        double MAD_flow = calMAD(gt, flow, fb, scale);
        double MAD_flow_optimized = calMAD(gt, flow_u_16, fb, scale);
        double MAD_flow_optimized_mask = calMAD_mask(gt, flow_u_16, fb, scale, valid_mask);

        if(show_every){
            cout<<"index: "<<index<<endl;
            cout<<"MAD deep "<< MAD_deep<<endl;
            cout<<"MAD flow "<<MAD_flow<<endl;
            cout<<"MAD deep mask "<< MAD_deep_mask<<endl;
            cout<<"MAD flow mask "<<MAD_flow_mask<<endl;
            cout<<"Optimized MAD  "<< MAD_flow_optimized<<endl;
            cout<<"Optimized MAD mask "<<MAD_flow_optimized_mask<<endl;
            cout<<"\n"<<endl;
        }

        error_deep = error_deep + MAD_deep;
        error_flow = error_flow + MAD_flow;
        error_deep_mask = error_deep_mask + MAD_deep_mask;
        error_flow_mask = error_flow_mask + MAD_flow_mask;
        error_op = error_op + MAD_flow_optimized;
        error_op_mask = error_op_mask + MAD_flow_optimized_mask;

    }
    cout<<"mean deep "<< error_deep/16<<endl;
    cout<<"mean flow "<<error_flow/16<<endl;
    cout<<"mean op "<< error_op/16<<endl;
    cout<<"mean deep mask "<< error_deep_mask/16<<endl;
    cout<<"mean flow mask "<<error_flow_mask/16<<endl;
    cout<<"mean op mask "<<error_op_mask/16<<endl;

}

void select_size(int level){

    int scale_im = 4;
    double error_deep =0;
    double error_flow =0;
    double error_deep_mask =0;
    double error_flow_mask =0;
    double error_op =0;
    double error_op_mask =0;
    bool use_alpha = false;
    bool visual = true;
    bool show_every = false;
    float lambda = 12;
    bool bInitializeWithPrior= true;

    //int index = 10;
    for (int index=1;index<17;index++){

        //boost::format fmt("%03d.png");
        boost::format fmt("%03d.png");
        string name = (fmt%index).str();
        boost::format fmt1("%03d_disp.png");

        string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
        string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/"+name;//013.png";
        string disp_deep =  "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/"+name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
        string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);
        string valid_mask_dir = regex_replace(disp_deep, regex(name), "valid_mask"+name);
        string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
        string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;


        Mat left_org = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
        Mat right_org = cv::imread(right_file,cv::IMREAD_GRAYSCALE);
       // Mat prior_mat_org = cv::imread(confidence);

        Mat valid_mask = cv::imread(valid_mask_dir,IMREAD_GRAYSCALE);
        valid_mask.setTo(1,valid_mask>0);

        Mat deep_mat_org = cv::imread(disp_deep,IMREAD_ANYDEPTH);

        Mat flow = cv::imread(optical_flow,IMREAD_ANYDEPTH);
        Mat gt = cv::imread(gt_path,IMREAD_ANYDEPTH);

//        int ImageCols_org = left_org.cols;
//        int ImageRows_org = left_org.rows;
        Mat left, right, deep_mat;// flow;

        cv::resize(left_org, left, Size(left_org.cols/scale_im, left_org.rows/scale_im));
        cv::resize(right_org, right, Size(left_org.cols/scale_im, left_org.rows/scale_im));

       // cv::resize(prior_mat, right, Size(left_org.cols/scale_im, left_org.cols/scale_im),scale_im,1);
        cv::resize(deep_mat_org, deep_mat, Size(left_org.cols/scale_im, left_org.rows/scale_im));
        deep_mat = deep_mat/scale_im;
       // cv::resize(flow, right, Size(left_org.cols/scale_im, left_org.cols/scale_im),scale_im,1);

//        double min, max;
//        cv::minMaxLoc(deep_mat, &min, &max);
//        cout<<"disparity deep the min "<< min/256<<endl;
//        cout<<"disparity deep the max "<< max/256<<endl;
//        cv::minMaxLoc(deep_mat_org, &min, &max);
//        cout<<"disparity deep org the min "<< min/256<<endl;
//        cout<<"disparity deep org the max "<< max/256<<endl;

        double scale = 256;
        double fx,fy,cx,cy,fb;

        if (index<=8){
            // 内参
            fx = 9.9640068207290187e+02, fy =  9.9640068207290187e+02, cx = 3.7502582168579102e+02, cy =  2.4026374816894531e+02;
            // 基线
            fb = 5.4301732344712009e+03;
        }
        else{
            fx = 9.3469009076912857e+02, fy =  9.3469009076912857e+02, cx = 3.2071961212158203e+02, cy =  2.7268288040161133e+02;
            fb = 5.1446987361551292e+03;
        }

//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
//    getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);
//
//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;
//    getPointCloud(left,flow,scale,cx,cy,fx,fy,fb,pointcloud_pred);
//
//    Visualize(pointcloud_gt,pointcloud_pred);

        int ImageCols = left.cols;
        int ImageRows = left.rows;

        float** image1 = AllocateImage<float>(ImageCols, ImageRows);
        float** image2 = AllocateImage<float>(ImageCols, ImageRows);

        float** u = AllocateImage<float>(ImageCols, ImageRows);
        float** v = AllocateImage<float>(ImageCols, ImageRows);
        float** alpha = AllocateImage<float>(ImageCols, ImageRows);

        //cout<< &alpha[155][155]<<endl;
        float** flow_raw = AllocateImage<float>(ImageCols, ImageRows); //u
        float** deep_raw =  AllocateImage<float>(ImageCols, ImageRows); //u
        float** v_raw =  AllocateImage<float>(ImageCols, ImageRows); //v

        Mat v_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        memset(v[0], 0, ImageCols * ImageRows * sizeof(float));

        Mat2Raw(image1,left);
        Mat2Raw(image2,right);
        //Mat2Raw(v_raw,v_mat);

        float disp_scale = -256.0;
      //  Mat2Raw_scale(flow_raw,flow, disp_scale);
        Mat2Raw_scale(deep_raw,deep_mat, disp_scale);
        //cout<<deep_raw[10][10]<<endl;

        // set parameters
        OpticalFlow_Huber_L2::Parms inputParms{};
        inputParms.epsilon = 0.001f;
        inputParms.iterations = 50;//50;
        inputParms.Lambda = lambda;//15.0f;//100.0f;
        //patch = 120  D1 = 15.0f;
        inputParms.levels = level;//6;//6; //6
        inputParms.minSize = 48;
        inputParms.scalefactor = 0.5f;
        inputParms.warps = 5;
        inputParms.descriptor = OpticalFlow_Huber_L2::D1;//PATCH_INTENSITY;
        inputParms.bInitializeWithPrior = true;//true;//true;

        OpticalFlow_Huber_L2::Data data{};
        data.I1 = image1;
        data.I2 = image2;
        data.u = u;
        data.v = v;
        data.nCols = ImageCols;
        data.nRows = ImageRows;

        data.alpha = alpha;// NULL; //alpha;//alpha;//alpha;//NULL;
        data.uprior = deep_raw;//deep_raw;//NULL;deep_raw;
        data.vprior = v_raw;//NULL; v_raw;

        OpticalFlow_Huber_L2 of;
        of.setup(data, inputParms);
        of.run();

        Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(u, flow_u);

        float** image1_warp;
        image1_warp = of.m_I2w;
        Mat image1_warp_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(image1_warp, image1_warp_mat);

        vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
        //getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);

        vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;

        Mat flow_u_16= Mat::zeros(ImageRows,ImageCols,CV_16UC1);
//        Mat flow_u_mask;
//        flow_u_mask = flow_u.mul(valid_mask);

        //  waitKey();
        flow_u.convertTo(flow_u_16,CV_16UC1, scale);
        resize(flow_u_16,flow_u_16,Size(ImageCols*scale_im,ImageRows*scale_im));
        flow_u_16 = flow_u_16*scale_im;

        double MAD_deep_mask = calMAD_mask(gt, deep_mat_org, fb, scale, valid_mask);
        double MAD_flow_mask = calMAD_mask(gt, flow, fb, scale, valid_mask);
        double MAD_deep = calMAD(gt, deep_mat_org, fb, scale);
        double MAD_flow = calMAD(gt, flow, fb, scale);

        double MAD_flow_optimized = calMAD(gt, flow_u_16, fb, scale);
        double MAD_flow_optimized_mask = calMAD_mask(gt, flow_u_16, fb, scale, valid_mask);

        if(show_every){
            cout<<"index: "<<index<<endl;
            cout<<"MAD deep "<< MAD_deep<<endl;
            cout<<"MAD flow "<<MAD_flow<<endl;
            cout<<"MAD deep mask "<< MAD_deep_mask<<endl;
            cout<<"MAD flow mask "<<MAD_flow_mask<<endl;
            cout<<"Optimized MAD  "<< MAD_flow_optimized<<endl;
            cout<<"Optimized MAD mask "<<MAD_flow_optimized_mask<<endl;
            cout<<"\n"<<endl;
        }

        error_deep = error_deep + MAD_deep;
        error_flow = error_flow + MAD_flow;
        error_deep_mask = error_deep_mask + MAD_deep_mask;
        error_flow_mask = error_flow_mask + MAD_flow_mask;
        error_op = error_op + MAD_flow_optimized;
        error_op_mask = error_op_mask + MAD_flow_optimized_mask;

    }
    cout<<"mean deep "<< error_deep/16<<endl;
    cout<<"mean flow "<<error_flow/16<<endl;
    cout<<"mean op "<< error_op/16<<endl;
    cout<<"mean deep mask "<< error_deep_mask/16<<endl;
    cout<<"mean flow mask "<<error_flow_mask/16<<endl;
    cout<<"mean op mask "<<error_op_mask/16<<endl;

}

void opt_initialization(){
    double error_deep =0;
    double error_flow =0;
    double error_deep_mask =0;
    double error_flow_mask =0;
    double error_op =0;
    double error_op_mask =0;
    bool use_alpha = false;
    bool visual = true;
    bool show_every = false;
    float lambda = 12;
    bool bInitializeWithPrior= true;

    //int index = 10;
    for (int index=1;index<17;index++){

        //boost::format fmt("%03d.png");
        boost::format fmt("%03d.png");
        string name = (fmt%index).str();
        boost::format fmt1("%03d_disp.png");

        string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
        string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/"+name;//013.png";
        string disp_deep =  "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/"+name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
        string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);
        string valid_mask_dir = regex_replace(disp_deep, regex(name), "valid_mask"+name);
        string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
        string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;


        Mat left = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
        Mat right = cv::imread(right_file,cv::IMREAD_GRAYSCALE);
        Mat prior_mat = cv::imread(confidence);
        Mat deep_mat = cv::imread(disp_deep,IMREAD_ANYDEPTH);
        Mat flow = cv::imread(optical_flow,IMREAD_ANYDEPTH);
        Mat gt = cv::imread(gt_path,IMREAD_ANYDEPTH);
        Mat valid_mask = cv::imread(valid_mask_dir,IMREAD_GRAYSCALE);
        valid_mask.setTo(1,valid_mask>0);

        double scale = 256;

        double fx,fy,cx,cy,fb;

        if (index<=8){
            // 内参
            fx = 9.9640068207290187e+02, fy =  9.9640068207290187e+02, cx = 3.7502582168579102e+02, cy =  2.4026374816894531e+02;
            // 基线
            fb = 5.4301732344712009e+03;
        }
        else{
            fx = 9.3469009076912857e+02, fy =  9.3469009076912857e+02, cx = 3.2071961212158203e+02, cy =  2.7268288040161133e+02;
            fb = 5.1446987361551292e+03;
        }

//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
//    getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);
//
//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;
//    getPointCloud(left,flow,scale,cx,cy,fx,fy,fb,pointcloud_pred);
//
//    Visualize(pointcloud_gt,pointcloud_pred);

        int ImageCols = left.cols;
        int ImageRows = left.rows;

        float** image1 = AllocateImage<float>(ImageCols, ImageRows);
        float** image2 = AllocateImage<float>(ImageCols, ImageRows);

        float** u = AllocateImage<float>(ImageCols, ImageRows);
        float** v = AllocateImage<float>(ImageCols, ImageRows);
        float** alpha = AllocateImage<float>(ImageCols, ImageRows);

        //cout<< &alpha[155][155]<<endl;
        float** flow_raw = AllocateImage<float>(ImageCols, ImageRows); //u
        float** deep_raw =  AllocateImage<float>(ImageCols, ImageRows); //u
        float** v_raw =  AllocateImage<float>(ImageCols, ImageRows); //v

        Mat v_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        memset(v[0], 0, ImageCols * ImageRows * sizeof(float));

        Mat2Raw(image1,left);
        Mat2Raw(image2,right);
        //Mat2Raw(v_raw,v_mat);

        float disp_scale = -256.0;
        Mat2Raw_scale(flow_raw,flow, disp_scale);
        Mat2Raw_scale(deep_raw,deep_mat, disp_scale);
        //cout<<deep_raw[10][10]<<endl;

        // set parameters
        OpticalFlow_Huber_L2::Parms inputParms{};
        inputParms.epsilon = 0.001f;
        inputParms.iterations = 50;//50;
        inputParms.Lambda = lambda;//15.0f;//100.0f;
        //patch = 120  D1 = 15.0f;
        inputParms.levels = 6;//6; //6
        inputParms.minSize = 48;
        inputParms.scalefactor = 0.5f;
        inputParms.warps = 5;
        inputParms.descriptor = OpticalFlow_Huber_L2::D1;//PATCH_INTENSITY;
        inputParms.bInitializeWithPrior = true;//true;//true;

        OpticalFlow_Huber_L2::Data data{};
        data.I1 = image1;
        data.I2 = image2;
        data.u = u;
        data.v = v;
        data.nCols = ImageCols;
        data.nRows = ImageRows;

        data.alpha = alpha;// NULL; //alpha;//alpha;//alpha;//NULL;
        data.uprior = deep_raw;//deep_raw;//NULL;deep_raw;
        data.vprior = v_raw;//NULL; v_raw;

        OpticalFlow_Huber_L2 of;
        of.setup(data, inputParms);
        of.run();

        Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(u, flow_u);

        float** image1_warp;
        image1_warp = of.m_I2w;
        Mat image1_warp_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(image1_warp, image1_warp_mat);

        vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
        //getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);

        vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;

        Mat flow_u_16= Mat::zeros(ImageRows,ImageCols,CV_16UC1);
        Mat flow_u_mask;
        flow_u_mask = flow_u.mul(valid_mask);

        //  waitKey();
        flow_u_mask.convertTo(flow_u_16,CV_16UC1, scale);

        double MAD_deep_mask = calMAD_mask(gt, deep_mat, fb, scale, valid_mask);
        double MAD_flow_mask = calMAD_mask(gt, flow, fb, scale, valid_mask);
        double MAD_deep = calMAD(gt, deep_mat, fb, scale);
        double MAD_flow = calMAD(gt, flow, fb, scale);
        double MAD_flow_optimized = calMAD(gt, flow_u_16, fb, scale);
        double MAD_flow_optimized_mask = calMAD_mask(gt, flow_u_16, fb, scale, valid_mask);

        if(show_every){
            cout<<"index: "<<index<<endl;
            cout<<"MAD deep "<< MAD_deep<<endl;
            cout<<"MAD flow "<<MAD_flow<<endl;
            cout<<"MAD deep mask "<< MAD_deep_mask<<endl;
            cout<<"MAD flow mask "<<MAD_flow_mask<<endl;
            cout<<"Optimized MAD  "<< MAD_flow_optimized<<endl;
            cout<<"Optimized MAD mask "<<MAD_flow_optimized_mask<<endl;
            cout<<"\n"<<endl;
        }

        error_deep = error_deep + MAD_deep;
        error_flow = error_flow + MAD_flow;
        error_deep_mask = error_deep_mask + MAD_deep_mask;
        error_flow_mask = error_flow_mask + MAD_flow_mask;
        error_op = error_op + MAD_flow_optimized;
        error_op_mask = error_op_mask + MAD_flow_optimized_mask;

    }
    cout<<"mean deep "<< error_deep/16<<endl;
    cout<<"mean flow "<<error_flow/16<<endl;
    cout<<"mean op "<< error_op/16<<endl;
    cout<<"mean deep mask "<< error_deep_mask/16<<endl;
    cout<<"mean flow mask "<<error_flow_mask/16<<endl;
    cout<<"mean op mask "<<error_op_mask/16<<endl;

}

void try_initialization_mask(){
    double error_deep =0;
    double error_flow =0;
    double error_deep_mask =0;
    double error_flow_mask =0;
    double error_op =0;
    double error_op_mask =0;
    bool use_alpha = false;
    bool visual = true;
    bool show_every = true;
    float lambda = 12;
    bool bInitializeWithPrior= true;

    //int index = 10;
    for (int index=1;index<17;index++){

        //boost::format fmt("%03d.png");
        boost::format fmt("%03d.png");
        string name = (fmt%index).str();
        boost::format fmt1("%03d_disp.png");

        string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;//"013.png";
        string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/"+name;//013.png";
        string disp_deep =  "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/STTR/"+name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
        string confidence =  regex_replace(disp_deep, regex(name), "mask"+name);
        string valid_mask_dir = regex_replace(disp_deep, regex(name), "valid_mask"+name);
        string optical_flow = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/Optical_flow/"+(fmt1%index).str();
        string gt_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/gt_disp/" + name;

        Mat left = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
        Mat right = cv::imread(right_file,cv::IMREAD_GRAYSCALE);
        Mat prior_mat = cv::imread(confidence,cv::IMREAD_GRAYSCALE);
        Mat deep_mat = cv::imread(disp_deep,IMREAD_ANYDEPTH);
        Mat flow = cv::imread(optical_flow,IMREAD_ANYDEPTH);
        Mat gt = cv::imread(gt_path,IMREAD_ANYDEPTH);

        Mat valid_mask = cv::imread(valid_mask_dir,IMREAD_GRAYSCALE);
        valid_mask.setTo(1,valid_mask>0);

        Mat prior_mat_mask = Mat::zeros(left.rows,left.cols,CV_16UC1);
        //prior_mat.convertTo(prior_mat_mask,CV_16UC1);
        prior_mat_mask.setTo(1,prior_mat>0.2*255);
        Mat deep_mat_mask = deep_mat.mul(prior_mat_mask);
//        imshow("deep mat mask",deep_mat_mask);
//        waitKey();


        double scale = 256;

        double fx,fy,cx,cy,fb;

        if (index<=8){
            // 内参
            fx = 9.9640068207290187e+02, fy =  9.9640068207290187e+02, cx = 3.7502582168579102e+02, cy =  2.4026374816894531e+02;
            // 基线
            fb = 5.4301732344712009e+03;
        }
        else{
            fx = 9.3469009076912857e+02, fy =  9.3469009076912857e+02, cx = 3.2071961212158203e+02, cy =  2.7268288040161133e+02;
            fb = 5.1446987361551292e+03;
        }

//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
//    getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);
//
//    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;
//    getPointCloud(left,flow,scale,cx,cy,fx,fy,fb,pointcloud_pred);
//
//    Visualize(pointcloud_gt,pointcloud_pred);

        int ImageCols = left.cols;
        int ImageRows = left.rows;

        float** image1 = AllocateImage<float>(ImageCols, ImageRows);
        float** image2 = AllocateImage<float>(ImageCols, ImageRows);

        float** u = AllocateImage<float>(ImageCols, ImageRows);
        float** v = AllocateImage<float>(ImageCols, ImageRows);
        float** alpha = AllocateImage<float>(ImageCols, ImageRows);

        //cout<< &alpha[155][155]<<endl;
        float** flow_raw = AllocateImage<float>(ImageCols, ImageRows); //u
        float** deep_raw =  AllocateImage<float>(ImageCols, ImageRows); //u
        float** v_raw =  AllocateImage<float>(ImageCols, ImageRows); //v

        Mat v_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        memset(v[0], 0, ImageCols * ImageRows * sizeof(float));

        Mat2Raw(image1,left);
        Mat2Raw(image2,right);
        //Mat2Raw(v_raw,v_mat);

        float disp_scale = -256.0;
        Mat2Raw_scale(flow_raw,flow, disp_scale);
        Mat2Raw_scale(deep_raw,deep_mat_mask, disp_scale);
        //cout<<deep_raw[10][10]<<endl;

        // set parameters
        OpticalFlow_Huber_L2::Parms inputParms{};
        inputParms.epsilon = 0.001f;
        inputParms.iterations = 50;//50;
        inputParms.Lambda = lambda;//15.0f;//100.0f;
        //patch = 120  D1 = 15.0f;
        inputParms.levels = 6;//6; //6
        inputParms.minSize = 48;
        inputParms.scalefactor = 0.5f;
        inputParms.warps = 5;
        inputParms.descriptor = OpticalFlow_Huber_L2::D1;//PATCH_INTENSITY;
        inputParms.bInitializeWithPrior = true;//true;//true;

        OpticalFlow_Huber_L2::Data data{};
        data.I1 = image1;
        data.I2 = image2;
        data.u = u;
        data.v = v;
        data.nCols = ImageCols;
        data.nRows = ImageRows;

        data.alpha = alpha;// NULL; //alpha;//alpha;//alpha;//NULL;
        data.uprior = deep_raw;//deep_raw;//NULL;deep_raw;
        data.vprior = v_raw;//NULL; v_raw;

        OpticalFlow_Huber_L2 of;
        of.setup(data, inputParms);
        of.run();

        Mat flow_u = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(u, flow_u);

        float** image1_warp;
        image1_warp = of.m_I2w;
        Mat image1_warp_mat = Mat::zeros(ImageRows,ImageCols,CV_8UC1);
        ToMat(image1_warp, image1_warp_mat);

        vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_gt;
        //getPointCloud(left,gt,scale,cx,cy,fx,fy,fb,pointcloud_gt);

        vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud_pred;

        Mat flow_u_16= Mat::zeros(ImageRows,ImageCols,CV_16UC1);
//        Mat flow_u_mask;
//        flow_u_mask = flow_u.mul(valid_mask);

        //  waitKey();
        flow_u.convertTo(flow_u_16,CV_16UC1, scale);

        double MAD_deep_mask = calMAD_mask(gt, deep_mat, fb, scale, valid_mask);
        double MAD_flow_mask = calMAD_mask(gt, flow, fb, scale, valid_mask);
        double MAD_deep = calMAD(gt, deep_mat, fb, scale);
        double MAD_flow = calMAD(gt, flow, fb, scale);
        double MAD_flow_optimized = calMAD(gt, flow_u_16, fb, scale);
        double MAD_flow_optimized_mask = calMAD_mask(gt, flow_u_16, fb, scale, valid_mask);

        if(show_every){
            cout<<"index: "<<index<<endl;
            cout<<"MAD deep "<< MAD_deep<<endl;
            cout<<"MAD flow "<<MAD_flow<<endl;
            cout<<"MAD deep mask "<< MAD_deep_mask<<endl;
            cout<<"MAD flow mask "<<MAD_flow_mask<<endl;
            cout<<"Optimized MAD  "<< MAD_flow_optimized<<endl;
            cout<<"Optimized MAD mask "<<MAD_flow_optimized_mask<<endl;
            cout<<"\n"<<endl;
        }

        error_deep = error_deep + MAD_deep;
        error_flow = error_flow + MAD_flow;
        error_deep_mask = error_deep_mask + MAD_deep_mask;
        error_flow_mask = error_flow_mask + MAD_flow_mask;
        error_op = error_op + MAD_flow_optimized;
        error_op_mask = error_op_mask + MAD_flow_optimized_mask;

    }
    cout<<"mean deep "<< error_deep/16<<endl;
    cout<<"mean flow "<<error_flow/16<<endl;
    cout<<"mean op "<< error_op/16<<endl;
    cout<<"mean deep mask "<< error_deep_mask/16<<endl;
    cout<<"mean flow mask "<<error_flow_mask/16<<endl;
    cout<<"mean op mask "<<error_op_mask/16<<endl;

}

void FillHolesInDispMap(int width_, int height_, float* disp_ptr)
{
    const int width = width_;
    const int height = height_;

    std::vector<float> disp_collects;
    int iter = 5;


    // 定义8个方向
    const float pi = 3.1415926f;
    float angle1[8] = { pi, 3 * pi / 4, pi / 2, pi / 4, 0, 7 * pi / 4, 3 * pi / 2, 5 * pi / 4 };
    float angle2[8] = { pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4 };
    float *angle = angle1;
    // 最大搜索行程，没有必要搜索过远的像素
    const int max_search_length = 40; // 1.0*std::max(abs(option_.max_disparity), abs(option_.min_disparity));

    //float* disp_ptr = disp_left_;
//    for (int k = 0; k < 5; k++) {
        // 第一次循环处理遮挡区，第二次循环处理误匹配区
//        auto& trg_pixels = (k == 0) ? occlusions_ : mismatches_;
//        if (trg_pixels.empty()) {
//            continue;
//        }


    for (int k = 0; k < iter; k++) {
        std::vector<std::pair<int, int>> inv_pixels;
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (disp_ptr[i * width + j] == 0) { // Invalid_Float
                    inv_pixels.emplace_back(i, j);
                }
            }
        }

        auto trg_pixels = inv_pixels;

        std::vector<float> fill_disps(trg_pixels.size());

        // 遍历待处理像素
        for (auto n = 0u; n < trg_pixels.size(); n++) {
            auto &pix = trg_pixels[n];
            const int y = pix.first;
            const int x = pix.second;

            if (y == height / 2) {
                angle = angle2;
            }

            // 收集8个方向上遇到的首个有效视差值
            disp_collects.clear();
            for (int s = 0; s < 8; s++) {
                const float ang = angle[s];
                const auto sina = float(sin(ang));
                const auto cosa = float(cos(ang));
                for (int m = 1; m < max_search_length; m++) {
                    const int yy = lround(y + m * sina);
                    const int xx = lround(x + m * cosa);
                    if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
                        break;
                    }
                    const auto &disp = *(disp_ptr + yy * width + xx);
                    if (disp != 0) {//Invalid_Float
                        disp_collects.push_back(disp);
                        break;
                    }
                }
            }
            if (disp_collects.empty()) {
                //disp_collects.push_back(0);
                fill_disps[n] = 0;

            }else{
                std::sort(disp_collects.begin(), disp_collects.end());
                fill_disps[n] = disp_collects[disp_collects.size() / 2];
            }

        }

        // assign values to pixels
        for (auto n = 0u; n < trg_pixels.size(); n++) {
            auto &pix = trg_pixels[n];
            const int y = pix.first;
            const int x = pix.second;
            disp_ptr[y * width + x] = fill_disps[n];
        }

    }

}

void FillHolesInDispMap_test()
{
   // int index = 1;

    for (int index=1;index<17;index++) {

        double th = 0.5;
        boost::format fmt("%03d.png");
        string name = (fmt % index).str();
        string disp_deep = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/STTR/" + name;//013.png"; // "/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/001.png";//
        string confidence = regex_replace(disp_deep, regex(name), "mask" + name);
        string valid_mask_dir = regex_replace(disp_deep, regex(name), "valid_mask" + name);
        string refine_disp_dir = regex_replace(disp_deep, regex(name), "refine_" + name);

        Mat prior_mat = cv::imread(confidence, cv::IMREAD_GRAYSCALE);
        Mat deep_mat = cv::imread(disp_deep, IMREAD_ANYDEPTH);
        Mat valid_mask = cv::imread(valid_mask_dir, IMREAD_GRAYSCALE);

        int ImageCols = prior_mat.cols;
        int ImageRows = prior_mat.rows;
//        cout << " width Cols" << ImageCols << endl;
//        cout << "height Rows" << ImageRows << endl;
        deep_mat.convertTo(deep_mat, CV_8UC1, 1 / 256.0);

        // apply mask
        //valid_mask.convertTo(valid_mask,CV_16UC1);
        valid_mask.setTo(1, valid_mask > 0);
        //Mat prior_mat_mask = Mat::zeros(ImageRows, ImageCols,CV_16UC1);

        prior_mat.setTo(255, prior_mat > th * 255);
        prior_mat.setTo(0, prior_mat <= th * 255);

        Mat deep_mat_mask = deep_mat.mul(prior_mat / 255);
        deep_mat_mask = deep_mat_mask.mul(valid_mask);
        // deep_mat_mask = deep_mat_mask/256.0;
        //  deep_mat_mask.convertTo(deep_mat_mask,CV_8UC1);
//        cv::imshow("deep mat mask", deep_mat_mask);
//        cv::imshow("deep mat ", deep_mat);
//        waitKey();

        int width = ImageCols;
        int height = ImageRows;

        auto disparity = new float[width * height]();
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                disparity[i * width + j] = deep_mat_mask.at<uint8_t>(i, j);
            }
        }

        FillHolesInDispMap(width, height, disparity);

        cv::Mat disp_mat = cv::Mat(height, width, CV_8UC1);

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                const float disp = disparity[i * width + j];
                if (disp == 0) {//Invalid_Float
                    disp_mat.data[i * width + j] = 0;
                } else {
                    disp_mat.data[i * width +
                                  j] = static_cast<uchar>(disp);// (disp - min_disp) / (max_disp - min_disp) * 255
                }
            }
        }

       // disp_mat = disp_mat.mul(valid_mask);
        disp_mat.convertTo(disp_mat, CV_16UC1, 256.0);

        imwrite(refine_disp_dir, disp_mat);
    }

//    cv::imshow("视差图", disp_mat);
//    waitKey();


//    float min_disp = width, max_disp = -width;
//    for (int i = 0; i < height; i++) {
//        for (int j = 0; j < width; j++) {
//            const float disp = disparity[i * width + j];
//            if (disp != 0) {//Invalid_Float
//                min_disp = std::min(min_disp, disp);
//                max_disp = std::max(max_disp, disp);
//            }
//        }
//    }
//
//    for (int i = 0; i < height; i++) {
//        for (int j = 0; j < width; j++) {
//            const float disp = disparity[i * width + j];
//            if (disp == 0) {//Invalid_Float
//                disp_mat.data[i * width + j] = 0;
//            }
//            else {
//                disp_mat.data[i * width + j] = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
//            }
//        }
//    }

//    float** v = AllocateImage<float>(ImageCols, ImageRows);
//    float** alpha = AllocateImage<float>(ImageCols, ImageRows);
}



int main(){

    boost::format fmt("%03d.png");
    int index =1;
    string name = (fmt % index).str();
    string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;
    string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/" + name;
    cv::Mat left = cv::imread(left_file, cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_file, cv::IMREAD_GRAYSCALE);

    cv::resize(left, left, Size(left.cols/2, left.rows/2));
    cv::resize(right, right, Size(left.cols/2, left.rows/2));

    int ImageCols = left.cols; // width
    int ImageRows = left.rows; // height

    float **image1 = AllocateImage<float>(ImageCols, ImageRows); // images
    float **image2 = AllocateImage<float>(ImageCols, ImageRows);
    Mat2Raw(image1, left);
    Mat2Raw(image2, right);

    string file_l = "left" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    string file_r = "right" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    string dirIn = "/home/yzx/CLionProjects/Stereo_Matching/Optical_Huber_L2/Data/";
    WriteImage(dirIn, file_l, image1, ImageCols, ImageRows);
    WriteImage(dirIn, file_r, image2, ImageCols, ImageRows);



    /**alpha**/
    //float alpha = 1;
//    vector<float> Alpha{0.0005, 0.00025, 0.000125, 0.00006125};
//    vector<float> Alpha{  2, 1, 0.9, 0.8, 0.7, 0.6, 0.5};
//    for(auto alpha:Alpha){
//        cout<<"alpha weight"<<alpha<<endl;
//        refinement_all(alpha);
//        cout<<"\n"<<endl;
//    }

   /** lambda **/
//    vector<float> Lambda{ 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5};
//    for(auto lambda:Lambda){
//        cout<<"lambda "<<lambda<<endl;
//        select_lambda(lambda);
//        cout<<"\n"<<endl;
//    }

   /**level**/
//    vector<int> Level{1,2,3,4,5,6};
//        for(auto level:Level){
//        cout<<"level "<<level<<endl;
//        initialize_level(level);
//        cout<<"\n"<<endl;
//    }

    /**size**/
//    vector<int> Level{1,2,3,4,5,6};
//    for(auto level:Level){
//        cout<<"level "<<level<<endl;
//        select_size(level);
//        cout<<"\n"<<endl;
//    }

    /**Initialization**/
    //try_initialization();
    /**Initialization mask**/
    // try_initialization_mask();
   /** Fill holes **/
    //FillHolesInDispMap_test();
   //try_initialization_ref();




//    float alpha = 0.0005;
//    cout<<"alpha "<<alpha<<endl;
//    refinement_all(alpha);
//    cout<<"\n"<<endl;



    //test_refinement();
    //refinement();
    //save_confidence_SERV();

//    boost::format fmt("%03d.png");
//    int index =1;
//    string method_name = "LEAStereo";
//    string name = (fmt % index).str();
//    string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;
//    string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/" + name;
//    string disp_deep = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/" + method_name + "/" + name; //"/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/013.png";// "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/AAnet/001.png";
//    string out_dir = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/" + method_name + "/mask" + name;
//
//    confidence(left_file, right_file, disp_deep, out_dir, true);

//    string test = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/AAnet/mask001.png";
//    Mat conf = imread(test);
//    imshow("confidence", conf);
//    waitKey();
    //test_prior();
    //test_lost();




};