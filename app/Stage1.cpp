//
// Created by yzx on 3/3/22.
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
#include <chrono>
#include <iostream>
#include<vector>

using namespace std::chrono;
#include <string>
#include <unistd.h>
#include <math.h>

using namespace std;
using namespace cv;
using namespace Eigen;

class StageOne{

public:
    string left_path;
    string disp_path;
    Mat disp;
    int Cols;
    int Rows;
    int m_nCols;
    int m_nRows;
    int m_nDescriptors = 8;


    float** image1;
    float** image2;
    float** image1_warp;
    float** deep_raw;

    // D1 diff
    float*** l_D1;
    float*** l_warp_D1;
    float** D1_diff;
    float** l_D1_mag;
    float** D1_confidence;

    // Photometric
    float** photo_conf;
    // Dist
    float** dist;
    float** dist_conf;
    // Valid Mask
    float** valid_mask;
    // Saturation mask
    float** mask_s;

    //  Result
    Mat valid_mask_mat;
    Mat final_mask_mat;
    Mat mask_s_mat;
    Mat ref_disp_mat;
    Mat deep_mat_mask;


    float atPt(float**image, int y, int x) const
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

    void compute_feature_map_D1(float*** D, float** img) const
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
                    if (norm != 0.0f){
                        D[d][i][j] /= norm;}
                    else{
                        D[d][i][j] = 0.0f;}//cout<<"zero!!!!"<<endl;}

                }

            }
        }
    }

    static float Tex2D(float** t, const int w, const int h, const float x, const float y)
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

    static void warp_u(const int Rows, const int Cols, float** m_u, float** I_source, float** I_warp)
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

    static void mul_all(const int Rows, const int Cols, float** im1, float** im2, float** im3, float** im4, float** output){

        for (int i = 0; i < Rows; i++) {
            for (int j = 0; j <Cols; j++) {
                //wnd_data.clear();
                // get window data around i,j
                float value = (im1[i][j]/255)*(im2[i][j]/255)*(im3[i][j]/255)*(im4[i][j]/255);//1- alpha*abs(dist[i][j]);//input[i*width+j];
                output[i][j] = value *255 ;
            }
        }

    }

//    void compute_D1_diff(float alpha){
//
//        float Dt;
//        l_D1_mag = AllocateImage<float>(m_nCols, m_nRows);
//        memset(l_D1_mag[0], 0, m_nCols * m_nRows * sizeof(float));
//        D1_diff = AllocateImage<float>(m_nCols, m_nRows);
//        memset(D1_diff[0], 0, m_nCols * m_nRows * sizeof(float));
//        D1_confidence = AllocateImage<float>(m_nCols, m_nRows);
//        memset(D1_confidence[0], 0, m_nCols * m_nRows * sizeof(float));
//
//        for (int i = 0; i < m_nRows; i++)
//        {
//            for (int j = 0; j < m_nCols; j++)
//            {
//                for (int d = 0; d < m_nDescriptors; d++)
//                {
//                    // The feature difference
//                    Dt = abs(l_D1[d][i][j] - l_warp_D1[d][i][j]);
//                    D1_diff[i][j] = D1_diff[i][j] + Dt*Dt;
//                    // The magnitude
//                    l_D1_mag[i][j] = l_D1_mag[i][j]+ pow(l_D1[d][i][j],2);
//                }
//                D1_diff[i][j] = sqrt(D1_diff[i][j]);
//                l_D1_mag[i][j] = sqrt(l_D1_mag[i][j]);
//                float confidence_value =l_D1_mag[i][j];// D1_diff[i][j]; //(D1_diff[i][j]/l_D1_mag[i][j]);
//                D1_confidence[i][j] = confidence_value;
//
//                if(confidence_value>0){
//                    D1_confidence[i][j] = confidence_value * 255;
//                }else{
//                    D1_confidence[i][j] = 0;
//                }
//
//            }
//        }
//    }
//    void run_D1(float alpha){
//        l_D1 = AllocateVolume<float>(m_nCols, m_nRows, m_nDescriptors);
//        l_warp_D1 = AllocateVolume<float>(m_nCols, m_nRows, m_nDescriptors);
//
//        compute_feature_map_D1(l_D1, image1);
//        compute_feature_map_D1(l_warp_D1, image1_warp);
//
//        compute_D1_diff(alpha);
//        //visualize(D1_confidence,"conf");
//
//        //confidence_map_warp_diff(m_nRows, m_nCols, l_D1, l_warp_D1, D1_diff,alpha);
//
//    }

    static void confidence_map_warp(const int Rows, const int Cols, float** input, float** output, float** map, float alpha){

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

    static void cal_dist_map(const int Rows, const int Cols, float** input, float** output,
                      int radius, int radius_step, int num_radius_step ){

        for (int i = 0; i < Rows; i++) {
            for (int j = 0; j <Cols; j++) {
                //wnd_data.clear();
                // get window data around i,j
                float disp_win = abs(input[i][j]);//input[i*width+j];
                float disp_sum = 0;
                int num = 0;
                for (int k = 0; k < num_radius_step; k++) {
                    int radius_c = k * radius_step + radius;
                    cout<<"radius c: "<< radius_c<<endl;
                    for (int r = -radius_c; r <= radius_c;) {
                        for (int c = -radius_c; c <= radius_c;) {
                            const int row = i + r;
                            const int col = j + c;
                            if (row >= 0 && row < Rows && col >= 0 && col < Cols ) { //&& r != 0 && c != 0
                                disp_sum = disp_sum + abs(input[row][col]);
                                num = num + 1;
                            }
                            //c = c + radius_c/2;
                            c = c + 1;
                        }
                        r= r + 1;
                        //r = r + radius_c/2;
                    }
                }
                float mean_disp_sum = disp_sum/num;
                //cout << " value " << abs(mean_disp_sum - disp_win)<<endl;
                output[i][j] = abs(mean_disp_sum - disp_win)/mean_disp_sum;//  abs(mean_disp_sum - disp_win);
            }
        }

    }

    static void cal_dist_map_win(const int Rows, const int Cols, float** input, float** output, int radius ){

        for (int i = 0; i < Rows; i++) {
            for (int j = 0; j <Cols; j++) {
                //wnd_data.clear();
                // get window data around i,j
                float disp_win = abs(input[i][j]);//input[i*width+j];
                float disp_sum = 0;
                int num = 0;
                for (int r = -radius; r <= radius;) {
                    for (int c = -radius; c <= radius;) {
                        const int row = i + r;
                        const int col = j + c;
                        if (row >= 0 && row < Rows && col >= 0 && col < Cols ) { //&& r != 0 && c != 0
                            disp_sum = disp_sum + abs(input[row][col]);
                            num = num + 1;
                        }
                        c = c + 1;
                    }
                    r= r + 1;
                }
                float mean_disp_sum = disp_sum/num;
                output[i][j] = abs(mean_disp_sum - disp_win)/mean_disp_sum;//  abs(mean_disp_sum - disp_win);
            }
        }

    }

    static void confidence_map_dist(const int Rows, const int Cols, float** dist,float** output, float alpha){

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

    void valid_range(const int Rows, const int Cols, float** m_u) const
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

    void FillHolesInDispMap(int width_, int height_, float* disp_ptr, int iter =5)
    {
        const int width = width_;
        const int height = height_;

        std::vector<float> disp_collects;
//        int iter = 5;


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

    void FillHolesMat(Mat disp, int iter =5){

        int width = Cols;
        int height = Rows;

        auto disparity = new float[width * height]();
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                disparity[i * width + j] = disp.at<unsigned short >(i,j); //disp.at<uint8_t>(i, j);
            }
        }

        FillHolesInDispMap(width, height, disparity, iter);

        cv::Mat disp_mat = cv::Mat(height, width, CV_16UC1);

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                const float disp_value = disparity[i * width + j];
                if (disp_value == 0) {//Invalid_Float
                    //disp_mat.data[i * width + j] = 0;
                    disp_mat.at<unsigned short >(i,j) = 0.0;
                } else {
                    //disp_mat.data[i * width + j] = static_cast<uchar>(disp_value);// (disp - min_disp) / (max_disp - min_disp) * 255
                    disp_mat.at<unsigned short >(i,j) = disp_value;
                }
            }
        }

        // disp_mat = disp_mat.mul(valid_mask);
        //disp_mat.convertTo(disp_mat, CV_16UC1, 256.0);
        ref_disp_mat = disp_mat;
        //return disp_mat;
    }


    void setup(const string& left_file, const string& right_file, const string& disp_file, int scale_im){

        left_path = left_file;
        disp_path = disp_file;

        Mat left = cv::imread(left_file,cv::IMREAD_GRAYSCALE);
        Mat right = cv::imread(right_file,cv::IMREAD_GRAYSCALE);
        Mat deep_mat = cv::imread(disp_file,IMREAD_ANYDEPTH);
        disp =deep_mat;

        Cols = left.cols;
        Rows = left.rows;
        m_nCols = left.cols/scale_im;
        m_nRows = left.rows/scale_im;

        if(scale_im>1){
            cv::resize(left, left, Size(m_nCols,m_nRows));
            cv::resize(right, right, Size(m_nCols, m_nRows));
            // cv::resize(prior_mat, right, Size(left_org.cols/scale_im, left_org.cols/scale_im),scale_im,1);
            cv::resize(deep_mat, deep_mat, Size(m_nCols, m_nRows));
            deep_mat = deep_mat/scale_im;
        }

        image1 = AllocateImage<float>(m_nCols, m_nRows);
        image2 = AllocateImage<float>(m_nCols, m_nRows);
        image1_warp = AllocateImage<float>(m_nCols, m_nRows);

        Mat2Raw(image1,left);
        Mat2Raw(image2,right);
        deep_raw =  AllocateImage<float>(m_nCols, m_nRows);
        float disp_scale = -256.0;
        Mat2Raw_scale(deep_raw,deep_mat, disp_scale);
        warp_u(m_nRows, m_nCols, deep_raw,  image2,image1_warp);
        //visualize(image1_warp,"warp");

    }

    void visualize(float **img, const string& title) const{
        Mat mat = Mat::zeros(m_nRows,m_nCols,CV_8UC1);
        ToMat(img, mat);

        imshow(title,mat);

        waitKey();
    }


    void run_ph(float alpha,bool show=true){
        photo_conf = AllocateImage<float>(m_nCols, m_nRows); // 0 ~ 255
        confidence_map_warp(m_nRows, m_nCols, image1, image1_warp, photo_conf, alpha);
        if (show){visualize(photo_conf,"photo conf");}
    }

    void run_dist(int radius, float alpha_dist, bool show=true){
        //  int radius_step, int num_radius_step,
        dist = AllocateImage<float>(m_nCols, m_nRows);
        dist_conf = AllocateImage<float>(m_nCols, m_nRows);
//        cal_dist_map(m_nRows, m_nCols, deep_raw, dist,
//                     radius, radius_step, num_radius_step);
        cal_dist_map_win(m_nRows, m_nCols, deep_raw, dist, radius);
        confidence_map_dist(m_nRows, m_nCols, dist, dist_conf, alpha_dist);

        if (show){visualize(dist_conf,"dist conf");}
    }

    void run_valid_mask(bool show= true){
        valid_mask = AllocateImage<float>(m_nCols, m_nRows);
        valid_range(m_nRows, m_nCols, deep_raw);
        if (show){visualize(valid_mask,"valid mask");}

    }

    void run_mask_s(float th_s, bool show= true){
        cv::Mat left_color = cv::imread(left_path);
        cv::resize(left_color, left_color, Size(m_nCols,m_nRows));
//        imshow("test",left_color);
//        waitKey();
        Mat hsv;
        cv::cvtColor(left_color, hsv, cv::COLOR_BGR2HSV);
        Mat im1_s;
        int channelIdx = 1;
        //double min, max;
        extractChannel(hsv, im1_s, channelIdx);
        mask_s_mat = cv::Mat(m_nCols, m_nRows, CV_8UC1);
        mask_s_mat = (im1_s >= (th_s * 255));
//        imshow("test",mask_s_mat);
//        waitKey();
        mask_s = AllocateImage<float>(m_nCols, m_nRows);
        Mat2Raw(mask_s, mask_s_mat);
        if (show){visualize(mask_s,"mask s");}

    }

    void save_process(const string& dir,const string& name){

        cv::resize(mask_s_mat, mask_s_mat, Size(Cols,Rows));
        cv::resize(valid_mask_mat, valid_mask_mat, Size(Cols,Rows));

        Mat dist_conf_mat = Mat::zeros(m_nRows,m_nCols,CV_8UC1);
        ToMat(dist_conf, dist_conf_mat);
        cv::resize(dist_conf_mat, dist_conf_mat, Size(Cols,Rows));

        Mat photo_mat = Mat::zeros(m_nRows,m_nCols,CV_8UC1);
        ToMat(photo_conf, photo_mat);
        cv::resize(photo_mat, photo_mat, Size(Cols,Rows));

        imwrite(dir+"org"+name,disp);
        imwrite(dir+"photo"+name,photo_mat);
        imwrite(dir+"dist"+name,dist_conf_mat);
        imwrite(dir+"valid"+name,valid_mask_mat);
        imwrite(dir+"s"+name,mask_s_mat);
        imwrite(dir+"disp_high"+name,deep_mat_mask);
        imwrite(dir+"ref"+name,ref_disp_mat);
    }

    double main_mask(float alpha_warp, int radius, float alpha_dist, float th_s, float th_conf,
                   bool vis = false){

        auto start = high_resolution_clock::now();
        run_ph(alpha_warp,vis); // ph conf
        run_dist(radius,  alpha_dist,vis); // dist conf
        run_valid_mask(vis);  // valid mask
        run_mask_s(th_s,vis); //  mask s
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);

       // cout << "Time taken by function: "
         //    << duration.count() *0.001 << " seconds" << endl;

        // Get the final mask
        float **final_mask = AllocateImage<float>(m_nCols, m_nRows);
        mul_all(m_nRows, m_nCols, valid_mask, mask_s, dist_conf, dist_conf, final_mask);

        // Get valid mask mat and final mask mat
        final_mask_mat = Mat::zeros(m_nRows, m_nCols, CV_8UC1);
        ToMat(final_mask, final_mask_mat);

        valid_mask_mat = Mat::zeros(m_nRows, m_nCols, CV_8UC1);
        ToMat(valid_mask, valid_mask_mat);

        cv::resize(final_mask_mat, final_mask_mat, Size(Cols,Rows));
        cv::resize(valid_mask_mat, valid_mask_mat, Size(Cols,Rows));

        //valid_mask_mat.setTo(1, valid_mask_mat > 0);
        final_mask_mat.setTo(255, final_mask_mat > th_conf * 255);
        final_mask_mat.setTo(0, final_mask_mat <= th_conf * 255);

        Mat deep_mat = cv::imread(disp_path, IMREAD_ANYDEPTH);
        //deep_mat.convertTo(deep_mat, CV_8UC1, 1 / 256.0);
        final_mask_mat.convertTo(final_mask_mat,CV_16UC1,1);
        deep_mat_mask = deep_mat.mul(final_mask_mat/255);

        FillHolesMat(deep_mat_mask,5); // iter, search range, 16UC1

        //Open
//        int morph_size = 5;
//        Mat element = getStructuringElement( MORPH_ELLIPSE, Size( 2*morph_size + 1, 2*morph_size+1 ), Point( morph_size, morph_size ) );
//        morphologyEx(final_mask_mat,final_mask_mat,MORPH_OPEN ,element);

        if (vis){
            imshow("valid mask", valid_mask_mat);
            imshow("final mask", final_mask_mat*255);
            imshow("org disp",deep_mat);
            imshow("Deep mat mask", deep_mat_mask);
            imshow("refinement", ref_disp_mat);
            waitKey();
        }

        return  duration.count() *0.001;
    }

    void save_results(const string& name){
        string valid_mask_dir = regex_replace(disp_path, regex(name), "valid_mask" + name);
        string refine_disp_dir = regex_replace(disp_path, regex(name), "refine_" + name);
        imwrite(valid_mask_dir,valid_mask_mat);
        imwrite(refine_disp_dir,ref_disp_mat);

    }

};


void test_serv_sample(){

    boost::format fmt("%03d.png");
    int index = 10;
    string method_name = "AAnet";
    string name = (fmt % index).str();
    string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;
    string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/" + name;
    string disp_deep = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/" + method_name + "/" + name; //"/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/013.png";// "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/AAnet/001.png";
    string out_dir = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Result_vis_stage1/";

//    int radius_step = 10;
//    int num_radius_step = 1;
    int radius = 10;

    float alpha_dist = 20;
    float alpha_ph = 2;

    float th_s = 0.1;

    int im_scale = 2;

    float th_conf = 0.5;

    StageOne Confidence;
    Confidence.setup(left_file, right_file, disp_deep, im_scale);
    Confidence.main_mask(alpha_ph, radius, alpha_dist, th_s, th_conf, false);
    Confidence.save_process(out_dir,name);
    //Confidence.save_results(name);

    //Confidence.run_D1(6);
    //Confidence.run_ph(alpha_warp);
    //Confidence.run_dist(radius,  alpha_dist);
    //Confidence.run_valid_mask();
    //Confidence.run_mask_s(th_s);
    //StageOne(left_file, right_file, disp_deep, out_dir, false);

}

void save_results_serv(){

    boost::format fmt("%03d.png");
   // int index = 1;
   double time = 0;

   vector<string> methods={"AAnet","PSMnet","LEAStereo","STTR"};

   for (auto &method_name:methods){
       cout<<"current method "<<method_name<<endl;

       for (int index=1;index<17;index++) {
           //string method_name = "AAnet";
           string name = (fmt % index).str();
           string left_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/left/" + name;
           string right_file = "/media/yzx/Elements/Dataset/SERV-CT-ALL/images/right/" + name;
           string disp_deep = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/" + method_name + "/" +
                              name;

//        cout<<index<<endl;

           //"/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/013.png";// "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/AAnet/001.png";

           int radius = 10;

           float alpha_dist = 20;
           float alpha_ph = 2;

           float th_s = 0.1;

           int im_scale = 2;

           float th_conf = 0.5;

           StageOne Confidence;
           Confidence.setup(left_file, right_file, disp_deep, im_scale);
           time = time+ Confidence.main_mask(alpha_ph, radius, alpha_dist, th_s, th_conf, false);
           Confidence.save_results(name);

       }
       cout<<"ave time"<<time/16.0<<endl;

   }
}

void save_results_SCARED(){
    boost::format fmt("%03d.png");
    string txt_name = "/media/yzx/Elements/Dataset/SCARED_Keyframes/left_list.txt";
    string root_dir = "/media/yzx/Elements/Dataset/SCARED_Keyframes";
    string result_dir = "/media/yzx/Elements/Dataset/SCARED_Keyframes/Result/";

    //string method = "AAnet";

    vector<string> methods={"PSMnet","LEAStereo","STTR","AAnet"};

    for (auto method:methods) {
        cout << "current method " << method << endl;

        int i = 0;
        std::ifstream file(txt_name);
        string left_str;

        while (std::getline(file, left_str)) {
            // Process str
            i = i + 1;
            left_str = root_dir + left_str;
            string right_str = regex_replace(left_str, regex("\\left"), "right");
            string name = (fmt % i).str();
            string disp_deep = result_dir + method + "/" + name;
            cout<<disp_deep<<endl;

            int radius = 10;

            float alpha_dist = 20;
            float alpha_ph = 2;

            float th_s = 0.1;



            float th_conf = 0.5;
            int im_scale = 2;

            StageOne Confidence;
            Confidence.setup(left_str, right_str, disp_deep, im_scale);
            Confidence.main_mask(alpha_ph, radius, alpha_dist, th_s, th_conf, false);
            Confidence.save_results(name);
        }
    }

}



void test_other(){

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
            string disp_deep = "/media/yzx/Elements/Dataset/Hamlyn_all/hamlyn_data/" + folder +"/result/" + name_disp;//0000.png"; //"/media/yzx/Elements/Dataset/SERV-CT-ALL/Experiment_1/Ground_truth_CT/Disparity/013.png";// "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/AAnet/001.png";
            string out_dir = "/media/yzx/Elements/Dataset/Hamlyn_all/hamlyn_data/" + folder +"/result/"+name_disp;

//    int radius_step = 10;
//    int num_radius_step = 1;
            int radius = 10;

            float alpha_dist = 20;
            float alpha_ph = 2;

            float th_s = 0.1;

            int im_scale = 2;

            float th_conf = 0.5;

            StageOne Confidence;
            Confidence.setup(left_file, right_file, disp_deep, im_scale);
            Confidence.main_mask(alpha_ph, radius, alpha_dist, th_s, th_conf, false);
            //Confidence.save_process(out_dir,name);
            Confidence.save_results(name_disp);

            //Confidence.run_D1(6);
            //Confidence.run_ph(alpha_warp);
            //Confidence.run_dist(radius,  alpha_dist);
            //Confidence.run_valid_mask();
            //Confidence.run_mask_s(th_s);
            //StageOne(left_file, right_file, disp_deep, out_dir, false);
        }
    }




}

int main(){
    //save_results_serv();
    save_results_SCARED();
    //test_serv_sample();
    //test_other();
}