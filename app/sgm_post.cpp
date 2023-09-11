//
// Created by yzx on 3/14/22.
//
#include <chrono>
#include <cstring>
#include <stdio.h>
#include <algorithm>
#include <utility>
#include <vector>
#include <cassert>
#include <chrono>
#include <cstring>
#include <math.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
using namespace std;
using namespace std::chrono;
using namespace cv;

#include <cstdint>
#include <limits>
#include <time.h>


/** \brief float无效值 */
constexpr auto Invalid_Float = std::numeric_limits<float>::infinity();

/** \brief 基础类型别名 */
typedef int8_t			sint8;		// 有符号8位整数
typedef uint8_t			uint8;		// 无符号8位整数
typedef int16_t			sint16;		// 有符号16位整数
typedef uint16_t		uint16;		// 无符号16位整数
typedef int32_t			sint32;		// 有符号32位整数
typedef uint32_t		uint32;		// 无符号32位整数
typedef int64_t			sint64;		// 有符号64位整数
typedef uint64_t		uint64;		// 无符号64位整数
typedef float			float32;	// 单精度浮点
typedef double			float64;	// 双精度浮点

class SGM_post{
public:
    /** \brief 影像宽	 */
    sint32 width_;

    /** \brief 影像高	 */
    sint32 height_;

    /** \brief 左影像视差图	*/
    float32* disp_left_;
    /** \brief 右影像视差图	*/
    float32* disp_right_;

    /** \brief 遮挡区像素集	*/
    std::vector<std::pair<int, int>> occlusions_;
    /** \brief 误匹配区像素集	*/
    std::vector<std::pair<int, int>> mismatches_;


    cv::Mat disp_ref;
    Mat disp;

    void setup( Mat disp_l, Mat disp_r){
        // 影像尺寸
        disp = disp_l;
        width_ = disp_l.cols;//width;
        height_ = disp_l.rows;
        const sint32 img_size = width_ * height_;
        disp_left_ = new float32[img_size]();
        disp_right_ = new float32[img_size]();
        disp_ref = cv::Mat(height_, width_, CV_8UC1);

        loadMat(std::move(disp_l), disp_left_);
        loadMat(std::move(disp_r),disp_right_);
    }

    double run(){
        //LRCheck();
        clock_t start, end;
        start = clock();
        RemoveSpeckles(disp_left_, width_, height_, 1, 50, Invalid_Float);
        //FillHolesInDispMap();
        MedianFilter(disp_left_, disp_left_, width_, height_, 3+10);
        end = clock();
        std::cout << "Time consumed : "
                  << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

        return (double)(end - start)/CLOCKS_PER_SEC;
    }

    void loadMat(Mat disp_l, float32* disp) const{
        for (int i = 0; i < height_; i++) {
            for (int j = 0; j < width_; j++) {
                disp[i * width_ + j] = (float) disp_l.at<unsigned short>(i, j)/256.0;
            }
        }
    }

    void ToMat(){

        for (int i = 0; i < height_; i++) {
            for (int j = 0; j < width_; j++) {
                const float disp_value = disp_left_[i * width_ + j];
                if (disp_value == 0) {//Invalid_Float
                    disp_ref.data[i * width_ + j] = 0;
                } else {
                    disp_ref.data[i * width_ + j] = static_cast<uchar>(disp_value);// (disp - min_disp) / (max_disp - min_disp) * 255
                }
            }
        }

        disp_ref.convertTo(disp_ref, CV_16UC1, 256.0);
    }

    void visualize(){

        imshow("before",disp);
        imshow("after",disp_ref);
        waitKey();

    }

    void LRCheck()
    {
        const sint32 width = width_;
        const sint32 height = height_;

        const float32& threshold = 1.0f;

        // 遮挡区像素和误匹配区像素
        auto& occlusions = occlusions_;
        auto& mismatches = mismatches_;
        occlusions.clear();
        mismatches.clear();

        // ---左右一致性检查
        for (sint32 i = 0; i < height; i++) {
            for (sint32 j = 0; j < width; j++) {
                // 左影像视差值
                auto& disp = disp_left_[i * width + j];
                if(disp == Invalid_Float){
                    mismatches.emplace_back(i, j);
                    continue;
                }

                // 根据视差值找到右影像上对应的同名像素
                const auto col_right = static_cast<sint32>(j - disp + 0.5);

                if(col_right >= 0 && col_right < width) {
                    // 右影像上同名像素的视差值
                    const auto& disp_r = disp_right_[i * width + col_right];

                    // 判断两个视差值是否一致（差值在阈值内）
                    if (abs(disp - disp_r) > threshold) {
                        // 区分遮挡区和误匹配区
                        // 通过右影像视差算出在左影像的匹配像素，并获取视差disp_rl
                        // if(disp_rl > disp)
                        //		pixel in occlusions
                        // else
                        //		pixel in mismatches
                        const sint32 col_rl = static_cast<sint32>(col_right + disp_r + 0.5);
                        if(col_rl > 0 && col_rl < width){
                            const auto& disp_l = disp_left_[i*width + col_rl];
                            if(disp_l > disp) {
                                occlusions.emplace_back(i, j);
                            }
                            else {
                                mismatches.emplace_back(i, j);
                            }
                        }
                        else{
                            mismatches.emplace_back(i, j);
                        }

                        // 让视差值无效
                        disp = Invalid_Float;
                    }
                }
                else{
                    // 通过视差值在右影像上找不到同名像素（超出影像范围）
                    disp = Invalid_Float;
                    mismatches.emplace_back(i, j);
                }
            }
        }

    }

    void FillHolesInDispMap()
    {
        const sint32 width = width_;
        const sint32 height = height_;

        std::vector<float32> disp_collects;

        // 定义8个方向
        const float32 pi = 3.1415926f;
        float32 angle1[8] = { pi, 3 * pi / 4, pi / 2, pi / 4, 0, 7 * pi / 4, 3 * pi / 2, 5 * pi / 4 };
        float32 angle2[8] = { pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4, 0, pi / 4, pi / 2, 3 * pi / 4 };
        float32 *angle = angle1;
        // 最大搜索行程，没有必要搜索过远的像素
        const sint32 max_search_length = 1.0*std::max(abs(100), abs(0));

        float32* disp_ptr = disp_left_;
        for (sint32 k = 0; k < 3; k++) {
            // 第一次循环处理遮挡区，第二次循环处理误匹配区
            auto& trg_pixels = (k == 0) ? occlusions_ : mismatches_;
            if (trg_pixels.empty()) {
                continue;
            }
            std::vector<float32> fill_disps(trg_pixels.size());
            std::vector<std::pair<sint32, sint32>> inv_pixels;
            if (k == 2) {
                //  第三次循环处理前两次没有处理干净的像素
                for (sint32 i = 0; i < height; i++) {
                    for (sint32 j = 0; j < width; j++) {
                        if (disp_ptr[i * width + j] == Invalid_Float) {
                            inv_pixels.emplace_back(i, j);
                        }
                    }
                }
                trg_pixels = inv_pixels;
            }

            // 遍历待处理像素
            for (auto n = 0u; n < trg_pixels.size(); n++) {
                auto& pix = trg_pixels[n];
                const sint32 y = pix.first;
                const sint32 x = pix.second;

                if (y == height / 2) {
                    angle = angle2;
                }

                // 收集8个方向上遇到的首个有效视差值
                disp_collects.clear();
                for (sint32 s = 0; s < 8; s++) {
                    const float32 ang = angle[s];
                    const float32 sina = float32(sin(ang));
                    const float32 cosa = float32(cos(ang));
                    for (sint32 m = 1; m < max_search_length; m++) {
                        const sint32 yy = lround(y + m * sina);
                        const sint32 xx = lround(x + m * cosa);
                        if (yy<0 || yy >= height || xx<0 || xx >= width) {
                            break;
                        }
                        const auto& disp = *(disp_ptr + yy*width + xx);
                        if (disp != Invalid_Float) {
                            disp_collects.push_back(disp);
                            break;
                        }
                    }
                }

                if(disp_collects.empty()) {
                    continue;
                }

                std::sort(disp_collects.begin(), disp_collects.end());

                // 如果是遮挡区，则选择第二小的视差值
                // 如果是误匹配区，则选择中值
                if (k == 0) {
                    if (disp_collects.size() > 1) {
                        fill_disps[n] = disp_collects[1];
                    }
                    else {
                        fill_disps[n] = disp_collects[0];
                    }
                }
                else{
                    fill_disps[n] = disp_collects[disp_collects.size() / 2];
                }
            }
            for (auto n = 0u; n < trg_pixels.size(); n++) {
                auto& pix = trg_pixels[n];
                const sint32 y = pix.first;
                const sint32 x = pix.second;
                disp_ptr[y * width + x] = fill_disps[n];
            }
        }
    }

    void MedianFilter(const float32* in, float32* out, const sint32& width, const sint32& height,
                      const sint32 wnd_size)
    {
        const sint32 radius = wnd_size / 2;
        const sint32 size = wnd_size * wnd_size;

        // 存储局部窗口内的数据
        std::vector<float32> wnd_data;
        wnd_data.reserve(size);

        for (sint32 i = 0; i < height; i++) {
            for (sint32 j = 0; j < width; j++) {
                wnd_data.clear();
                // 获取局部窗口数据
                for (sint32 r = -radius; r <= radius; r++) {
                    for (sint32 c = -radius; c <= radius; c++) {
                        const sint32 row = i + r;
                        const sint32 col = j + c;
                        if (row >= 0 && row < height && col >= 0 && col < width) {
                            wnd_data.push_back(in[row * width + col]);
                        }
                    }
                }

                // 排序
                std::sort(wnd_data.begin(), wnd_data.end());
                // 取中值
                out[i * width + j] = wnd_data[wnd_data.size() / 2];
            }
        }

    }

    void RemoveSpeckles(float32* disparity_map, const sint32& width, const sint32& height,
    const sint32& diff_insame, const uint32& min_speckle_aera, const float32& invalid_val)
    {
        assert(width > 0 && height > 0);
        if (width < 0 || height < 0) {
            return;
        }

        // 定义标记像素是否访问的数组
        std::vector<bool> visited(uint32(width*height),false);
        for(sint32 i=0;i<height;i++) {
            for(sint32 j=0;j<width;j++) {
                if (visited[i * width + j] || disparity_map[i*width+j] == invalid_val) {
                    // 跳过已访问的像素及无效像素
                    continue;
                }
                // 广度优先遍历，区域跟踪
                // 把连通域面积小于阈值的区域视差全设为无效值
                std::vector<std::pair<sint32, sint32>> vec;
                vec.emplace_back(i, j);
                visited[i * width + j] = true;
                uint32 cur = 0;
                uint32 next = 0;
                do {
                    // 广度优先遍历区域跟踪
                    next = vec.size();
                    for (uint32 k = cur; k < next; k++) {
                        const auto& pixel = vec[k];
                        const sint32 row = pixel.first;
                        const sint32 col = pixel.second;
                        const auto& disp_base = disparity_map[row * width + col];
                        // 8邻域遍历
                        for(int r=-1;r<=1;r++) {
                            for(int c=-1;c<=1;c++) {
                                if(r==0&&c==0) {
                                    continue;
                                }
                                int rowr = row + r;
                                int colc = col + c;
                                if (rowr >= 0 && rowr < height && colc >= 0 && colc < width) {
                                    if(!visited[rowr * width + colc] &&
                                       (disparity_map[rowr * width + colc] != invalid_val) &&
                                       abs(disparity_map[rowr * width + colc] - disp_base) <= diff_insame) {
                                        vec.emplace_back(rowr, colc);
                                        visited[rowr * width + colc] = true;
                                    }
                                }
                            }
                        }
                    }
                    cur = next;
                } while (next < vec.size());

                // 把连通域面积小于阈值的区域视差全设为无效值
                if(vec.size() < min_speckle_aera) {
                    for(auto& pix:vec) {
                        disparity_map[pix.first * width + pix.second] = invalid_val;
                    }
                }
            }
        }
    }

    void save(string path,int imscale){
        resize(disp_ref,disp_ref,Size(disp_ref.cols*imscale,disp_ref.rows*imscale));
        disp_ref = disp_ref* (double)imscale;
        imwrite(path,disp_ref);
    }


};




int main(){
    boost::format fmt("%03d.png");

    //int index = 1;
    double time = 0;
    int imscale =2;
    for (int index=1;index<17;index++) {
        string name = (fmt % index).str();
        string left = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereo/" + name;
        string right = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/PSMnet_right/" + name;
        string out_path = "/media/yzx/Elements/Dataset/SERV-CT-ALL/Results/LEAStereoSGM/" + name;

        Mat disp_l = imread(left, IMREAD_ANYDEPTH);
        Mat disp_r = imread(right, IMREAD_ANYDEPTH);
        resize(disp_l,disp_l,Size(disp_l.cols/imscale,disp_l.rows/imscale));
        disp_l = disp_l/(double) imscale;
        SGM_post post;
        post.setup(disp_l, disp_r);
        time = time + post.run();
        post.ToMat();
        //post.visualize();
        post.save(out_path,imscale);
    }

    cout<<"Ave time"<< time/16.0;



}
