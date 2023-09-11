// AlgorithmDriver.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "ImageIO.h"
//#include "Matrix3D.h"
//#include "MinimalSurfaceRegularization.h"
#include "OpticalFlow_Huber_L2.h"
//#include "cudaOpticalFlow_Huber_L2.h"
//#include "cudaOpticalFlow_Huber_L1.h"
#include "OpticalFlow_Huber_L1.h"
#include "Stereo_Huber_L2.h"
//#include "cudaStereo_Huber_L2.h"
//#include "cudaStereo_Huber_L1.h"
#include "Stereo_Huber_L1.h"
//#include "cudaDepth_Huber_L1.h"
//#include "Depth_Huber_L1.h"
#include <ctime>

//
//void testDepthMap()
//{
//    //intrinsicParms cameraIntrinsicParms1;
//    //cameraIntrinsicParms1.fx = 3249.0f;
//    //cameraIntrinsicParms1.fy = 3245.0f;
//    //cameraIntrinsicParms1.gamma = 0.0f;
//    //cameraIntrinsicParms1.uc = 688.07f;
//    //cameraIntrinsicParms1.vc = 514.59;
//    //cameraIntrinsicParms1.k0 = 0.0f;// -0.380835f;
//    //cameraIntrinsicParms1.k1 = 0.0f;// 0.031188f;
//    //cameraIntrinsicParms1.p0 = cameraIntrinsicParms1.p1 = 0.0f;
//
//    //intrinsicParms cameraIntrinsicParms2;
//    //cameraIntrinsicParms2.fx = 3227.0f;
//    //cameraIntrinsicParms2.fy = 3221.0f;
//    //cameraIntrinsicParms2.gamma = 0.0f;
//    //cameraIntrinsicParms2.uc = 778.45f;
//    //cameraIntrinsicParms2.vc = 543.3f;
//    //cameraIntrinsicParms2.k0 = 0.0f;// -0.339019f;
//    //cameraIntrinsicParms2.k1 = 0.0f;// 0.244582f;
//    //cameraIntrinsicParms2.p0 = cameraIntrinsicParms1.p1 = 0.0f;
//
//    //extrinsicParms cameraExtrinsicParms1;
//    //cameraExtrinsicParms1.tx = 0.0f;
//    //cameraExtrinsicParms1.ty = 0.0f;
//    //cameraExtrinsicParms1.tz = 0.0f;
//    //cameraExtrinsicParms1.thetaX = 0.0f;
//    //cameraExtrinsicParms1.thetaY = 0.0f;
//    //cameraExtrinsicParms1.thetaZ = 0.0f;
//
//    //extrinsicParms cameraExtrinsicParms2;
//    //cameraExtrinsicParms2.tx = 45.573f;
//    //cameraExtrinsicParms2.ty = -0.634f;
//    //cameraExtrinsicParms2.tz = -1.518f;
//    //cameraExtrinsicParms2.thetaX = 0.005291f;
//    //cameraExtrinsicParms2.thetaY = 6.226643f;
//    //cameraExtrinsicParms2.thetaZ = -0.002760f;
//
//    //int ImageCols = 1280;// 1024;
//    //int ImageRows = 1024;// 512;
//
//    intrinsicParms cameraIntrinsicParms1;
//    cameraIntrinsicParms1.fx = 0.5f * 1120.226f;//592.1550293f;//0.5f * 1120.226f;
//    cameraIntrinsicParms1.fy = 0.5f * 1120.226f;//592.1550293f;//0.5f * 1120.226f;
//    cameraIntrinsicParms1.gamma = 0.0f;
//    cameraIntrinsicParms1.uc = 0.5f * 642.187f; //337.59899902f; //0.5f * 642.187f;
//    cameraIntrinsicParms1.vc = 0.5f * 520.724f; //199.2460022f; //0.5f * 520.724f;
//    cameraIntrinsicParms1.k0 = 0.0f;// -0.380835f;
//    cameraIntrinsicParms1.k1 = 0.0f;// 0.031188f;
//    cameraIntrinsicParms1.p0 = cameraIntrinsicParms1.p1 = 0.0f;
//
//    intrinsicParms cameraIntrinsicParms2;
//    cameraIntrinsicParms2.fx = 0.5f * 1120.226f;//592.1550293f;
//    cameraIntrinsicParms2.fy = 0.5f * 1120.226f;//592.1550293f;
//    cameraIntrinsicParms2.gamma = 0.0f;
//    cameraIntrinsicParms2.uc = 0.5f * 642.187f; //337.59899902f;
//    cameraIntrinsicParms2.vc = 0.5f * 520.724f; //199.2460022f;
//    cameraIntrinsicParms2.k0 = 0.0f;// -0.339019f;
//    cameraIntrinsicParms2.k1 = 0.0f;// 0.244582f;
//    cameraIntrinsicParms2.p0 = cameraIntrinsicParms1.p1 = 0.0f;
//
//    extrinsicParms cameraExtrinsicParms1;
//    cameraExtrinsicParms1.tx = 0.0f;
//    cameraExtrinsicParms1.ty = 0.0f;
//    cameraExtrinsicParms1.tz = 0.0f;
//    cameraExtrinsicParms1.thetaX = 0.0f;
//    cameraExtrinsicParms1.thetaY = 0.0f;
//    cameraExtrinsicParms1.thetaZ = 0.0f;
//
//    extrinsicParms cameraExtrinsicParms2;
//    cameraExtrinsicParms2.tx = -0.464161f;
//    cameraExtrinsicParms2.ty = 0.0f;
//    cameraExtrinsicParms2.tz = 0.0f;
//    cameraExtrinsicParms2.thetaX = 0.0f;
//    cameraExtrinsicParms2.thetaY = 0.0f;
//    cameraExtrinsicParms2.thetaZ = 0.0f;
//
//    /*   Matrix3D g1, g2;
//       g1[0][0] = 0.995119f;
//       g1[0][1] = 0.00802287f;
//       g1[0][2] = -0.0983521f;
//       g1[0][3] = -0.217509f;
//
//       g1[1][0] = -0.064339f;
//       g1[1][1] = -0.702954f;
//       g1[1][2] = -0.708319f;
//       g1[1][3] = 0.216215f;
//
//       g1[2][0] = -0.0748197f;
//       g1[2][1] = 0.71119f;
//       g1[2][2] = -0.699007f;
//       g1[2][3] = 1.57562f;
//
//       g1[3][0] = 0.0f;
//       g1[3][1] = 0.0f;
//       g1[3][2] = 0.0f;
//       g1[3][3] = 1.0f;
//
//       g2[0][0] = 0.995653f;
//       g2[0][1] = 0.0232385f;
//       g2[0][2] = -0.0901938f;
//       g2[0][3] = -0.239879f;
//
//       g2[1][0] = -0.0480011f;
//       g2[1][1] = -0.70184f;
//       g2[1][2] = -0.710716f;
//       g2[1][3] = 0.215676f;
//
//       g2[2][0] = -0.0798176f;
//       g2[2][1] = 0.711956f;
//       g2[2][2] = -0.697673f;
//       g2[2][3] = 1.57992f;
//
//       g2[3][0] = 0.0f;
//       g2[3][1] = 0.0f;
//       g2[3][2] = 0.0f;
//       g2[3][3] = 1.0f;
//
//       float thetaX, thetaY, thetaZ, tx, ty, tz;
//
//       g1.RotationMatrixToEuler(thetaX, thetaY, thetaZ);
//
//       extrinsicParms cameraExtrinsicParms1;
//       cameraExtrinsicParms1.tx = g1[0][3];
//       cameraExtrinsicParms1.ty = g1[1][3];
//       cameraExtrinsicParms1.tz = g1[2][3];
//       cameraExtrinsicParms1.thetaX = thetaX;
//       cameraExtrinsicParms1.thetaY = thetaY;
//       cameraExtrinsicParms1.thetaZ = thetaZ;
//
//       g2.RotationMatrixToEuler(thetaX, thetaY, thetaZ);
//
//       extrinsicParms cameraExtrinsicParms2;
//       cameraExtrinsicParms2.tx = g2[0][3];
//       cameraExtrinsicParms2.ty = g2[1][3];
//       cameraExtrinsicParms2.tz = g2[2][3];
//       cameraExtrinsicParms2.thetaX = thetaX;
//       cameraExtrinsicParms2.thetaY = thetaY;
//       cameraExtrinsicParms2.thetaZ = thetaZ;*/
//
//    int ImageCols = 640;
//    int ImageRows = 512;
//
//
//    float** depthMap = AllocateImage<float>(ImageCols, ImageRows);
//    float** image1 = AllocateImage<float>(ImageCols, ImageRows);
//    float** image2 = AllocateImage<float>(ImageCols, ImageRows);
//
//    /*   string dirIn = "D:\\Source(2019)\\Endoscope\\Data6\\DepthMapData\\";
//       string fileIn = "kidney1_1280x1024.raw";*/
//    string dirIn = "D:\\RickSimon\\SourceCode\\DeformableIGS\\TestData\\DepthMapData\\";
//    string fileIn = "re_left_640x512.raw";// "I1_640x480.raw";//"re_left_1280x1024.raw";
//    ReadImage<float>(dirIn, fileIn, image1, ImageCols, ImageRows);
//
//    //fileIn = "kidney2_1280x1024.raw";
//    fileIn = "re_right_640x512.raw";//"I2_640x480.raw";
//    ReadImage<float>(dirIn, fileIn, image2, ImageCols, ImageRows);
//
//    cudaDepth_Huber_L1 myDepth;
//
//    cudaDepth_Huber_L1::Data inputData;
//    inputData.depth = depthMap;
//    inputData.I1 = image1;
//    inputData.I2 = image2;
//    inputData.alpha = NULL;
//    inputData.intrinsicParms1 = cameraIntrinsicParms1;
//    inputData.intrinsicParms2 = cameraIntrinsicParms2;
//    inputData.extrinsicParms1 = cameraExtrinsicParms1;
//    inputData.extrinsicParms2 = cameraExtrinsicParms2;
//
//    inputData.nCols = ImageCols;
//    inputData.nRows = ImageRows;
//
//    cudaDepth_Huber_L1::Parms inputParms;
//    //inputParms.warps = 5;// 15;
//    //inputParms.iterations = 50;// 30;
//    //inputParms.scalefactor = 0.5f;
//    //inputParms.levels = 5;
//    //inputParms.minSize = 48;
//    //inputParms.stop_level = 0;
//    //inputParms.check = 10;
//    //inputParms.Lambda = 0.1f;// 0.005f;
//    //inputParms.ref = 0;
//    //inputParms.epsilon = 0.01f; // huber epsilon
//    //inputParms.minz = 4.0f; // lower threshold for scene depth
//    //inputParms.maxz = 8.0f;
//
//    inputParms.warps = 10;
//    inputParms.outerIters = 30;
//    inputParms.innerIters = 5;
//    inputParms.scalefactor = 0.5f;
//    inputParms.levels = 6;
//    inputParms.minSize = 48;
//    inputParms.lambda = 0.1f; //weight parameter for data term
//    inputParms.theta = 0.3f; //weight parameter for (u-1)^2
//    inputParms.epsilon = 0.101f; // huber epsilon
//    inputParms.minDepth = 4.0f; // lower threshold for scene depth
//    inputParms.maxDepth = 8.0f;
//    inputParms.descriptor = cudaDepth_Huber_L1::CENSUS;
//    inputParms.bUseImageWts = false;
//
//    for (int i = 0; i < ImageRows; i++)
//    {
//        for (int j = 0; j < ImageCols; j++)
//        {
//
//            depthMap[i][j] = 5.0f;
//        }
//    }
//
//    myDepth.setup(inputData, inputParms);
//    myDepth.run();
//
//    fileIn = "DepthMap_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
//    WriteImage(dirIn, fileIn, depthMap, ImageCols, ImageRows);
//
//    FreeImage(depthMap);
//    FreeImage(image1);
//    FreeImage(image2);
//
//
//
//}

void OpticalFlowTest()
{
    int ImageCols = 640;
    int ImageRows = 512;


    float** u = AllocateImage<float>(ImageCols, ImageRows);
    float** v = AllocateImage<float>(ImageCols, ImageRows);

    float** image1 = AllocateImage<float>(ImageCols, ImageRows);
    float** image2 = AllocateImage<float>(ImageCols, ImageRows);

    float** alpha = AllocateImage<float>(ImageCols, ImageRows);
    float** uPrior = AllocateImage<float>(ImageCols, ImageRows);
    float** vPrior = AllocateImage<float>(ImageCols, ImageRows);


    string dirIn = "/home/yzx/CLionProjects/Stereo_Matching/Optical_Huber_L2/Data/";
    string fileIn = "re_left_640x512.raw";
    ReadImage<float>(dirIn, fileIn, image1, ImageCols, ImageRows);

    fileIn = "re_right_640x512.raw";
    ReadImage<float>(dirIn, fileIn, image2, ImageCols, ImageRows);

    //fileIn = "001_uPrior_720x576.raw";
    //ReadImage<float>(dirIn, fileIn, uPrior, ImageCols, ImageRows);

    //fileIn = "001_vPrior_720x576.raw";
    //ReadImage<float>(dirIn, fileIn, vPrior, ImageCols, ImageRows);

    for (int i = 0; i < ImageRows; i++)
    {
        for (int j = 0; j < ImageCols; j++)
        {
            alpha[i][j] = 0.05f;
            uPrior[i][j] *= -1.0f;
        }
    }

    //cudaOpticalFlow_Huber_L2::Parms inputParms;

    //inputParms.epsilon = 0.001f;
    //inputParms.iterations = 50;
    //inputParms.lambda = 15.0f; //patch = 150  D1 = 15.0f;
    //inputParms.levels = 6;
    //inputParms.minSize = 48;
    //inputParms.scalefactor = 0.5f;
    //inputParms.warps = 20;
    //inputParms.descriptor = cudaOpticalFlow_Huber_L2::D1;
    //inputParms.bUseImageWts = false;


    OpticalFlow_Huber_L1::Parms inputParms;

    inputParms.epsilon = 0.001f;
    inputParms.outerIters = 30;
    inputParms.innerIters = 5;
    inputParms.theta = 0.3f;
    inputParms.Lambda = 5.0f; //patch = 120  D1 = 15.0f;
    inputParms.levels = 6;
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 20;
    inputParms.descriptor = OpticalFlow_Huber_L1::D1;

    OpticalFlow_Huber_L1::Data data;
    data.alpha = NULL;
    data.I1 = image1;
    data.I2 = image2;
    data.u = u;
    data.v = v;
    data.uprior = uPrior;
    data.vprior = vPrior;
    data.nCols = ImageCols;
    data.nRows = ImageRows;

    OpticalFlow_Huber_L1 of;
    of.setup(data, inputParms);
    of.run();

    fileIn = "u_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    WriteImage(dirIn, fileIn, u, ImageCols, ImageRows);
    fileIn = "v_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    WriteImage(dirIn, fileIn, v, ImageCols, ImageRows);

    fileIn = "I2w_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    WriteImage(dirIn, fileIn, image2, ImageCols, ImageRows);

    //float flowscale = 100.0f;
    //unsigned char** R = AllocateImage<unsigned char>(ImageCols, ImageRows);
    //unsigned char** G = AllocateImage<unsigned char>(ImageCols, ImageRows);
    //unsigned char** B = AllocateImage<unsigned char>(ImageCols, ImageRows);

    //FlowToColor(u, v, R, G, B, ImageRows, ImageCols, flowscale);

    //fileIn = "R_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    //WriteImage(dirIn, fileIn, R, ImageCols, ImageRows);
    //fileIn = "G_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    //WriteImage(dirIn, fileIn, G, ImageCols, ImageRows);
    //fileIn = "B_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    //WriteImage(dirIn, fileIn, B, ImageCols, ImageRows);

    FreeImage(image1);
    FreeImage(image2);
    FreeImage(u);
    FreeImage(v);
    //FreeImage(R);
    //FreeImage(G);
    //FreeImage(B);
}

void StereoTest()
{
    int ImageCols = 640;
    int ImageRows = 512;


    float** u = AllocateImage<float>(ImageCols, ImageRows);

    float** image1 = AllocateImage<float>(ImageCols, ImageRows);
    float** image2 = AllocateImage<float>(ImageCols, ImageRows);

    float** alpha = AllocateImage<float>(ImageCols, ImageRows);
    float** uPrior = AllocateImage<float>(ImageCols, ImageRows);


    string dirIn = "/home/yzx/CLionProjects/Stereo_Matching/Optical_Huber_L2/Data/";
    string fileIn = "re_left_640x512.raw";
    ReadImage<float>(dirIn, fileIn, image1, ImageCols, ImageRows);

    fileIn = "re_right_640x512.raw";
    ReadImage<float>(dirIn, fileIn, image2, ImageCols, ImageRows);

    /*   fileIn = "001_uPrior_720x576.raw";
       ReadImage<float>(dirIn, fileIn, uPrior, ImageCols, ImageRows);*/

    for (int i = 0; i < ImageRows; i++)
    {
        for (int j = 0; j < ImageCols; j++)
        {
            alpha[i][j] = 0.05f;
            uPrior[i][j] *= -1.0f;
        }
    }

    Stereo_Huber_L2::Parms inputParms;

    inputParms.epsilon = 0.001f;
    inputParms.iterations = 50;
    inputParms.lambda = 15.0f; //patch = 150  D1 = 15.0f;
    inputParms.levels = 6;
    inputParms.minSize = 48;
    inputParms.scalefactor = 0.5f;
    inputParms.warps = 5;
    inputParms.descriptor = Stereo_Huber_L2::D1;

//    inputParms.epsilon = 0.1f;
//    inputParms.outerIters = 30;
//    inputParms.innerIters = 5;
//    inputParms.theta = 0.3f;
//    inputParms.lambda = 1.0f; //patch = 120  D1 = 15.0f;
//    inputParms.levels = 6;
//    inputParms.minSize = 48;
//    inputParms.scalefactor = 0.5f;
//    inputParms.warps = 5;//20
//    inputParms.descriptor = Stereo_Huber_L1::D1;

    Stereo_Huber_L2::Data data;
    data.alpha = NULL;
    data.I1 = image1;
    data.I2 = image2;
    data.u = u;
    data.uprior = uPrior;
    data.nCols = ImageCols;
    data.nRows = ImageRows;

    Stereo_Huber_L2 of;
    of.setup(data, inputParms);

    double Time;
    time_t start, end;
    Time = 0.0;
    time(&start);

    of.run();

    time(&end);
    Time += difftime(end, start);
    printf("OpticalFlow time:  %6.3f seconds\n\n", Time);

    fileIn = "u_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    WriteImage(dirIn, fileIn, u, ImageCols, ImageRows);

    fileIn = "I2w_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    WriteImage(dirIn, fileIn, image2, ImageCols, ImageRows);

    FreeImage(image1);
    FreeImage(image2);
    FreeImage(u);
    FreeImage(uPrior);
}


void StereoTest_L1()
{
    int ImageCols = 640;
    int ImageRows = 512;


    float** u = AllocateImage<float>(ImageCols, ImageRows);

    float** image1 = AllocateImage<float>(ImageCols, ImageRows);
    float** image2 = AllocateImage<float>(ImageCols, ImageRows);

    float** alpha = AllocateImage<float>(ImageCols, ImageRows);
    float** uPrior = AllocateImage<float>(ImageCols, ImageRows);


    string dirIn = "/home/yzx/CLionProjects/Stereo_Matching/Optical_Huber_L2/Data/";
    string fileIn = "re_left_640x512.raw";
    ReadImage<float>(dirIn, fileIn, image1, ImageCols, ImageRows);

    fileIn = "re_right_640x512.raw";
    ReadImage<float>(dirIn, fileIn, image2, ImageCols, ImageRows);

    /*   fileIn = "001_uPrior_720x576.raw";
       ReadImage<float>(dirIn, fileIn, uPrior, ImageCols, ImageRows);*/

    for (int i = 0; i < ImageRows; i++)
    {
        for (int j = 0; j < ImageCols; j++)
        {
            alpha[i][j] = 0.05f;
            uPrior[i][j] *= -1.0f;
        }
    }

    Stereo_Huber_L1::Parms inputParms;

//    inputParms.epsilon = 0.001f;
//    inputParms.iterations = 50;
//    inputParms.lambda = 15.0f; //patch = 150  D1 = 15.0f;
//    inputParms.levels = 6;
//    inputParms.minSize = 48;
//    inputParms.scalefactor = 0.5f;
//    inputParms.warps = 5;
//    inputParms.descriptor = Stereo_Huber_L2::D1;

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

    Stereo_Huber_L1::Data data;
    data.alpha = NULL;
    data.I1 = image1;
    data.I2 = image2;
    data.u = u;
    data.uprior = uPrior;
    data.nCols = ImageCols;
    data.nRows = ImageRows;

    Stereo_Huber_L1 of;
    of.setup(data, inputParms);

    double Time;
    time_t start, end;
    Time = 0.0;
    time(&start);

    of.run();

    time(&end);
    Time += difftime(end, start);
    printf("OpticalFlow time:  %6.3f seconds\n\n", Time);

    fileIn = "L_1_u_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    WriteImage(dirIn, fileIn, u, ImageCols, ImageRows);

    fileIn = "L_1_I2w_" + to_string(ImageCols) + "x" + to_string(ImageRows) + ".raw";
    WriteImage(dirIn, fileIn, image2, ImageCols, ImageRows);

    FreeImage(image1);
    FreeImage(image2);
    FreeImage(u);
    FreeImage(uPrior);
}

int main()
{


     //testDepthMap();

    //OpticalFlowTest();

    StereoTest_L1();


}

