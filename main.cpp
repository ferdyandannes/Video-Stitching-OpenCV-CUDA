#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <iostream>
#include <stdio.h>
//#include <chrono>
#include <vector>
#include <string>
#include <time.h>
#include <ctime>

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::cuda;

 
int main( int argc, char** argv ) {
  
  /*VideoCapture cap_1("right.mp4");
  VideoCapture cap_2("left.mp4");*/

  /*VideoCapture cap_1("seq86_2.mp4");
  VideoCapture cap_2("seq86_1.mp4");*/

  VideoCapture cap_1("wb1.mp4");
  VideoCapture cap_2("wb2.mp4");

  cv::cuda::printCudaDeviceInfo(0);

  detail::MultiBandBlender blender(false, 5);

  // DECLARE ROI
  int width = cap_1.get(CV_CAP_PROP_FRAME_WIDTH);
  int height = cap_1.get(CV_CAP_PROP_FRAME_HEIGHT);
  int x1 = width / 4;
  int x2 = width - x1;
  Rect Rec2(x1, 0, x2, height);
  Rect Rec1(x1, 0, x2, height);

  // FOR MEASURE THE FPS
  long frameCounter = 0;
  std::time_t timeBegin = std::time(0);
  int tick = 0;
  int hit = 0;

  // DECLARE SOME VARIABLE IN HERE
  Ptr<cuda::ORB> orb = cuda::ORB::create(9000);
  GpuMat keypoints1GPU, keypoints2GPU;
  GpuMat descriptors1GPU, descriptors2GPU;
  vector< KeyPoint > keypoints_scene, keypoints_object;
  Ptr< cuda::DescriptorMatcher > matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
  vector< vector< DMatch> > matches;

  // FOR BLENDING
  double alpha = 0.5; 
  double beta = 0;

  for (;;)
  {
    UMat seq_1, seq_2, hasil;
    cap_2 >> seq_2;
    cap_1 >> seq_1;

    cv::cuda::GpuMat dst2, src2;
    cv::cuda::GpuMat dst1, src1;
    cv::cuda::GpuMat temp1, temp2;

    // Convert into the GPU Mat
    temp2.upload(seq_2);
    temp1.upload(seq_1);

    // USING ROI
    /*UMat Roi2 = seq_2(Rec2);
    UMat Roi1 = seq_1(Rec1);
    src2.upload(Roi2);
    src1.upload(Roi1);
    temp2.upload(seq_2);
    temp1.upload(seq_1);*/

    // NO ROI
    src2.upload(seq_2);
    src1.upload(seq_1);

    cv::cuda::cvtColor(src2, src2, COLOR_BGR2GRAY);
    cv::cuda::cvtColor(src1, src1, COLOR_BGR2GRAY);

    // CHECK ROI
   /* UMat a, b;
    src2.download(a);
    imshow("Result a", a);
    src1.download(b);
    imshow("Result b", b);*/

    orb->detectAndComputeAsync(src1, noArray(), keypoints1GPU, descriptors1GPU, false);
    orb->detectAndComputeAsync(src2, noArray(), keypoints2GPU, descriptors2GPU, false);
    orb->convert(keypoints1GPU, keypoints_object);
    orb->convert(keypoints2GPU, keypoints_scene);

    //cout << "KPTS = " << keypoints_scene.size() << endl;

    matcher->knnMatch(descriptors1GPU, descriptors2GPU, matches, 2);

    std::vector< DMatch > good_matches;

    for (int z = 0; z < std::min(keypoints_object.size() - 1, matches.size()); z++)
    {
      if (matches[z][0].distance < 0.75*(matches[z][1].distance))
      {
        good_matches.push_back(matches[z][0]);
      }
    }

    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for (int y = 0; y < good_matches.size(); y++)
    {
      obj.push_back(keypoints_object[good_matches[y].queryIdx].pt);
      scene.push_back(keypoints_scene[good_matches[y].trainIdx].pt);
    }

    //cout << "Match points = " << good_matches.size() << endl;

    Mat H = findHomography(obj, scene, RANSAC);
    
    GpuMat result, H_gpu, store;
    H_gpu.upload(H);
    UMat result_mat, cek_mat, dst;

    // cv::cuda::warpPerspective(temp1, result, H, cv::Size(temp2.cols + temp1.cols, temp2.rows));
    //cv::cuda::warpPerspective(temp1, result, H, cv::Size(1400, temp2.rows));
    cv::cuda::warpPerspective(temp1, result, H, cv::Size(2000, temp2.rows));
    //result.copyTo(store);
    GpuMat half(result, cv::Rect(0, 0, temp2.cols, temp2.rows));
    temp2.copyTo(half);

    result.download(result_mat);

    //cout << result_mat.size() << endl;

    imshow("Result Image", result_mat);

    // ALPHA BLEND ###################################################
    // ROI
    /*store.download(result_mat);
    temp2.download(cek_mat);
    int min_x = ( result_mat.cols - cek_mat.cols)/2;
    int min_y = ( result_mat.rows - cek_mat.rows)/2;

    cout << min_x << " " << min_y << endl;

    Rect roi(min_x, min_y, cek_mat.cols, cek_mat.rows);
    UMat out_image = result_mat.clone();
    UMat A_roi = result_mat(roi);
    UMat out_image_roi = out_image(roi);
    cv::addWeighted(A_roi, alpha, cek_mat, 1 - alpha, 0.0, out_image_roi);*/

    //beta = (1.0 - alpha);
    //cv::addWeighted(result_mat, alpha, cek_mat, beta, 0.0, dst);

    //alphaBlend(result_mat, cek_mat, alpha, dst);

    //imshow("Result Image", result_mat);
    //imshow("Result Image2", cek_mat);
    //imshow("Resultssss ssImage", out_image_roi);
    // ##############################################################






    /*drawKeypoints(seq_2, keypoints_scene, hasil);
    imshow("kpt", hasil);*/

    frameCounter++;
    hit++;
    std::time_t timeNow = std::time(0) - timeBegin;
    if (timeNow - tick >= 1)
    {
        tick++;
        cout << "Frame = " << hit << "  Frames per second: " << frameCounter << endl << endl;
        frameCounter = 0;
    }

    //cout << "Counter = " << hit << endl;
    imwrite("a.jpg", result_mat);

    //waitKey(0);
    if ((char)waitKey(33) >= 0) break;
   
    keypoints_object.clear();
    keypoints_scene.clear();
  }
  matcher.release();
  //orb.releaseMemory();
  cap_1.release();
  cap_2.release();
  return 0;
}
