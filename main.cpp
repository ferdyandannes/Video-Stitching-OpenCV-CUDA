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

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::cuda;
 
int main( int argc, char** argv ) {
  
  VideoCapture cap_1("seq2.mp4");
  VideoCapture cap_2("seq1.mp4");

  cv::cuda::printCudaDeviceInfo(0);

  time_t start, end;
  double fps;
  int counter = 0;
  double sec;

  time(&start);
  

  for (;;)
  {
    UMat seq_1, seq_2, hasil;
    cap_2 >> seq_2;
    cap_1 >> seq_1;

    cv::cuda::GpuMat dst2, src2;
    cv::cuda::GpuMat dst1, src1;
    cv::cuda::GpuMat temp1, temp2;

    // Convert into the GPU Mat
    src2.upload(seq_2);
    src1.upload(seq_1);
    temp2.upload(seq_2);
    temp1.upload(seq_1);



    cv::cuda::cvtColor(src2, src2, COLOR_BGR2GRAY);
    cv::cuda::cvtColor(src1, src1, COLOR_BGR2GRAY);

    SURF_CUDA detector(100);
    GpuMat keypoints1GPU, keypoints2GPU;
    GpuMat descriptors1GPU, descriptors2GPU;
    vector<KeyPoint> keypoints_tmp_CPU1, keypoints_tmp_CPU2;
    detector(src1, GpuMat(), keypoints1GPU, descriptors1GPU);
    detector(src2, GpuMat(), keypoints2GPU, descriptors2GPU);

    detector.downloadKeypoints(keypoints1GPU, keypoints_tmp_CPU1);
    detector.downloadKeypoints(keypoints2GPU, keypoints_tmp_CPU2);

    Ptr< cuda::DescriptorMatcher > matcher = cuda::DescriptorMatcher::createBFMatcher();
    vector< vector< DMatch> > matches;
    matcher->knnMatch(descriptors1GPU, descriptors2GPU, matches, 2);

    vector< KeyPoint > keypoints_scene, keypoints_object;
    detector.downloadKeypoints(keypoints2GPU, keypoints_scene);
    detector.downloadKeypoints(keypoints1GPU, keypoints_object);

    std::vector< DMatch > good_matches;
    for (int k = 0; k < std::min(keypoints_object.size() - 1, matches.size()); k++)
    {
      if ((matches[k][0].distance < 0.6*(matches[k][1].distance)) &&
        ((int)matches[k].size() <= 2 && (int)matches[k].size() > 0))
      {
        // take the first result only if its distance is smaller than 0.6*second_best_dist
        // that means this descriptor is ignored if the second distance is bigger or of similar
        good_matches.push_back(matches[k][0]);
      }
    }

    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for (int i = 0; i < good_matches.size(); i++)
    {
      //-- Get the keypoints from the good matches
      obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
      scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }


    Mat H = findHomography(obj, scene, RANSAC);

    ///////////////////////////////////////////////////
    std::vector<Point2f> imageCorners(4);
    imageCorners[0] = cvPoint(0, 0);
    imageCorners[1] = cvPoint(seq_1.cols, 0);
    imageCorners[2] = cvPoint(seq_1.cols, seq_1.rows);
    imageCorners[3] = cvPoint(0, seq_1.rows);
    std::vector<Point2f> projectedCorners(4);

    perspectiveTransform(imageCorners, projectedCorners, H);
    //cout << "4 = " << round(projectedCorners[2].x) << endl;
    ///////////////////////////////////////////////////
    
    GpuMat result, H_gpu;
    H_gpu.upload(H);
    UMat result_mat;

    //cout << "H = " << H << endl;
    //cv::cuda::warpPerspective(temp1, result, H, cv::Size(projectedCorners[2].x, temp1.rows));
    cv::cuda::warpPerspective(temp1, result, H, cv::Size(1350, temp1.rows));
    GpuMat half(result, cv::Rect(0, 0, temp2.cols, temp2.rows));
    temp2.copyTo(half);

    result.download(result_mat);
    imshow("Result Image", result_mat);

    time(&end);
    ++counter;
    sec = difftime(end, start);
    fps = counter / sec;
    cout << "Fps = " << fps << endl;


    if ((char)waitKey(33) >= 0) break;
  }
  
  cap_1.release();
  cap_2.release();
  return 0;
}