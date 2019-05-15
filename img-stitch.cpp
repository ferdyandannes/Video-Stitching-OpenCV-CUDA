// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
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
#include <chrono>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::cuda;

int main()
{
	VideoCapture cap_1("seq86_2.mp4");
	VideoCapture cap_2("seq86_1.mp4");

	/*cv::cuda::printCudaDeviceInfo(0);
	const std::string fname = "seq86_2.mp4";
	cv::cuda::GpuMat d_frame;
	cv::Ptr<cv::cudacodec::VideoReader> cap_1 = cv::cudacodec::createVideoReader(fname);*/

	int i = 0;

	for (;;)
	{
		//cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
		auto start = std::chrono::system_clock::now();
		UMat seq_1, seq_2, hasil;
		cap_2 >> seq_2;
		cap_1 >> seq_1;

		cv::cuda::GpuMat dst2, src2;
		cv::cuda::GpuMat dst1, src1;
		cv::cuda::GpuMat temp1, temp2;

		// Convert ke GpuMat
		src2.upload(seq_2);
		src1.upload(seq_1);
		temp2.upload(seq_2);
		temp1.upload(seq_1);

		//cv::cuda::GpuMat dst, src;
		//src.upload(src_host);

		cv::cuda::cvtColor(src2, src2, COLOR_BGR2GRAY);
		cv::cuda::cvtColor(src1, src1, COLOR_BGR2GRAY);

		SURF_CUDA detector;
		GpuMat keypoints1GPU, keypoints2GPU;
		GpuMat descriptors1GPU, descriptors2GPU;
		vector<KeyPoint> keypoints_tmp_CPU1, keypoints_tmp_CPU2;
		detector(src1, GpuMat(), keypoints1GPU, descriptors1GPU);
		detector(src2, GpuMat(), keypoints2GPU, descriptors2GPU);

		detector.downloadKeypoints(keypoints1GPU, keypoints_tmp_CPU1);
		detector.downloadKeypoints(keypoints2GPU, keypoints_tmp_CPU2);

		// matching descriptors
		Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(detector.defaultNorm());
		vector<DMatch> matches;
		matcher->match(descriptors1GPU, descriptors2GPU, matches);

		double max_dist = 0; double min_dist = 2000;

		//-- Quick calculation of max and min distances between keypoints
		for (int i = 0; i < descriptors1GPU.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		//printf("-- Max dist : %f \n", max_dist);
		//printf("-- Min dist : %f \n", min_dist);

		std::vector< DMatch > good_matches;

		for (int i = 0; i < descriptors1GPU.rows; i++)
		{
			if (matches[i].distance < 50 * min_dist)
			{
				good_matches.push_back(matches[i]);
			}
		}

		//-- Localize the object
		std::vector<Point2f> obj;
		std::vector<Point2f> scene;

		for (int i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(keypoints_tmp_CPU1[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints_tmp_CPU2[good_matches[i].trainIdx].pt);
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

		cv::cuda::warpPerspective(temp1, result, H, cv::Size(1350, temp1.rows));
		GpuMat half(result, cv::Rect(0, 0, temp2.cols, temp2.rows));
		temp2.copyTo(half);

		result.download(result_mat);
		imshow("Result Image", result_mat);


		// Konvert ke UMat
		//dst2.download(hasil);
		//cv::imshow("GPU", hasil);

		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = end - start;
		std::cout << "Time to process last frame (seconds): " << diff.count()
			<< " FPS: " << 1.0 / diff.count() << endl;

		i++;
		if ((char)cv::waitKey(33) >= 0) break;
	}
	//cap_1.release();
	cap_2.release();
	destroyAllWindows();
	waitKey(1);
	return 0;
}