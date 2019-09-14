#include <opencv2/opencv.hpp>
#include <iostream>
#include "./src/image_matcher.h"

int main(int argc, char** argv) {
	
	std::string img0_file = argv[1], img1_file = argv[2];
	cv::Mat img0 = cv::imread(img0_file);
	cv::Mat img1 = cv::imread(img1_file);

	cv::TickMeter tm;
	tm.start();

	ImageMatcher image_matcher(img0, img1, FEATURE_AKAZE,
		MATCHER_BF, PRUNER_GMS, true, 2);

	std::vector<cv::Point2f> src_points, dst_points;
	image_matcher.GetMatchedPoints(src_points, dst_points);

	tm.stop();
	std::cout << "cost time: " << tm.getTimeMilli() << " ms" << std::endl;

	cv::Mat concat_img;
	cv::hconcat(img0, img1, concat_img);

	for (size_t i = 0; i < src_points.size(); ++i) {
		cv::line(concat_img, src_points[i], cv::Point2f(float(img0.cols), 0.f)
			+ dst_points[i], CV_RGB(0, 255, 0), 1, 16);
	}

	cv::namedWindow("matching result", CV_WINDOW_NORMAL);
	cv::resizeWindow("matching result", 1000, 500);
	cv::imshow("matching result", concat_img);
	cv::waitKey();
	return 0;
}