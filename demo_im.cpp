#include <opencv2/opencv.hpp>
#include <iostream>
#include "./src/image_matcher.h"
#include "./src/match_pruner.h"

int main() {
	
	std::string img0_path = std::string(SOURCE_DIR) + "/data/biscuit1.jpg";
	std::string img1_path = std::string(SOURCE_DIR) + "/data/biscuit2.jpg";
	cv::Mat img0 = cv::imread(img0_path);
	cv::Mat img1 = cv::imread(img1_path);

	cv::TickMeter tm;
	tm.start();

	//=========================== Image matching ===========================//
	std::vector<cv::KeyPoint> keypts0, keypts1;
	std::vector<std::vector<cv::DMatch> > putative_matches;

	ImageMatcher image_matcher(img0, img1, FEATURE_AKAZE, MATCHER_BF, 2);
	image_matcher.GetKeyPoints(keypts0, keypts1);
	image_matcher.GetMatches(putative_matches);

	//=========================== Matches pruning ===========================//
	MatchPruner match_pruner(img0, img1, keypts0, keypts1, putative_matches, PRUNER_GMS);

	std::vector<cv::Point2f> src_points, dst_points;
	match_pruner.GetMatchedPoints(src_points, dst_points);

	tm.stop();
	std::cout << "cost time: " << tm.getTimeMilli() << " ms" << std::endl;

	//=========================== Draw results ===========================//
	cv::Mat concat_img;
	cv::hconcat(img0, img1, concat_img);

	for (size_t i = 0; i < src_points.size(); ++i) {
		cv::line(concat_img, src_points[i], cv::Point2f(float(img1.cols), 0.f)
			+ dst_points[i], CV_RGB(0, 255, 0), 1, 16);
	}
	
	cv::namedWindow("matching result", CV_WINDOW_NORMAL);
	cv::resizeWindow("matching result", 1000, 500);
	cv::imshow("matching result", concat_img);
	cv::waitKey();
	
	return 0;
}