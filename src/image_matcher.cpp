#include "image_matcher.h"
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "./lib_akaze/AKAZE.h"
#include "./libGMS/gms_matcher.h"
#include "./libLPM/lpm_matcher.h"

static void ExtractAkazeFeatures(const cv::Mat& img,
	std::vector<cv::KeyPoint>& kpts, cv::Mat& desc) {

	cv::Mat gray_img;
	if (img.channels() == 3)
		cv::cvtColor(img, gray_img, CV_BGR2GRAY);
	else
		img.copyTo(gray_img);

	cv::Mat img_32;
	gray_img.convertTo(img_32, CV_32F, 1.0 / 255.0, 0);

	AKAZEOptions options = AKAZEOptions(); // default parameters
	options.img_width = img_32.cols;
	options.img_height = img_32.rows;

	libAKAZE::AKAZE evolution(options);

	evolution.Create_Nonlinear_Scale_Space(img_32);
	evolution.Feature_Detection(kpts);
	evolution.Compute_Descriptors(kpts, desc);
}

ImageMatcher::ImageMatcher() {}

ImageMatcher::~ImageMatcher() {}

ImageMatcher::ImageMatcher(const cv::Mat& img0, const cv::Mat& img1,
	FeatureType method1, MatcherType method2, PrunerType method3,
	bool sorted_matches, int knn)
	:query_image_(img0), refer_image_(img1), feature_method_(method1),
	matcher_method_(method2), pruner_method_(method3),
	sorted_matches_(sorted_matches) {

	ExtractFeatures();
	MatchFeatures(knn);
}

void ImageMatcher::ExtractFeatures() {

	cv::Ptr<cv::FeatureDetector> detector;
	cv::Ptr<cv::DescriptorExtractor> extractor;

	int i = 0;
	int jhist = 0, jori = 0;
	cv::Mat temp;
	switch (feature_method_) {
	case FEATURE_SIFT:
		cv::initModule_nonfree();
		detector = cv::FeatureDetector::create("SIFT");
		detector->detect(query_image_, query_kpts_);
		detector->detect(refer_image_, refer_kpts_);

		extractor = cv::DescriptorExtractor::create("SIFT");
		extractor->compute(query_image_, query_kpts_, query_des_);
		extractor->compute(refer_image_, refer_kpts_, refer_des_);
		break;

	case FEATURE_SURF:
		cv::initModule_nonfree();
		detector = cv::FeatureDetector::create("SURF");
		detector->detect(query_image_, query_kpts_);
		detector->detect(refer_image_, refer_kpts_);

		extractor = cv::DescriptorExtractor::create("SURF");
		extractor->compute(query_image_, query_kpts_, query_des_);
		extractor->compute(refer_image_, refer_kpts_, refer_des_);
		break;

	case FEATURE_ORB:
		detector = cv::FeatureDetector::create("ORB");
		detector->detect(query_image_, query_kpts_);
		detector->detect(refer_image_, refer_kpts_);

		extractor = cv::DescriptorExtractor::create("ORB");
		extractor->compute(query_image_, query_kpts_, query_des_);
		extractor->compute(refer_image_, refer_kpts_, refer_des_);
		break;

	case FEATURE_AKAZE:
		ExtractAkazeFeatures(query_image_, query_kpts_, query_des_);
		ExtractAkazeFeatures(refer_image_, refer_kpts_, refer_des_);
		break;
	case FEATURE_ROOTSIFT:
		cv::initModule_nonfree();
		detector = cv::FeatureDetector::create("SIFT");
		detector->detect(query_image_, query_kpts_);
		detector->detect(refer_image_, refer_kpts_);

		extractor = cv::DescriptorExtractor::create("SIFT");
		extractor->compute(query_image_, query_kpts_, query_des_);
		extractor->compute(refer_image_, refer_kpts_, refer_des_);

		for (i = 0; i < query_des_.rows; ++i) {
			cv::normalize(query_des_.row(i), temp, 1, cv::NORM_L1);
			cv::sqrt(temp, temp);
			temp.row(0).copyTo(query_des_.row(i));
		}

		for (i = 0; i < refer_des_.rows; ++i) {
			cv::normalize(refer_des_.row(i), temp, 1, cv::NORM_L1);
			cv::sqrt(temp, temp);
			temp.row(0).copyTo(refer_des_.row(i));
		}
		break;
	case FEATURE_HALFSIFT:
		cv::initModule_nonfree();
		detector = cv::FeatureDetector::create("SIFT");
		detector->detect(query_image_, query_kpts_);
		detector->detect(refer_image_, refer_kpts_);

		extractor = cv::DescriptorExtractor::create("SIFT");
		extractor->compute(query_image_, query_kpts_, query_des_);
		extractor->compute(refer_image_, refer_kpts_, refer_des_);

		for (i = 0; i < query_des_.rows; ++i) {
			for (jhist = 0; jhist < 16; ++jhist) {
				for (jori = 0; jori < 4; ++jori) {
					temp = (query_des_.row(i).col(jhist * 8 + jori) +
						query_des_.row(i).col(jhist * 8 + jori + 4));
					temp.copyTo(query_des_.row(i).col(jhist * 8 + jori));
					temp.copyTo(query_des_.row(i).col(jhist * 8 + jori + 4));
				}
			}		
		}

		for (i = 0; i < refer_des_.rows; ++i) {
			for (jhist = 0; jhist < 16; ++jhist) {
				for (jori = 0; jori < 4; ++jori) {
					temp = (refer_des_.row(i).col(jhist * 8 + jori) +
						refer_des_.row(i).col(jhist * 8 + jori + 4));
					temp.copyTo(refer_des_.row(i).col(jhist * 8 + jori));
					temp.copyTo(refer_des_.row(i).col(jhist * 8 + jori + 4));
				}
			}
		}

		break;
	}

	// Make sure the types of the descriptors support the FlannBasedMatcher.
	query_des_.convertTo(query_des_, CV_32F);
	refer_des_.convertTo(refer_des_, CV_32F);
}

void ImageMatcher::PruneMatchesByRatioTest(
	const std::vector<std::vector<cv::DMatch> >& matches, const double ratio) {

	for (size_t i = 0; i < matches.size(); ++i) {
		if (matches[i][0].distance < matches[i][1].distance * ratio) {
			matches_.push_back(matches[i][0]);
		}
	}
}

void ImageMatcher::PruneMatchesByGMS(
	const std::vector<std::vector<cv::DMatch> >& matches) {

	std::vector<cv::DMatch> initial_matches;
	for (size_t i = 0; i < matches.size(); ++i) {
		initial_matches.push_back(matches[i][0]);
	}

	// gms matcher
	GMS_Matcher gms_matcher(query_kpts_, query_image_.size(),
		refer_kpts_, refer_image_.size(), initial_matches, cv::Size(15, 15), 6);

	std::vector<bool> labels;
	gms_matcher.GetInlierMask(labels, true, true);

	for (size_t i = 0; i < labels.size(); ++i) {
		if (labels[i]) {
			matches_.push_back(initial_matches[i]);
		}
	}

}

void ImageMatcher::PruneMatchesByLPM(
	const std::vector<std::vector<cv::DMatch> >& matches) {

	std::vector<cv::DMatch> initial_matches;
	for (size_t i = 0; i < matches.size(); ++i) {
		initial_matches.push_back(matches[i][0]);
	}

	std::vector<cv::Point2d> query_pts(initial_matches.size());
	std::vector<cv::Point2d> refer_pts(initial_matches.size());

	for (size_t i = 0; i < initial_matches.size(); ++i) {
		query_pts[i] = cv::Point2d(query_kpts_[initial_matches[i].queryIdx].pt);
		refer_pts[i] = cv::Point2d(refer_kpts_[initial_matches[i].trainIdx].pt);
	}

	// Iteration 1
	LPM_Matcher lpm0(query_pts, refer_pts, 8, 0.8, 0.2);
	cv::Mat cost0;
	std::vector<bool> labels0;
	lpm0.Match(cost0, labels0);

	// Iteration 2
	LPM_Matcher lpm1(query_pts, refer_pts, 8, 0.5, 0.2, labels0);
	cv::Mat cost1;
	std::vector<bool> labels1;
	lpm1.Match(cost1, labels1);

	for (size_t i = 0; i < labels1.size(); ++i) {
		if (labels1[i]) {
			matches_.push_back(initial_matches[i]);
		}
	}
}

void ImageMatcher::MatchFeatures(int knn) {

	std::vector<std::vector<cv::DMatch> > initial_matches;

	cv::Ptr<cv::DescriptorMatcher> matcher;
	switch (matcher_method_)
	{
	case MATCHER_BF:
		matcher = cv::DescriptorMatcher::create("BruteForce");
		break;
	case MATCHER_FLANN:
		matcher = cv::DescriptorMatcher::create("FlannBased");
		break;
	}
	matcher->knnMatch(query_des_, refer_des_, initial_matches, knn);

	double ratio = 0.8;
	switch (pruner_method_) {
	case PRUNER_RATIO:
		CV_Assert(knn >= 2);
		PruneMatchesByRatioTest(initial_matches, ratio);
		break;
	case PRUNER_GMS:
		PruneMatchesByGMS(initial_matches);
		break;
	case PRUNER_LPM:
		PruneMatchesByLPM(initial_matches);
		break;
	}

	// Sorting.
	if (sorted_matches_)
		std::sort(matches_.begin(), matches_.end());

	knn_distances_ = cv::Mat::zeros((int)matches_.size(), knn, CV_64F);
	for (size_t i = 0; i < matches_.size(); ++i) {
		query_mpts_.push_back(query_kpts_[matches_[i].queryIdx].pt);
		refer_mpts_.push_back(refer_kpts_[matches_[i].trainIdx].pt);
		/**
		 * For EVSAC
		 */
		double* pdata = (double*)knn_distances_.ptr((int)i);
		for (int j = 0; j < knn; ++j) {
			pdata[j] = initial_matches[matches_[i].queryIdx][j].distance;
		}
	}
}

void ImageMatcher::GetKeyPoints(std::vector<cv::KeyPoint>& key_points0,
	std::vector<cv::KeyPoint>& key_points1) const {
	key_points0 = query_kpts_;
	key_points1 = refer_kpts_;
}

void ImageMatcher::GetMatches(std::vector<cv::DMatch>& matches) const {
	matches = matches_;
}

void ImageMatcher::GetMatchedPoints(std::vector<cv::Point2f>& points0,
	std::vector<cv::Point2f>& points1) const {
	points0 = query_mpts_;
	points1 = refer_mpts_;
}

void ImageMatcher::GetKnnDistances(cv::Mat& knn_distances) const {
	knn_distances_.copyTo(knn_distances);
}

void ImageMatcher::DrawPointMatches(cv::Mat& image) const {
	cv::drawMatches(query_image_, query_kpts_, refer_image_, refer_kpts_,
		matches_, image);
}