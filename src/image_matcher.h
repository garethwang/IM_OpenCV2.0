/****************************************************************************//**
 * @file image_matcher.h
 * @brief A c++ implementation of image matching.
 *
 * @author Gareth Wang <gareth.wang@hotmail.com>
 * @version 0.1
 * @date 2018-05-19
 * 
 * @copyright Copyright (c) 2018
 * 
********************************************************************************/
#ifndef _IMAGE_MATCHER_H_
#define _IMAGE_MATCHER_H_
#include <opencv2/opencv.hpp>

//! Types of a feature detector and a descriptor extractor.
enum FeatureType{
	FEATURE_SIFT = 0,         //!< SIFT
	FEATURE_SURF = 1,         //!< SURF
	FEATURE_ORB = 2,          //!< ORB
	FEATURE_AKAZE = 3,        //!< AKAZE
	FEATURE_ROOTSIFT = 4,     //!< ROOTSIFT
	FEATURE_HALFSIFT = 5      //!< HALFSIFT
};

//! Matcher types.
enum MatcherType{
	MATCHER_BF = 0,        //!< BruteForce-L2
	MATCHER_FLANN = 1      //!< FlannBased
};

/**
 * Class for image matching.
 */
class ImageMatcher {
public:
	/**
	 * @brief  Default constructor.
	 *
	 */
	ImageMatcher();

	/**
	 * @brief  Destructor.
	 *
	 */
	~ImageMatcher();

	/**
	 * @brief  Constructor with parameters.
	 *
	 * @param  img0 [in] Query image.
	 * @param  img1 [in] Reference image.
	 * @param  method1 [in] Feature detector type.
	 * @param  method2 [in] Descriptor matcher type.
	 * @param  knn [in] Count of best matches found per each query descriptor.
	 */
	ImageMatcher(const cv::Mat& img0, const cv::Mat& img1,
		FeatureType method1 = FEATURE_SIFT, MatcherType method2 = MATCHER_BF,
		const int knn = 1);

	/**
	 * @brief  Gets the keypoints from both the query and reference image.
	 *
	 * @return void
	 * @param  key_points0 [out] Keypoints from the query image.
	 * @param  key_points1 [out] Keypoints from the reference image.
	 */
	void GetKeyPoints(std::vector<cv::KeyPoint>& key_points0,
		std::vector<cv::KeyPoint>& key_points1) const;

	/**
	 * @brief  Gets the matches after pruning bad correspondences.
	 *
	 * @return void
	 * @param  matches [out] Matches.
	 */
	void GetMatches(std::vector<std::vector<cv::DMatch> >& matches) const;

private:
	/**
	 * @brief  Detects keypoints in the query and reference images and computes
	 *         the descriptors for the corresponding keypoints.
	 *
	 * @return void
	 */
	void ExtractFeatures();

	/**
	 * @brief  Finds the best matches and rejects false matches.
	 *
	 * @return void
	 * @param  knn [in] Count of best matches found per each query descriptor.
	 */
	void MatchFeatures(int knn);

private:
	cv::Mat query_image_;    //!< Query image.
	cv::Mat refer_image_;    //!< Reference image.

	FeatureType feature_method_;  //!< Local Features.
	MatcherType matcher_method_;  //!< Matching methods.

	std::vector<cv::KeyPoint> query_kpts_; //!< Key points from the query image.	
	std::vector<cv::KeyPoint> refer_kpts_; //!< Key points from the reference image.

	cv::Mat query_des_; //!< Keypoint descriptors from the query image.	
	cv::Mat refer_des_; //!< Keypoint descriptors from the reference image.

	std::vector<std::vector<cv::DMatch> > matches_; //!< Matchers of keypoint descriptors.
};
#endif