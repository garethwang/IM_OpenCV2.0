/****************************************************************************//**
 * @file image_matcher.h
 * @brief A c++ implementation of image matching.
 *
 * The major stages of image matching consist of generating features, matching 
 * descriptors and pruning matches. The corresponding methods in each stage 
 * are available here:
 * - Generating features 
 *  + SIFT
 *  + SURF
 *  + ORB
 *  + AKAZE
 *  + ROOTSIFT
 *  + HALFSIFT
 * - Matching descriptors
 *  + BruteForce
 *  + FlannBased
 * - Pruning matches
 *  + GMS
 *  + Ratio test
 *  + LPM
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

//! Matches pruning algorithms.
enum PrunerType{
	PRUNER_GMS = 0,       //!< GMS
	PRUNER_RATIO = 1,     //!< Ratio test
	PRUNER_LPM = 2        //!< LPM
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
	 * @param  method3 [in] Pruning algorithms type.
	 * @param  sorted_matches [in] Flag indicates whether the matches are sorted.
	 * @param  knn [in] Count of best matches found per each query descriptor.
	 */
	ImageMatcher(const cv::Mat& img0, const cv::Mat& img1,
		FeatureType method1 = FEATURE_SIFT, MatcherType method2 = MATCHER_BF,
		PrunerType method3 = PRUNER_GMS, bool sorted_matches = false, int knn = 1);
	
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
	void GetMatches(std::vector<cv::DMatch>& matches) const;

	/**
	 * @brief  Gets the matched points.
	 *
	 * @return void 
	 * @param  points0 [out] Matched points from the query image.
	 * @param  points1 [out] Matched points from the reference image.
	 */
	void GetMatchedPoints(std::vector<cv::Point2f>& points0, 
		std::vector<cv::Point2f>& points1) const;

	/**
	 * @brief  Get \f$k\f$ nearest neighbor distances.
	 *
	 * @return void 
	 * @param  knn_distances [out] \f$k\f$ nearest neighbor distances.
	 */
	void GetKnnDistances(cv::Mat& knn_distances) const;

	/**
	 * @brief  Draws the matches.
	 *
	 * @return void 
	 * @param  image [in/out] Output image.
	 */
	void DrawPointMatches(cv::Mat& image) const;

private:
	/**
	 * @brief  Detects keypoints in the query and reference images and computes 
	 *         the descriptors for the corresponding keypoints.
	 *
	 * @return void 
	 */
	void ExtractFeatures();

	/**
	 * @brief  Prunes the matches using Lowe's ratio test.
	 *
	 * @return void 
	 * @param  matches [in] Matches.
	 * @param  ratio [in] Threshold for ratio test.
	 */
	void PruneMatchesByRatioTest(
		const std::vector<std::vector<cv::DMatch> >& matches,
		const double ratio = 0.8);

	/**
	 * @brief  Prunes the matches by GMS algorithm.
	 *
	 * @return void 
	 * @param  matches [in] Matches.
	 */
	void PruneMatchesByGMS(const std::vector<std::vector<cv::DMatch> >& matches);

	/**
	 * @brief  Prunes the matches by LPM algorithm.
	 *
	 * @return void 
	 * @param  matches [in/out] Matches.
	 */
	void PruneMatchesByLPM(const std::vector<std::vector<cv::DMatch> >& matches);

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
	PrunerType pruner_method_;    //!< Pruning methods.
	
	bool sorted_matches_;  //!< Flag indicates whether the matches are sorted.
	
	std::vector<cv::KeyPoint> query_kpts_; //!< Key points from the query image.	
	std::vector<cv::KeyPoint> refer_kpts_; //!< Key points from the reference image.
	
	cv::Mat query_des_; //!< Keypoint descriptors from the query image.	
	cv::Mat refer_des_; //!< Keypoint descriptors from the reference image.
	
	std::vector<cv::DMatch> matches_; //!< Matchers of keypoint descriptors.
	
	std::vector<cv::Point2f> query_mpts_; //!< Matched points from the query image.	
	std::vector<cv::Point2f> refer_mpts_; //!< Matched points from the reference image.

	cv::Mat knn_distances_; //!< \f$k\f$ nearest neighbor distances.
};
#endif
