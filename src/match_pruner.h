/****************************************************************************//**
 * @file match_pruner.h
 * @brief A c++ implementation of matches pruning.
 *
 * @author Gareth Wang <gareth.wang@hotmail.com>
 * @version 0.1
 * @date 2019-09-13
 * 
 * @copyright Copyright (c) 2019
 * 
********************************************************************************/
#ifndef _MATCH_PRUNER_H
#define _MATCH_PRUNER_H
#include <opencv2/opencv.hpp>

//! Matches pruning algorithms.
enum PrunerType{
	PRUNER_RATIO = 0,     //!< Ratio test
	PRUNER_GMS = 1,       //!< GMS
	PRUNER_LPM = 2        //!< LPM
};

/**
 * Class for matches pruning.
 */
class MatchPruner
{
public:

	/**
	 * @brief  Default constructor.
	 *  
	 */
	MatchPruner();

	/**
	 * @brief  Destructor.
	 *
	 */
	~MatchPruner();

	/**
	 * @brief  Constructor with parameters.
	 *
	 * @param  img0 [in] Query image.
	 * @param  img1 [in] Reference image.
	 * @param  keypts0 [in] Keypoints from the query image.
	 * @param  keypts1 [in] Keypoints from the reference image.
	 * @param  matches [in] Putative matches.
	 * @param  method [in] Matches pruning algorithm.
	 */
	MatchPruner(const cv::Mat& img0, const cv::Mat& img1, 
		const std::vector<cv::KeyPoint>& keypts0,
		const std::vector<cv::KeyPoint>& keypts1, 
		const std::vector<std::vector<cv::DMatch> >& matches, PrunerType method);
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
	 * @brief  Gets \f$k\f$ nearest neighbor distances.
	 *
	 * @return void
	 * @param  knn_distances [out] \f$k\f$ nearest neighbor distances.
	 */
	void GetKnnDistances(cv::Mat& knn_distances) const;

	/**
	 * @brief  Gets the matching scores.
	 *
	 * @return void
	 * @param  scores [out] Matching scores.
	 */
	void GetMatchingScores(std::vector<double>& scores) const;

private:

	/**
	 * @brief  PruneMatches
	 *
	 * @return void 
	 */
	void PruneMatches();

	/**
	 * @brief  Prunes the matches using Lowe's ratio test.
	 *
	 * @return void 
	 * @param  ratio [in] Threshold for ratio test.
	 */
	void PruneMatchesByRatioTest(const double ratio = 0.8);

	/**
	 * @brief  Prunes the matches using GMS algorithm.
	 *
	 * @return void 
	 * @param  grid_size [in] Size of the grid.
	 * @param  alpha [in] Scale factor \f$ \alpha\f$.
	 */
	void PruneMatchesByGMS(const cv::Size& grid_size = cv::Size(20, 20), 
		const double alpha = 6.0);

	/**
	 * @brief  Prunes the matches using LPM algorithm.
	 *
	 * @return void 
	 * @param  knn0 [in] Number of nearest neighbors for the first time using LPM.
	 * @param  lambda0 [in] \f$ \lambda\f$ for the first time using LPM.
	 * @param  tau0 [in] \f$ \tau\f$ for the first time using LPM.
	 * @param  knn1 [in] Number of nearest neighbors for the second time using LPM.
	 * @param  lambda1 [in] \f$ \lambda\f$ for the second time using LPM.
	 * @param  tau1 [in] \f$ \tau\f$ for the second time using LPM.
	 */
	void PruneMatchesByLPM(const int knn0 = 8, const double lambda0 = 0.8,
		const double tau0 = 0.2, const int knn1 = 8, const double lambda1 = 0.5,
		const double tau1 = 0.2);
private:
	const cv::Mat query_image_;    //!< Query image.
	const cv::Mat refer_image_;    //!< Reference image.

	const std::vector<cv::KeyPoint> query_kpts_; //!< Key points from the query image.	
	const std::vector<cv::KeyPoint> refer_kpts_; //!< Key points from the reference image.

	const std::vector<std::vector<cv::DMatch> >& putative_matches_; //!< Matches before pruning.

	PrunerType pruner_method_;    //!< Pruning methods.

	std::vector<cv::DMatch> pruned_matches_; //!< Matches after pruning.

	std::vector<cv::Point2f> query_mpts_; //!< Matched points from the query image.	
	std::vector<cv::Point2f> refer_mpts_; //!< Matched points from the reference image.

	cv::Mat knn_distances_; //!< \f$k\f$ nearest neighbor distances.

	std::vector<double> scores_;  //!< Matching scores. The lower score, the greater 
	                              //!< the possibility of correct match. 

};


#endif