/****************************************************************************//**
 * @file gms_matcher.h
 *
 * @brief The c++ implementation of GMS (Grid-based Motion Statistics) 
 *        feature matching strategy.
 * 
 * Read the paper "GMS: Grid-Based Motion Statistics for Fast, Ultra-Robust 
 * Feature Correspondence" by JiaWang Bian, et al for details.
 *
 * The original version was written by Jiawang Bian, which is modified by 
 * Gareth Wang. The modifications are as following:
 * - Split the original file into the head file and the source file.
 * - Comment the code in detail. 
 * - Modify the constructor and add two parameters for the \f$\alpha\f$ value 
 * and the size of the grid.
 * 
 * @author Jiawang Bian
 * @author Gareth Wang <gareth.wang@hotmail.com>
 * @date 2018-05-19
 * 
 * @copyright Copyright (c) 2018
 * 
********************************************************************************/
#ifndef _GMS_MATCHER_H_
#define _GMS_MATCHER_H_
#include <opencv2/opencv.hpp>

/**
 * Class for GMS algorithm.
 */
class GMS_Matcher {
public:
	/**
	 * @brief  Default constructor.
	 */
	GMS_Matcher();

	/**
	 * @brief  Destructor.
	 */
	~GMS_Matcher();
	
	/**
	 * @brief  Constructor with parameters.
	 *
	 * @param  vkp1 [in] The keypoints from the left image.
	 * @param  size1 [in] The size of the left image.
	 * @param  vkp2 [in] The keypoints from the right image.
	 * @param  size2 [in] The size of the right image.
	 * @param  vDMatches [in] The nearest neighbor matches.
	 * @param  grid_size [in] The size of the grid.
	 * @param  alpha [in] The factor \f$\alpha\f$ of the desired threshold.
	 */
	GMS_Matcher(const std::vector<cv::KeyPoint>& vkp1, const cv::Size& size1,
		const std::vector<cv::KeyPoint>& vkp2, const cv::Size& size2,
		const std::vector<cv::DMatch>& vDMatches, const cv::Size& grid_size,
		const double alpha);
	
	/**
	 * @brief  Gets the mask of inliers/outliers.
	 *
	 * @return int Number of inliers.
	 * @param  vbInliers [out] Output vector that contains the Boolean values 
	 *                         indicating whether the matches are true.
	 * @param  WithScale [in] Parameter defining whether scale invariance 
	 *                        should be enabled.
	 * @param  WithRotation [in] Parameter defining whether rotational 
	 *                           invariance should be enabled.
	 */
	int GetInlierMask(std::vector<bool>& vbInliers, bool WithScale = false,
		bool WithRotation = false);

private:
	
	/**
	 * @brief  Normalizes the keypoints to the range from 0 to 1.
	 *
	 * @return void 
	 * @param  kp [in] The keypoints.
	 * @param  size [in] The size of the image.
	 * @param  npts [out] The normalized keypoints.
	 */
	void NormalizePoints(const std::vector<cv::KeyPoint>& kp,
		const cv::Size& size, std::vector<cv::Point2f>& npts);

	/**
	 * @brief  Converts matches from the OpenCV format to the user-defined format.
	 *
	 * @return void 
	 * @param  vDMatches [in] The matches of the OpenCV format.
	 * @param  vMatches [out] The matches of the user-defined format.
	 */
	void ConvertMatches(const std::vector<cv::DMatch>& vDMatches,
		std::vector<std::pair<int, int> >& vMatches);

	/**
	 * @brief  Gets the index of the cell for the point from the left image.
	 *
	 * @return int Index.
	 * @param  pt [in] Point.
	 * @param  type [in] Grid patterns shift flag. 
	 */
	int GetGridIndexLeft(const cv::Point2f& pt, int type);

	/**
	 * @brief  Gets the index of the cell for the point from the right image.
	 *
	 * @return int Index.
	 * @param  pt [in] Point.
	 */
	int GetGridIndexRight(const cv::Point2f& pt);

	/**
	 * @brief  Assigns the matches to the cell-pairs.
	 *
	 * @return void 
	 * @param  GridType [in] Grid patterns shift flag.
	 */
	void AssignMatchPairs(int GridType);

	/**
	 * @brief  Verifies the cell-pairs and divide them into true and false set.
	 *
	 * @return void 
	 * @param  RotationType [in] Rotation type.
	 */
	void VerifyCellPairs(int RotationType);

	/**
	 * @brief  Gets the indices of the grid-cell and its eight neighborhoods.
	 *
	 * @return std::vector<int> Indices of the cell and the 8-neighborhoods.
	 * @param  idx [in] Index of the cell.
	 * @param  GridSize [in] Grid size.
	 */
	std::vector<int> GetNB9(const int idx, const cv::Size& GridSize);

	/**
	 * @brief  Gets the neighborhoods of all grid-cells.
	 *
	 * @return void 
	 * @param  neighbor [out] Neighborhoods of all grid-cells, \f$N\times9\f$.
	 * @param  GridSize [in] Grid size.
	 */
	void InitializeNeighbors(cv::Mat& neighbor, const cv::Size& GridSize);

	/**
	 * @brief  Sets the scale and compute the grid size of the right image.
	 *
	 * @return void 
	 * @param  Scale [in] Scale.
	 */
	void SetScale(int Scale);

	
	/**
	 * @brief  Gets the mask of inliers/outliers with the fixed rotation type.
	 *
	 * @return int Number of inliers.
	 * @param  RotationType [in] Rotation type.
	 */
	int Run(int RotationType);
private:
	
	std::vector<cv::Point2f> mvP1;  //!< Normalized points from the left image.
	std::vector<cv::Point2f> mvP2;  //!< Normalized points from the right image.

	std::vector<std::pair<int, int> > mvMatches; //!< Matches.

	size_t mNumberMatches; //!< Number of matches.

	cv::Size mGridSizeLeft;  //!< Grid size of the left image.
	cv::Size mGridSizeRight; //!< Grid size of the right image.
	int mGridNumberLeft;  //!< Number of grid-cells of the left image.
	int mGridNumberRight; //!< Number of grid-cells of the right image.

	/**
	 * x	  : left grid idx.\n
	 * y      : right grid idx.\n
	 * value  : how many matches from idx_left to idx_right.
	 */
	cv::Mat mMotionStatistics; //!< The number of matches between cells.

	std::vector<int> mNumberPointsInPerCellLeft; /**< Number of the points from the 
	                                             grid-cells of the left image. */
	/**
	 * Index  : grid_idx_left.\n
	 * Value  : grid_idx_right.
	 */	
	std::vector<int> mCellPairs; //!< Indices of the cell-pairs.

	/**
	 * Every matches has a cell-pair.\n 
	 * first  : grid_idx_left.\n
	 * second : grid_idx_right.
	 */
	std::vector<std::pair<int, int> > mvMatchPairs; //!< Cell-pairs of the matches.

	std::vector<bool> mvbInlierMask; //!< Mask of inliers/outliers.

	cv::Mat mGridNeighborLeft;  //!< Neighborhoods of the grid-cells of the left image.	
	cv::Mat mGridNeighborRight; //!< Neighborhoods of the grid-cells of the right image.
	
	double mAlpha;  /**< The factor \f$\alpha\f$ of the desired threshold 
	                /to divide cell-pairs into true and false sets. */
};
#endif
