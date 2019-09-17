#include "match_pruner.h"
#include "./libGMS/gms_matcher.h"
#include "./libLPM/lpm_matcher.h"

MatchPruner::MatchPruner(const cv::Mat& img0, const cv::Mat& img1,
	const std::vector<cv::KeyPoint>& keypts0,
	const std::vector<cv::KeyPoint>& keypts1,
	const std::vector<std::vector<cv::DMatch> >& matches, PrunerType method)
	:query_image_(img0), refer_image_(img1), query_kpts_(keypts0),
	refer_kpts_(keypts1), putative_matches_(matches), pruner_method_(method) {

	PruneMatches();
}

MatchPruner::~MatchPruner() {}

void MatchPruner::PruneMatches() {

	switch (pruner_method_) {
	case PRUNER_RATIO:
		PruneMatchesByRatioTest(0.8);
		break;
	case PRUNER_GMS:
		PruneMatchesByGMS(cv::Size(15, 15), 6);
		break;
	case PRUNER_LPM:
		PruneMatchesByLPM(8, 0.8, 0.2, 8, 0.5, 0.2);
		break;
	}
	
	int num_pruned_matches = static_cast<int>(pruned_matches_.size());
	// Matched points from the query and the reference image.
	query_mpts_.resize(num_pruned_matches);
	refer_mpts_.resize(num_pruned_matches);

	int knn = static_cast<int>(putative_matches_[0].size());
	knn_distances_ = cv::Mat::zeros(num_pruned_matches, knn, CV_64F);	
	for (int i = 0; i < num_pruned_matches; ++i) {
		query_mpts_[i] = query_kpts_[pruned_matches_[i].queryIdx].pt;
		refer_mpts_[i] = refer_kpts_[pruned_matches_[i].trainIdx].pt;

		// Compute for EVSAC
		double* pdata = (double*)knn_distances_.ptr(i);
		for (int j = 0; j < knn; ++j) {
			pdata[j] = putative_matches_[pruned_matches_[i].queryIdx][j].distance;
		}
	}
}

void MatchPruner::PruneMatchesByRatioTest(const double ratio) {

	double score;
	for (size_t i = 0; i < putative_matches_.size(); ++i) {
		std::vector<cv::DMatch> tmp_matches(putative_matches_[i]);
		score = tmp_matches[0].distance / tmp_matches[1].distance;
		if (score < ratio) {
			pruned_matches_.push_back(tmp_matches[0]);
			scores_.push_back(score);
		}
	}
}

void MatchPruner::PruneMatchesByGMS(const cv::Size& grid_size, const double alpha) {

	std::vector<cv::DMatch> initial_matches(putative_matches_.size());
	for (size_t i = 0; i < putative_matches_.size(); ++i) {
		initial_matches[i] = putative_matches_[i][0];
	}

	// GMS matcher
	GMS_Matcher gms_matcher(query_kpts_, query_image_.size(),
		refer_kpts_, refer_image_.size(), initial_matches, grid_size, alpha);

	std::vector<bool> labels;
	gms_matcher.GetInlierMask(labels, true, true);

	for (size_t i = 0; i < labels.size(); ++i) {
		if (labels[i]) {
			pruned_matches_.push_back(initial_matches[i]);
			scores_.push_back(1.0);
		}
	}
}

void MatchPruner::PruneMatchesByLPM(const int knn0, const double lambda0,
	const double tau0, const int knn1, const double lambda1, const double tau1) {

	std::vector<cv::DMatch> initial_matches(putative_matches_.size());
	for (size_t i = 0; i < putative_matches_.size(); ++i) {
		initial_matches[i] = putative_matches_[i][0];
	}

	std::vector<cv::Point2d> query_pts(initial_matches.size());
	std::vector<cv::Point2d> refer_pts(initial_matches.size());

	cv::Point2d pt;
	for (size_t i = 0; i < initial_matches.size(); ++i) {
		pt.x = (double)query_kpts_[initial_matches[i].queryIdx].pt.x;
		pt.y = (double)query_kpts_[initial_matches[i].queryIdx].pt.y;
		query_pts[i] = pt;
		pt.x = (double)refer_kpts_[initial_matches[i].trainIdx].pt.x;
		pt.y = (double)refer_kpts_[initial_matches[i].trainIdx].pt.y;
		refer_pts[i] = pt;
	}

	// Iteration 1
	LPM_Matcher lpm0(query_pts, refer_pts, knn0, lambda0, tau0);
	cv::Mat cost0;
	std::vector<bool> labels0;
	lpm0.Match(cost0, labels0);

	// Iteration 2
	LPM_Matcher lpm1(query_pts, refer_pts, knn1, lambda1, tau1, labels0);
	cv::Mat cost1;
	std::vector<bool> labels1;
	lpm1.Match(cost1, labels1);

	double* pcost = (double*)cost1.data;
	for (size_t i = 0; i < labels1.size(); ++i) {
		if (labels1[i]) {
			pruned_matches_.push_back(initial_matches[i]);
			scores_.push_back(pcost[i]);
		}
	}
}

void MatchPruner::GetMatches(std::vector<cv::DMatch>& matches) const {
	matches = pruned_matches_;
}

void MatchPruner::GetMatchedPoints(std::vector<cv::Point2f>& points0,
	std::vector<cv::Point2f>& points1) const {
	points0 = query_mpts_;
	points1 = refer_mpts_;
}

void MatchPruner::GetKnnDistances(cv::Mat& knn_distances) const {
	knn_distances_.copyTo(knn_distances);
}

void MatchPruner::GetMatchingScores(std::vector<double>& scores) const {
	scores = scores_;
}